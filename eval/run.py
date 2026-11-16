"""评测入口：一次算全检索、重排、拒答、引用四类指标。

用法：

    uv run python -m eval.run                      # 不调用模型，快且可复现
    uv run python -m eval.run --with-llm           # 额外测引用精度（8 次真实调用）
    uv run python -m eval.run --k 3 --out eval/baseline.json

指标口径：

    hit_rate_at_recall  金标块是否落在粗召回 top-(recall_k) 内 —— 整条链路的天花板
    hit_rate_at_k       重排后 top-k 是否含金标块
    mrr_at_k            首个命中排名的倒数均值（第 1 名记 1，第 3 名记 1/3）
    refusal_accuracy    无关题被正确拒掉的比例
    false_refusal_rate  相关题被误拒的比例
    citation_precision  回答里的引用指向 gold_sources 的比例（需 --with-llm）

★ 默认不调用模型，理由是**可复现性**：
  检索与重排是确定性的，同一条题目跑两次结果一致；生成不是。
  两类指标混进同一个数字，回归对比时分不清是「索引变了」还是「采样变了」。
  `--with-llm` 单独打开生成类指标，并在结果里留下 `with_llm: true` 标记。

★ `false_refusal_rate` 必须与 `refusal_accuracy` 一起看：
  只报后者的话，把阈值调到无穷大就能得满分，而系统已经不能回答任何问题。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.client.llm import build_client
from app.core.config import Settings
from app.rag.rerank import Reranker, build_reranker
from app.rag.retriever import retrieve_with_score
from app.schema.rag import RagRequest
from app.service.rag_service import RagService

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCATE = ROOT / "eval" / "gold" / "rag_locate.jsonl"
DEFAULT_REFUSE = ROOT / "eval" / "gold" / "rag_refuse.jsonl"
DEFAULT_OUT = ROOT / "eval" / "baseline.json"
CORPUS = ROOT / "data" / "corpus" / "chunks.jsonl"


# ══════════════════════ 一、读题与判定 ══════════════════════


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到金标文件：{path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def is_hit(doc: Document, case: dict[str, Any]) -> bool:
    """命中判定：来源在 gold_sources 内，且正文含任一金标符号。

    两个条件缺一不可 —— 只看来源，同文件几十个块都算命中，指标会虚高；
    只看符号，同名符号在多个文件出现时无法定位。
    """
    meta = doc.metadata or {}
    return meta.get("source") in case["gold_sources"] and any(
        symbol in doc.page_content for symbol in case["gold_symbols"]
    )


def first_hit_rank(pairs: list[tuple[Document, float]], case: dict[str, Any]) -> int:
    """返回首个命中的名次（从 1 开始）；未命中返回 0。"""
    for rank, (doc, _score) in enumerate(pairs, start=1):
        if is_hit(doc, case):
            return rank
    return 0


def rank_candidates(
    question: str,
    cfg: Settings,
    reranker: Reranker,
) -> tuple[list[tuple[Document, float]], list[tuple[Document, float]]]:
    """返回 (粗召回列表, 重排后 top_n 列表)，两者各自有序。

    这里与 `RagService` 走同一套调用，但不经 service：评测需要看到**完整**的
    召回列表（算 Hit@recall 与 MRR），而 service 只返回最终答案。
    """
    pairs = retrieve_with_score(question, k=cfg.rag_recall_k)
    documents = [doc for doc, _ in pairs]
    if not documents:
        return pairs, []

    scores = reranker.score(question, [doc.page_content for doc in documents])
    if len(scores) != len(documents):
        raise RuntimeError(f"重排分数个数与候选不一致：{len(scores)} != {len(documents)}")

    ranked = sorted(zip(documents, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return pairs, ranked[: cfg.rag_top_k]


# ══════════════════════ 二、跑题 ══════════════════════


def _resolve(raw: str) -> Path:
    """相对路径按项目根解析，与 scripts/ 下的脚本保持一致。

    ★ 不用 `Path(raw)` 直接当结果：`--out eval/x.json` 会得到相对路径，
      后面拿它做 `relative_to(ROOT)` 会抛 ValueError —— 文件已经写成功了，
      却因为打印路径而让整次评测以异常退出。
    """
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


async def evaluate(args: argparse.Namespace) -> int:
    cfg = Settings()
    reranker = build_reranker(cfg)
    locate = load_cases(_resolve(args.locate))
    refuse = load_cases(_resolve(args.refuse))
    if not locate or not refuse:
        print("金标为空，无法评测", file=sys.stderr)
        return 2

    print(
        f"题目 {len(locate)} 条相关 / {len(refuse)} 条无关"
        f"｜K={args.k}｜recall_k={cfg.rag_recall_k}"
        f"｜阈值 {cfg.rag_min_rerank_score:.2f}"
    )
    print(f"embed={cfg.rag_embed_model}｜rerank={cfg.rag_rerank_model}\n")

    # 相关题：先算排名（不产生模型调用）。
    rows: list[dict[str, Any]] = []
    hit_recall = 0
    hit_top = 0
    reciprocal = 0.0
    false_refusals = 0

    for case in locate:
        pairs, ranked = rank_candidates(case["question"], cfg, reranker)
        rank_recall = first_hit_rank(pairs, case)
        rank_top = first_hit_rank(ranked, case)
        # Hit@k 只看前 k 名。`ranked` 长度是 top_n（可能大于 k），
        # 落在 k 之外的命中不算数 —— 否则 --k 改小也不影响结果。
        hit_at_k = bool(rank_top) and rank_top <= args.k
        top_score = float(ranked[0][1]) if ranked else None
        # 与 service 内是同一条判据：最高分低于阈值即拒答。
        refused = top_score is None or top_score < cfg.rag_min_rerank_score

        hit_recall += 1 if rank_recall else 0
        if hit_at_k:
            hit_top += 1
            reciprocal += 1.0 / rank_top
        false_refusals += 1 if refused else 0

        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "rank_recall": rank_recall,
                "rank_top": rank_top,
                "hit_at_k": hit_at_k,
                "top_score": _round(top_score),
                "refused": refused,
                "recalled": len(pairs),
                "reranked": len(ranked),
            }
        )

    # 无关题：走 service 的真实决策路径。拒答发生在调用模型之前，因此不产生费用。
    svc = RagService(
        build_client(cfg),
        reranker,
        recall_k=cfg.rag_recall_k,
        top_n=cfg.rag_top_k,
        min_score=cfg.rag_min_rerank_score,
    )
    refuse_rows: list[dict[str, Any]] = []
    refusal_correct = 0
    for case in refuse:
        answer = await svc.ask(RagRequest(question=case["question"]))
        refusal_correct += 1 if answer.refused else 0
        refuse_rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "refused": answer.refused,
                "reason": answer.reason,
                "top_score": _round(answer.top_score),
                "recalled": answer.recalled,
            }
        )

    metrics: dict[str, Any] = {
        "hit_rate_at_recall": _round(hit_recall / len(locate)),
        "hit_rate_at_k": _round(hit_top / len(locate)),
        "mrr_at_k": _round(reciprocal / len(locate)),
        "refusal_accuracy": _round(refusal_correct / len(refuse)),
        "false_refusal_rate": _round(false_refusals / len(locate)),
        "citation_precision": None,
        "answers_without_citation": None,
        "invalid_citations": None,
    }

    if args.with_llm:
        await _measure_citations(svc, locate, rows, metrics)

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "with_llm": bool(args.with_llm),
        "k": args.k,
        "config": {
            "embed_model": cfg.rag_embed_model,
            "rerank_model": cfg.rag_rerank_model,
            "recall_k": cfg.rag_recall_k,
            "top_n": cfg.rag_top_k,
            "min_rerank_score": cfg.rag_min_rerank_score,
        },
        "corpus": {"chunks": _count_chunks(), "index_dir": cfg.rag_persist_dir},
        "metrics": metrics,
        "cases": {"locate": rows, "refuse": refuse_rows},
        "git": {"commit": _git("rev-parse", "--short", "HEAD"), "dirty": _git_dirty()},
        "notes": [
            "hit_rate_at_recall 是整条链路的天花板：候选里没有金标块时，重排无法补救",
            "citation_precision 需要 --with-llm，且受采样影响，回归对比时应固定 with_llm",
        ],
    }

    _print_report(metrics, rows, refuse_rows, cfg, args.k)

    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n已写入 {_display(out)}")
    return 0


async def _measure_citations(
    svc: RagService,
    locate: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """生成类指标：需要真实模型，逐条调用 service。"""
    total_cited = 0
    correct_cited = 0
    without_citation = 0
    invalid = 0
    decision_mismatch = 0

    for case, row in zip(locate, rows, strict=True):
        answer = await svc.ask(RagRequest(question=case["question"]))
        # 一致性检查：评测侧算出的拒答判定应与 service 一致，否则口径已经漂移。
        if answer.refused != row["refused"]:
            decision_mismatch += 1

        if answer.citations:
            total_cited += len(answer.citations)
            correct_cited += sum(
                1 for citation in answer.citations if citation.source in case["gold_sources"]
            )
        else:
            without_citation += 1
        invalid += answer.invalid_citations

        row["answer_refused"] = answer.refused
        row["citations"] = [f"{c.source}:{c.line_start}" for c in answer.citations]

    metrics["citation_precision"] = _round(correct_cited / total_cited) if total_cited else None
    metrics["answers_without_citation"] = without_citation
    metrics["invalid_citations"] = invalid
    metrics["refusal_decision_mismatch"] = decision_mismatch


# ══════════════════════ 三、输出 ══════════════════════


def _print_report(
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    refuse_rows: list[dict[str, Any]],
    cfg: Settings,
    k: int,
) -> None:
    print(f"{'用例':<6}{'排名@recall':>12}{'排名@top':>10}{'命中@k':>8}{'最高分':>10}{'拒答':>6}")
    for row in rows:
        print(
            f"{row['id']:<6}"
            f"{row['rank_recall'] or '-':>12}"
            f"{row['rank_top'] or '-':>10}"
            f"{'是' if row['hit_at_k'] else '否':>8}"
            f"{_fmt(row['top_score']):>10}"
            f"{'是' if row['refused'] else '否':>6}"
        )

    print(f"\n{'无关题':<8}{'拒答':>6}{'最高分':>10}")
    for row in refuse_rows:
        print(f"{row['id']:<8}{'是' if row['refused'] else '否':>6}{_fmt(row['top_score']):>10}")

    locate_total = len(rows)
    refuse_total = len(refuse_rows)
    print("\n指标")
    print(f"  Hit@recall     {metrics['hit_rate_at_recall']:.1%}（天花板）")
    print(
        f"  Hit@{k}          {metrics['hit_rate_at_k']:.1%}"
        f"（{round(metrics['hit_rate_at_k'] * locate_total)}/{locate_total}）"
    )
    print(f"  MRR@{k}          {metrics['mrr_at_k']:.3f}")
    print(
        f"  拒答正确率     {metrics['refusal_accuracy']:.1%}"
        f"（{round(metrics['refusal_accuracy'] * refuse_total)}/{refuse_total}）"
    )
    print(f"  误拒率         {metrics['false_refusal_rate']:.1%}")

    if metrics["citation_precision"] is not None:
        print(f"  引用精度       {metrics['citation_precision']:.1%}")
        print(f"  无引用回答     {metrics['answers_without_citation']} 条")
        print(f"  越界引用       {metrics['invalid_citations']} 处")
        if metrics.get("refusal_decision_mismatch"):
            print(f"  判定不一致     {metrics['refusal_decision_mismatch']} 条（口径漂移）")
    else:
        print("  引用精度       未测（需要 --with-llm）")

    print(
        f"\n阈值 {cfg.rag_min_rerank_score:.2f}｜recall_k {cfg.rag_recall_k}｜top_n {cfg.rag_top_k}"
    )


def _count_chunks() -> int | None:
    if not CORPUS.is_file():
        return None
    return sum(1 for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip())


def _git(*argv: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *argv],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def _git_dirty() -> bool | None:
    try:
        done = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(done.stdout.strip())


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


# ══════════════════════ 四、入口 ══════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG 全链路评测")
    parser.add_argument("--locate", default=str(DEFAULT_LOCATE), help="相关题金标")
    parser.add_argument("--refuse", default=str(DEFAULT_REFUSE), help="无关题金标")
    parser.add_argument("--k", type=int, default=3, help="重排后的 Top-K")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="基线 JSON 输出路径")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="额外测引用精度（会真实调用模型，结果受采样影响）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(evaluate(args))


if __name__ == "__main__":
    raise SystemExit(main())
