"""检索评测：跑定位题，算 Hit Rate@K 与 MRR@K。

用法：
    uv run python scripts/eval_retrieval.py
    uv run python scripts/eval_retrieval.py --k 5 --out eval/runs/retrieval.md

`runs/` 下的产物已被 .gitignore 忽略；需要长期保留的基线写 `eval/baseline.json`。

判定口径（刻意简单，只验证「能不能定位到」）：

- 命中：返回块的 `source` 等于 `gold_source`，且正文包含 `gold_symbol`
- Hit@K：Top-K 中至少一个命中
- MRR@K：第一个命中所在排名的倒数（第 1 名记 1，第 3 名记 1/3，未命中记 0）

不评文采、不评答案完整性 —— 这两项要等重排与生成接入后才可度量。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from app.core.config import Settings
from app.rag.retriever import retrieve_with_score

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_GOLD = PROJECT_ROOT / "eval" / "gold" / "rag_locate.jsonl"


def _resolve(raw: str) -> pathlib.Path:
    path = pathlib.Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_cases(path: pathlib.Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evaluate(cases: list[dict], k: int, settings: Settings) -> tuple[list[dict], float, float]:
    rows: list[dict] = []
    hits = 0
    reciprocal = 0.0

    for case in cases:
        results = retrieve_with_score(case["question"], k=k, settings=settings)
        first_rank = 0
        first_meta: dict | None = None
        first_distance = 0.0

        for rank, (doc, distance) in enumerate(results, start=1):
            meta = doc.metadata
            matched = meta.get("source") in case["gold_sources"] and any(
                symbol in doc.page_content for symbol in case["gold_symbols"]
            )
            if matched:
                first_rank, first_meta, first_distance = rank, meta, distance
                break

        if first_rank:
            hits += 1
            reciprocal += 1.0 / first_rank

        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "rank": first_rank,
                "distance": first_distance,
                "where": (
                    f"{first_meta['source']}:{first_meta['line_start']}-{first_meta['line_end']}"
                    if first_meta
                    else "-"
                ),
                "returned": [
                    f"{d.metadata.get('source')}:{d.metadata.get('line_start')}" for d, _ in results
                ],
            }
        )

    total = len(cases)
    return rows, hits / total if total else 0.0, reciprocal / total if total else 0.0


def render(rows: list[dict], k: int, hit_rate: float, mrr: float) -> str:
    lines = [
        f"| 用例 | 问题 | 首个命中排名 | 距离 | 命中块 | 命中@{k} |",
        "|---:|---|---:|---:|---|:--:|",
    ]
    for row in rows:
        rank = row["rank"]
        mark = "是" if rank else "否"
        rank_text = str(rank) if rank else "-"
        dist_text = f"{row['distance']:.4f}" if rank else "-"
        lines.append(
            f"| {row['id']} | {row['question']} | {rank_text} | {dist_text} "
            f"| {row['where']} | {mark} |"
        )
    lines.append("")
    hit_count = sum(1 for r in rows if r["rank"])
    lines.append(f"- **Hit Rate@{k}**：{hit_rate:.1%}（{hit_count}/{len(rows)}）")
    lines.append(f"- **MRR@{k}**：{mrr:.3f}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检索评测")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD), help="金标文件")
    parser.add_argument("--k", type=int, default=3, help="Top-K")
    parser.add_argument("--out", default="", help="把结果写成 Markdown 文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gold = _resolve(args.gold)
    if not gold.is_file():
        print(f"找不到金标文件：{gold}", file=sys.stderr)
        return 2

    cases = load_cases(gold)
    settings = Settings()
    print(f"用例 {len(cases)} 条 | K={args.k} | model={settings.rag_embed_model}\n")

    rows, hit_rate, mrr = evaluate(cases, args.k, settings)
    report = render(rows, args.k, hit_rate, mrr)
    print(report)

    if args.out:
        out = _resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
        print(f"\n已写入 {out}")

    return 0 if hit_rate > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
