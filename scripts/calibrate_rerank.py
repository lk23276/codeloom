"""标定拒答阈值：跑相关题与无关题，打印分数分布与不同阈值下的取舍。

用法：
    uv run python scripts/calibrate_rerank.py

判定口径：

- 相关题：取重排最高分，阈值高于它就会误拒（把能答的问题拒掉）
- 无关题：取重排最高分，阈值低于它就会漏拒（把无关问题交给模型）
- 可用区间 = （无关题最高分, 相关题最低分）。区间不存在时，
  说明该重排模型分不开两者，阈值方案不成立 —— 应放弃拒答而非硬凑一个值。

★ 负例要放**同领域但语料未覆盖**的问题（Kafka / Redis / MySQL 之类）。
  只用「股市、菜谱」这类容易的负例，分数天然很低，会把阈值算得过于乐观。
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.config import Settings
from app.rag.rerank import build_reranker
from app.rag.retriever import retrieve_with_score

GOLD = Path("eval/gold/rag_locate.jsonl")

NEGATIVES_EASY = [
    "分析一下 A 股下周走势",
    "红烧肉怎么做才不腻",
    "我的猫连续两天不吃饭怎么办",
    "推荐一部适合周末看的科幻电影",
]

NEGATIVES_HARD = [
    "Kafka 的 ISR 副本同步是怎么实现的",
    "Redis Cluster 的槽位迁移如何保证一致性",
    "MySQL 的 next-key lock 解决了什么问题",
]

NEGATIVES = NEGATIVES_EASY + NEGATIVES_HARD

THRESHOLDS = [-1.0, 0.0, 0.2, 1.0, 2.0, 3.0, 4.0]


def top_score(question: str, reranker, k: int = 20) -> tuple[float | None, int]:
    pairs = retrieve_with_score(question, k=k)
    if not pairs:
        return None, 0
    documents = [doc for doc, _ in pairs]
    scores = reranker.score(question, [doc.page_content for doc in documents])
    return max(scores), len(documents)


def main() -> int:
    gold = Path(GOLD)
    if not gold.is_file():
        print(f"找不到金标文件：{gold}")
        return 2

    cfg = Settings()
    reranker = build_reranker(cfg)
    cases = [
        json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    print(f"模型 {cfg.rag_rerank_model}｜相关题 {len(cases)} 条｜无关题 {len(NEGATIVES)} 条")
    print(f"\n{'类别':<6}{'编号':<6}{'最高分':>10}{'候选数':>8}")

    positives: list[tuple[str, float | None]] = []
    for case in cases:
        score, recalled = top_score(case["question"], reranker)
        positives.append((case["id"], score))
        print(f"{'相关':<6}{case['id']:<6}{_fmt(score):>10}{recalled:>8}")

    negatives: list[tuple[str, float | None]] = []
    for index, question in enumerate(NEGATIVES, start=1):
        score, recalled = top_score(question, reranker)
        negatives.append((f"N{index}", score))
        print(f"{'无关':<6}{f'N{index}':<6}{_fmt(score):>10}{recalled:>8}")

    pos_values = [score for _, score in positives if score is not None]
    neg_values = [score for _, score in negatives if score is not None]
    if not pos_values or not neg_values:
        print("\n数据不足，无法标定")
        return 1

    low, high = max(neg_values), min(pos_values)
    print(f"\n相关题最低分 = {min(pos_values):.4f}")
    print(f"无关题最高分 = {max(neg_values):.4f}")
    if high > low:
        print(f"可用区间 = ({low:.4f}, {high:.4f})，中点 {(low + high) / 2:.4f}")
    else:
        print(f"区间重叠（{low:.4f} >= {high:.4f}），阈值方案不成立")

    print(f"\n{'阈值':>8}{'误拒':>10}{'挡掉':>10}")
    for threshold in THRESHOLDS:
        missed = sum(1 for value in pos_values if value < threshold)
        blocked = sum(1 for value in neg_values if value < threshold)
        print(
            f"{threshold:>8.1f}"
            f"{f'{missed}/{len(pos_values)}':>10}"
            f"{f'{blocked}/{len(neg_values)}':>10}"
        )
    return 0


def _fmt(score: float | None) -> str:
    return "-" if score is None else f"{score:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
