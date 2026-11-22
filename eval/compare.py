"""回归门禁：把当前结果与基线比，退化则返回非零退出码。

用法：

    uv run python -m eval.run --out eval/runs/v1.json      # 先跑一次，产出当前结果
    uv run python -m eval.compare --current eval/runs/v1.json

    # 演示门禁真的会拦人：
    uv run python -m eval.compare --current eval/runs/v1.json --tolerance-pp 0

退出码（这是门禁的本体，表格只是给人看的）：

    0  全部在阈值内
    1  有指标退化
    2  跑不起来（文件缺失 / 口径不一致）

★ 为什么把「口径不一致」单独给 2：
  `k` 不同、`with_llm` 不同、语料块数不同，两边的指标根本不可比。
  把它归到 1（退化）会让人去查模型，而真正的问题是比较条件变了。

★ 阈值为什么同时有「百分点」和「条数」两套：
  小样本下百分比分辨率很粗 —— 8 条题里 1 条翻转就是 12.5pp，
  5pp 的阈值永远触发不了。所以比率看百分点，计数类（越界引用等）看绝对条数。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_REGRESSED = 1
EXIT_ERROR = 2

# 越大越好：下降超过阈值即退化
HIGHER_IS_BETTER = (
    "hit_rate_at_recall",
    "hit_rate_at_k",
    "mrr_at_k",
    "refusal_accuracy",
    "citation_precision",
)

# 越小越好、且是计数：容许小幅波动，超过条数阈值即退化
LOWER_IS_BETTER_COUNT = ("answers_without_citation", "invalid_citations")
# 越小越好、比率
LOWER_IS_BETTER_RATE = ("false_refusal_rate",)

# 必须是同一口径才能比较的字段
COMPARABLE_KEYS = ("k", "with_llm")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(
    name: str,
    base: Any,
    cur: Any,
    *,
    verdict: str,
    delta_pp: float | None = None,
    delta: int | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"name": name, "base": base, "cur": cur, "verdict": verdict}
    if delta_pp is not None:
        row["delta_pp"] = delta_pp
    if delta is not None:
        row["delta"] = delta
    return row


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def check_config(baseline: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """口径校验：不一致就没有比较的意义。"""
    problems: list[str] = []
    for key in COMPARABLE_KEYS:
        if baseline.get(key) != current.get(key):
            base_value = _fmt(baseline.get(key))
            cur_value = _fmt(current.get(key))
            problems.append(f"{key} 不同：baseline={base_value} current={cur_value}")

    base_chunks = (baseline.get("corpus") or {}).get("chunks")
    cur_chunks = (current.get("corpus") or {}).get("chunks")
    if base_chunks != cur_chunks:
        problems.append(f"语料块数不同：baseline={base_chunks} current={cur_chunks}")
    return problems


def compare(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    tolerance_pp: float,
    max_extra: int,
) -> tuple[list[dict[str, Any]], bool]:
    """逐项比较，返回 (行, 是否退化)。"""
    base_metrics = baseline.get("metrics") or {}
    cur_metrics = current.get("metrics") or {}
    names = list(dict.fromkeys([*base_metrics, *cur_metrics]))

    rows: list[dict[str, Any]] = []
    regressed = False

    for name in names:
        base = base_metrics.get(name)
        cur = cur_metrics.get(name)

        if name in HIGHER_IS_BETTER:
            if base is None or cur is None:
                rows.append({"name": name, "base": base, "cur": cur, "verdict": "N/A"})
                continue
            delta_pp = round((cur - base) * 100, 2)
            bad = delta_pp < -tolerance_pp
            verdict = "REGRESSED" if bad else ("improved" if delta_pp > 0 else "same")
            rows.append(_row(name, base, cur, delta_pp=delta_pp, verdict=verdict))
            regressed = regressed or bad

        elif name in LOWER_IS_BETTER_RATE:
            if base is None or cur is None:
                rows.append({"name": name, "base": base, "cur": cur, "verdict": "N/A"})
                continue
            delta_pp = round((cur - base) * 100, 2)
            bad = delta_pp > tolerance_pp
            verdict = "REGRESSED" if bad else ("improved" if delta_pp < 0 else "same")
            rows.append(_row(name, base, cur, delta_pp=delta_pp, verdict=verdict))
            regressed = regressed or bad

        elif name in LOWER_IS_BETTER_COUNT:
            if base is None or cur is None:
                rows.append({"name": name, "base": base, "cur": cur, "verdict": "N/A"})
                continue
            delta = cur - base
            bad = delta > max_extra
            verdict = "REGRESSED" if bad else ("improved" if delta < 0 else "same")
            rows.append(_row(name, base, cur, delta=delta, verdict=verdict))
            regressed = regressed or bad

        else:
            rows.append({"name": name, "base": base, "cur": cur, "verdict": "未定义判据"})

    return rows, regressed


def _print_table(rows: list[dict[str, Any]]) -> None:
    print(f"{'指标':<24}{'baseline':>12}{'current':>12}{'变化':>12}  判定")
    print("-" * 74)
    for row in rows:
        if "delta_pp" in row:
            delta = f"{row['delta_pp']:+.2f}pp"
        elif "delta" in row:
            delta = f"{row['delta']:+d}"
        else:
            delta = "—"
        cells = f"{row['name']:<24}{_fmt(row['base']):>12}{_fmt(row['cur']):>12}"
        print(f"{cells}{delta:>12}  {row['verdict']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="回归门禁：对比 baseline 与 current")
    parser.add_argument("--baseline", default="eval/baseline.json", help="基线文件")
    parser.add_argument("--current", required=True, help="本次结果文件（eval.run --out 产出）")
    parser.add_argument(
        "--tolerance-pp",
        type=float,
        default=5.0,
        help="比率类指标允许的下降百分点（默认 5）",
    )
    parser.add_argument(
        "--max-extra",
        type=int,
        default=0,
        help="计数类指标允许的增加条数（默认 0）",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出对比结果")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        baseline = _load(Path(args.baseline))
        current = _load(Path(args.current))
    except FileNotFoundError as exc:
        print(f"[门禁] 文件不存在：{exc.filename}", file=sys.stderr)
        return EXIT_ERROR
    except json.JSONDecodeError as exc:
        print(f"[门禁] 结果文件不是合法 JSON：{exc}", file=sys.stderr)
        return EXIT_ERROR

    problems = check_config(baseline, current)
    if problems:
        print("[门禁] 口径不一致，拒绝比较：", file=sys.stderr)
        for item in problems:
            print(f"  - {item}", file=sys.stderr)
        return EXIT_ERROR

    rows, regressed = compare(
        baseline, current, tolerance_pp=args.tolerance_pp, max_extra=args.max_extra
    )

    if args.json:
        print(json.dumps({"rows": rows, "regressed": regressed}, ensure_ascii=False, indent=2))
    else:
        generated = current.get("generated_at", "?")
        print(f"baseline: {args.baseline} | current: {args.current} ({generated})")
        print(f"口径: k={current.get('k')} with_llm={_fmt(current.get('with_llm'))} "
              f"chunks={(current.get('corpus') or {}).get('chunks')}")
        print()
        _print_table(rows)
        print()
        if regressed:
            print(f"[门禁] 有指标退化（阈值 {args.tolerance_pp}pp / {args.max_extra} 条）")
        else:
            print("[门禁] 通过")

    return EXIT_REGRESSED if regressed else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
