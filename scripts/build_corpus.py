"""构建语料索引：加载 → 切分 → 换算行号 → 写 JSONL。

用法：
    uv run python scripts/build_corpus.py
    uv run python scripts/build_corpus.py --show 5
    uv run python scripts/build_corpus.py --chunk-size 1000 --chunk-overlap 150
    uv run python scripts/build_corpus.py --root data/corpus/raw --out data/corpus/chunks.jsonl

产物只落在 `data/` 下，已被 `.gitignore` 覆盖：语料是派生物，
可由本脚本随时重建，不入库。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from app.rag.loader import (
    Chunk,
    build_chunks,
    discover,
    find_duplicate_ids,
    verify_round_trip,
)
from app.rag.splitter import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, kind_of

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _print_stats(chunks: list[Chunk]) -> None:
    sizes = [c.chars for c in chunks]
    print(
        f"      {len(chunks)} 块 | 字符数 min={min(sizes)} "
        f"avg={sum(sizes) // len(sizes)} max={max(sizes)}"
    )

    by_source = Counter(c.source for c in chunks)
    for source, count in sorted(by_source.items()):
        kind = next(c.kind for c in chunks if c.source == source)
        print(f"        {source:<28} {kind:<5} {count:>3} 块")


def _show(chunks: list[Chunk], count: int) -> None:
    print(f"\n前 {min(count, len(chunks))} 块预览：")
    for chunk in chunks[:count]:
        head = chunk.text.splitlines()[0][:56] if chunk.text else ""
        print(f"      {chunk.id}")
        print(f"        {chunk.chars} 字 | 首行 {head!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建 RAG 语料索引")
    parser.add_argument("--root", default="data/corpus/raw", help="语料根目录")
    parser.add_argument("--out", default="data/corpus/chunks.jsonl", help="输出 JSONL 路径")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--show", type=int, default=0, help="打印前 N 块的内容预览")
    parser.add_argument("--no-verify", action="store_true", help="跳过抽样回读校验")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _resolve(args.root)
    out = _resolve(args.out)

    print(f"[1/4] 扫描语料\n      {root}")
    try:
        files = discover(root)
    except NotADirectoryError as exc:
        print(f"      失败：{exc}", file=sys.stderr)
        return 2

    kinds = Counter(kind_of(f.name) for f in files)
    print(f"      {len(files)} 个文件（code {kinds['code']} / note {kinds['note']}）")
    if not files:
        print("      失败：目录下没有可切分的语料", file=sys.stderr)
        return 2

    print(f"\n[2/4] 切分\n      chunk_size={args.chunk_size} chunk_overlap={args.chunk_overlap}")
    chunks = build_chunks(root, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    _print_stats(chunks)

    print("\n[3/4] 校验")
    duplicates = find_duplicate_ids(chunks)
    print(f"      重复 id：{len(duplicates)}")
    if duplicates:
        for dup in duplicates[:5]:
            print(f"        {dup}", file=sys.stderr)

    failures: list[str] = []
    if args.no_verify:
        print("      抽样回读：已跳过（--no-verify）")
    else:
        failures = verify_round_trip(chunks, root, sample=3)
        print(f"      抽样 3 块回读：{'全部一致' if not failures else f'{len(failures)} 处不一致'}")
        for failure in failures:
            print(f"        {failure}", file=sys.stderr)

    print("\n[4/4] 写入")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(c.as_dict(), ensure_ascii=False) for c in chunks]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"      {out} → {len(lines)} 行 / {out.stat().st_size / 1024:.1f} KB")

    if args.show:
        _show(chunks, args.show)

    if duplicates or failures:
        print("\n校验未通过，索引不可用于后续环节。", file=sys.stderr)
        return 1
    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
