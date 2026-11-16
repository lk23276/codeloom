"""语料加载：遍历目录、按后缀分派切分策略、组装可写入 JSONL 的记录。

`Chunk` 是索引层的记录单位，比 `Document` 多两样东西：

- `line_start` / `line_end`：由 `start_index` 换算而来，供引用展示。
- `id`：`source:起始行-结束行`，便于去重、更新单块与排查。

`start_index` 同时保留，原因是它是原始证据：行号由它算出，
校验时必须回到它才能判断换算有没有错。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from app.rag.splitter import (
    CODE_SUFFIXES,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    NOTE_SUFFIXES,
    line_range_of,
    split_text,
)

SUPPORTED_SUFFIXES: frozenset[str] = CODE_SUFFIXES | NOTE_SUFFIXES

# 遍历时跳过的目录：版本库与虚拟环境里不会有语料，但可能有大量文本文件
_SKIP_DIRS: frozenset[str] = frozenset({".git", ".venv", "__pycache__", "node_modules"})


@dataclass(frozen=True, slots=True)
class Chunk:
    """一个可检索的文本块，带可回溯到原文的位置信息。"""

    id: str
    source: str
    kind: str
    start_index: int
    line_start: int
    line_end: int
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)

    def as_dict(self) -> dict[str, object]:
        """JSONL 行。字段顺序固定，便于 diff。"""
        return {
            "id": self.id,
            "source": self.source,
            "kind": self.kind,
            "start_index": self.start_index,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "chars": self.chars,
            "text": self.text,
        }


def load_text(path: Path) -> str:
    """读取语料文件。

    `utf-8-sig` 移除可能存在的 BOM，否则第一行会带一个不可见字符，
    展示时表现为引用内容的开头多出一个空格。
    `errors="replace"` 而不是 `ignore`：遇到非 UTF-8 字节时保留字符位，
    块内容与偏移的对应关系不会因此错位。
    """
    return path.read_text(encoding="utf-8-sig", errors="replace")


def discover(root: Path) -> list[Path]:
    """递归列出语料文件，按路径排序保证每次构建结果一致。"""
    if not root.is_dir():
        raise NotADirectoryError(f"语料目录不存在：{root}")

    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return files


def build_chunks(
    root: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """把 `root` 下的全部语料切块。

    `source` 用相对 `root` 的 POSIX 路径，天然支持子目录且不暴露本机路径。
    """
    chunks: list[Chunk] = []
    for path in discover(root):
        source = path.relative_to(root).as_posix()
        text = load_text(path)
        for doc in split_text(text, source, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            start_line, end_line = line_range_of(doc, text)
            chunks.append(
                Chunk(
                    id=f"{source}:{start_line}-{end_line}",
                    source=source,
                    kind=str(doc.metadata["kind"]),
                    start_index=int(doc.metadata["start_index"]),
                    line_start=start_line,
                    line_end=end_line,
                    text=doc.page_content,
                )
            )
    return chunks


def find_duplicate_ids(chunks: list[Chunk]) -> list[str]:
    """重复 id 说明两块的起止行相同，通常是切分或偏移换算出了问题。"""
    seen: set[str] = set()
    duplicated: list[str] = []
    for chunk in chunks:
        if chunk.id in seen:
            duplicated.append(chunk.id)
        seen.add(chunk.id)
    return duplicated


def verify_round_trip(
    chunks: list[Chunk],
    root: Path,
    *,
    sample: int = 3,
    seed: int = 0,
) -> list[str]:
    """抽样校验：用 `start_index` 精确回读原文，并复核行号。

    返回不一致的描述列表，空列表表示通过。`seed` 固定使抽到的样本可复现。
    """
    if not chunks:
        return ["没有可校验的块"]

    picked = chunks if len(chunks) <= sample else random.Random(seed).sample(chunks, sample)
    cache: dict[str, str] = {}
    failures: list[str] = []

    for chunk in picked:
        if chunk.source not in cache:
            cache[chunk.source] = load_text(root / chunk.source)
        text = cache[chunk.source]

        if text[chunk.start_index : chunk.start_index + len(chunk.text)] != chunk.text:
            failures.append(f"{chunk.id}: 按 start_index 回读的内容与块内容不一致")
            continue

        actual_line = text.count("\n", 0, chunk.start_index) + 1
        if actual_line != chunk.line_start:
            failures.append(
                f"{chunk.id}: 行号不一致（记录 {chunk.line_start}，实算 {actual_line}）"
            )
    return failures
