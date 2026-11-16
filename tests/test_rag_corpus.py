"""语料切分与加载的契约测试。

这一层守住的是「可溯源」：任何一块都必须能用自己记录的偏移，
从原文精确取回同样的内容。偏移错了不会报错，只会让引用指到错误的位置 ——
和字段映射那类静默失败同源，所以断言要落在具体数值上。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.rag.loader import (
    Chunk,
    build_chunks,
    discover,
    find_duplicate_ids,
    load_text,
    verify_round_trip,
)
from app.rag.splitter import (
    MissingStartIndexError,
    UnsupportedSuffixError,
    kind_of,
    line_range_of,
    separators_for,
    split_text,
)

CPP = """// 示例注释
void alpha() {
    int a = 1;
}

void beta() {
    int b = 2;
}
"""

NOTE = """# 一级标题

引言。

## 二级标题

小节内容。

## 另一个二级

更多内容。
"""


# ══════════════════════ 一、类型分派 ══════════════════════


def test_kind_of_dispatches_by_suffix() -> None:
    """后缀决定切分策略，不能靠调用方传参猜。"""
    assert kind_of("server2.cpp") == "code"
    assert kind_of("server2.h") == "code"
    assert kind_of("README.md") == "note"
    assert kind_of("x.MD") == "note", "后缀大小写不应影响判断"


def test_kind_of_rejects_unknown_suffix() -> None:
    """不支持的格式要立刻报错，避免被当成 note 静默切一遍。"""
    with pytest.raises(UnsupportedSuffixError):
        kind_of("data.json")


# ══════════════════════ 二、切分 ══════════════════════


def test_split_text_attaches_source_and_offset() -> None:
    """每块都要带 source / kind / start_index —— 少一个后面就无法引用。"""
    docs = split_text(CPP, "demo.cpp", chunk_size=40, chunk_overlap=0)

    assert docs, "没切出任何块"
    for doc in docs:
        assert doc.metadata["source"] == "demo.cpp"
        assert doc.metadata["kind"] == "code"
        assert isinstance(doc.metadata["start_index"], int)


def test_split_text_prefers_function_boundary() -> None:
    """按结构切分时，块首应落在函数签名或注释上。

    ★ 若分隔符里混入 `\\n}`，切点会落在函数末尾之后，
      下一块以一个孤立的 `}` 起头 —— 函数被切断，检索命中也没有上下文价值。
    """
    docs = split_text(CPP, "demo.cpp", chunk_size=40, chunk_overlap=0)

    heads = [d.page_content.lstrip() for d in docs]
    assert all(not h.startswith("}") for h in heads), f"有块以 }} 开头：{heads}"


def test_split_text_drops_punctuation_only_fragments() -> None:
    """只含标点的碎片没有检索价值，应在切分阶段丢弃。"""
    text = "int a = 1;\n}\n\nint b = 2;\n}\n"

    docs = split_text(text, "demo.cpp", chunk_size=12, chunk_overlap=0)

    assert all(any(ch.isalnum() for ch in d.page_content) for d in docs)


def test_split_text_keeps_short_but_meaningful_chunks() -> None:
    """短不等于无用：单行注释、短声明要保留。

    ★ 判据是「是否含字母或数字」，不是长度阈值 —— 长度阈值会误删短函数。
    """
    docs = split_text("// 说明\nint f();\n", "demo.cpp", chunk_size=200, chunk_overlap=0)

    assert len(docs) == 1
    assert "// 说明" in docs[0].page_content


def test_split_text_rejects_overlap_not_smaller_than_size() -> None:
    """overlap >= size 会让切分无法前进，必须在入口拦住。"""
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_text(CPP, "demo.cpp", chunk_size=50, chunk_overlap=50)


def test_split_text_handles_markdown_headings() -> None:
    """Markdown 用标题当分隔符，且只切一次以保住全局偏移。"""
    docs = split_text(NOTE, "note.md", chunk_size=30, chunk_overlap=0)

    assert len(docs) >= 2
    assert all(d.metadata["start_index"] >= 0 for d in docs)
    # 第二条起应有块落在二级标题上，说明标题参与切分
    assert any(d.page_content.lstrip().startswith("##") for d in docs)


# ══════════════════════ 三、行号换算 ══════════════════════


def test_line_range_of_counts_from_offset() -> None:
    """偏移换算出的起始行必须与原文一致。"""
    text = "line1\nline2\nline3\nvoid target() {\n}\n"
    offset = text.index("void target")
    doc = Document(page_content="void target() {", metadata={"start_index": offset})

    assert line_range_of(doc, text) == (4, 4)


def test_line_range_of_trims_trailing_newlines() -> None:
    """块尾带换行时不能多算一行。

    ★ 否则引用会显示成「第 87-103 行」，比实际多一行，
      而这类偏差在抽查时很难被发现。
    """
    text = "a\nb\nc\nd\n"
    doc = Document(page_content="b\nc\n", metadata={"start_index": 2})

    assert line_range_of(doc, text) == (2, 3)


def test_line_range_of_raises_without_start_index() -> None:
    """缺偏移时不能降级成「第 1 行」，否则整份索引的行号都是错的。"""
    doc = Document(page_content="x", metadata={"source": "demo.cpp"})

    with pytest.raises(MissingStartIndexError):
        line_range_of(doc, "x\n")


# ══════════════════════ 四、加载与校验 ══════════════════════


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    (root / "demo.cpp").write_text(CPP, encoding="utf-8")
    (root / "note.md").write_text(NOTE, encoding="utf-8")
    (root / "skip.json").write_text("{}", encoding="utf-8")
    return root


def test_discover_filters_by_suffix(corpus: Path) -> None:
    """只收语料格式，其他文件不进索引。"""
    names = sorted(p.name for p in discover(corpus))

    assert names == ["demo.cpp", "note.md"]


def test_discover_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        discover(tmp_path / "nope")


def test_build_chunks_round_trips(corpus: Path) -> None:
    """端到端：块数、id、行号、回读四项都要成立。"""
    chunks = build_chunks(corpus, chunk_size=40, chunk_overlap=0)

    assert chunks
    assert all(c.line_start >= 1 and c.line_end >= c.line_start for c in chunks)
    assert all(c.id == f"{c.source}:{c.line_start}-{c.line_end}" for c in chunks)
    assert verify_round_trip(chunks, corpus, sample=10) == []


def test_verify_round_trip_detects_broken_offset(corpus: Path) -> None:
    """校验必须能发现偏移错误，否则它形同虚设。

    ★ 手工把 start_index 改错一位，回读内容就会对不上。
    """
    chunks = build_chunks(corpus, chunk_size=40, chunk_overlap=0)
    broken = Chunk(
        id=chunks[0].id,
        source=chunks[0].source,
        kind=chunks[0].kind,
        start_index=chunks[0].start_index + 1,
        line_start=chunks[0].line_start,
        line_end=chunks[0].line_end,
        text=chunks[0].text,
    )

    failures = verify_round_trip([broken], corpus, sample=1)

    assert failures, "偏移错了却校验通过"


def test_find_duplicate_ids_flags_same_range() -> None:
    """同文件同起止行说明切分重复产块，会在索引里互相覆盖。"""
    base = dict(source="demo.cpp", kind="code", start_index=0, line_start=1, line_end=3, text="x")
    a = Chunk(id="demo.cpp:1-3", **base)
    b = Chunk(id="demo.cpp:1-3", **base)

    assert find_duplicate_ids([a, b]) == ["demo.cpp:1-3"]
    assert find_duplicate_ids([a]) == []


def test_load_text_tolerates_bom(tmp_path: Path) -> None:
    """带 BOM 的文件读出来不能带不可见字符，否则引用内容开头会多一个空白。

    `utf-8-sig` 编码时自行写入 BOM，读取时再剥掉 —— 这里刻意让编码器生成，
    而不是手写 `\\ufeff`，否则会变成两个 BOM，测的就不是目标行为。
    """
    path = tmp_path / "bom.md"
    path.write_bytes("# 标题\n\n内容\n".encode("utf-8-sig"))

    assert load_text(path).startswith("# 标题")


# ══════════════════════ 五、序列化与遍历细节 ══════════════════════


def test_chunk_as_dict_is_jsonl_ready() -> None:
    """导出字段必须齐全且顺序固定，JSONL 才能逐行 diff。"""
    chunk = Chunk(
        id="demo.cpp:1-3",
        source="demo.cpp",
        kind="code",
        start_index=0,
        line_start=1,
        line_end=3,
        text="abc",
    )

    row = chunk.as_dict()

    assert list(row) == [
        "id",
        "source",
        "kind",
        "start_index",
        "line_start",
        "line_end",
        "chars",
        "text",
    ]
    assert row["chars"] == 3, "chars 是冗余字段，便于不解析正文就统计"


def test_discover_skips_vcs_and_cache_dirs(tmp_path: Path) -> None:
    """版本库与缓存目录里的文本文件不应进语料。"""
    root = tmp_path / "raw"
    (root / ".git").mkdir(parents=True)
    (root / "__pycache__").mkdir()
    (root / ".git" / "config.cpp").write_text(CPP, encoding="utf-8")
    (root / "__pycache__" / "x.md").write_text(NOTE, encoding="utf-8")
    (root / "real.cpp").write_text(CPP, encoding="utf-8")

    assert [p.name for p in discover(root)] == ["real.cpp"]


def test_verify_round_trip_rejects_empty_input(tmp_path: Path) -> None:
    """空索引要报错，而不是静默返回「通过」。"""
    assert verify_round_trip([], tmp_path) == ["没有可校验的块"]


def test_verify_round_trip_detects_wrong_line(tmp_path: Path) -> None:
    """内容对但行号错，同样要判失败。

    ★ 行号错意味着引用会指向错误位置，而内容回读是通过的 ——
      只校验内容会漏掉这一类。
    """
    root = tmp_path / "raw"
    root.mkdir()
    (root / "demo.cpp").write_text(CPP, encoding="utf-8")
    chunks = build_chunks(root, chunk_size=40, chunk_overlap=0)
    first = chunks[0]
    shifted = Chunk(
        id=f"demo.cpp:{first.line_start + 5}-{first.line_end + 5}",
        source=first.source,
        kind=first.kind,
        start_index=first.start_index,
        line_start=first.line_start + 5,
        line_end=first.line_end + 5,
        text=first.text,
    )

    failures = verify_round_trip([shifted], root, sample=1)

    assert failures
    assert "行号不一致" in failures[0]


def test_separators_for_rejects_unknown_kind() -> None:
    """类型名写错要立刻失败，不能回退到某套默认分隔符。"""
    with pytest.raises(ValueError, match="未知语料类型"):
        separators_for("binary")
