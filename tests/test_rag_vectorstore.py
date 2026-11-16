"""向量库与检索的契约测试。

这一层守两件事：

1. **幂等**：同一批块写两次，索引条数不变。不传稳定 id 时框架会生成 uuid4，
   重跑一次就翻倍，而检索结果里同一个块重复出现，Top-K 只覆盖一半候选。
2. **不对称**：建索引走 passage 编码，查询走 query 编码。bge 只在 query 侧
   加检索指令，两边混用不会报错，只掉召回。

全部用 `FakeEmbeddings(size=8)`：不下载 90 MB 模型、不需要网络，
测试只验证链路与契约，不验证语义质量。
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings, FakeEmbeddings

from app.core.config import Settings
from app.rag.retriever import get_store, reset_store, retrieve, retrieve_with_score
from app.rag.vectorstore import (
    FastEmbedEmbeddings,
    build_embeddings,
    count_chunks,
    ensure_indexed,
    get_vectorstore,
)

RAG_DIR = "chroma"


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, rag_persist_dir=str(tmp_path / RAG_DIR))


@pytest.fixture
def db(tmp_path: Path):
    """真 Chroma + 假向量。持久化到 tmp_path，用完即弃。"""
    settings = make_settings(tmp_path)
    return get_vectorstore(
        settings,
        embeddings=FakeEmbeddings(size=8),
        persist_dir=str(tmp_path / RAG_DIR),
    )


def sample_chunks() -> list[dict]:
    return [
        {
            "id": "demo.cpp:1-3",
            "source": "demo.cpp",
            "kind": "code",
            "line_start": 1,
            "line_end": 3,
            "start_index": 0,
            "text": "void alpha() { return; }",
        },
        {
            "id": "demo.cpp:5-7",
            "source": "demo.cpp",
            "kind": "code",
            "line_start": 5,
            "line_end": 7,
            "start_index": 20,
            "text": "void beta() { return; }",
        },
        {
            "id": "note.md:1-4",
            "source": "note.md",
            "kind": "note",
            "line_start": 1,
            "line_end": 4,
            "start_index": 0,
            "text": "# 标题\n\n段落。",
        },
    ]


# ══════════════════════ 一、embedding 的两条路径 ══════════════════════


class _RecordingModel:
    """记录调用的是 passage 还是 query 编码。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def passage_embed(self, texts: list[str]):
        self.calls.append(("passage", list(texts)))
        return [np.array([1.0, 0.0]) for _ in texts]

    def query_embed(self, text: str):
        self.calls.append(("query", text))
        return iter([np.array([0.0, 1.0])])


class _DeterministicEmbeddings(Embeddings):
    """按文本内容生成固定向量。

    `FakeEmbeddings` 每次采样随机向量，无法用来验证「距离越小越近」；
    这里用内容哈希做种子，相同文本必然得到相同向量。
    """

    def __init__(self, size: int = 8) -> None:
        self.size = size

    def _vec(self, text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        return [rng.gauss(0.0, 1.0) for _ in range(self.size)]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def test_embeddings_split_passage_and_query() -> None:
    """建索引与查询必须走不同方法。

    ★ bge 系列只在 query 侧加检索指令。两边都用同一个方法不会抛异常，
      代价是召回率下降 —— 在建立评测基线之前无法察觉。
    """
    emb = FastEmbedEmbeddings("dummy")
    recorder = _RecordingModel()
    emb._model = recorder  # 跳过真实模型加载

    vectors = emb.embed_documents(["a", "b"])
    query_vector = emb.embed_query("q")

    assert recorder.calls == [("passage", ["a", "b"]), ("query", "q")]
    assert len(vectors) == 2
    assert query_vector == [0.0, 1.0]


def test_embeddings_return_plain_lists() -> None:
    """返回必须是 list[float]，不是 numpy 数组：Chroma 直接比对元素类型。"""
    emb = FastEmbedEmbeddings("dummy")
    emb._model = _RecordingModel()

    assert isinstance(emb.embed_query("q"), list)
    assert all(isinstance(v, list) for v in emb.embed_documents(["a"]))


def test_build_embeddings_puts_cache_under_data(tmp_path: Path) -> None:
    """模型缓存放 data/ 下，不放系统临时目录 —— 后者会被清理，重下 90 MB。"""
    settings = make_settings(tmp_path)
    embeddings = build_embeddings(settings)

    assert isinstance(embeddings, FastEmbedEmbeddings)
    assert embeddings.model_name == settings.rag_embed_model
    assert (tmp_path / "fastembed").is_dir()


# ══════════════════════ 二、建库与幂等 ══════════════════════


def test_ensure_indexed_writes_all_chunks(db) -> None:
    chunks = sample_chunks()

    written = ensure_indexed(db, chunks)

    assert written == len(chunks)
    assert count_chunks(db) == len(chunks)


def test_ensure_indexed_is_idempotent(db) -> None:
    """同一批块写两次，条数不变。

    ★ 不传 ids 时 LangChain 用 uuid4 生成主键，重复写入不会覆盖，
      库内条目会翻倍，检索结果里同一块出现两次。显式传稳定 id 才能幂等。
    """
    chunks = sample_chunks()
    ensure_indexed(db, chunks)
    first = count_chunks(db)

    ensure_indexed(db, chunks)

    assert count_chunks(db) == first


def test_add_without_ids_duplicates(db) -> None:
    """反向验证：不传 ids 会重复插入。

    ★ 本用例验证的是框架行为，不是本项目代码。它的作用是锁定前提：
      一旦 `add_documents` 改变主键生成策略，此断言会失败，
      从而暴露「必须显式传 id」这一假设的失效。
    """
    docs = [Document(page_content="x", metadata={"source": "a"})]
    db.add_documents(docs)
    first = count_chunks(db)

    db.add_documents(docs)

    assert count_chunks(db) == 2 * first


def test_vectorstore_uses_cosine_space(tmp_path: Path) -> None:
    """space 必须显式设为 cosine。

    ★ 默认是 `l2`（平方 L2）。两者在归一化向量上排序等价，但数值口径不同：
      余弦距离 ∈ [0,2]，平方 L2 是它的两倍。距离阈值只能绑定其中一种，
      不固定则同一段文本的分数会随配置漂移，拒答判定随之失效。
    """
    settings = make_settings(tmp_path)
    store = get_vectorstore(
        settings, embeddings=FakeEmbeddings(size=8), persist_dir=str(tmp_path / RAG_DIR)
    )

    space = store._collection.configuration["hnsw"]["space"]

    assert space == "cosine"


def test_metadata_keeps_location_fields(db) -> None:
    """位置字段必须进 metadata，否则检索回来无法拼引用。"""
    ensure_indexed(db, sample_chunks())

    hits = db.similarity_search("alpha", k=1)
    meta = hits[0].metadata

    assert {"source", "kind", "line_start", "line_end", "start_index"} <= set(meta)


# ══════════════════════ 三、检索入口 ══════════════════════


def test_retrieve_rejects_blank_query() -> None:
    """空查询直接返回 []，不发起检索。"""
    assert retrieve("   ") == []
    assert retrieve("") == []


def test_filter_restricts_to_one_source(db) -> None:
    """metadata 过滤后结果必须全部来自指定文件。"""
    ensure_indexed(db, sample_chunks())

    hits = db.similarity_search("return", k=5, filter={"source": "note.md"})

    assert hits
    assert all(h.metadata["source"] == "note.md" for h in hits)


def test_count_reflects_batch_writes(db) -> None:
    """分批写入后总数正确（BATCH_SIZE=64，用 3 条不足以跨批，这里只验总数）。"""
    chunks = sample_chunks()

    ensure_indexed(db, chunks, batch_size=2)

    assert count_chunks(db) == len(chunks)


# ══════════════════════ 四、retriever 单例与检索 ══════════════════════


@pytest.fixture(autouse=True)
def _clean_singleton():
    """每个用例前后都清空模块级单例，避免用例之间互相污染。"""
    reset_store()
    yield
    reset_store()


def test_get_store_returns_same_instance(tmp_path: Path) -> None:
    """同一进程内复用同一个向量库客户端。

    ★ 每次检索都新建 Chroma 会重复打开文件句柄，并发下容易出问题；
      单例也保证了 embedding 模型只加载一次（90 MB，加载约 1 秒）。
    """
    settings = make_settings(tmp_path)

    first = get_store(settings, embeddings=FakeEmbeddings(size=8), recreate=True)
    second = get_store(settings)

    assert first is second


def test_reset_store_forces_new_instance(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    first = get_store(settings, embeddings=FakeEmbeddings(size=8), recreate=True)

    reset_store()
    second = get_store(settings, embeddings=FakeEmbeddings(size=8))

    assert first is not second


def test_retrieve_returns_hits(tmp_path: Path) -> None:
    """检索入口返回 Document 列表，且带位置元数据。"""
    settings = make_settings(tmp_path)
    db = get_store(settings, embeddings=FakeEmbeddings(size=8), recreate=True)
    ensure_indexed(db, sample_chunks())

    hits = retrieve("alpha", settings=settings, k=2)

    assert 0 < len(hits) <= 2
    assert all("source" in h.metadata for h in hits)


def test_retrieve_with_score_distance_zero_for_identical_text(tmp_path: Path) -> None:
    """距离语义：相同文本距离为 0，越大越远。

    ★ 不能用 `FakeEmbeddings` 验证这一条：它的文档写明「从正态分布采样」，
      每次调用都产生新的随机向量，相同文本的距离也不会是 0。
      这里改用按内容哈希生成的确定性向量。
    """
    settings = make_settings(tmp_path)
    db = get_vectorstore(
        settings,
        embeddings=_DeterministicEmbeddings(size=8),
        persist_dir=str(tmp_path / RAG_DIR),
    )
    ensure_indexed(db, sample_chunks())

    rows = db.similarity_search_with_score("void alpha() { return; }", k=2)

    assert rows[0][1] < 0.01, f"相同文本的距离应为 0，实际 {rows[0][1]}"
    assert rows[1][1] > rows[0][1], "不同文本的距离应更大"


def test_retrieve_honours_where_filter(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db = get_store(settings, embeddings=FakeEmbeddings(size=8), recreate=True)
    ensure_indexed(db, sample_chunks())

    hits = retrieve("return", settings=settings, where={"source": "note.md"}, k=5)

    assert all(h.metadata["source"] == "note.md" for h in hits)


def test_retrieve_with_score_rejects_blank_query() -> None:
    """带分数的检索同样要拦空查询，避免无谓地打向量库。"""
    assert retrieve_with_score("  ") == []
