"""检索入口：上层唯一需要知道的 RAG 接口。

为什么要在 Chroma 之外再包一层：

- 上层拿到的永远是 `list[Document]`，不接触 `Chroma` 实例。
  换向量库（FAISS / Qdrant）只改本文件与 `vectorstore.py`。
- 默认参数（`k`、持久化路径、模型）集中在一处，不在调用点重复。
- 测试可以注入现成的 `store`，不必建真索引。

★ 向量库实例是模块级单例：`chromadb` 的客户端持有文件句柄，
  每个请求 new 一个既慢又容易在并发下触发句柄问题。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from app.core.config import Settings
from app.rag.vectorstore import build_embeddings, get_vectorstore

logger = logging.getLogger(__name__)

_store: Any = None


@lru_cache(maxsize=1)
def default_settings() -> Settings:
    """进程内共享一份配置，避免每次检索都读一遍 .env。"""
    return Settings()


def get_store(
    settings: Settings | None = None,
    *,
    embeddings: Any = None,
    recreate: bool = False,
) -> Any:
    """取向量库单例。

    `embeddings` 只在真正创建时生效（单例已存在则忽略），
    供测试注入假向量，避免为了跑一条断言去加载 90 MB 模型。
    `recreate=True` 用于测试隔离或切换配置后重建。
    """
    global _store
    if _store is None or recreate:
        cfg = settings or default_settings()
        _store = get_vectorstore(
            cfg,
            embeddings=embeddings or build_embeddings(cfg),
            persist_dir=cfg.rag_persist_dir,
        )
        logger.info("向量库已打开 dir=%s", cfg.rag_persist_dir)
    return _store


def reset_store() -> None:
    """释放单例。测试与脚本切换索引时使用。"""
    global _store
    _store = None


def retrieve(
    query: str,
    *,
    k: int | None = None,
    where: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> list[Document]:
    """按语义检索返回最相关的块。

    `where` 是 metadata 过滤条件，例如 `{"source": "server2.cpp"}`。
    Chroma 的过滤语法与 SQL 不同：等值用 `{"field": value}`，
    范围与逻辑组合需要 `{"$and": [...]}` 这类操作符。
    """
    if not query.strip():
        return []

    cfg = settings or default_settings()
    store = get_store(cfg)
    top_k = k if k is not None else cfg.rag_top_k

    if where:
        return store.similarity_search(query, k=top_k, filter=where)
    return store.similarity_search(query, k=top_k)


def retrieve_with_score(
    query: str,
    *,
    k: int | None = None,
    where: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> list[tuple[Document, float]]:
    """带距离的检索结果。

    ★ 第二个值是**距离**，不是相似度：`hnsw.space=cosine` 时范围为 [0, 2]，
      越小越近，0 表示方向完全相同。按「分数越高越好」解读会把排序读反。
    """
    if not query.strip():
        return []

    cfg = settings or default_settings()
    store = get_store(cfg)
    top_k = k if k is not None else cfg.rag_top_k
    return store.similarity_search_with_score(query, k=top_k, filter=where or None)
