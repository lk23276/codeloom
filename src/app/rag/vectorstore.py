"""向量库：Chroma 的封装、embedding 适配、幂等写入。

两层职责：

- `FastEmbedEmbeddings`：把 fastembed 的 ONNX 推理接到 LangChain 的 `Embeddings` 接口。
- `get_vectorstore` / `ensure_indexed`：Chroma 的建库与写入。

★ 为什么不用 `langchain_community.embeddings.FastEmbedEmbeddings`：
  `langchain-community` 会引入十个以上与向量化无关的依赖（numpy 之外的
  sqlalchemy、aiohttp 等）。本模块只需要 embedding 一件事，适配代码不足 30 行。

★ 为什么区分 `embed_documents` 与 `embed_query`：
  bge 系列在训练时对 query 一侧加了检索指令，document 一侧不加。
  两者混用不会报错，只会降低召回 —— 属于「结果变差但没有异常」的一类，
  在建立评测基线之前无法发现。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.config import Settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "codeloom"
BATCH_SIZE = 64


class EmbeddingNotAvailableError(RuntimeError):
    """fastembed 或模型不可用。"""


@lru_cache(maxsize=1)
def _fastembed_cls() -> Any:
    """延迟导入 fastembed。未安装时在构造期报错，而不是模块导入期。"""
    try:
        from fastembed import TextEmbedding
    except ModuleNotFoundError as exc:  # pragma: no cover - 取决于运行环境
        raise EmbeddingNotAvailableError("缺少 fastembed：执行 `uv add fastembed` 后重试") from exc
    return TextEmbedding


class FastEmbedEmbeddings(Embeddings):
    """fastembed 的 LangChain 适配层。

    模型首次使用时会下载（`bge-small-zh-v1.5` 约 90 MB），之后走本地缓存。
    延迟构造实例，避免「只是导入模块」就触发模型加载。
    """

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            cls = _fastembed_cls()
            logger.info("加载 embedding 模型 model=%s", self.model_name)
            try:
                self._model = cls(model_name=self.model_name, cache_dir=self._cache_dir)
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingNotAvailableError(
                    f"向量模型 {self.model_name} 加载失败（下载失败时设 HF_ENDPOINT 或预下载）："
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            logger.info("embedding 模型就绪 model=%s", self.model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """建索引一侧。**不加** query 前缀。"""
        vectors = self.model.passage_embed(list(texts))
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """查询一侧。bge 会在这里加检索指令前缀。"""
        return next(iter(self.model.query_embed(text))).tolist()


@lru_cache(maxsize=1)
def _chroma_cls() -> Any:
    """延迟导入 langchain_chroma：该包会连带导入 chromadb，耗时约 1 秒。"""
    try:
        from langchain_chroma import Chroma
    except ModuleNotFoundError as exc:  # pragma: no cover - 取决于运行环境
        raise EmbeddingNotAvailableError(
            "缺少 langchain-chroma：执行 `uv add langchain-chroma` 后重试"
        ) from exc
    return Chroma


def get_vectorstore(
    settings: Settings,
    *,
    embeddings: Embeddings | None = None,
    persist_dir: str | None = None,
    collection: str = COLLECTION_NAME,
) -> Any:
    """打开（或创建）持久化向量库。

    `hnsw.space` 显式设为 `cosine`：默认是 `l2`（平方 L2），两者在归一化向量上
    排序等价，但数值口径不同（余弦距离 ∈ [0,2]），阈值只能绑定其中一种。
    这里固定为余弦，便于后续用「距离 < 阈值」做拒答。
    """
    chroma = _chroma_cls()
    path = Path(persist_dir or settings.rag_persist_dir)
    path.mkdir(parents=True, exist_ok=True)
    return chroma(
        collection_name=collection,
        embedding_function=embeddings or build_embeddings(settings),
        persist_directory=str(path),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def build_embeddings(settings: Settings) -> Embeddings:
    """按配置构造 embedding。

    模型缓存放在 `data/` 下（与索引同目录），而不是系统临时目录：
    临时目录会被清理，重下一次 90 MB 不划算；`data/` 已被 `.gitignore` 覆盖。
    """
    cache_dir = Path(settings.rag_persist_dir).parent / "fastembed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return FastEmbedEmbeddings(settings.rag_embed_model, cache_dir=str(cache_dir))


def ensure_indexed(
    db: Any,
    chunks: list[dict[str, Any]],
    *,
    batch_size: int = BATCH_SIZE,
) -> int:
    """把块写入向量库，按稳定 id 去重，返回**新增**条数。

    ★ 必须传 `ids`。不传时 LangChain 会为每个 `Document` 生成 uuid4，
      同一批数据写两次就是两条记录 —— 实测 3 条变 6 条。
      表现是检索结果里同一个块重复出现，Top-K 实际只覆盖一半的候选。

    ★ 已存在的 id 直接跳过，不做 upsert。原因：Chroma 对同 id 的 upsert 是
      「删旧 + 插新」，会重写 HNSW 段；若这次写入没走到完整收尾，段目录可能只剩
      `index_metadata.pickle`，下一个进程打开即 `Error loading hnsw index`。
      跳过之后，增量运行是零写入，索引文件不会被无谓重写。
    """
    wanted = [str(item["id"]) for item in chunks]
    existing: set[str] = set()
    for start in range(0, len(wanted), 512):
        existing.update(db._collection.get(ids=wanted[start : start + 512]).get("ids", []))

    todo = [(str(item["id"]), item) for item in chunks if str(item["id"]) not in existing]
    total = 0
    for start in range(0, len(todo), batch_size):
        part = todo[start : start + batch_size]
        documents = [
            Document(page_content=item["text"], metadata=_metadata_of(item)) for _, item in part
        ]
        db.add_documents(documents, ids=[cid for cid, _ in part])
        total += len(part)
    return total


def _metadata_of(chunk: dict[str, Any]) -> dict[str, Any]:
    """写入向量库的元数据。

    只放标量：Chroma 的 metadata 不支持嵌套结构，且过滤条件按这些字段走。
    """
    return {
        "source": chunk["source"],
        "kind": chunk["kind"],
        "line_start": int(chunk["line_start"]),
        "line_end": int(chunk["line_end"]),
        "start_index": int(chunk["start_index"]),
    }


def count_chunks(db: Any) -> int:
    """当前集合的向量数。用于写入前后的一致性检查。"""
    return int(db._collection.count())


def close_store(db: Any) -> None:
    """显式关闭 Chroma 客户端，触发 HNSW 段落盘。

    ★ 为什么必须有：写入只保证 sqlite 与内存一致，段目录可能只留下
      `index_metadata.pickle`。此时计数校验完全正常，但下一个进程打开会报
      `Error loading hnsw index` —— 只有在写完后显式关闭才会补齐二进制文件。
    """
    client = getattr(db, "_client", None)
    close = getattr(client, "close", None)
    if callable(close):
        close()
