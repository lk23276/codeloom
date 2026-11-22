"""重排：用 cross-encoder 给「问题 + 块」这一对打分。

为什么需要第二段检索：

    向量检索把 query 与 document **分别**压缩成一个向量，两边的细粒度对应
    在压缩中丢失。典型表现是自然语言问法与代码里的符号名分处两侧 ——
    问「边沿触发怎么设置」，答案那块里写的是 `EPOLLET`，两个向量的夹角不够小。

    cross-encoder 把两者**拼接后一起编码**，注意力同时看到问法与符号名，
    判定依据是「这一对」，因此比向量点积准得多。

代价是它无法预计算：每个候选都要一次前向，所以只对粗召回结果做，不做全库。
粗召回 20 条、精排取 5 条，是本项目规模下的固定组合。

★ 用 fastembed 的 ONNX 实现而不是 `sentence-transformers`：
  后者会引入 GB 级的 PyTorch 依赖，而这里只需要「打分」这一件事。
  与 embedding 共用同一套 ONNX 运行时和同一个缓存目录。

★ 输出是**原始 logits**，没有经过 sigmoid：可以为负，也可以远大于 1。
  因此阈值只能在标注集上标定，不能按「0.5 就是一半概率」去理解。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.core.config import Settings

logger = logging.getLogger(__name__)


class RerankNotAvailableError(RuntimeError):
    """fastembed 或重排模型不可用。"""


@runtime_checkable
class Reranker(Protocol):
    """业务层看到的全部能力：给一批候选打分，返回值与输入同序。"""

    def score(self, query: str, documents: list[str]) -> list[float]: ...


@lru_cache(maxsize=1)
def _cross_encoder_cls() -> Any:
    """延迟导入 fastembed 的 cross-encoder。

    未安装依赖时在**构造期**报错，而不是模块导入期 —— 与 `rag/vectorstore.py`
    的处理一致：只要不调用重排，缺少该依赖不应影响进程启动。
    """
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ModuleNotFoundError as exc:  # pragma: no cover - 取决于运行环境
        raise RerankNotAvailableError("缺少 fastembed：执行 `uv add fastembed` 后重试") from exc
    return TextCrossEncoder


class FastEmbedReranker:
    """fastembed cross-encoder 的适配层。

    模型参数规模远大于 embedding 模型（`bge-reranker-base` 约 1 GB），
    因此**首次调用才加载**：构造对象本身不触发下载，进程启动也不会因此变慢。
    """

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._encoder: Any = None

    @property
    def encoder(self) -> Any:
        if self._encoder is None:
            cls = _cross_encoder_cls()
            logger.info("加载重排模型 model=%s", self.model_name)
            try:
                self._encoder = cls(
                    model_name=self.model_name,
                    cache_dir=self._cache_dir,
                    lazy_load=True,
                )
            except Exception as exc:  # noqa: BLE001 —— 下载/加载失败形态很杂（OSError/ValueError/SSL）
                # ★ 转成领域异常：上层据此回 502 + 可操作文案，
                #   而不是让原始异常冒到全局处理器变成「服务器内部错误」。
                raise RerankNotAvailableError(
                    f"重排模型 {self.model_name} 加载失败（下载失败时设 HF_ENDPOINT 或预下载）："
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            logger.info("重排模型就绪 model=%s", self.model_name)
        return self._encoder

    def score(self, query: str, documents: list[str]) -> list[float]:
        """按文档顺序返回分数，越大越相关。

        空列表直接返回，不调用模型：fastembed 在空输入下仍会走一次批处理路径，
        而这里调用点很多，短路掉可以省掉无意义的开销。
        """
        if not documents:
            return []
        return [float(value) for value in self.encoder.rerank(query, list(documents))]


def build_reranker(settings: Settings) -> Reranker:
    """按配置构造重排器。

    缓存目录与 embedding 共用 `data/fastembed`：两者都是 ONNX 模型，
    放在同一处便于整体清理，也避免出现「索引删了、模型散落在系统临时目录」。

    这里只算出路径、不创建目录：构造对象不触碰磁盘，
    模型下载时由 fastembed 自行创建。启动路径因此保持零写入 ——
    测试里 `create_app()` 不会在仓库下留下空目录。
    """
    cache_dir = Path(settings.rag_persist_dir).parent / "fastembed"
    return FastEmbedReranker(settings.rag_rerank_model, cache_dir=str(cache_dir))
