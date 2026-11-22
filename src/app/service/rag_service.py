"""知识库问答的编排：召回 → 重排 → 阈值判定 → 作答 → 引用映射。

★ 分层规矩与 `ChatService` 相同，违反即破坏分层，而且不会报错：

    - 不 `import fastapi` —— 业务层不认识 HTTP
    - 不 `import fastembed` —— 模型细节全在 `rag/rerank.py`
    - 不 `import openai`  —— SDK 细节全在 `client/llm.py`
    - 只认识本项目自己的类型：`RagRequest` / `RagAnswer` / `LLMClient` / `Reranker`

★ 顺序上有一处不可调换：**先判阈值，再调模型**。
  反过来（先让模型作答，再判断该不该拒）有两个代价：多一次调用费用；
  「模型是否愿意承认不知道」不可控，指标无法复现。拒答必须由规则决定。

★ 重排与检索都是同步的 CPU 密集操作，必须经 `run_sync` 进线程池。
  直接调用会堵死事件循环 —— 与 `sqlite3` 是同一类问题（见 `core/async_utils.py`）。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from langchain_core.documents import Document

from app.client.llm import LLMClient, LLMError
from app.core.async_utils import run_sync
from app.core.errors import UpstreamError
from app.obs.local_trace import NullTracer, Tracer
from app.rag.prompt import build_rag_messages, parse_citation_indices
from app.rag.rerank import Reranker, RerankNotAvailableError
from app.rag.retriever import retrieve_with_score
from app.rag.vectorstore import EmbeddingNotAvailableError
from app.schema.chat import UsageInfo
from app.schema.rag import Citation, RagAnswer, RagRequest

logger = logging.getLogger(__name__)

EMPTY_KB_REASON = "知识库为空或尚未建立索引"
NO_EVIDENCE_REASON = "知识库中无相关依据"

# 检索函数签名：问题 -> 候选列表（Document 与距离）。抽成类型是为了可注入。
RetrieveFn = Callable[[str, int], list[tuple[Document, float]]]


def default_retrieve(question: str, k: int) -> list[tuple[Document, float]]:
    """默认走向量检索。测试通过构造函数注入替身，不必建真索引。"""
    return retrieve_with_score(question, k=k)


class RagService:
    """一次知识库问答的业务编排。

    无状态：所有可变的部分（llm、reranker、配置）都在构造时传入，
    每次请求新建一个实例即可（与 `ChatService` 的取舍一致）。
    """

    def __init__(
        self,
        llm: LLMClient,
        reranker: Reranker,
        *,
        recall_k: int = 20,
        top_n: int = 5,
        min_score: float = 0.2,
        retrieve_fn: RetrieveFn | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._llm = llm
        self._reranker = reranker
        self._recall_k = recall_k
        self._top_n = top_n
        self._min_score = min_score
        self._retrieve: RetrieveFn = retrieve_fn or default_retrieve
        # 默认不记：测试与脚本不该产生 trace 文件；线上由 deps 注入真 tracer
        self._tracer: Tracer = tracer or NullTracer()
        self.model = llm.model

    async def ask(self, req: RagRequest) -> RagAnswer:
        """完整回答一次提问。拒答与作答都返回 `RagAnswer`，不抛业务异常。"""
        started = time.perf_counter()
        top_n = req.top_n or self._top_n

        # ① 粗召回。宽一些，给重排留出捞回的空间。
        with self._tracer.span("retrieve", k=self._recall_k) as span:
            try:
                candidates = await run_sync(self._retrieve, req.question, self._recall_k)
            except EmbeddingNotAvailableError as exc:
                # ★ 502 而不是 500：这是「依赖没就绪」，不是「服务写错了」。
                logger.warning("rag_embedding_unavailable: %s", exc)
                raise UpstreamError(
                    f"{exc}（下载失败时设置 HF_ENDPOINT，或执行 scripts/prefetch_models.py 预下载）"
                ) from exc
            span["hits"] = len(candidates)
        if not candidates:
            logger.info("rag_empty reason=%s", EMPTY_KB_REASON)
            return self._refuse(req, EMPTY_KB_REASON, started)

        # ② 精排。cross-encoder 逐对打分，只对候选集做，不做全库。
        documents = [doc for doc, _ in candidates]
        with self._tracer.span("rerank", candidates=len(documents)) as span:
            try:
                scores = await run_sync(
                    self._reranker.score, req.question, [doc.page_content for doc in documents]
                )
            except RerankNotAvailableError as exc:
                logger.warning("rag_rerank_unavailable: %s", exc)
                raise UpstreamError(
                    f"{exc}（下载失败时设置 HF_ENDPOINT，或执行 scripts/prefetch_models.py 预下载）"
                ) from exc
            span["scored"] = len(scores)
        # 防御：长度不一致会让候选与分数错位，而错位不报错、只是答案对应到别的文件。
        if len(scores) != len(documents):
            raise UpstreamError(
                f"重排返回的分数个数与候选不一致：{len(scores)} != {len(documents)}"
            )

        # strict=True：上面已校验长度，这里再保一层 —— 万一将来有人删掉上面的校验，
        # 长度不一致会立刻抛错，而不是按较短的序列静默截断。
        ranked = sorted(
            zip(documents, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )[:top_n]
        top_score = float(ranked[0][1])

        # ③ 阈值判定。低于阈值直接拒，不产生模型调用。
        if top_score < self._min_score:
            logger.info(
                "rag_refused top_score=%.4f threshold=%.4f recalled=%d",
                top_score,
                self._min_score,
                len(documents),
            )
            return self._refuse(
                req,
                NO_EVIDENCE_REASON,
                started,
                recalled=len(documents),
                reranked=len(ranked),
                top_score=top_score,
            )

        # ④ 作答。提示词只允许引用编号，位置由下一步映射。
        try:
            with self._tracer.span("llm", model=self.model, top_n=len(ranked)) as span:
                result = await self._llm.achat(
                    build_rag_messages(req.question, ranked),
                    max_tokens=req.max_tokens,
                )
                # llm 这层必须带 token：否则 trace 里看不到成本归因
                span["prompt_tokens"] = result.usage.prompt_tokens
                span["completion_tokens"] = result.usage.completion_tokens
        except LLMError as exc:
            logger.warning("rag_llm_failed: %s", exc)
            raise UpstreamError(f"模型服务暂时不可用：{type(exc).__name__}") from exc

        # ⑤ 引用映射 + 越界计数。
        citations, invalid = self._map_citations(result.text, ranked)
        usage = result.usage
        return RagAnswer(
            question=req.question,
            refused=False,
            answer=result.text,
            citations=citations,
            invalid_citations=invalid,
            top_score=top_score,
            recalled=len(documents),
            reranked=len(ranked),
            usage=UsageInfo(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ),
            latency_ms=self._elapsed_ms(started),
        )

    # ══════════════════════ 内部：拒答与引用 ══════════════════════

    def _refuse(
        self,
        req: RagRequest,
        reason: str,
        started: float,
        *,
        recalled: int = 0,
        reranked: int = 0,
        top_score: float | None = None,
    ) -> RagAnswer:
        """构造拒答结果。没有调用模型，因此 usage 全为 0。"""
        return RagAnswer(
            question=req.question,
            refused=True,
            reason=reason,
            top_score=top_score,
            recalled=recalled,
            reranked=reranked,
            latency_ms=self._elapsed_ms(started),
        )

    def _map_citations(
        self,
        text: str,
        ranked: list[tuple[Document, float]],
    ) -> tuple[list[Citation], int]:
        """把正文里的 `[n]` 映射回块的位置。

        越界编号只计数、不抛异常：它是**模型行为异常**的信号，
        应当体现在 `invalid_citations` 上，而不是变成 500 让整次请求失败。
        """
        citations: list[Citation] = []
        invalid = 0
        for index in parse_citation_indices(text):
            if not 1 <= index <= len(ranked):
                invalid += 1
                continue
            doc, score = ranked[index - 1]
            meta = doc.metadata or {}
            citations.append(
                Citation(
                    index=index,
                    source=str(meta.get("source", "")),
                    line_start=_as_int(meta.get("line_start")),
                    line_end=_as_int(meta.get("line_end")),
                    score=float(score),
                )
            )
        return citations, invalid

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)


def _as_int(value: object) -> int:
    """metadata 里的行号按整数用；缺失或类型异常时退回 0，不抛异常。"""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
