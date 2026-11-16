"""知识库问答端点。

★ 这一层只做三件事：接参数、调 service、包响应体。
  「该不该拒答」「引用怎么映射」都属于业务判断，写在 service 里才能被单测直接覆盖。

★ 拒答返回 **200** 而不是 4xx：
  拒答是一个正常结果（知识库确实没有依据），不是请求错误。
  用 4xx 会让客户端把它当故障并重试，而重试不会改变检索结果 ——
  问题出在语料，不在这一次请求。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.core.deps import get_rag_service
from app.core.errors import ok
from app.core.ratelimit import rate_limit
from app.schema.rag import RagAnswer, RagRequest
from app.service.rag_service import RagService

router = APIRouter(tags=["rag"])


def _rid(request: Request) -> str:
    return getattr(request.state, "rid", "")


@router.post(
    "/rag/ask",
    summary="基于知识库回答（带引用与拒答）",
    dependencies=[Depends(rate_limit)],
)
async def rag_ask(
    body: RagRequest,
    request: Request,
    svc: RagService = Depends(get_rag_service),
) -> dict:
    """检索本地语料后回答，并返回引用位置。

    响应里的 `data` 形如：

        {
          "question": "...",
          "refused": false,
          "answer": "eventfd 用于唤醒子反应堆 [1] ...",
          "citations": [{"index": 1, "source": "server2.cpp",
                         "line_start": 345, "line_end": 362, "score": 2.71}],
          "invalid_citations": 0,
          "top_score": 2.71, "recalled": 20, "reranked": 5,
          "usage": {...}, "latency_ms": 1830
        }

    `citations` 里的位置来自语料的 metadata，不是模型写的；
    `invalid_citations` 大于 0 表示模型引用了不存在的块编号。
    """
    answer: RagAnswer = await svc.ask(body)
    return ok(answer.model_dump(), request_id=_rid(request))
