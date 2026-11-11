"""对话端点：非流式 + 流式。

★ 这一层只做三件事：接参数、调 service、包响应体。
  任何"业务判断"（比如该不该带历史、要不要拒答）都不属于这里——
  写在这里就没法被单测直接覆盖（得走 HTTP），也没法在别处复用。

★ 限流挂在**路由级**（`dependencies=[Depends(...)]`），不进函数签名：
  它只做副作用（检查配额，超了抛 RateLimited），没有返回值要注入。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.core.deps import get_chat_service
from app.core.errors import AppError, ok
from app.core.ratelimit import rate_limit
from app.core.sse import SSE_HEADERS, SSE_MEDIA_TYPE, done_frame, event_frame, json_frame
from app.schema.chat import ChatRequest
from app.service.chat_service import ChatService

router = APIRouter(tags=["chat"])


def _rid(request: Request) -> str:
    return getattr(request.state, "rid", "")


# ══════════════════════ 已实现：非流式 ══════════════════════


@router.post(
    "/chat",
    summary="一轮问答（非流式）",
    dependencies=[Depends(rate_limit)],
)
async def chat(
    body: ChatRequest,
    request: Request,
    svc: ChatService = Depends(get_chat_service),
) -> dict:
    """一次完整问答，等模型全部生成完才返回。

    ★ 这个 handler 只是接线——真正的五步在 `ChatService.chat()` 里。
      所以它不需要 try/except：service 已经把异常翻译成 UpstreamError，
      统一异常处理器会接住并返回 502 + envelope。

    ★ 限流走 `dependencies=[Depends(rate_limit)]`：它是纯副作用（检查配额），
      没有值要注入，所以不进函数签名。超限时抛 `RateLimited` → 429 + Retry-After。
    """
    answer = await svc.chat(body)
    return ok(answer.model_dump(), request_id=_rid(request))


# ══════════════════════ 已实现：流式 ══════════════════════


@router.post(
    "/chat/stream",
    summary="一轮问答（SSE 流式）",
    dependencies=[Depends(rate_limit)],
)
async def chat_stream(
    body: ChatRequest,
    request: Request,
    svc: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """逐 token 推送。

    事件序列（客户端按这个顺序处理）：
        meta   → {"session_id": ..., "model": ...}   前端拿 session_id 续聊
        delta  → 正文增量，可能几十上百个
        : ping → 长时间无 token 时保活
        done   → "[DONE]" 明确结束（★ 不能靠"连接关闭"表示结束，
                  那样客户端分不清"正常结束"和"网络断了"）

    ★ 这个 handler 只做一件事：把 service 产出的「纯文本增量」翻译成 SSE 帧。
      真正的流式逻辑在 `svc.open_stream()`。三个审查要点：

        ① `meta` 必须在第一个 delta 之前发 —— 前端要拿 `session_id` 才能续聊
        ② `except AppError` 只接业务异常；service 若让 `LLMError` 冒出来，
           这里接不住，流会断在半路（客户端既没有 error 也没有 done）
        ③ `done` 在 `try` 之外 —— 无论成功还是出错，都要给客户端一个明确结束信号。
           靠"连接关闭"表示结束是不行的：客户端分不清正常结束和网络断了

    ★ 为什么不是 `await svc.open_stream(...)` 之后 `await` 每个 delta：
      `deltas` 是异步迭代器，只能 `async for`，不能 `await`。
      而且必须在这里边收边 `yield`，一旦先收完再发，流式就失效了。

    ★ `open_stream()` 的签名是本骨架定下的：
          返回 (session_id, 异步生成器)
      这样 router 能先拿到 session_id 发 meta，再开始转发正文。
      service 签名变更时必须同步修改这里。
    """
    session_id, deltas = await svc.open_stream(body)

    async def gen() -> AsyncIterator[str]:
        try:
            yield json_frame("meta", {"session_id": session_id, "model": svc.model})
            async for delta in deltas:
                yield event_frame("delta", delta)
        except AppError as exc:
            # ★ 流内错误：状态码已经是 200，只能用事件告知
            yield json_frame("error", {"code": exc.code.value, "msg": exc.msg})
        yield done_frame()

    return StreamingResponse(gen(), media_type=SSE_MEDIA_TYPE, headers=SSE_HEADERS)
