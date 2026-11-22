"""Agent 端点：多步工具调用，带记忆，可暂停审批。

★ 与 `/chat`、`/rag/ask` 的职责边界（三者不要互相替代）：

    /chat      轻问答，一轮或短多轮，不需要工具
    /rag/ask   文档问答，无状态，带引用与拒答
    /agent     工具 / 多步，有记忆（checkpointer），可停在审批点

★ 为什么 `awaiting_approval` 也返回 **200**：
  图停在断点是正常结果（设计如此），不是请求失败。客户端据此展示「待确认」界面，
  再用 `approve=true` 继续；用 4xx 表达会把「等待用户」误报成错误率。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.core.deps import get_agent_service
from app.core.errors import ok
from app.core.ratelimit import rate_limit
from app.schema.agent import AgentAnswer, AgentHistory, AgentRequest
from app.service.agent_service import AgentService

router = APIRouter(tags=["agent"])


def _rid(request: Request) -> str:
    return getattr(request.state, "rid", "")


@router.post(
    "/agent",
    summary="多步工具调用（带记忆，可暂停审批）",
    dependencies=[Depends(rate_limit)],
)
async def run_agent(
    body: AgentRequest,
    request: Request,
    svc: AgentService = Depends(get_agent_service),
) -> dict:
    """跑一次状态图。

    两种用法：

        {"message": "12*7 等于几"}                          -> 新建 thread，直接作答
        {"message": "...", "thread_id": "s-001"}            -> 续聊
        {"message": "...", "hitl": true}                    -> 工具执行前暂停
        {"thread_id": "s-001", "approve": true}             -> 继续被暂停的图

    `data` 形如：

        {
          "thread_id": "s-001",
          "status": "answered",          // 或 awaiting_approval
          "answer": "12 × 7 = 84",
          "pending": [],                  // 待执行的工具调用
          "tool_calls": ["calculator({\"expression\": \"12*7\"})"],
          "steps": 3, "usage": {...}, "latency_ms": 1520
        }
    """
    answer: AgentAnswer = await svc.ask(body)
    return ok(answer.model_dump(), request_id=_rid(request))


@router.get(
    "/agent/{thread_id}/messages",
    summary="某个 thread 的全部对话历史",
)
async def agent_history(
    thread_id: str,
    request: Request,
    svc: AgentService = Depends(get_agent_service),
) -> dict:
    """读图状态里的历史，供页面刷新后恢复。

    `data` 形如：

        {
          "thread_id": "a304b533…",
          "status": "answered",              // 或 awaiting_approval
          "messages": [{"role": "user", "content": "12*7 等于几"},
                       {"role": "assistant", "content": "12 × 7 = **84**"}],
          "pending": [],                      // 待审批的工具调用
          "count": 2
        }

    ★ 数据源是 checkpointer，不是另一张表：图和页面看同一份状态，不会分叉。
    """
    history: AgentHistory = await svc.history(thread_id)
    return ok(history.model_dump(), request_id=_rid(request))
