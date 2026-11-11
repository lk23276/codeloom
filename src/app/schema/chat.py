"""请求 / 响应契约。

★ 这些类就是「接口文档的源码」：`/docs` 里的字段说明、约束区间、示例，
  全部来自这里的类型注解与 Field(...)。

★ 一个容易踩的点：Pydantic 类的 docstring 会进 `model_json_schema()`。
  如果某个模型将来被当作 Function Calling 的参数 schema，它的 docstring 会被【发给模型】，
  所以要按「给模型看的文案」写，而不是按「给同事看的注释」写。
  纯 HTTP 契约（下面这些）不受影响，可以自由写注释。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """一轮对话的请求。"""

    message: str = Field(min_length=1, max_length=2000, description="用户这一轮说的话")

    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="不传则新建会话；传了就接着聊",
    )

    history_limit: int = Field(
        default=20,
        ge=1,
        le=200,
        description="带多少条历史给模型（越少越省钱）",
    )

    max_tokens: int = Field(default=200, ge=1, le=1000, description="输出 token 上限")


class UsageInfo(BaseModel):
    """token 用量。字段缺失时给 0，避免前端判空。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatAnswer(BaseModel):
    """一轮对话的结果（非流式）。"""

    reply: str = Field(description="模型回复的正文")
    session_id: str = Field(description="会话 id；下一轮带回来就能续聊")
    usage: UsageInfo = Field(default_factory=UsageInfo)


class MessageOut(BaseModel):
    """一条消息（对外展示用）。"""

    role: str
    content: str
    created_at: str


class SessionMessages(BaseModel):
    """某个会话的全部消息。"""

    session_id: str
    messages: list[MessageOut]
    count: int


class SessionBrief(BaseModel):
    """会话列表里的一项。"""

    id: str
    created_at: str
    updated_at: str


class SessionList(BaseModel):
    sessions: list[SessionBrief]


class TrimResult(BaseModel):
    session_id: str
    removed: int
    left: int


class DeleteResult(BaseModel):
    session_id: str
    deleted: bool


class HealthOut(BaseModel):
    """探针的响应体。

    ★ 探针不套统一 envelope（定下的规矩）：消费者是编排系统，它只看状态码。
    """

    status: str
    version: str
    uptime_s: float = 0.0
    db: str | None = None
