"""Agent 接口的请求 / 响应契约。

与 `/chat`、`/rag/ask` 的区别在**状态**：`/agent` 的图状态存在 checkpointer 里，
按 `thread_id` 分桶，因此多出两个概念：

    thread_id  记忆分桶键（等价于 `/chat` 的 session_id）
    status     可能停在「等待审批」，而不是每次都以答案收尾

两种用法互斥：发起（给 message）与恢复（approve=true）。放在同一个模型里而不是拆两个端点，
是因为它们操作的是同一份图状态 —— 拆开会让「恢复的是哪个 thread」变成两处校验。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schema.chat import UsageInfo


class AgentRequest(BaseModel):
    """一次 agent 调用。"""

    message: str | None = Field(
        default=None,
        min_length=1,
        max_length=2000,
        description="这一轮的问题；恢复时不要提供",
    )
    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str | None) -> str | None:
        """恢复（approve）时 message 为 None；只校验真正给了内容的情况。"""
        if value is None:
            return None
        text = value.strip()
        if not text:
            raise ValueError("message 不能只包含空白字符")
        return text

    thread_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="记忆分桶键；不传则新建一个",
    )
    approve: bool = Field(default=False, description="继续上一次被暂停的图")
    hitl: bool = Field(
        default=False,
        description="为 True 时在工具执行前暂停，等 approve 再继续",
    )

    @model_validator(mode="after")
    def _exactly_one_intent(self) -> AgentRequest:
        if self.approve and self.message is not None:
            raise ValueError("approve 与 message 不能同时提供")
        if not self.approve and not self.message:
            raise ValueError("message 与 approve 至少提供一个")
        if self.approve and not self.thread_id:
            # 恢复必须指向某个具体的图状态，没有 thread_id 就无从恢复
            raise ValueError("approve 必须与 thread_id 一起提供")
        return self


class AgentAnswer(BaseModel):
    """一次 agent 调用的结果。

    两个状态：

        answered           本轮已收敛，answer 为最终答案
        awaiting_approval  图停在工具执行前，pending 列出待执行调用
    """

    thread_id: str = Field(description="本次使用的分桶键；新建时由服务端生成")
    status: Literal["answered", "awaiting_approval"] = Field(description="本轮结束时的图状态")
    answer: str = Field(default="", description="最终答案；awaiting_approval 时为空")
    pending: list[str] = Field(
        default_factory=list,
        description="待执行的工具调用（名称与参数），仅 awaiting_approval 时非空",
    )
    tool_calls: list[str] = Field(default_factory=list, description="本轮已执行的工具名")
    steps: int = Field(default=0, description="图执行了多少帧（agent/tools 各算一帧）")
    usage: UsageInfo = Field(default_factory=UsageInfo, description="累计 token 用量")
    latency_ms: int = Field(default=0, description="端到端耗时（毫秒）")


class AgentMessage(BaseModel):
    """历史里的一条消息（只保留对话内容，工具中间态不进历史）。"""

    role: Literal["user", "assistant"] = Field(description="消息角色")
    content: str = Field(description="消息正文")


class AgentHistory(BaseModel):
    """某个 thread 的全部对话历史。

    ★ 数据来源是 **checkpointer**，不是另建一张表：
      图的每条消息本来就存在状态快照里，再存一份必然分叉。
    """

    thread_id: str = Field(description="分桶键")
    status: Literal["answered", "awaiting_approval"] = Field(description="当前图状态")
    messages: list[AgentMessage] = Field(default_factory=list, description="按时间正序")
    pending: list[str] = Field(default_factory=list, description="待执行的工具调用")
    count: int = Field(default=0, description="消息条数")
