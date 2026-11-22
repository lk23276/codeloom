"""测试共用的替身。

放在单独模块而不是某个 test_*.py 里：`tests` 是命名空间包，跨文件 `import tests._fakes`
可用；写在测试文件里则要 `from test_xxx import ...`，依赖收集顺序，容易踩到循环导入。
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """按顺序吐出预置消息的假模型。

    `bind_tools` 直接返回自身：图会对模型调用它，但脚本已经决定了下一条消息，
    这里的语义是「工具都已绑好」，不需要真的改行为。
    """

    responses: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 超出脚本长度时重复最后一条，避免「少写一条就报错」的噪声
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        template = self.responses[index]

        # ★ 每次都要造新消息、新 id：`add_messages` 按 id 合并，重复返回同一个对象
        #   会让第二轮的回复**替换**第一轮而不是追加（历史长度不变，但内容变了）。
        calls = [
            {**call, "id": f"call_{uuid4().hex[:12]}"} for call in (template.tool_calls or [])
        ]
        message = AIMessage(
            content=template.content,
            id=f"msg_{uuid4().hex[:12]}",
            tool_calls=calls,  # type: ignore[arg-type]
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        return self


def tool_call(expression: str, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": expression},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )
