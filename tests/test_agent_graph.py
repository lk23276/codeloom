"""ReAct 状态图的契约测试：不联网、不花钱。

锁住四条**不会报错的错**：

1. `messages` 有没有 reducer —— 漏了会把历史冲成一条，模型「失忆」但接口 200
2. 帧序列是否 agent -> tools -> agent —— 少了回边，工具执行完图就结束
3. 工具异常是否变成 ToolMessage —— 抛穿会把整次请求变成 500
4. `@tool` 的名字与参数 Schema 是否与手写路径一致 —— 不一致时模型会用错

★ 用一个脚本化的假模型（`tests/_fakes.py` 的 `ScriptedChatModel`）而不是真模型：
  图的结构问题是确定性的，用真模型只会把「图错了」和「模型今天不听话」混在一起。
"""

from __future__ import annotations

from typing import get_type_hints

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from app.agent.graph import AgentState, build_agent_graph
from app.core.config import Settings
from app.tools.lc_tools import LC_TOOLS, calculator
from tests._fakes import ScriptedChatModel, tool_call

# ══════════════════════ 替身 ══════════════════════


def build(responses: list[AIMessage]):
    settings = Settings(_env_file=None)  # 不读本机 .env
    return build_agent_graph(settings, model=ScriptedChatModel(responses=responses))


# ══════════════════════ 一、状态声明 ══════════════════════


def test_messages_field_declares_add_messages_reducer() -> None:
    """`messages` 必须带 reducer。去掉 `Annotated[...]` 时这条会失败。

    ★ 用 `get_type_hints(include_extras=True)` 而不是 `__annotations__`：
      模块开了 `from __future__ import annotations`，注解在源码里是字符串，
      直接读只会拿到 "Annotated[list[AnyMessage], add_messages]" 这段文本。
    """
    hints = get_type_hints(AgentState, include_extras=True)
    assert getattr(hints["messages"], "__metadata__", ()) == (add_messages,)


async def test_history_accumulates_instead_of_overwritten() -> None:
    """跑完一轮后历史应是 4 条；若 reducer 失效，只剩 1 条。

    ★ 必须用 `ainvoke`：agent 节点是 `async def`，同步入口会报
      「No synchronous function provided to agent」。
    """
    graph = build([tool_call("12*7"), AIMessage(content="12 × 7 = 84")])

    result = await graph.ainvoke(
        {"messages": [("user", "12*7 等于几")]}, config={"recursion_limit": 8}
    )

    history = result["messages"]
    assert len(history) == 4, [type(m).__name__ for m in history]
    assert isinstance(history[0], HumanMessage)
    assert history[-1].content == "12 × 7 = 84"


# ══════════════════════ 二、图结构与执行 ══════════════════════


async def test_frame_sequence_is_agent_tools_agent() -> None:
    """`updates` 的帧序列必须是 agent -> tools -> agent。"""
    graph = build([tool_call("12*7"), AIMessage(content="12 × 7 = 84")])

    frames: list[str] = []
    tool_outputs: list[str] = []
    async for event in graph.astream({"messages": [("user", "12*7 等于几")]}):
        for node, payload in event.items():
            frames.append(node)
            for message in payload.get("messages", []):
                if getattr(message, "type", "") == "tool":
                    tool_outputs.append(str(message.content))

    assert frames == ["agent", "tools", "agent"]
    assert tool_outputs == ["84"]


async def test_tool_exception_becomes_tool_message() -> None:
    """工具抛异常要被 ToolNode 转成文本，而不是让整次调用失败。"""
    graph = build([tool_call("1/0"), AIMessage(content="这个表达式不合法")])

    frames: list[str] = []
    tool_outputs: list[str] = []
    async for event in graph.astream({"messages": [("user", "1/0 等于几")]}):
        for node, payload in event.items():
            frames.append(node)
            for message in payload.get("messages", []):
                if getattr(message, "type", "") == "tool":
                    tool_outputs.append(str(message.content))

    assert frames == ["agent", "tools", "agent"]
    assert tool_outputs, "没有产生 ToolMessage"
    assert "ZeroDivisionError" in tool_outputs[0]


# ══════════════════════ 三、工具封装 ══════════════════════


def test_tool_name_and_schema_match_handwritten_path() -> None:
    """名字与参数名必须与 `registry.TOOLS` 一致，否则模型会调不到。"""
    assert calculator.name == "calculator"
    assert calculator.args == {"expression": {"title": "Expression", "type": "string"}}
    assert "计算" in (calculator.description or "")


def test_lc_tools_exports_the_registry_tool() -> None:
    assert [tool.name for tool in LC_TOOLS] == ["calculator"]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("12*7", "84"),
        ("2**40", "1099511627776"),
        ("123456789*987654321", "121932631112635269"),
    ],
)
def test_lc_tool_delegates_to_registry(expression: str, expected: str) -> None:
    """封装层不重新实现计算，结果必须与 `registry._calculator` 完全一致。"""
    assert calculator.invoke({"expression": expression}) == expected
