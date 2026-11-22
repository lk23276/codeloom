"""ReAct 状态图：把 v0.1 手写的 Function Calling 循环换成 LangGraph。

对照关系（左侧是 `service/chat_service.py` 里的手写实现）：

    手写循环                             本模块
    ─────────────────────────────────    ────────────────────────────────
    while step < MAX_TOOL_STEPS          图的回路 + `recursion_limit`
    if not resp.tool_calls: break        add_conditional_edges(..., tools_condition)
    _DISPATCH[name](args) + 循环体       ToolNode(LC_TOOLS)
    messages.append(...) 手工维护        add_messages reducer

★ 本层的职责只有三件：声明状态、注册节点、连边。
  「什么时候停」不在这里判断 —— 交给 `tools_condition`；
  「消息怎么合并」也不在这里 —— 交给 `add_messages`。
  把这两件事写进节点函数，就退化成手写循环了。

★ 功能等价是验收标准：同一句「12*7 等于几」，图版与 `/chat` 手写版最终答案必须一致。
  工具实现共用一个（见 `tools/lc_tools.py`），所以差异只可能出在图结构上。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.client.lc_client import build_chat_model
from app.core.config import Settings
from app.tools.lc_tools import LC_TOOLS

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """图的状态。只有一个字段，但**必须带 reducer**。

    `Annotated[list[AnyMessage], add_messages]` 里的 `add_messages` 决定合并规则：
    节点返回 `{"messages": [新消息]}` 时按 id 追加或更新，而不是覆盖整个历史。

    ★ 去掉 reducer 的后果：agent 节点第二轮返回时把历史冲成一条，模型「失忆」，
      接口照常 200，答案却答非所问 —— 不报错的错，最难查。
    """

    messages: Annotated[list[AnyMessage], add_messages]


def build_agent_graph(
    settings: Settings,
    *,
    model: Any | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    interrupt_before: Sequence[str] | None = None,
) -> CompiledStateGraph:
    """构造并编译图。

    三个注入点，各有明确用途：

    | 参数 | 用途 |
    |---|---|
    | `model` | 测试注入假模型，断言图结构而不联网、不花钱 |
    | `checkpointer` | 记忆：不传则无状态；传 `InMemorySaver` / `AsyncSqliteSaver` 才有多轮 |
    | `interrupt_before` | 静态断点：`("tools",)` 表示工具执行前暂停，等待 `Command`/`None` 恢复 |

    ★ `checkpointer` 传了之后，**每次调用都必须带 `configurable.thread_id`**，
      否则 LangGraph 直接报错 —— 没有「默认单线程」这回事。
    """
    llm = model or build_chat_model(settings)
    llm_with_tools = llm.bind_tools(LC_TOOLS)

    async def agent(state: AgentState) -> dict[str, Any]:
        """调用模型。返回的是**增量**，由 `add_messages` 负责合并。"""
        reply = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [reply]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    # handle_tool_errors=True：把工具异常转成 ToolMessage 文本回给模型。
    # ★ 与手写路径 dispatch() 的 `_fail()` 对齐 —— 工具是「不可信输入的执行现场」，
    #   除零、格式错都该由模型转述；不设这个参数（1.x 默认只捕获 ToolException）
    #   会让 ZeroDivisionError 直接抛穿，整次请求变成 500。
    graph.add_node("tools", ToolNode(LC_TOOLS, handle_tool_errors=True))

    graph.add_edge(START, "agent")
    # 有 tool_calls -> tools，否则 -> END。这一行等价于手写循环里的终止判断
    graph.add_conditional_edges("agent", tools_condition)
    # ★ 回边是「循环」的来源：少了它，工具执行完图就结束，拿不到最终答案
    graph.add_edge("tools", "agent")

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=list(interrupt_before or ()),
    )
    logger.debug(
        "agent 图已编译：节点=%s 工具=%s",
        list(compiled.nodes),
        [t.name for t in LC_TOOLS],
    )
    return compiled
