"""Agent 编排的契约测试：记忆、隔离、HITL、错误边界。

四条「不会报错的错」：

1. 记忆没生效 —— 第二轮看不到第一轮，但接口照常 200
2. 分桶串了 —— 换了 `thread_id` 却读到别人的历史
3. 断点没停下来 —— 声称支持审批，实际直接执行了工具
4. `approve` 无效也能过 —— 回复 200 但什么也没做

★ 第 6 条用例专门验证「进程重启后仍记得」：`InMemorySaver` 也能通过前几条，
  只有换后端并重开连接，才能区分两者。
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent.checkpoint import open_checkpointer
from app.agent.graph import build_agent_graph
from app.core.config import Settings
from app.core.errors import InvalidInput
from app.schema.agent import AgentRequest
from app.service.agent_service import AgentService
from tests._fakes import ScriptedChatModel, tool_call

FINAL = "12 × 7 = 84"


def _script(rounds: int = 5) -> list:
    """造一段「每轮都调一次工具」的脚本。

    ★ 为什么不是两条就够：脚本模型的调用计数**跨轮累积**，写死两条的话第二轮会落到
      「直接回答」那条上，历史只增加 2 条，断言就随轮次漂移。每轮固定消耗两条，
      测试的期望值才是稳定的。
    """
    responses = []
    for i in range(rounds):
        responses.append(tool_call(f"{i + 1}*2", call_id=f"c{i}"))
        responses.append(_ai(f"{(i + 1) * 2}"))
    return responses


def make_service(
    *,
    responses: list | None = None,
    checkpointer=None,
    limit: int = 8,
) -> tuple[AgentService, object]:
    """组装一个可注入的 service，返回 (service, 图) 便于断言内部状态。"""
    settings = Settings(_env_file=None)
    cp = checkpointer or _memory_saver()
    model = ScriptedChatModel(responses=responses or _script())
    graph = build_agent_graph(settings, model=model, checkpointer=cp)
    graph_hitl = build_agent_graph(
        settings, model=model, checkpointer=cp, interrupt_before=("tools",)
    )
    service = AgentService(graph=graph, graph_hitl=graph_hitl, recursion_limit=limit)
    return service, graph


def _ai(text: str):
    from langchain_core.messages import AIMessage

    return AIMessage(content=text)


def _memory_saver():
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ══════════════════════ 一、基本流程 ══════════════════════


async def test_first_turn_answers_and_generates_thread_id() -> None:
    service, _ = make_service()

    answer = await service.ask(AgentRequest(message="12*7 等于几"))

    assert answer.status == "answered"
    assert answer.answer == "2"
    assert answer.tool_calls == ["calculator"]
    assert answer.steps == 3           # agent -> tools -> agent
    assert len(answer.thread_id) == 32  # uuid4().hex
    assert answer.pending == []


async def test_same_thread_accumulates_history() -> None:
    service, graph = make_service()
    first = await service.ask(AgentRequest(message="12*7 等于几"))

    await service.ask(AgentRequest(message="那 3*5 呢", thread_id=first.thread_id))

    state = await graph.aget_state(_config(first.thread_id))
    # 每轮 4 条：user / ai(tool_calls) / tool / ai
    assert len(state.values["messages"]) == 8


async def test_new_thread_starts_empty() -> None:
    service, graph = make_service()
    first = await service.ask(AgentRequest(message="12*7 等于几"))

    second = await service.ask(AgentRequest(message="12*7 等于几"))

    assert second.thread_id != first.thread_id
    state = await graph.aget_state(_config(second.thread_id))
    assert len(state.values["messages"]) == 4


# ══════════════════════ 二、HITL ══════════════════════


async def test_hitl_pauses_before_tools() -> None:
    service, _ = make_service()

    paused = await service.ask(AgentRequest(message="12*7 等于几", hitl=True))

    assert paused.status == "awaiting_approval"
    assert paused.answer == ""
    assert paused.pending == ['calculator({"expression": "1*2"})']
    assert paused.tool_calls == []      # 工具还没执行


async def test_approve_resumes_and_finishes() -> None:
    service, _ = make_service()
    paused = await service.ask(AgentRequest(message="12*7 等于几", hitl=True))

    resumed = await service.ask(AgentRequest(thread_id=paused.thread_id, approve=True))

    assert resumed.status == "answered"
    assert resumed.answer == "2"
    assert resumed.tool_calls == ["calculator"]


async def test_approve_without_pending_is_rejected() -> None:
    """已有 thread 但没有待审批操作：InvalidInput（400）。

    ★ 与「thread 不存在」是两件事：后者应为 NotFound（404），见下面的用例。
    """
    service, _ = make_service()
    first = await service.ask(AgentRequest(message="12*7 等于几"))

    with pytest.raises(InvalidInput):
        await service.ask(AgentRequest(thread_id=first.thread_id, approve=True))


# ══════════════════════ 三、持久化与并发 ══════════════════════


async def test_sqlite_checkpointer_survives_reopen(tmp_path) -> None:
    """关掉连接再重开，历史必须还在 —— 这是「重启不丢」的最小验证。"""
    path = tmp_path / "agent_memory.db"
    settings = Settings(
        _env_file=None, agent_checkpoint="sqlite", agent_memory_path=str(path)
    )

    async with open_checkpointer(settings) as cp:
        model = ScriptedChatModel(responses=[tool_call("12*7"), _ai(FINAL)])
        graph = build_agent_graph(settings, model=model, checkpointer=cp)
        service = AgentService(graph=graph)
        first = await service.ask(AgentRequest(message="12*7 等于几"))
        before = await graph.aget_state(_config(first.thread_id))

    async with open_checkpointer(settings) as cp:
        model2 = ScriptedChatModel(responses=[tool_call("12*7"), _ai(FINAL)])
        graph2 = build_agent_graph(settings, model=model2, checkpointer=cp)
        after = await graph2.aget_state(_config(first.thread_id))

    assert len(before.values["messages"]) == 4
    assert len(after.values["messages"]) == 4     # 重开连接后仍在
    assert after.values["messages"][0].content == "12*7 等于几"


async def test_same_thread_calls_are_serialized() -> None:
    """同一 thread 并发：串行化之后历史是 8 条，而不是互相覆盖成 4 条。"""
    service, graph = make_service()

    first = await service.ask(AgentRequest(message="12*7 等于几"))
    tid = first.thread_id

    await asyncio.gather(
        service.ask(AgentRequest(message="第一路", thread_id=tid)),
        service.ask(AgentRequest(message="第二路", thread_id=tid)),
    )

    state = await graph.aget_state(_config(tid))
    # 每轮固定 4 条（user / ai(tool_calls) / tool / ai）：首轮 + 两路 = 12。
    # 若两路并发写同一桶，其中一路会被覆盖，结果小于 12。
    assert len(state.values["messages"]) == 12


# ══════════════════════ 四、历史读取 ══════════════════════


async def test_history_returns_user_and_assistant_only() -> None:
    """历史只保留对话内容：中间态的 ai(tool_calls) 与 tool 结果都不出现。"""
    service, _ = make_service()
    first = await service.ask(AgentRequest(message="12*7 等于几"))
    await service.ask(AgentRequest(message="再加 1", thread_id=first.thread_id))

    history = await service.history(first.thread_id)

    assert [m.role for m in history.messages] == ["user", "assistant", "user", "assistant"]
    assert history.count == 4
    assert history.status == "answered"
    assert history.pending == []


async def test_history_reports_paused_thread() -> None:
    """暂停中的 thread：status 与 pending 都要能从历史里读出来（刷新后可恢复审批按钮）。"""
    service, _ = make_service()
    paused = await service.ask(AgentRequest(message="12*7 等于几", hitl=True))

    history = await service.history(paused.thread_id, hitl=True)

    assert history.status == "awaiting_approval"
    assert history.pending == [paused.pending[0]]
    assert [m.role for m in history.messages] == ["user"]


async def test_unknown_thread_is_rejected_not_created() -> None:
    """供了不存在的 thread_id：必须 404，不能静默新建一个同 id 的空 thread。

    ★ 与 `/chat` 的 `session_id` 口径一致。不校验时「id 拼错」会被伪装成 200，
      用户以为在续聊，实际开了新对话。
    """
    from app.core.errors import NotFound

    service, _ = make_service()
    with pytest.raises(NotFound):
        await service.ask(AgentRequest(message="1+1 等于几", thread_id="ghost-thread"))


async def test_approve_unknown_thread_is_rejected() -> None:
    from app.core.errors import NotFound

    service, _ = make_service()
    with pytest.raises(NotFound):
        await service.ask(AgentRequest(thread_id="ghost-thread", approve=True))


async def test_history_of_unknown_thread_raises() -> None:
    from app.core.errors import NotFound

    service, _ = make_service()
    with pytest.raises(NotFound):
        await service.history("never-started")


async def test_recursion_limit_is_reported_as_failure() -> None:
    """模型永远要求调工具时，图会撞上限 —— 异常应向上抛，而不是静默返回空答案。"""
    from langgraph.errors import GraphRecursionError

    service, _ = make_service(
        responses=[tool_call("1+1", call_id="c1")],  # 一直重复调用
        limit=4,
    )

    with pytest.raises(GraphRecursionError):
        await service.ask(AgentRequest(message="算一下"))
