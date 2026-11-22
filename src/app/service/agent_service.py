"""Agent 编排：把请求翻译成「对图的调用」，再把图的结果翻译成响应。

★ 这一层不认识 HTTP，也不认识 langgraph 的内部结构：
  它只做三件事 —— 选图、串行化同一 thread 的调用、把状态翻译成 `AgentAnswer`。

★ 为什么必须按 thread 串行化：
  同一 `thread_id` 的两个请求会读写同一个 checkpointer 桶。并发时后写的覆盖前面的，
  表现为「两条回复都正常，但第二轮只记得其中一条分支」。进程内用 `asyncio.Lock` 挡住；
  多实例部署时这把锁不成立，要换 Postgres checkpointer —— 那不在当前范围内。

★ 为什么 `approve` 要校验「确实有待审批的操作」：
  `approve=true` 但该 thread 没有暂停的图时，LangGraph 会把它当成一次空输入继续执行，
  拿到的是旧状态或直接结束。这种「成功但什么也没做」的响应最难排查，所以在边界上拒绝。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from app.core.errors import InvalidInput, NotFound
from app.schema.agent import AgentAnswer, AgentHistory, AgentMessage, AgentRequest
from app.schema.chat import UsageInfo

logger = logging.getLogger(__name__)


def _text_of(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


def _describe(call: dict[str, Any]) -> str:
    """把一次工具调用压成一行可读文本，例如 `calculator({"expression": "12*7"})`。"""
    args = json.dumps(call.get("args", {}), ensure_ascii=False)
    return f"{call.get('name')}({args})"


class _ThreadLocks:
    """按 thread_id 分配锁，带容量上限的 LRU。

    ★ 为什么不用无限字典：thread 数量等于会话数量，长期运行会无界增长。
      淘汰时跳过正被持有的锁，否则同一个 thread 会短暂出现两把锁。
    """

    def __init__(self, capacity: int = 1024) -> None:
        self._capacity = capacity
        self._locks: OrderedDict[str, asyncio.Lock] = OrderedDict()

    def _get(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
            self._evict()
        else:
            self._locks.move_to_end(key)
        return lock

    def _evict(self) -> None:
        while len(self._locks) > self._capacity:
            for key, lock in list(self._locks.items()):
                if not lock.locked():
                    del self._locks[key]
                    break
            else:  # 所有锁都在用，放弃本次淘汰
                return

    @asynccontextmanager
    async def acquire(self, key: str) -> AsyncIterator[None]:
        lock = self._get(key)
        async with lock:
            yield


class AgentService:
    """图的调用方。两张图只有「是否带断点」的区别，checkpointer 共用。"""

    def __init__(
        self,
        *,
        graph: CompiledStateGraph,
        graph_hitl: CompiledStateGraph | None = None,
        recursion_limit: int = 8,
    ) -> None:
        self._graph = graph
        self._graph_hitl = graph_hitl or graph
        self._recursion_limit = recursion_limit
        self._locks = _ThreadLocks()

    def _pick(self, request: AgentRequest) -> CompiledStateGraph:
        # 带断点的图用在两种场景：显式要求审批，或继续一次已暂停的执行
        return self._graph_hitl if (request.hitl or request.approve) else self._graph

    async def ask(self, request: AgentRequest) -> AgentAnswer:
        graph = self._pick(request)
        thread_id = request.thread_id or _new_thread_id()
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._recursion_limit,
        }

        started = time.perf_counter()
        async with self._locks.acquire(thread_id):
            # ★ 供了 thread_id 却没有这个 thread：与 /chat 一致报 404，不以新建代替。
            #   不校验的后果是「id 拼错」被伪装成正常 200，用户以为在续聊、实际开了新对话。
            if request.thread_id:
                snapshot0 = await graph.aget_state(config)
                if not (snapshot0.values or {}).get("messages"):
                    raise NotFound(f"thread {thread_id} 不存在")

            if request.approve and not await self._has_pending(graph, config):
                raise InvalidInput(f"thread {thread_id} 没有待审批的操作")

            payload = None if request.approve else {"messages": [("user", request.message)]}
            collected = await self._drain(graph, payload, config)
            snapshot = await graph.aget_state(config)

        return AgentAnswer(
            thread_id=thread_id,
            status="awaiting_approval" if snapshot.next else "answered",
            answer="" if snapshot.next else collected["answer"],
            pending=collected["pending"] if snapshot.next else [],
            tool_calls=collected["executed"],
            steps=collected["steps"],
            usage=collected["usage"],
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def history(self, thread_id: str, *, hitl: bool = False) -> AgentHistory:
        """读某个 thread 的历史。

        ★ 只读 checkpointer，不另存一份：图的每条消息本来就在状态快照里。
          另存一份的后果是「页面显示的历史」和「模型看到的上下文」会分叉。
        """
        graph = self._pick_for_history(hitl)
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(config)

        raw = (snapshot.values or {}).get("messages") or []
        if not raw and not snapshot.next:
            raise NotFound(f"thread {thread_id} 不存在或没有历史")

        messages: list[AgentMessage] = []
        pending: list[str] = []
        for message in raw:
            if isinstance(message, HumanMessage):
                messages.append(AgentMessage(role="user", content=_text_of(message)))
                continue
            if isinstance(message, AIMessage):
                calls = message.tool_calls or []
                if calls:
                    # 有 tool_calls 的中间态不进历史；如果是最后一条且图没走完，就是待审批项
                    pending = [_describe(call) for call in calls]
                    continue
                text = _text_of(message)
                if text.strip():
                    messages.append(AgentMessage(role="assistant", content=text))

        if not snapshot.next:
            pending = []

        return AgentHistory(
            thread_id=thread_id,
            status="awaiting_approval" if snapshot.next else "answered",
            messages=messages,
            pending=pending,
            count=len(messages),
        )

    def _pick_for_history(self, hitl: bool) -> CompiledStateGraph:
        # 两张图共用同一个 checkpointer，读哪个都行；带断点的那张能拿到完整的 state.next
        return self._graph_hitl if hitl else self._graph

    async def _has_pending(self, graph: CompiledStateGraph, config: dict[str, Any]) -> bool:
        snapshot = await graph.aget_state(config)
        return bool(snapshot.next)

    async def _drain(
        self,
        graph: CompiledStateGraph,
        payload: Any,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """跑完图（或跑到断点），把散落在各帧里的信息收集起来。"""
        answer = ""
        steps = 0
        pending: list[str] = []
        executed: list[str] = []
        usage = UsageInfo()

        async for event in graph.astream(payload, config, stream_mode="updates"):
            for node, update in event.items():
                # ★ 断点会让流里多出一帧 `__interrupt__`，值是元组而不是 dict。
                #   它不是节点执行，跳过；不跳会让整次请求 500（`tuple.get` 不存在）。
                if not isinstance(update, dict):
                    logger.debug("agent_interrupt node=%s", node)
                    continue
                steps += 1
                messages = update.get("messages", [])
                logger.debug("agent_frame node=%s messages=%d", node, len(messages))
                for message in messages:
                    calls = getattr(message, "tool_calls", None) or []
                    if isinstance(message, AIMessage):
                        usage = _add_usage(usage, message)
                        if calls:
                            # 模型刚提出的调用：先记为待执行，工具跑完就清空
                            pending = [_describe(call) for call in calls]
                        else:
                            answer = _text_of(message)
                        continue

                    # ToolMessage：说明上一批调用已经执行
                    if getattr(message, "type", "") == "tool":
                        # ★ 工具调用必须留痕：没有这行，线上只能靠猜「工具到底调没调」
                        logger.info(
                            "agent_tool name=%s status=%s chars=%d",
                            getattr(message, "name", "?"),
                            getattr(message, "status", "?"),
                            len(_text_of(message)),
                        )
                        # 用工具名而不是完整的 pending 描述：恢复执行时（approve）
                        # 这一轮没有「模型提出调用」的帧，只能从 ToolMessage 自身取名字
                        executed.append(str(getattr(message, "name", "?")))
                        pending = []

        return {
            "answer": answer,
            "pending": pending,
            "executed": executed,
            "steps": steps,
            "usage": usage,
        }


def _new_thread_id() -> str:
    return uuid.uuid4().hex


def _add_usage(total: UsageInfo, message: AIMessage) -> UsageInfo:
    meta = getattr(message, "usage_metadata", None) or {}
    return UsageInfo(
        prompt_tokens=total.prompt_tokens + int(meta.get("input_tokens") or 0),
        completion_tokens=total.completion_tokens + int(meta.get("output_tokens") or 0),
        total_tokens=total.total_tokens + int(meta.get("total_tokens") or 0),
    )
