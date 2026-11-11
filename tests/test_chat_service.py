"""ChatService 的关键路径测试。

★ 这个文件测的不是「覆盖率」，是**十类不会报错的错误**。
  它们共同的特点：程序不崩、状态码正常、字段名也对，只是悄悄做错事。

      ① 顺序错：先调模型再落库        -> 用户的问题会丢（只有读代码才发现）
      ② 不拼历史：多轮等于单轮        -> 模型"失忆"，但接口照样 200
      ③ usage 丢失：字段名写错大小写   -> 成本统计全 0（真实踩过）
      ④ 静默新建：拼错 session_id     -> 200 但开了个新会话（决策是 404）
      ⑤ 异常没翻译：LLMError 直接冒   -> router 接不住，流断在半路
      ⑥ 流式分段/顺序错：拼接后才发现 -> 逐字断言才看得出来
      ⑦ 流式异常不翻译：同上           -> 客户端既没 error 也没 done
      ⑧ 流式 404 拖进生成器           -> 状态码从 404 变成 200
      ⑨ 假流式：先收完再吐            -> 用户等全部生成完，需按时序断言
      ⑩ 重复注入 system               -> prompt 变长、每轮白花钱

用法：
    uv run pytest tests/test_chat_service.py -q
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest

from app.client.llm import ChatResult, FakeClient, LLMError, ToolCall, Usage
from app.core.errors import NotFound, UpstreamError
from app.db.store import MessageStore
from app.schema.chat import ChatRequest
from app.service.chat_service import MAX_TOOL_STEPS, TOOL_FALLBACK, ChatService

# ══════════════════════ 夹具与替身 ══════════════════════


@pytest.fixture
def store(tmp_path) -> Iterator[MessageStore]:
    """每个测试一个独立的真实数据库文件。

    ★ 不能用 `:memory:` —— MessageStore 直接拒绝，因为每个连接会拿到
      各自独立的内存库，多线程下会出现「两个请求看到两个库」。
    """
    s = MessageStore(str(tmp_path / "sessions.db"))
    yield s
    s.close()


def make_service(
    store: MessageStore,
    *,
    reply: str = "假回复",
    chunks: list[str] | None = None,
) -> tuple[ChatService, FakeClient]:
    """造一个「不联网、不花钱、结果可预测」的 service。"""
    llm = FakeClient(reply=reply, chunks=chunks)
    return ChatService(llm=llm, store=store), llm


class FailOnChat(FakeClient):
    """非流式必失败的替身。"""

    async def achat(self, messages, **kw):  # type: ignore[no-untyped-def]
        raise LLMError("模拟上游超时")


class FailMidStream(FakeClient):
    """先吐一段再失败，用来验证「部分输出保留 + 异常翻译」。"""

    async def astream(self, messages, **kw):  # type: ignore[no-untyped-def]
        yield "第一段"
        raise LLMError("模拟上游断开")


class RecordingClient(FakeClient):
    """在 FakeClient 之上记录**每轮真正发出去的东西**。

    ★ 两个细节都不是多余的：
      1. `sent` 存的是 `list(messages)` 的**快照** —— `chat()` 拿到同一个 list
         之后还会往里 append（assistant 申请单 + tool 结果），
         直接存引用的话，所有轮次看到的都是"最终状态"，断言第 1 轮就失真了。
      2. `kwargs` 用来验证 `tools` **每轮都在** —— FakeClient 原本只记 messages。
    """

    def __init__(self, script: list[ChatResult]) -> None:
        super().__init__(script=script)
        self.sent: list[list[dict[str, Any]]] = []
        self.kwargs: list[dict[str, Any]] = []

    async def achat(self, messages, **kw):  # type: ignore[no-untyped-def]
        self.sent.append(list(messages))
        self.kwargs.append(dict(kw))
        return await super().achat(messages, **kw)


# ══════════════════════ ① 五步顺序 ══════════════════════


async def test_chat_sends_system_then_user(store: MessageStore) -> None:
    """模型收到的第一条必须是 system，第二条才是用户这句话。

    ★ 为什么测这个：`_build_messages` 里 system 是「每轮都发」的，
      一旦顺序反了（用户消息在前），模型会把 system 当成用户说的话。
      接口仍然返回 200，只有断言 messages 才看得出来。
    """
    svc, llm = make_service(store)

    answer = await svc.chat(ChatRequest(message="用一句话介绍 epoll"))

    assert answer.reply == "假回复"
    assert answer.session_id, "不传 session_id 时应自动生成一个"

    sent = llm.calls[-1]
    assert [m["role"] for m in sent] == ["system", "user"], sent
    assert sent[0]["content"], "system 提示不能为空"
    assert sent[1]["content"] == "用一句话介绍 epoll"


# ══════════════════════ ② 多轮记忆 ══════════════════════


async def test_chat_second_turn_carries_history(store: MessageStore) -> None:
    """第二轮要把上一轮的 user + assistant 带上，而不是只发这一句。

    ★ 这是「会话记忆」的唯一实现方式：模型是无状态的，记忆完全靠
      每次把历史重新拼进去。拼错了接口不会报错，只是模型「失忆」。
    """
    svc, llm = make_service(store, reply="epoll 是 I/O 多路复用")

    first = await svc.chat(ChatRequest(message="用一句话介绍 epoll"))
    await svc.chat(ChatRequest(message="刚才我问了什么？", session_id=first.session_id))

    sent = llm.calls[-1]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"], sent
    assert sent[1]["content"] == "用一句话介绍 epoll"
    assert sent[2]["content"] == "epoll 是 I/O 多路复用"
    assert sent[3]["content"] == "刚才我问了什么？"


# ══════════════════════ ③ usage 透传（回归测试） ══════════════════════


async def test_chat_propagates_usage(store: MessageStore) -> None:
    """`usage` 必须原样透传，不能是 0。

    ★ 这是一个真实 bug 的回归测试：
      构造 `ChatAnswer` 时把字段名写成 `Usage=`（大写），Pydantic 默认
      `extra="ignore"` 会**直接丢弃**这个多余参数，而 `usage` 有默认值，
      于是安静地变成 0 —— 不报错、不告警、接口结构全对。

      FakeClient 固定返回 10 / 5 / 15，所以断言具体数值就能卡住它。
    """
    svc, _ = make_service(store)

    answer = await svc.chat(ChatRequest(message="hi"))

    assert answer.usage.prompt_tokens == 10
    assert answer.usage.completion_tokens == 5
    assert answer.usage.total_tokens == 15


# ══════════════════════ ④ 未知 session_id ══════════════════════


async def test_chat_unknown_session_raises_not_found(store: MessageStore) -> None:
    """传了不存在的 session_id -> NotFound，而且**不能**被静默创建。

    ★ 设计决策：`ensure_conversation` 内部是 `INSERT OR IGNORE`，
      直接用会把「客户端拼错 id」伪装成 200。所以先判存在性。
      断言 `conversation_exists` 为 False，就是验证「没有偷偷写库」。
    """
    svc, llm = make_service(store)

    with pytest.raises(NotFound):
        await svc.chat(ChatRequest(message="hi", session_id="typo123"))

    assert store.conversation_exists("typo123") is False
    assert llm.calls == [], "校验失败时不应该调模型（别白花钱）"


# ══════════════════════ ⑤ 上游异常翻译 + 保留 user ══════════════════════


async def test_chat_llm_error_becomes_upstream_error(store: MessageStore) -> None:
    """上游失败 -> UpstreamError，且历史里留着 user 消息。

    ★ 两个断言各管一件事：
      1. 异常被翻译成本层语言（router 不需要认识 LLMError）
      2. 「先落库再调模型」的收益 —— 失败时用户的问题不丢，可以重试
    """
    svc = ChatService(llm=FailOnChat(), store=store)

    with pytest.raises(UpstreamError):
        await svc.chat(ChatRequest(message="hi"))

    conversations = store.conversations()
    assert len(conversations) == 1
    roles = [m["role"] for m in store.all_messages(conversations[0]["id"])]
    assert roles == ["user"], "失败时只应留下 user，不该有假回答"


# ══════════════════════ ⑥ 流式：增量顺序 ══════════════════════


async def test_open_stream_yields_deltas_in_order(store: MessageStore) -> None:
    """流式必须逐段产出，顺序与分段都不能变。

    ★ 先划清这条测的边界：它只验证「产出顺序和分段」，
      **测不出「先收完再拆开发」** —— 那种实现产出的列表一模一样。
      真正的「边收边发」由下面第 ⑨ 条（门控替身）负责。
    """
    svc, _ = make_service(store, chunks=["你", "好", "呀"])

    session_id, deltas = await svc.open_stream(ChatRequest(message="hi"))

    assert session_id
    got = [d async for d in deltas]
    assert got == ["你", "好", "呀"]
    assert all(isinstance(d, str) for d in got)


# ══════════════════════ ⑦ 流式：异常发生在流内 ══════════════════════


async def test_open_stream_llm_error_raised_during_iteration(store: MessageStore) -> None:
    """上游异常要在**迭代时**抛出，且已产出的增量保留。

    ★ 三个断言分别验证：
      1. `await open_stream(...)` 本身**不抛** —— 说明错误没有提前到
         await 阶段（那会变成 500，而不是流内 error 事件）
      2. 已产出的 "第一段" 没丢（客户端至少看到了一部分）
      3. 抛的是 UpstreamError（router 的 `except AppError` 能接住）
    """
    svc = ChatService(llm=FailMidStream(), store=store)

    _session_id, deltas = await svc.open_stream(ChatRequest(message="hi"))  # 不应抛

    got: list[str] = []
    with pytest.raises(UpstreamError):
        async for delta in deltas:
            got.append(delta)

    assert got == ["第一段"]


# ══════════════════════ ⑧ 流式：未知 id 在 await 阶段抛 ══════════════════════


async def test_open_stream_unknown_session_raises_before_streaming(
    store: MessageStore,
) -> None:
    """流式的 404 必须在 `await` 阶段抛，不能拖进生成器。

    ★ 这是 SSE 的关键约束：`StreamingResponse` 一旦开始发送，响应头 200
      就已经出去了，改不了。如果 404 拖到生成器里，客户端拿到的是
      「200 + 流内 error」，HTTP 语义就废了。
      所以断言点在 **`await` 那一行**，而不是迭代过程。
    """
    svc, _ = make_service(store)

    with pytest.raises(NotFound):
        await svc.open_stream(ChatRequest(message="hi", session_id="typo123"))

    assert store.conversation_exists("typo123") is False


# ══════════════════════ ⑨ 流式：真的边收边发 ══════════════════════


class GatedClient(FakeClient):
    """只有消费者取走第一段之后，才允许产出第二段。

    ★ 用它区分「真流式」和「假流式（先收完、再逐条吐）」：
      假流式会在 `await open_stream()` 里就想把增量收完，于是卡在
      `await self.gate.wait()` 上 —— 而开门的前提是消费者先拿到第一段，
      此时消费者还没拿到控制权，形成死锁。
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()

    async def astream(self, messages, **kw):  # type: ignore[no-untyped-def]
        yield "第一段"
        await self.gate.wait()
        yield "第二段"


async def test_open_stream_is_truly_incremental(store: MessageStore) -> None:
    """必须「消费一段才产下一段」，不能先把整段收完再返回。

    ★ 判据是**时序**，不是内容 —— 这是第 ⑥ 条测不出来的部分：
      1. `await open_stream()` 必须立刻返回（假流式会在这里卡死）
      2. 消费者拿到第一段之后放行，第二段才可能出现

      `asyncio.wait_for` 兜底：卡住就超时失败，而不是让测试永远挂着。
    """
    llm = GatedClient()
    svc = ChatService(llm=llm, store=store)

    _, deltas = await asyncio.wait_for(svc.open_stream(ChatRequest(message="hi")), timeout=1.0)

    first = await asyncio.wait_for(deltas.__anext__(), timeout=1.0)
    assert first == "第一段"

    llm.gate.set()  # 放行第二段

    rest = [d async for d in deltas]
    assert rest == ["第二段"]


# ══════════════════════ ⑩ 拼 messages：不重复注入 system ══════════════════════


async def test_build_messages_skips_duplicate_system(store: MessageStore) -> None:
    """历史里已经有 system 消息时，不再追加一条。

    ★ 这条守的是**成本**不是正确性：system 提示每轮都发，如果历史里
      存过一条又追加一条，prompt 会多出一整段 —— 接口照样正常，
      只是每轮都白花钱。属于「不报错的浪费」。
    """
    session_id = store.ensure_conversation(None)
    store.append(session_id, "system", "旧的系统提示")
    store.append(session_id, "user", "hi")

    svc, llm = make_service(store)
    await svc.chat(ChatRequest(message="再来一轮", session_id=session_id))

    sent = llm.calls[-1]
    assert [m["role"] for m in sent] == ["system", "user", "user"], sent
    assert sent[0]["content"] == "旧的系统提示", "有现成的就用现成的，别再插一条"


# ══════════════════════ ⑪ FC：一轮工具调用 → 最终答案 ══════════════════════


async def test_chat_executes_tool_calls_and_returns_final_answer(store: MessageStore) -> None:
    """模型要求调工具时，必须真的执行、按 id 配对回传，再拿第二次的回答。

    ★ 这条一次守住五件事，每一件破了都不报错：
      1. assistant 的 "工具申请单" **原样带回**（含 `content` 空串）——
         少了 `tool_calls`，下一条 tool 消息就没有归属
      2. `tool_call_id` **一一对应**且顺序不乱 —— 错位不报错，但答案会串
      3. 工具结果真的被执行（"51" / ZeroDivisionError），不是空字符串
      4. `usage` **逐轮累加** —— 只算最后一轮就是少报成本
      5. 只有最终回答落库 —— 中间的 tool 消息不进数据库

    ★ `tools` 每轮都要传（它不在 messages 里，少一轮模型就"忘了会工具"），
      所以用 RecordingClient 记 kwargs，而不是只看 messages。
    """
    llm = RecordingClient(
        script=[
            # 第 1 轮：同一轮要两个工具（一个成功、一个除零），验证批量 + 配对
            ChatResult(
                text="",
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_ok",
                        name="calculator",
                        arguments='{"expression": "(12+5)*3"}',
                    ),
                    ToolCall(id="call_bad", name="calculator", arguments='{"expression": "1/0"}'),
                ],
            ),
            # 第 2 轮：拿到工具结果，给出最终答案
            ChatResult(
                text="(12+5)*3 = 51",
                usage=Usage(prompt_tokens=3, completion_tokens=4, total_tokens=7),
                finish_reason="stop",
            ),
        ]
    )
    svc = ChatService(llm=llm, store=store)

    answer = await svc.chat(ChatRequest(message="(12+5)*3 等于多少"))

    assert answer.reply == "(12+5)*3 = 51"
    assert len(llm.sent) == 2, "两轮脚本应该正好调两次模型"

    # ── 第 1 轮：只有 system + user ──
    assert [m["role"] for m in llm.sent[0]] == ["system", "user"]

    # ── 第 2 轮：申请单 + 两条工具结果，顺序不能乱 ──
    assert [m["role"] for m in llm.sent[1]] == ["system", "user", "assistant", "tool", "tool"]

    assistant = llm.sent[1][2]
    assert assistant["content"] == "", "工具轮的正文可能为空，但字段必须留着"
    assert [tc["id"] for tc in assistant["tool_calls"]] == ["call_ok", "call_bad"]
    assert [tc["function"]["name"] for tc in assistant["tool_calls"]] == [
        "calculator",
        "calculator",
    ]
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"expression": "(12+5)*3"}'
    assert all(tc["type"] == "function" for tc in assistant["tool_calls"])

    tool_msgs = llm.sent[1][3:]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_ok", "call_bad"], (
        "工具结果必须按 id 一一配对：错位不报错，但模型会把结果算到别的问题上"
    )

    ok_payload, bad_payload = (json.loads(m["content"]) for m in tool_msgs)
    assert ok_payload == {"ok": True, "result": "51"}
    assert bad_payload["ok"] is False
    assert "ZeroDivisionError" in bad_payload["error"], "除零要变成文本，不能让整轮 500"

    # ── tools 每轮都传 ──
    assert all(kw.get("tools") for kw in llm.kwargs), "少传一轮，那一轮模型就不会用工具"
    assert all(kw.get("tool_choice") == "auto" for kw in llm.kwargs)

    # ── usage 逐轮累加（10+3 / 5+4 / 15+7）──
    assert answer.usage.prompt_tokens == 13
    assert answer.usage.completion_tokens == 9
    assert answer.usage.total_tokens == 22

    # ── 落库：只有 user + 最终 assistant ──
    saved = store.all_messages(answer.session_id)
    assert [m["role"] for m in saved] == ["user", "assistant"]
    assert saved[-1]["content"] == "(12+5)*3 = 51"


# ══════════════════════ ⑫ FC：步数用尽 → 兜底文案 ══════════════════════


async def test_chat_returns_fallback_when_tool_steps_exhausted(store: MessageStore) -> None:
    """连续 `MAX_TOOL_STEPS` 轮都要求调工具 -> 停止循环，返回兜底文案。

    ★ 关键是 `for...else`：Python 的 for 循环正常跑完才走 else，
      一路 `break` 出来的**不会**走。写错成"循环外直接返回 answer"的话，
      上面第 ⑪ 条仍然通过（那条会 break），只有这条能抓到 ——
      表现是返回了空字符串或上一轮的残留文本，接口照样 200。

    ★ 断言"模型只被调了 3 次"：脚本故意多给一轮，证明不会偷偷跑第 4 次。
      少调一次是省钱问题，多调一次是**死循环**问题。
    """
    llm = RecordingClient(
        script=[
            ChatResult(
                text="",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(id=f"call_{i}", name="calculator", arguments='{"expression": "1+1"}')
                ],
            )
            for i in range(MAX_TOOL_STEPS + 1)
        ]
    )
    svc = ChatService(llm=llm, store=store)

    answer = await svc.chat(ChatRequest(message="一直调工具，永远不回答"))

    assert len(llm.sent) == MAX_TOOL_STEPS, "步数上限必须真的生效"
    assert answer.reply == TOOL_FALLBACK
    assert store.all_messages(answer.session_id)[-1]["content"] == TOOL_FALLBACK


# ══════════════════════ ⑬ FC：未知工具不打断循环 ══════════════════════


async def test_chat_survives_unknown_tool_name(store: MessageStore) -> None:
    """模型编了一个没注册的工具名 -> 把错误当结果回传，循环继续。

    ★ 这是最现实的失败形态：上下文里提到"天气""搜索"时，模型很容易
      编出一个 `get_weather`。如果 dispatch 抛异常，整个请求 500；
      返回 `ok: false` 文本，模型下一轮就会改口 —— 用户看到的是一次正常回答。
    """
    llm = RecordingClient(
        script=[
            ChatResult(
                text="",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(id="call_x", name="get_weather", arguments='{"city": "北京"}')
                ],
            ),
            ChatResult(text="我这边没有天气工具，先按常识回答。", finish_reason="stop"),
        ]
    )
    svc = ChatService(llm=llm, store=store)

    answer = await svc.chat(ChatRequest(message="明天北京天气怎么样"))

    assert answer.reply == "我这边没有天气工具，先按常识回答。"

    tool_msg = llm.sent[1][-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_x"
    payload = json.loads(tool_msg["content"])
    assert payload["ok"] is False
    assert "未知工具" in payload["error"]
