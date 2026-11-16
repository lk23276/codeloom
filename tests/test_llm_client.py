"""适配层测试：`OpenAICompatClient` 的「SDK 响应 → 自己的类型」。

★ 为什么单独测这一层：
  其余测试全走 `FakeClient`，真实实现里「解析 SDK 响应」那段一行没跑。
  而那段代码的缺陷没有明显信号：

      - 漏写 `or []`        -> 有工具时正常，无工具时抛 TypeError
      - 字段名拼错/漏映射   -> `tool_calls` 永远为空，
                              业务层循环第一轮即认为「模型未调用工具」，
                              接口返回 200、工具从未执行（不报错）

★ 如何做到不联网、不产生调用费用：
  用 `SimpleNamespace` 造一个假 SDK 响应，再把 `client._client` 整体换掉。
  这正是依赖注入的价值 —— 构造完之后可以替换内部依赖，不用起真服务。

★ 这几条测的是**契约**不是实现：
  只要「SDK 的 tool_calls 能变成我们的 ToolCall」这条契约不变，
  内部怎么改都不该让测试红。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from app.client.llm import FakeClient, LLMError, OpenAICompatClient, build_client
from app.core.config import Settings

# ══════════════════════ 桩：假 SDK ══════════════════════


class _StubCompletions:
    """记录调用参数、返回预设响应（或抛预设异常）。"""

    def __init__(self, resp: Any) -> None:
        self.resp = resp
        self.kwargs: dict[str, Any] = {}
        self.raise_exc: BaseException | None = None

    async def create(self, **kw: Any) -> Any:
        self.kwargs = kw
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.resp


def _client(resp: Any) -> tuple[OpenAICompatClient, _StubCompletions]:
    """造一个真实实现，但内部 SDK 对象被替换成桩。"""
    client = OpenAICompatClient(Settings(llm_api_key="sk-fake-for-test"))
    stub = _StubCompletions(resp)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=stub))
    return client, stub


def _sdk_tool_call(call_id: str, name: str, arguments: str) -> Any:
    """模拟 SDK 的 ToolCall 对象：`id` + `function.{name, arguments}`。"""
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _sdk_response(
    *,
    content: str = "",
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    usage: tuple[int, int, int] = (1, 2, 3),
) -> Any:
    """模拟 SDK 的 ChatCompletion 响应。"""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(
            prompt_tokens=usage[0],
            completion_tokens=usage[1],
            total_tokens=usage[2],
        ),
    )


# ══════════════════════ ① 解析 tool_calls（含顺序） ══════════════════════


async def test_achat_parses_tool_calls_in_order() -> None:
    """同一轮返回多个工具时，顺序与 id 都必须原样保留。

    ★ 为什么要断言顺序：`tool_call_id` 是"结果归谁"的唯一凭据。
      如果实现用了 set 或按名字重排，多个同类工具的结果就会串
      —— 不报错，但答案是错的。

    ★ `text == ""` 是正常的：模型决定调工具的那一轮通常不产出正文。
    """
    client, _ = _client(
        _sdk_response(
            finish_reason="tool_calls",
            tool_calls=[
                _sdk_tool_call("call_a", "calculator", '{"expression": "1+2"}'),
                _sdk_tool_call("call_b", "now", "{}"),
            ],
        )
    )

    result = await client.achat([{"role": "user", "content": "算一下"}])

    assert [(t.id, t.name, t.arguments) for t in result.tool_calls] == [
        ("call_a", "calculator", '{"expression": "1+2"}'),
        ("call_b", "now", "{}"),
    ]
    assert result.text == ""
    assert result.finish_reason == "tool_calls"


# ══════════════════════ ② 没有工具调用时是空列表 ══════════════════════


async def test_achat_without_tool_calls_returns_empty_list() -> None:
    """SDK 在没有工具调用时给的是 `None`，必须转成 `[]`。

    ★ 这是 `choice.message.tool_calls or []` 那句的意义。
      少了它，业务层 `for tc in result.tool_calls` 会直接
      `TypeError: 'NoneType' is not iterable`，而且只在**普通问答**路径触发，
      带工具的用例反而正常，容易被视为偶发问题。
    """
    client, _ = _client(_sdk_response(content="你好", tool_calls=None, usage=(5, 2, 7)))

    result = await client.achat([{"role": "user", "content": "hi"}])

    assert result.tool_calls == []
    assert result.text == "你好"
    assert result.usage.total_tokens == 7


# ══════════════════════ ③ 参数透传 ══════════════════════


async def test_achat_forwards_tools_and_kwargs() -> None:
    """`tools` / `tool_choice` / `max_tokens` 必须原样进到 SDK 调用里。

    ★ 这条守的是 FC 最隐蔽的一类 bug：`tools` 是**顶层参数**，不在 messages 里。
      如果适配层把它吞掉（或被 `**kw` 覆盖顺序错乱），该轮模型即认为无可用工具，
      而请求本身完全成功 —— 200、有回复、只是永远不调用工具。
    """
    client, stub = _client(_sdk_response(content="ok"))
    tools = [{"type": "function", "function": {"name": "calculator"}}]
    messages = [{"role": "user", "content": "hi"}]

    await client.achat(messages, tools=tools, tool_choice="auto", max_tokens=200)

    assert stub.kwargs["tools"] == tools
    assert stub.kwargs["tool_choice"] == "auto"
    assert stub.kwargs["max_tokens"] == 200
    assert stub.kwargs["messages"] is messages
    # 思考开关必须显式下发：不传只是走服务端默认，而默认值可能变
    assert stub.kwargs["extra_body"]["thinking"]["type"] in {"enabled", "disabled"}


# ══════════════════════ ④ 异常翻译 ══════════════════════


async def test_achat_translates_unexpected_error() -> None:
    """任何 SDK/底层异常都要变成 `LLMError`，不许原样冒出去。

    ★ 判据：上层只 `except LLMError`，不需要 `import openai`。
      如果此处遗漏翻译，service 的 `except LLMError` 接不住，
      最终会变成 500「我们自己的 bug」，而真实原因是上游故障 —— 定级就错了。
    """
    client, stub = _client(_sdk_response())
    stub.raise_exc = RuntimeError("模拟网络中断")

    with pytest.raises(LLMError) as excinfo:
        await client.achat([{"role": "user", "content": "hi"}])

    assert "RuntimeError" in str(excinfo.value)


# ══════════════════════ ⑤ FakeClient 的脚本化 ══════════════════════


async def test_fake_client_script_runs_in_order_then_falls_back() -> None:
    """脚本按顺序消费，用完回落到 `reply`。

    ★ 为什么这条也要测：后面写 FC 循环的测试全靠它。
      如果脚本被"重放"或提前回落，FC 测试就会在错误的轮次上通过 ——
      测试本身变成假绿。
    """
    from app.client.llm import ChatResult, ToolCall

    llm = FakeClient(
        reply="兜底回复",
        script=[
            ChatResult(
                text="", tool_calls=[ToolCall(id="call_1", name="calculator", arguments="{}")]
            ),
            ChatResult(text="最终答案"),
        ],
    )

    first = await llm.achat([{"role": "user", "content": "算"}])
    second = await llm.achat([{"role": "user", "content": "继续"}])
    third = await llm.achat([])

    assert [tc.id for tc in first.tool_calls] == ["call_1"]
    assert second.text == "最终答案"
    assert third.text == "兜底回复", "脚本用完后要回落，否则普通用例会被带偏"
    assert len(llm.calls) == 3, "每次 achat 都要记录 messages，方便断言轮次"


# ══════════════════════ 工厂：后端分派 ══════════════════════


def test_build_client_defaults_to_handwritten_impl() -> None:
    """默认后端必须是手写实现。

    ★ 该断言的作用是锁定前提：ADR-7 的结论建立在「默认路径不依赖框架」上。
      默认值一旦改为 langchain，主干会新增一个第三方运行时依赖，
      而测试全部注入 `FakeClient`，无法发现这一变化。
    """
    settings = Settings(_env_file=None, llm_api_key="sk-x")

    assert settings.llm_backend == "openai", "默认值变了要先改 ADR-7"
    assert type(build_client(settings)).__name__ == "OpenAICompatClient"


def test_build_client_dispatches_to_langchain(monkeypatch: pytest.MonkeyPatch) -> None:
    """`llm_backend=langchain` 时走 LCEL 适配层。

    ★ 替换的是 `lc_client` 模块里的工厂函数：`build_client` 在**函数内部**才
      import 它，所以 monkeypatch 模块属性即可生效 —— 不需要真的装 langchain，
      测试也就不会因为环境差异而红。
    """
    from app.client import lc_client

    sentinel = object()
    monkeypatch.setattr(lc_client, "build_lc_client", lambda _settings: sentinel)

    settings = Settings(_env_file=None, llm_api_key="sk-x", llm_backend="langchain")

    assert build_client(settings) is sentinel, "配置成 langchain 却没有走适配层"


def test_invalid_backend_value_fails_fast() -> None:
    """写错后端名必须在**构造 `Settings` 时**失败，而不是静默 fallback。

    ★ 这是 Literal 相对 str 的全部价值：`APP_LLM_BACKEND=langchian`（拼错）
      用 str 时会一路跑到运行时，被 `== "langchain"` 判假后退化为手写实现。
      该路径不抛任何异常，容易被误判为「框架路径已生效」。
    """
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_backend="langchian")
