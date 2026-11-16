"""LangChain 适配层的字段映射测试。

★ 这一层只有两个纯函数，但它们是双实现里唯一会「静默出错」的地方：

    - usage 映射写错 -> token 用量恒为 0，成本统计与日志失真，接口照常 200
    - 工具映射写错   -> `tool_calls` 恒为空，业务层以为模型没调工具，直接空回复

  两者都不抛异常，因此断言必须落在**具体数值**上，不能只断言"没报错"。

★ 全部用 `SimpleNamespace` 造「假 AIMessage」：
  不构造真实 `ChatOpenAI`、不发起网络请求、不读取 `.env` 里的 key，
  因此这组测试在任何环境都能跑，也不会产生费用。
  真实 `AIMessage` 的字段形状已在本机实测核对过（键名、None、invalid_tool_calls 走向）。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.client import lc_client
from app.client.llm import Usage

LOGGER = "app.client.lc_client"


# ══════════════════════ 一、usage 映射 ══════════════════════


def test_map_usage_reads_langchain_keys() -> None:
    """LangChain 的键名是 input/output，不是 SDK 的 prompt/completion。

    ★ 这条是整组测试的地基：键名一旦拼错，下面所有断言都会得到 0，
      而接口返回的一切看起来都正常。
    """
    msg = SimpleNamespace(
        usage_metadata={"input_tokens": 443, "output_tokens": 114, "total_tokens": 557}
    )

    assert lc_client._map_usage(msg) == Usage(443, 114, 557)


def test_map_usage_tolerates_missing_usage() -> None:
    """流式或部分供应商不给用量时返回全 0，不能抛异常。

    ★ 两种情况都要覆盖：属性缺失、属性存在但值为 None。
    """
    assert lc_client._map_usage(SimpleNamespace(usage_metadata=None)) == Usage()
    assert lc_client._map_usage(SimpleNamespace()) == Usage()


def test_map_usage_derives_total_when_absent() -> None:
    """total 缺失时用 input + output 补。

    ★ 下游按 total_tokens 做的成本统计依赖这个值；上游少给一个字段，
      统计就归零，而这不会触发任何异常。
    """
    msg = SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 5})

    assert lc_client._map_usage(msg).total_tokens == 15


def test_map_usage_warns_on_key_drift(caplog: pytest.LogCaptureFixture) -> None:
    """键名整体对不上时必须留下日志。

    ★ 这是「上游未提供」与「键名读取错误」的区别：
      前者属正常情况，后者是集成缺陷。两者都会得到 0，只能靠日志区分。
    """
    msg = SimpleNamespace(usage_metadata={"prompt_tokens": 10, "completion_tokens": 5})

    with caplog.at_level("WARNING", logger=LOGGER):
        assert lc_client._map_usage(msg) == Usage()

    assert "键名与预期不符" in caplog.text


# ══════════════════════ 二、tool_calls 映射 ══════════════════════


def test_map_tool_calls_serializes_args_to_json_string() -> None:
    """LangChain 的 args 是 dict，而 `ToolCall.arguments` 约定是 JSON 字符串。

    ★ 断言用 `json.loads` 回解，不写死字符串：
      `json.dumps` 的空格差异会让断言无意义地变红。
    """
    msg = SimpleNamespace(
        tool_calls=[{"name": "calculator", "args": {"expression": "(12+5)*3"}, "id": "call_1"}]
    )

    calls = lc_client._map_tool_calls(msg)

    assert len(calls) == 1
    assert calls[0].id == "call_1"
    assert calls[0].name == "calculator"
    assert json.loads(calls[0].arguments) == {"expression": "(12+5)*3"}


def test_map_tool_calls_keeps_chinese_readable() -> None:
    """`ensure_ascii=False`：否则日志与评测集里全是 \\uXXXX，排查时读不出参数。"""
    msg = SimpleNamespace(
        tool_calls=[{"name": "weather", "args": {"city": "示例城市"}, "id": "c1"}]
    )

    assert "示例城市" in lc_client._map_tool_calls(msg)[0].arguments


def test_map_tool_calls_passes_through_string_args() -> None:
    """args 已是字符串时不能再 dumps 一次。

    ★ 二次编码会得到 `"{\\"expression\\": \\"1+2\\"}"`，
      下游 `json.loads` 解出来是**字符串**而非对象，dispatch 会报「参数必须是 JSON 对象」。
    """
    msg = SimpleNamespace(
        tool_calls=[{"name": "calculator", "args": '{"expression": "1+2"}', "id": "c1"}]
    )

    arguments = lc_client._map_tool_calls(msg)[0].arguments

    assert arguments == '{"expression": "1+2"}'
    assert json.loads(arguments) == {"expression": "1+2"}


def test_map_tool_calls_returns_empty_list_when_absent() -> None:
    """没有工具调用时返回空列表 —— 业务层用 `if not tool_calls` 判断终止。"""
    assert lc_client._map_tool_calls(SimpleNamespace(tool_calls=[])) == []
    assert lc_client._map_tool_calls(SimpleNamespace()) == []


def test_map_tool_calls_warns_when_arguments_unparsable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """参数不是合法 JSON 时，LangChain 把它挪到 `invalid_tool_calls`，`tool_calls` 变空。

    ★ 本条针对最易遗漏的失败模式：缺少告警时，模型已发起工具调用，
      业务层却按「未调用」处理，把空回复返回给客户端 —— 接口 200、工具从未执行。
    """
    msg = SimpleNamespace(
        tool_calls=[],
        invalid_tool_calls=[
            {"name": "calculator", "args": "{bad", "id": "c1", "error": "bad json"}
        ],
    )

    with caplog.at_level("WARNING", logger=LOGGER):
        assert lc_client._map_tool_calls(msg) == []

    assert "无法解析的工具调用" in caplog.text


# ══════════════════════ 三、协议一致性 ══════════════════════


def test_langchain_client_exposes_protocol_methods() -> None:
    """`LangChainClient` 必须提供 `LLMClient` 要求的两个入口。

    ★ 不直接写 `isinstance(instance, LLMClient)`：那需要构造实例，
      而构造函数依赖 langchain（未安装时必失败），测试会因环境差异而红。
      这里检查的是**类层面**的契约，装好依赖后可用 ADR-7 里的完整断言复核。
    """
    assert callable(getattr(lc_client.LangChainClient, "achat", None))
    assert callable(getattr(lc_client.LangChainClient, "astream", None))


# ══════════════════════ 四、调用参数整理 ══════════════════════


def _client_without_init(thinking: bool = False) -> lc_client.LangChainClient:
    """绕过 `__init__` 造实例：不构造 ChatOpenAI，避免测试依赖网络与密钥。"""
    client = object.__new__(lc_client.LangChainClient)
    client._thinking = thinking
    return client


def test_call_kwargs_moves_max_tokens_into_extra_body() -> None:
    """`max_tokens` 必须改道 `extra_body`。

    ★ 实测依据：框架 1.x 会把它改名为 `max_completion_tokens`，而 OpenAI 兼容端点
      不认这个字段 —— 传 60 实际生成 143，成本闸门形同虚设且毫无报错。
      放进 `extra_body` 后实测被正确截断（60 → 60）。
    """
    kwargs = _client_without_init()._call_kwargs({"max_tokens": 120})

    assert "max_tokens" not in kwargs, "顶层 max_tokens 会被框架改名，必须移除"
    assert kwargs["extra_body"]["max_tokens"] == 120


def test_call_kwargs_always_disables_thinking() -> None:
    """思考模式必须显式关。

    ★ 实测同一问题：思考开 130 完成 token、关 50，正文反而更短。
      不传不等于关闭 —— 不传只是走服务端默认。
    """
    kwargs = _client_without_init(thinking=False)._call_kwargs({})

    assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}


def test_call_kwargs_honours_explicit_thinking() -> None:
    """配置打开思考时，仍要显式传 enabled，而不是省略。"""
    kwargs = _client_without_init(thinking=True)._call_kwargs({})

    assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}


def test_call_kwargs_lets_caller_override_extra_body() -> None:
    """调用方显式给的 extra_body 优先级最高，且不会覆盖掉思考开关。"""
    kwargs = _client_without_init()._call_kwargs(
        {"max_tokens": 100, "extra_body": {"thinking": {"type": "enabled"}}}
    )

    assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}
    assert kwargs["extra_body"]["max_tokens"] == 100
    assert "extra_body" in kwargs
