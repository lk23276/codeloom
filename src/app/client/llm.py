"""LLM 客户端：协议 + 真实实现 + 异常翻译。

★ 全项目**只有这一个文件**允许 `import openai`。
  业务层（service）只认识 `LLMClient` 这个 Protocol 和 `LLMError` 这个异常家族，
  所以把 DeepSeek 换成 GLM / vLLM 只需要改这里的 `base_url` 和 `model`。

★ 依赖倒置的要点不是"用接口"，而是「**接口归谁所有**」：
  `LLMClient` 是【业务层自己定义的】，openai SDK 反过来实现它。
  所以这个 Protocol 的签名里**不许出现任何 SDK 类型**（比如 openai.ChatCompletion）。

★ 关于 `iscoroutinefunction` 的坑（openai 3.x 实测）：
  `inspect.iscoroutinefunction(client.chat.completions.create)` 返回 **False**，
  但它**实际可以 await**（SDK 用了装饰器）。判断能不能 await 只能靠真 await 一次。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import openai
from openai import AsyncOpenAI

from app.core.config import Settings

logger = logging.getLogger(__name__)


# ══════════════════════ 一、异常家族 ══════════════════════


class LLMError(Exception):
    """所有 LLM 相关异常的基类。上层只需 catch 这一个。"""


class LLMAuthError(LLMError):
    """key 无效 / 无权限（401/403）——重试没用。"""


class LLMRateLimitError(LLMError):
    """限流（429）——可退避重试。"""


class LLMTimeoutError(LLMError):
    """超时——可重试。"""


class LLMConnectionError(LLMError):
    """连不上（网络/DNS）——可重试。"""


class LLMBadRequestError(LLMError):
    """请求本身不合法（400/422）——重试没用，是代码问题。"""


# SDK 异常 → 我们的异常。★ 顺序重要：子类要排在父类前面
_ERROR_MAP: tuple[tuple[type[BaseException], type[LLMError]], ...] = (
    (openai.AuthenticationError, LLMAuthError),
    (openai.PermissionDeniedError, LLMAuthError),
    (openai.RateLimitError, LLMRateLimitError),
    (openai.APITimeoutError, LLMTimeoutError),
    (openai.APIConnectionError, LLMConnectionError),
    (openai.BadRequestError, LLMBadRequestError),
    (openai.UnprocessableEntityError, LLMBadRequestError),
    (openai.APIStatusError, LLMError),
)


def translate(exc: BaseException) -> LLMError:
    """把 SDK 异常翻译成本项目的异常。

    ★ 这样做的价值：上层 `except LLMError` 就够，不需要 import openai。
      否则 router/service 里会出现 `except openai.RateLimitError` —— 依赖就泄漏了。
    """
    for sdk_exc, our_exc in _ERROR_MAP:
        if isinstance(exc, sdk_exc):
            return our_exc(f"{our_exc.__name__}: {exc}")
    return LLMError(f"未预期的模型调用错误: {type(exc).__name__}: {exc}")


# ══════════════════════ 二、结果对象 ══════════════════════


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ToolCall:
    """模型要求执行的一次工具调用。

    ★ 三个字段各有不可替代的作用，少一个就出错：

        id         配对用的。`role: "tool"` 的消息必须带回同一个 id，
                   否则服务端不知道这条结果属于哪一次调用
        name       工具名，要和 `tools/registry.py` 里 `_DISPATCH` 的键一致
        arguments  是 **JSON 字符串**（例如 `'{"expression": "1+2"}'`），
                   不是 dict。SDK 原样给的就是字符串，
                   "解析 + 校验"是业务层（dispatch）的职责，不是这一层的
    """

    id: str
    name: str
    arguments: str


@dataclass
class ChatResult:
    """一次非流式调用的完整结果。

    ★ 不直接返回 `str`：usage（成本）、finish_reason（是否被截断）、
      latency（性能）都是后面要用的信息，返回裸字符串就丢了。
    """

    text: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    latency_ms: int = 0

    # ★ 默认空列表：绝大多数调用不是工具调用。
    #   给默认值之后，所有旧的构造点（含 FakeClient、测试里的直接构造）都不用改。
    #   反过来说，如果哪天这里忘了解析，`tool_calls` 会**永远是空**，
    #   业务层的循环第一轮就以为"模型没调工具" —— 接口正常、工具从没被调用。
    tool_calls: list[ToolCall] = field(default_factory=list)


# ══════════════════════ 三、协议 ══════════════════════


@runtime_checkable
class LLMClient(Protocol):
    """业务层看到的全部能力。两个方法，不多不少。"""

    model: str

    # ★ messages 的值类型是 Any，不是 str —— 为什么放宽：
    #   普通消息是 {"role": "user", "content": "..."}，
    #   但带工具调用的 assistant 消息里 `tool_calls` 的值是**列表**，
    #   tool 消息还多一个 `tool_call_id`。写成 dict[str, str] 会一直报类型错。
    async def achat(self, messages: list[dict[str, Any]], **kw: Any) -> ChatResult: ...

    def astream(self, messages: list[dict[str, Any]], **kw: Any) -> AsyncIterator[str]: ...


# ══════════════════════ 四、真实实现 ══════════════════════


class OpenAICompatClient:
    """覆盖所有「OpenAI 兼容」端点：DeepSeek / GLM / 通义 / vLLM / Ollama。

    ★ 一个类服务四家，不是偷懒 —— 它们共用同一套 HTTP 协议，
      差异收敛到 Settings 的能力开关里，而不是膨胀成四个类。
    """

    def __init__(self, settings: Settings) -> None:
        self.model = settings.llm_model
        self._thinking = settings.llm_thinking
        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key or "sk-not-configured",
            base_url=settings.llm_base_url,
            timeout=settings.timeout_s,
            # ★ 必须关掉 SDK 的自动重试：它的重试会和我们的重试叠加。
            #   SDK 2 次 × 我们 2 次 = 4 次真实调用，账单和延迟都失控。
            max_retries=0,
        )

    def _extra(self) -> dict[str, Any]:
        """思考模式的开关。

        ★ 默认值不能靠「不传」:不传只是走服务端默认，而服务端默认可能变。
          要显式传 disabled，开关才真的由我们控制。
        """
        return {"extra_body": {"thinking": {"type": "enabled" if self._thinking else "disabled"}}}

    async def achat(self, messages: list[dict[str, Any]], **kw: Any) -> ChatResult:
        t0 = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                **self._extra(),
                **kw,
            )
        except Exception as exc:  # noqa: BLE001 —— 在边界处统一翻译
            raise translate(exc) from exc

        latency = int((time.perf_counter() - t0) * 1000)
        choice = resp.choices[0]
        u = resp.usage
        return ChatResult(
            text=choice.message.content or "",
            usage=Usage(
                prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                total_tokens=getattr(u, "total_tokens", 0) or 0,
            ),
            finish_reason=choice.finish_reason,
            latency_ms=latency,
            # ★ SDK 类型在这里就地转换，不往业务层漏：
            #   tc.function.arguments 是字符串，业务层要自己 json.loads。
            #   `or []` 不能省：没有工具调用时它是 None，不是空列表。
            tool_calls=[
                ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                for tc in (choice.message.tool_calls or [])
            ],
        )

    async def astream(  # type: ignore[override]
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[str]:
        """逐 token 产出增量。

        ★ `finally` 里必须关掉上游流：客户端中途断开时，`async for` 不会自动
          close，连接会泄漏，而且模型还在继续生成（白花钱）。
        """
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                stream=True,
                **self._extra(),
                **kw,
            )
        except Exception as exc:  # noqa: BLE001
            raise translate(exc) from exc

        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # noqa: BLE001 —— 关流失败不影响主流程
                    logger.debug("关闭上游流失败", exc_info=True)


class FakeClient:
    """测试用的假客户端：不联网、不花钱、结果可预测。

    ★ 它的存在就是这套抽象的价值证明：业务层只认 `LLMClient`，
      测试时一行 `app.dependency_overrides[get_llm] = lambda: FakeClient()` 就换掉了。

    ★ 脚本化（`script`）：按顺序返回预设结果，专门用来测 Function Calling。

        llm = FakeClient(
            script=[
                # 第 1 轮：模型要求调工具（text 为空很正常，这时没有正文）
                ChatResult(
                    text="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="calculator",
                            arguments='{"expression": "(12+5)*3"}',
                        )
                    ],
                ),
                # 第 2 轮：模型拿到工具结果后给出最终答案
                ChatResult(text="答案是 51"),
            ],
        )

      第一次 `achat` 返回"要工具"，第二次返回"最终答案" —— 正好走完两轮循环。
      脚本用完后自动回落到 `reply`，所以测普通问答时不用管 `script`。

      ★ 没有这个参数，FC 链路只能靠真模型 + 真 key 验证：一次几毛钱、
        结果不可复现、CI 里跑不了。
    """

    def __init__(
        self,
        reply: str = "这是假客户端的回复",
        chunks: list[str] | None = None,
        script: list[ChatResult] | None = None,
    ) -> None:
        self.model = "fake-model"
        self.reply = reply
        self.chunks = chunks or [reply]
        self.calls: list[list[dict[str, Any]]] = []
        # 脚本化的返回序列 + 当前下标（不要用 pop，留着方便断言第几轮）
        self.script: list[ChatResult] = list(script or [])
        self._script_i = 0

    async def achat(self, messages: list[dict[str, Any]], **kw: Any) -> ChatResult:
        self.calls.append(messages)
        if self._script_i < len(self.script):
            result = self.script[self._script_i]
            self._script_i += 1
            return result
        return ChatResult(
            text=self.reply,
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
            latency_ms=1,
        )

    async def astream(  # type: ignore[override]
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[str]:
        self.calls.append(messages)
        for c in self.chunks:
            await asyncio.sleep(0)
            yield c


# ══════════════════════ 五、工厂 ══════════════════════


def build_client(settings: Settings) -> LLMClient:
    """按配置造客户端。这是唯一"new 具体实现"的地方（除测试外）。"""
    if not settings.has_llm_key:
        logger.warning(
            "APP_LLM_API_KEY 未配置或仍是占位符：/chat 会返回 502，"
            "探针与会话接口不受影响。配置方法见 .env.example"
        )
    return OpenAICompatClient(settings)
