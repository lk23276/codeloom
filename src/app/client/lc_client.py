"""LangChain 后端：用 LCEL 实现同一个 `LLMClient` 协议。

设计约束见 `docs/adr/ADR-7-langchain-boundary.md`：

1. 只在 `client/` 层出现。`service/` 与 `router/` 既不 import 本模块，也不 import langchain。
2. 对外暴露的仍是 `LLMClient` 协议，返回值仍是 `ChatResult`，业务层感知不到后端差异。
3. 目的是「可切换 + 可对比」，不是替换手写的 `OpenAICompatClient`。
4. langchain 延迟导入：未安装依赖时本模块仍可被导入，只有真正构造客户端时才报错。

★ 为什么延迟导入而不是写在文件顶部：
  默认后端是 openai，绝大多数运行路径不需要 langchain。顶部导入会把
  「未安装 langchain」升级成模块级的硬依赖，连带影响不需要该后端的测试环境。

★ 与手写实现的关键差异（适配层存在的全部理由）：
  手写版拿到的是 SDK 的 `resp.usage` / `message.tool_calls`（arguments 为 JSON 字符串）；
  LangChain 给的是 `AIMessage.usage_metadata` / `AIMessage.tool_calls`（args 已是 dict）。
  两者形状不同，不转换就会得到「usage 恒为 0」「工具永远调不到」这类静默失败。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

from app.client.llm import (
    ChatResult,
    LLMClient,
    LLMError,
    ToolCall,
    Usage,
    translate,
)
from app.core.config import Settings

logger = logging.getLogger(__name__)


# ══════════════════════ 一、依赖装载 ══════════════════════


class LangChainNotInstalledError(LLMError):
    """缺少 langchain 依赖。

    ★ 继承 `LLMError` 而不是 ImportError：上层只需要 `except LLMError` 一种写法，
      不需要为「依赖没装」单独写分支。
    """


@lru_cache(maxsize=1)
def _lc() -> SimpleNamespace:
    """导入并缓存 langchain 的相关符号。

    `lru_cache` 让导入只发生一次；模块导入本身也有缓存，但这里顺带把
    「缺依赖」的报错文案收敛到一个地方。
    """
    try:
        from langchain_core.messages import convert_to_messages
        from langchain_core.output_parsers import StrOutputParser
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as exc:  # pragma: no cover - 取决于运行环境
        raise LangChainNotInstalledError(
            "langchain 后端不可用：缺少 langchain-core / langchain-openai。"
            "安装后重试，或把 APP_LLM_BACKEND 设回 openai。"
        ) from exc
    return SimpleNamespace(
        ChatOpenAI=ChatOpenAI,
        StrOutputParser=StrOutputParser,
        convert_to_messages=convert_to_messages,
    )


# ══════════════════════ 二、结果映射 ══════════════════════


def _map_usage(msg: Any) -> Usage:
    """`AIMessage.usage_metadata` → `Usage`。"""
    meta = getattr(msg, "usage_metadata", None) or {}

    # ① 上游没给用量：流式常见，部分供应商不支持。返回全 0 属正常，不告警
    if not meta:
        return Usage()

    # ② 有数据但键名完全对不上 —— 只可能是字段随版本改名，需要立刻发现
    if "input_tokens" not in meta and "output_tokens" not in meta:
        logger.warning("usage_metadata 键名与预期不符，可能随版本变更：%s", sorted(meta))
        return Usage()

    prompt = int(meta.get("input_tokens") or 0)
    completion = int(meta.get("output_tokens") or 0)
    # ③ total 缺失时自行相加，不依赖上游一定给
    total = int(meta.get("total_tokens") or 0) or prompt + completion
    return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def _map_tool_calls(msg: Any) -> list[ToolCall]:
    """`AIMessage.tool_calls` → `list[ToolCall]`。"""
    raw = getattr(msg, "tool_calls", None) or []

    # ① 模型确实要调工具，但参数不是合法 JSON：LangChain 把它挪到
    #    invalid_tool_calls，tool_calls 变成 []。不处理就等于「静默丢弃工具请求」
    invalid = getattr(msg, "invalid_tool_calls", None) or []
    if invalid:
        logger.warning(
            "模型返回了 %d 个无法解析的工具调用，已丢弃：%s",
            len(invalid),
            [c.get("error") for c in invalid],
        )

    calls: list[ToolCall] = []
    for tc in raw:
        args = tc.get("args") or {}
        arguments = (
            args
            if isinstance(args, str)  # ② 少数实现直接给字符串，原样透传（避免二次编码）
            else json.dumps(args, ensure_ascii=False)
        )
        calls.append(
            ToolCall(
                id=str(tc.get("id") or ""),
                name=str(tc.get("name") or ""),
                arguments=arguments,
            )
        )
    return calls


def _text_of(msg: Any) -> str:
    """从 `AIMessage.content` 取纯文本。

    LangChain 1.x 起 `content` 可能是 `str`，也可能是内容块列表
    （多模态 / 结构化输出场景），这里只取其中的文本部分。
    """
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


# ══════════════════════ 三、实现 ══════════════════════


class LangChainClient:
    """`LLMClient` 协议的 LangChain 实现。

    ★ 不继承 Protocol：`LLMClient` 是结构化协议（`runtime_checkable`），
      按签名实现即可，运行时用 `isinstance(client, LLMClient)` 可验证。
    """

    def __init__(self, settings: Settings) -> None:
        lc = _lc()
        self.model = settings.llm_model
        self._thinking = settings.llm_thinking
        self._chat = lc.ChatOpenAI(
            # model / base_url / api_key 是别名，对应 model_name / openai_api_base / openai_api_key
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key or "sk-not-configured",
            temperature=0.0,
            request_timeout=settings.timeout_s,
            # ★ 与手写实现一致：关掉 SDK 自动重试，避免与业务层重试叠加
            max_retries=0,
            # 流式路径的用量依赖该开关：部分供应商默认不在流式响应中返回 usage
            stream_usage=True,
        )
        # 流式链只做「模型 → 纯文本」，不含 usage；
        # 需要 usage 或工具调用时走 achat，拿原始 AIMessage。
        self._stream_chain = self._chat | lc.StrOutputParser()

    def _extra(self) -> dict[str, Any]:
        """provider 私有参数，与手写实现 `OpenAICompatClient._extra()` 对齐。

        ★ 思考模式必须显式关：实测同一问题，思考开 130 完成 token、关 50，
          正文长度反而更短。不传不等于关闭 —— 不传只是走服务端默认，而默认会变。
        """
        return {"thinking": {"type": "enabled" if self._thinking else "disabled"}}

    def _call_kwargs(self, kw: dict[str, Any]) -> dict[str, Any]:
        """把调用参数整理成最终传给模型的 kwargs。

        ★ 为什么 `max_tokens` 必须改道 `extra_body`：
          框架 1.x 把它改名为 `max_completion_tokens`（跟随 OpenAI 新命名），
          而 OpenAI 兼容端点（DeepSeek 实测）不认这个字段 —— 参数被**静默忽略**，
          传 60 实际生成 143。放进 `extra_body` 才由 SDK 原样合并进请求体。
        """
        extra = dict(self._extra())

        max_tokens = kw.pop("max_tokens", None)
        if max_tokens is not None:
            extra["max_tokens"] = max_tokens

        # 调用方显式传的 extra_body 优先级最高
        caller_extra = kw.pop("extra_body", None)
        if isinstance(caller_extra, dict):
            extra.update(caller_extra)

        kw["extra_body"] = extra
        return kw

    async def achat(self, messages: list[dict[str, Any]], **kw: Any) -> ChatResult:
        lc = _lc()
        t0 = time.perf_counter()
        try:
            msg = await self._chat.ainvoke(
                lc.convert_to_messages(messages), **self._call_kwargs(dict(kw))
            )
        except Exception as exc:  # noqa: BLE001 —— 在边界处统一翻译
            raise translate(exc) from exc

        latency = int((time.perf_counter() - t0) * 1000)
        return ChatResult(
            text=_text_of(msg),
            usage=_map_usage(msg),
            # finish_reason 在 response_metadata 里，不在顶层字段
            finish_reason=(getattr(msg, "response_metadata", {}) or {}).get("finish_reason"),
            latency_ms=latency,
            tool_calls=_map_tool_calls(msg),
        )

    async def astream(  # type: ignore[override]
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[str]:
        """逐段产出增量文本。

        ★ 必须用 `astream`：`Runnable.stream()` 是**同步生成器**
          （`inspect.isgeneratorfunction` 为 True），在 async 路由里调用会
          阻塞整个事件循环，而不只是当前请求。

        ★ 调用方在 `finally` 里 `aclose()` 时，关闭会沿生成器链传播到上游，
          与手写实现的「客户端断开即停止计费」行为一致。
        """
        lc = _lc()
        try:
            params = self._call_kwargs(dict(kw))
            async for delta in self._stream_chain.astream(
                lc.convert_to_messages(messages), **params
            ):
                if delta:
                    yield delta
        except Exception as exc:  # noqa: BLE001
            raise translate(exc) from exc


# ══════════════════════ 四、工厂 ══════════════════════


def build_lc_client(settings: Settings) -> LLMClient:
    """构造 LangChain 后端。

    供 `client/llm.py::build_client()` 在 `llm_backend == "langchain"` 时调用。
    """
    return LangChainClient(settings)
