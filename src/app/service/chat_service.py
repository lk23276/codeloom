"""对话业务逻辑。

★ 这一层的规矩（违反就是分层破了，而且不会报错）：
  - 不许 `import fastapi` —— 业务层不认识 HTTP
  - 不许写 SQL —— 数据库细节全在 db/store.py
  - 不许 `import openai` —— SDK 细节全在 client/llm.py
  - 只认识本项目自己的类型：ChatRequest / ChatAnswer / LLMClient / MessageStore

★ 自检命令（改完 service 后跑一次）：
      grep -rn "import fastapi\\|import openai" src/app/service/
      grep -rn "SELECT" src/app/service/
  两条都应该没有输出。

═══════════════════════════════════════════════════════════════════
  v0.1 已实现：chat()（五步顺序 + 异常翻译 + usage 透传）
               open_stream() / Function Calling / 限流 / 结构化日志 / 容器化
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from app.client.llm import LLMClient, LLMError
from app.core.async_utils import run_sync
from app.core.errors import NotFound, UpstreamError
from app.db.store import MessageStore
from app.prompts.system import build_system_message
from app.schema.chat import ChatAnswer, ChatRequest, UsageInfo
from app.tools.registry import TOOLS, dispatch

logger = logging.getLogger(__name__)

MAX_TOOL_STEPS = 3
TOOL_FALLBACK = "这个问题需要多步工具调用，当前版本最多 3 步，暂时无法给出完整答案。"


class ChatService:
    """一轮对话的业务编排。

    只做三件事：组织 messages、调模型、落库。
    不关心 HTTP（那是 router 的事），不关心 SQL（那是 store 的事）。
    """

    def __init__(self, llm: LLMClient, store: MessageStore) -> None:
        self._llm = llm
        self._store = store
        # 暴露给 router 用（meta 事件要告诉客户端实际用的哪个模型）
        self.model = llm.model

    # ══════════════════════ ① 非流式对话 ══════════════════════

    async def chat(self, req: ChatRequest) -> ChatAnswer:
        """非流式一轮对话。

        ★ 「续聊」必须先确认会话存在：
          `ensure_conversation` 内部是 `INSERT OR IGNORE`，所以传一个不存在的 id
          会【静默新建】——用户以为在续聊，实际开了个新会话，而且不报错。
          所以在第 ① 步之前先判一次存在性：传了 session_id 就必须是已有会话，
          否则 `NotFound` → 404。这样和 `GET /sessions/{id}/messages` 的行为一致
          （同一个 id 在两个端点上不会一个 404 一个 200）。

          代价：会话被清理后，客户端拿旧 id 重试会 404 而不是"自动续上"。
          这是有意的——静默新建会把「客户端 bug」伪装成正常返回，更难查。
        """
        # ①-0 续聊场景：传了 id 就必须已存在，不允许静默新建
        if req.session_id is not None and not await run_sync(
            self._store.conversation_exists, req.session_id
        ):
            raise NotFound(f"会话 {req.session_id} 不存在")

        session_id = await run_sync(self._store.ensure_conversation, req.session_id)
        await run_sync(self._store.append, session_id, "user", req.message)
        messages = await self._build_messages(session_id, req.history_limit)

        prompt_tokens = completion_tokens = total_tokens = 0
        answer = TOOL_FALLBACK

        # ★ 空列表不能传：部分 provider 会因 tools=[] 直接 400
        tool_kwargs = {"tools": TOOLS, "tool_choice": "auto"} if TOOLS else {}

        try:
            for _step in range(MAX_TOOL_STEPS):
                result = await self._llm.achat(messages, max_tokens=req.max_tokens, **tool_kwargs)
                # 逐轮累加（每轮独立计费，不累加就是少报成本）
                prompt_tokens += result.usage.prompt_tokens
                completion_tokens += result.usage.completion_tokens
                total_tokens += result.usage.total_tokens

                # ★ 结构化日志：一行一次模型调用。
                #   为什么要逐轮打而不是最后打一条汇总：排查「为什么这次贵」
                #   必须能看到每一轮各自的 token —— 工具轮往往 prompt 很大（带 schema）。
                logger.info(
                    "llm_call",
                    extra={
                        "model": self.model,
                        "step": _step + 1,
                        "prompt_tokens": result.usage.prompt_tokens,
                        "completion_tokens": result.usage.completion_tokens,
                        "tool_calls": len(result.tool_calls),
                    },
                )

                if not result.tool_calls:  # 终止条件
                    answer = result.text
                    break

                messages.append(  # ① 工具的"申请单"原样回传
                    {
                        "role": "assistant",
                        "content": result.text or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": tc.arguments},
                            }
                            for tc in result.tool_calls
                        ],
                    }
                )
                for tc in result.tool_calls:  # ② 这一轮全部执行完再进下一轮
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": dispatch(tc.name, tc.arguments),
                        }
                    )
            else:
                # 跑满没 break，说明步数用尽
                logger.warning("工具调用达到上限 %d 轮，返回兜底文案", MAX_TOOL_STEPS)
        except LLMError as e:  # 整个循环包一个 try
            raise self._wrap_llm_error(e, "achat") from e

        # ⑤ 落库 + 返回（只存最终回答）
        await run_sync(self._store.append, session_id, "assistant", answer)
        return ChatAnswer(
            reply=answer,
            session_id=session_id,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

    # ══════════════════════ ② 流式对话（SSE） ══════════════════════

    async def open_stream(self, req: ChatRequest) -> tuple[str, AsyncIterator[str]]:
        """流式一轮对话。返回 `(session_id, 只产正文增量的异步迭代器)`。"""
        if req.session_id is not None and not await run_sync(
            self._store.conversation_exists, req.session_id
        ):
            raise NotFound(f"会话 {req.session_id} 不存在")
        session_id = await run_sync(self._store.ensure_conversation, req.session_id)
        await run_sync(self._store.append, session_id, "user", req.message)
        messages = await self._build_messages(session_id, req.history_limit)

        async def gen() -> AsyncIterator[str]:
            stream = self._llm.astream(messages, max_tokens=req.max_tokens)
            try:
                async for delta in stream:
                    yield delta
            except LLMError as e:
                raise self._wrap_llm_error(e, "astream") from e
            finally:
                await stream.aclose()

        return session_id, gen()

    # ══════════════════════ 拼 messages ══════════════════════

    async def _build_messages(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        """把「系统提示 + 历史」拼成模型要的 messages 列表。

        两个细节：
        - system 提示放在最前面，且**每轮都发**（无状态 API，模型不记得上一轮）
        - 历史里可能已经有 system 消息（如果之前存过），这里只追加一条，
          避免重复注入导致 prompt 变长
        """
        history = await run_sync(self._store.history, session_id, limit)
        messages: list[dict[str, Any]] = []
        if not any(m.get("role") == "system" for m in history):
            messages.append(build_system_message())
        messages.extend({"role": str(m["role"]), "content": str(m["content"])} for m in history)
        return messages

    # ══════════════════════ 异常翻译示例 ══════════════════════

    def _wrap_llm_error(self, exc: LLMError, context: str) -> UpstreamError:
        """把 LLM 层异常翻成本层异常。

        ★ 每一层只认识自己那一层的语言：
          SDK 异常 →（client 翻译）→ LLMError →（service 翻译）→ UpstreamError
          router 只认识 UpstreamError，不需要 import 任何 LLM 相关的东西。
        """
        logger.warning("%s 失败: %s", context, exc)
        return UpstreamError(f"模型服务暂时不可用：{type(exc).__name__}")
