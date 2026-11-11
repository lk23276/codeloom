"""会话管理：列表 / 消息 / 删除。

★ 这三个端点**不限流**：它们只查本地 SQLite，不打模型、不花钱。
  限流应该只加在花钱的端点（/chat、/chat/stream）上。

★ 路径选择（design.md ADR-5）：
  用 `/sessions/{id}/messages` 而不是 `/sessions/{id}`，
  把 `/sessions/{id}` 这个资源位留给将来的「会话元信息」（标题/消息数/最后活跃）。
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.core.async_utils import run_sync
from app.core.deps import get_store
from app.core.errors import NotFound, ok

router = APIRouter(tags=["sessions"])


def _rid(request: Request) -> str:
    return getattr(request.state, "rid", "")


@router.get("/sessions", summary="会话列表")
async def list_sessions(request: Request, limit: int = 50) -> dict:
    """最近更新的会话在前。

    ★ `limit` 是 **query 参数**（`?limit=50`），不是 body —— 它是"筛选条件"，
      不是"要创建的资源"。放错位置不会报错，但语义会变形，也会影响缓存和日志。
    """
    store = get_store(request)
    rows = await run_sync(store.conversations, limit)
    return ok({"sessions": rows}, request_id=_rid(request))


@router.get("/sessions/{session_id}/messages", summary="某个会话的全部消息")
async def list_messages(session_id: str, request: Request) -> dict:
    """按时间正序返回全部消息。

    ★ 会话不存在要**显式查一次**，不能"查不到消息就返回空列表"——
      那让客户端分不清「会话不存在」和「会话存在但还没说话」。
    """
    store = get_store(request)
    if not await run_sync(store.conversation_exists, session_id):
        raise NotFound(f"会话 {session_id} 不存在")

    rows = await run_sync(store.all_messages, session_id)
    return ok(
        {"session_id": session_id, "messages": rows, "count": len(rows)},
        request_id=_rid(request),
    )


@router.delete("/sessions/{session_id}", summary="删除会话（连带消息）")
async def delete_session(session_id: str, request: Request) -> dict:
    """删除会话，消息靠外键 `ON DELETE CASCADE` 连带删除。

    ★ 前提是每条连接都设了 `PRAGMA foreign_keys=ON`（SQLite 默认是 OFF）。
      不设的话消息会变成孤儿，而且**不报错**——这是踩过的坑，
      现在由 MessageStore.__init__ 统一保证。

    ★ 返回 200 + body 而不是 204 No Content：
      204 要求响应体为空，而本项目所有响应用统一 envelope，两者冲突。
      选能自洽的那个。
    """
    store = get_store(request)
    deleted = await run_sync(store.delete_conversation, session_id)
    if not deleted:
        raise NotFound(f"会话 {session_id} 不存在")
    return ok({"session_id": session_id, "deleted": True}, request_id=_rid(request))


# ── 刻意没做的一个端点 ──
# POST /sessions/{id}/trim?keep=N（只保留最近 N 条，控成本用）
# v0.1 的接口只列了 7 个端点，这里保持一致。
# 如果确实需要，先更新接口设计再加实现 —— 不允许"实现偷偷多一个"。
