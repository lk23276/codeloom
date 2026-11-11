"""SQLite 会话存储。

设计要点（都是实测出来的，不是照抄教程）：

1. **WAL 模式**：读写不互斥。`/chat` 在一个请求里既有读（取历史）又有写（落库），
   默认的 DELETE 模式下写会锁住整个库，读要排队。
2. **PRAGMA foreign_keys=ON 必须每次连接都设**：SQLite 默认是 OFF，
   不设的话 `ON DELETE CASCADE` 静默失效——删了会话，消息还留着，而且不报错。
   （`journal_mode` 是持久化的，设一次就永久生效；`foreign_keys` 是连接级的。）
3. **排序用 `id` 不用 `created_at`**：`created_at` 只到秒，同一秒内多条会乱序。
   写法是 `ORDER BY id DESC LIMIT ?` 再 `reversed()`——先取最近 N 条，再翻回正序。
4. **拒绝 `:memory:`**：内存库每个连接都是独立的一份，多线程下"两个请求看到两个库"。
5. **单连接 + 锁**：`sqlite3` 的连接跨线程默认不安全（`check_same_thread`），
   这里设 `False` 并用 `RLock` 把整个操作串起来。上层用 `asyncio.to_thread` 调，
   不会阻塞事件循环。

★ 所有方法都是**同步**的。不要在这里 import asyncio——存储层不认识"协程"这件事，
  异步化是上层（service / router）的职责。这样这一层能被普通的单元测试直接调。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id     TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('system','user','assistant','tool')),
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    token_est   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id, id);
"""


def _now() -> str:
    """UTC 秒级时间戳（ISO 8601）。只用于展示，排序一律用 id。"""
    return datetime.now(UTC).isoformat(timespec="seconds")


class MessageStore:
    """会话与消息的持久化。一个实例对应一个数据库文件。"""

    def __init__(self, db_path: str) -> None:
        if db_path == ":memory:" or db_path.startswith("file::memory:"):
            raise ValueError(
                "不支持 :memory: —— 每个连接会拿到各自独立的内存库，"
                "多线程下会出现「两个请求看到两个库」。测试请用 tmp_path 下的真实文件。"
            )

        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False：允许该连接被 to_thread 的工作线程使用；
        # 真正的并发安全由 self._lock 保证。
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()

        with self._lock:
            # WAL：读写不互斥（持久化设置，设一次即可）
            self._conn.execute("PRAGMA journal_mode=WAL")
            # 外键：连接级设置，必须每次连接都开
            self._conn.execute("PRAGMA foreign_keys=ON")
            # 遇到锁时等 5 秒再报 "database is locked"，而不是立刻失败
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ── 生命周期 ──

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── 会话 ──

    def ensure_conversation(self, conv_id: str | None = None) -> str:
        """没有就建，有就返回原 id（幂等）。返回最终使用的 conv_id。"""
        cid = conv_id or uuid.uuid4().hex
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
                (cid, now, now),
            )
            self._conn.commit()
        return cid

    def conversation_exists(self, conv_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
        return row is not None

    def conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        """会话列表，最近更新的在前。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, conv_id: str) -> bool:
        """删会话（连带消息靠 ON DELETE CASCADE）。返回是否真的删掉了。"""
        with self._lock:
            cur = self._conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ── 消息 ──

    def append(self, conv_id: str, role: str, content: str, token_est: int = 0) -> int:
        """追加一条消息，返回它的自增 id。"""
        now = _now()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages (conv_id, role, content, created_at, token_est) "
                "VALUES (?, ?, ?, ?, ?)",
                (conv_id, role, content, now, token_est),
            )
            self._conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id)
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def history(self, conv_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """最近 limit 条消息，**按时间正序**返回（可直接拼给模型）。

        ★ 先 `ORDER BY id DESC LIMIT ?` 取"最近 N 条"，再 reversed() 翻正。
          反过来写（`ORDER BY id ASC LIMIT ?`）会取到"最早的 N 条"，是另一个意思。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages WHERE conv_id = ? ORDER BY id DESC LIMIT ?",
                (conv_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def all_messages(self, conv_id: str) -> list[dict[str, Any]]:
        """会话全部消息，正序。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, created_at FROM messages WHERE conv_id = ? ORDER BY id ASC",
                (conv_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self, conv_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE conv_id = ?", (conv_id,)
            ).fetchone()
        return int(row["n"]) if row else 0

    def trim(self, conv_id: str, keep: int = 20) -> int:
        """只保留最近 keep 条，返回删掉多少条（控成本用）。"""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM messages WHERE conv_id = ? AND id NOT IN ("
                "  SELECT id FROM messages WHERE conv_id = ? ORDER BY id DESC LIMIT ?"
                ")",
                (conv_id, conv_id, keep),
            )
            self._conn.commit()
            return int(cur.rowcount)

    # ── 探针用 ──

    def ping(self) -> None:
        """真查一张业务表（不是 SELECT 1）。

        ★ `SELECT 1` 不碰任何表，库文件被删了它照样"成功"；
          查 conversations 才能同时验证「文件在」「schema 在」「没被锁死」。
        """
        with self._lock:
            self._conn.execute("SELECT id FROM conversations LIMIT 1").fetchone()
