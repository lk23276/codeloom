"""checkpointer：把图的状态存到图外面，按 `thread_id` 分桶。

两种后端，接口一致，差别只在「存哪儿」：

    memory   InMemorySaver      进程内，重启即丢 —— 测试与本地演示
    sqlite   AsyncSqliteSaver   落盘 `data/agent_memory.db` —— 默认

★ 为什么用 **Async** 版本而不是同步的 `SqliteSaver`：
  图的节点是 `async def`，checkpoint 的读写也在事件循环里发生。同步版会把 sqlite 的
  文件 IO 压在 loop 上 —— 与 `rag/` 里 `sqlite3` 的问题同一类：单个慢写会拖住所有请求。
  `AsyncSqliteSaver` 走 aiosqlite，IO 在线程池里。

★ 为什么返回上下文管理器而不是裸对象：
  sqlite 连接必须在应用关闭时释放。用 `async with` 把「打开」与「关闭」绑在一起，
  比在 lifespan 里手工配对 `try/finally` 更不容易漏。

★ 目录会在打开前创建：`data/` 已被 `.gitignore` 忽略，首次运行可能不存在这个目录，
  而 aiosqlite 建库时不会自己建父目录。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.core.config import Settings


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else Path.cwd() / path


@asynccontextmanager
async def open_checkpointer(settings: Settings) -> AsyncIterator[BaseCheckpointSaver]:
    """按配置打开 checkpointer，退出上下文时释放连接。"""
    if settings.agent_checkpoint == "memory":
        # InMemorySaver 同时实现了同步与异步接口，可以直接给 async 图用
        yield InMemorySaver()
        return

    path = _resolve(settings.agent_memory_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield saver
