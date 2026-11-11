"""把同步函数丢进线程池的小工具。

★ 为什么必须有它：
  `sqlite3` 是同步的。在 `async def` 里直接调它会**堵死整个事件循环**——
  所有并发请求一起卡住。这不是"慢一点"，是"整个服务停住"。

  这和 server2 里「绝不能在 Reactor 回调里做阻塞 IO」是同一条纪律，
  只是 C++ 里要手写线程池，Python 里就是 `asyncio.to_thread`。

★ 用法：
      rows = await run_sync(store.conversations, limit)
  而不是：
      rows = store.conversations(limit)        # ← 阻塞事件循环
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any


async def run_sync[T](fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """在线程池里跑同步函数，返回它的结果。

    ★ 用了 PEP 695 的类型参数语法（`def f[T](...)`，Python 3.12+），
      比 `TypeVar("T")` 更短，而且泛型参数只在这一个函数里有效。
    """
    return await asyncio.to_thread(fn, *args, **kwargs)
