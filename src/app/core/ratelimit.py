"""限流：滑动窗口 + 按 key 分桶（v0.1 单进程版）。

★ 为什么手写而不是 slowapi：
  本项目只需要「按 IP 限 /chat」这一条规则，引入一个框架等于为一个函数装一套机制；
  更要紧的是**限流的坑都在语义上**（窗口边界、Retry-After 精度、key 的选取），
  这些坑只有真正实现过才会暴露。

★ 为什么选「滑动窗口」而不是令牌桶：
  令牌桶允许突发，适合「用户平时不用、一用就连点」的产品；本项目的对话接口
  每次调用都要花真金白银，我们**不想要突发**。滑动窗口判定的是「过去 window 秒
  内到底发了多少次」，边界处不会像固定窗口那样放过 2 倍流量。

      固定窗口：limit=5/s，t=0.99 发 5 次 + t=1.01 发 5 次 → 0.02 秒内放过 10 次
      滑动窗口：同样两次，第二次会被拒（因为过去 1 秒内有 10 次记录）

★ ★ v0.1 只支持**单进程**：
  limiter 是进程内 `dict`，`uvicorn --workers 4` 下每个 worker 各限各的，
  实际额度变成 4 × limit。这不是 bug，是**刻意写清的边界**（Dockerfile 里 workers=1）。
  多进程要换成 Redis：`ZREMRANGEBYSCORE` 清过期 + `ZCARD` 计数 + `EXPIRE`，用 Lua 保证原子。

★ ★ 为什么**不**信任 `X-Forwarded-For`：
  这个头客户端可以随便伪造。没有「可信代理」配置时信它，等于把限流开关交给攻击者：
  每次请求换一个假 IP，就永远打不到限额。
  所以默认只用 TCP 层的 `request.client.host`。真部署在 nginx 后面时，
  正确做法是配置「可信代理列表」，只对来自这些代理的请求采信 XFF，
  并且**从右往左**取第一跳（最左边那段是客户端自己写的，不可信）。
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass

from fastapi import Request

from app.core.config import Settings
from app.core.errors import RateLimited

# ══════════════════════ 一、判定结果 ══════════════════════


@dataclass(frozen=True)
class Decision:
    """一次限流判定的结果。

    ★ `retry_after_s` 必须由算法算出来，不能写死 1 秒：
      100 次/分钟的规则下，客户端可能只要等 0.6 秒。写死 1 秒会让它
      1 秒后重试又被拒，表现就是「越重试越像被封」。
    """

    allowed: bool
    limit: int
    remaining: int
    retry_after_s: float = 0.0

    def headers(self) -> dict[str, str]:
        """转成响应头（GitHub / Stripe 的惯例字段名）。"""
        out = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(self.remaining, 0)),
        }
        if not self.allowed:
            # HTTP 规范要求 Retry-After 是整数秒（或 HTTP-date），所以向上取整
            out["Retry-After"] = str(max(1, math.ceil(self.retry_after_s)))
        return out


# ══════════════════════ 二、滑动窗口 ══════════════════════


class SlidingWindow:
    """记录最近 `window_s` 秒内每一次放行的时间戳。

    ★ 代价要讲清：每个 key 最多存 `limit` 个 float。
      1 万 key × 30 次 × 8 字节 ≈ 2.4 MB，可以接受；
      但换成「10 万 key × 1000 次」就是 800 MB —— 这是它上不了大流量的原因，
      也是 KeyedLimiter 必须有 `max_keys` 的原因。
    """

    def __init__(self, limit: int, window_s: float) -> None:
        if limit <= 0 or window_s <= 0:
            raise ValueError("limit 与 window_s 必须为正")
        self.limit = limit
        self.window_s = float(window_s)
        self._hits: deque[float] = deque()

    def allow(self, *, now: float | None = None) -> Decision:
        t = time.monotonic() if now is None else now

        # ① 先把滑出窗口的记录丢掉（这才是「滑动」的含义）
        cutoff = t - self.window_s
        while self._hits and self._hits[0] <= cutoff:
            self._hits.popleft()

        # ② 判断 + 记账
        if len(self._hits) < self.limit:
            self._hits.append(t)
            return Decision(True, self.limit, self.limit - len(self._hits))

        # ③ 拒绝：等到「最早那次」滑出窗口，就有空位了
        wait = self._hits[0] + self.window_s - t if self._hits else self.window_s
        return Decision(False, self.limit, 0, max(wait, 0.0))

    @property
    def stored(self) -> int:
        return len(self._hits)


# ══════════════════════ 三、按 key 分桶 + LRU ══════════════════════


class KeyedLimiter:
    """每个 key 一个窗口，带 LRU 上限。

    ★ `max_keys` 不是可选项：没有它，攻击者用 100 万个不同 IP 打过来，
      就会为每个 IP 建一个窗口 —— **限流器自己变成内存炸弹**。
      这是限流实现里最常见的漏洞。

    ★ 用 LRU 而不是 TTL 定时清理：一个 OrderedDict 就够，
      而且「最久没来的 key」本来就该被忘掉；TTL 还要额外定时任务。
    """

    def __init__(self, limit: int, window_s: float, *, max_keys: int = 10_000) -> None:
        self.limit = limit
        self.window_s = float(window_s)
        self.max_keys = max_keys
        self._windows: OrderedDict[str, SlidingWindow] = OrderedDict()
        self.evictions = 0

    def check(self, key: str, *, now: float | None = None) -> Decision:
        window = self._windows.get(key)
        if window is None:
            if len(self._windows) >= self.max_keys:
                self._windows.popitem(last=False)  # 最久未用的先丢
                self.evictions += 1
            window = SlidingWindow(self.limit, self.window_s)
            self._windows[key] = window
        else:
            self._windows.move_to_end(key)  # 标记为最近用过
        return window.allow(now=now)

    @property
    def size(self) -> int:
        return len(self._windows)


def build_limiter(settings: Settings) -> KeyedLimiter:
    """按配置造 limiter（lifespan 里调一次，放进 app.state）。"""
    return KeyedLimiter(settings.rate_limit, settings.rate_window_s)


# ══════════════════════ 四、FastAPI 接线 ══════════════════════


def client_key(request: Request) -> str:
    """限流的 key：客户端 IP。

    ★ 这里刻意**不看** `X-Forwarded-For`，理由见模块 docstring 最后一段。
      拿不到 client（极少见，比如 ASGI 测试直接构造 scope）时退回 "unknown"，
      这样所有拿不到 IP 的请求共享一个桶 —— 宁可误伤，不可漏放。
    """
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    """路由级依赖：超限抛 `RateLimited`（429），没超什么都不做。

    ★ 为什么挂在 `dependencies=[Depends(rate_limit)]` 而不是函数参数：
      它只做副作用，没有值要注入。写成参数会让每个 handler 都多一个用不到的形参。

    ★ 为什么不用中间件：中间件要按路径前缀判断「哪些端点限流」，
      规则会散在两处（路由表写着限流、中间件写着路径匹配），改一处忘一处。
      挂在路由上，端点和规则在同一行。
    """
    limiter: KeyedLimiter = request.app.state.ratelimit
    decision = limiter.check(client_key(request))
    if decision.allowed:
        return

    wait = max(1, math.ceil(decision.retry_after_s))
    raise RateLimited(
        f"请求过于频繁，请 {wait} 秒后重试",
        # ★ 字段名与 design.md §4.2 的错误示例一致
        data={"retry_after_s": wait},
        # 429 也要给标准头：网关和客户端库会直接读 Retry-After 做退避
        headers=decision.headers(),
    )
