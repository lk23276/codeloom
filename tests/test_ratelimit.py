"""限流算法与接线测试。

★ 这个文件守的是三类「限流看起来生效、其实没有」的错误：

      ① 窗口边界放过 2 倍突发 —— 固定窗口的结构性 bug，只能靠时间序列断言抓
      ② Retry-After 写死 1 秒    —— 客户端越重试越像被封
      ③ key 选取错误 / 桶无限增长 —— 要么一人拖垮所有人，要么限流器自己成内存炸弹

★ 时间一律**注入**（`now=`），不 sleep：
  限流的正确性全在时间轴上，靠真实时钟测就只能写成「等 1.1 秒」的慢测试，
  而且边界（0.99 / 1.01）根本测不准。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from starlette.requests import Request

from app.core.config import Settings
from app.core.errors import RateLimited
from app.core.ratelimit import (
    Decision,
    KeyedLimiter,
    SlidingWindow,
    build_limiter,
    client_key,
    rate_limit,
)

# ══════════════════════ ① 滑动窗口：额度与拒绝 ══════════════════════


def test_sliding_window_allows_then_rejects() -> None:
    """limit=3：前 3 次放行，第 4 次拒绝，且 remaining 归 0。"""
    w = SlidingWindow(limit=3, window_s=60)

    results = [w.allow(now=0.0) for _ in range(4)]

    assert [r.allowed for r in results] == [True, True, True, False]
    assert [r.remaining for r in results] == [2, 1, 0, 0]
    assert results[-1].retry_after_s == pytest.approx(60.0)


def test_sliding_window_has_no_boundary_burst() -> None:
    """窗口边界处不能放过 2 倍流量 —— 这正是选滑动窗口而不是固定窗口的理由。

    ★ 场景（limit=5 / 1 秒）：
        t=0.99 连发 5 次  → 放行
        t=1.01 再发 5 次  → 固定窗口会全部放行（跨格子了），
                            滑动窗口会全部拒绝（过去 1 秒内已有 5 次）
      只用「连续发 N 次」的测试是抓不到这个 bug 的。
    """
    w = SlidingWindow(limit=5, window_s=1.0)

    first = [w.allow(now=0.99).allowed for _ in range(5)]
    second = [w.allow(now=1.01).allowed for _ in range(5)]

    assert first == [True] * 5
    assert second == [False] * 5, "边界突刺：0.02 秒内放过了 10 次"


def test_sliding_window_retry_after_shrinks_as_time_passes() -> None:
    """Retry-After 必须「等最早那次滑出窗口」，不能写死。

    ★ 写死 1 秒的后果：真实等待可能是 0.6 秒，客户端 1 秒后重试仍被拒，
      又等 1 秒 —— 表现就是「越重试越像被封」。
    """
    w = SlidingWindow(limit=1, window_s=10)
    assert w.allow(now=100.0).allowed

    early = w.allow(now=100.5)  # 才过 0.5 秒
    late = w.allow(now=109.9)  # 快滑出去了

    assert early.allowed is False and early.retry_after_s == pytest.approx(9.5)
    assert late.allowed is False and late.retry_after_s == pytest.approx(0.1)


def test_sliding_window_forgets_old_hits() -> None:
    """滑出窗口的记录要真的丢掉，否则「等一会儿再试」永远不生效。"""
    w = SlidingWindow(limit=2, window_s=5)

    assert w.allow(now=0.0).allowed
    assert w.allow(now=1.0).allowed
    assert w.allow(now=2.0).allowed is False
    assert w.allow(now=5.1).allowed is True, "0.0 那次已经滑出窗口，应该有空位"
    assert w.stored == 2, "过期的记录必须清掉（否则内存只增不减）"


def test_sliding_window_rejects_bad_config() -> None:
    """limit / window 非正时启动即报错，而不是运行期才出怪行为。"""
    with pytest.raises(ValueError):
        SlidingWindow(limit=0, window_s=60)
    with pytest.raises(ValueError):
        SlidingWindow(limit=10, window_s=0)


# ══════════════════════ ② 按 key 分桶 + LRU ══════════════════════


def test_keyed_limiter_isolates_clients() -> None:
    """A 打满额度不能影响 B —— 否则一个用户就能把所有人挡在门外。"""
    limiter = KeyedLimiter(limit=2, window_s=60)

    assert limiter.check("1.1.1.1", now=0.0).allowed
    assert limiter.check("1.1.1.1", now=0.0).allowed
    assert limiter.check("1.1.1.1", now=0.0).allowed is False

    assert limiter.check("2.2.2.2", now=0.0).allowed is True
    assert limiter.size == 2


def test_keyed_limiter_evicts_lru_when_full() -> None:
    """key 数量到上限必须淘汰最久未用的 —— 没有上限的限流器就是内存炸弹。

    ★ 攻击者用 100 万个 IP 打过来，每个 IP 建一个窗口；
      没有 `max_keys` 的话，限流器自己先 OOM。
    """
    limiter = KeyedLimiter(limit=5, window_s=60, max_keys=2)

    limiter.check("a", now=0.0)
    limiter.check("b", now=0.0)
    limiter.check("a", now=0.0)  # a 变成「最近用过」
    limiter.check("c", now=0.0)  # 触发淘汰

    assert limiter.evictions == 1
    assert limiter.size == 2

    # b 被淘汰了：再进来是一个全新的窗口（额度重新开始）
    decision = limiter.check("b", now=0.0)
    assert decision.allowed is True and decision.remaining == 4


def test_decision_headers_follow_http_conventions() -> None:
    """429 的响应头：Retry-After 取整、X-RateLimit-* 不能为负。"""
    denied = Decision(False, limit=30, remaining=0, retry_after_s=0.2).headers()
    assert denied["Retry-After"] == "1", "0.2 秒也要向上取整成 1（HTTP 只认整数秒）"
    assert denied["X-RateLimit-Limit"] == "30"
    assert denied["X-RateLimit-Remaining"] == "0"

    allowed = Decision(True, limit=30, remaining=29).headers()
    assert "Retry-After" not in allowed, "放行的响应不该带 Retry-After"


def test_build_limiter_reads_settings() -> None:
    """配置里的两个数字要真的进到 limiter 里（写反了就是查不出的小 bug）。"""
    settings = Settings(_env_file=None, rate_limit=7, rate_window_s=30)

    limiter = build_limiter(settings)

    assert (limiter.limit, limiter.window_s) == (7, 30.0)


# ══════════════════════ ③ 接线：client_key 与依赖 ══════════════════════


def make_request(
    host: str | None = None,
    *,
    state: object | None = None,
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Request:
    """造一个最小可用的 Request（不经过网络）。

    ★ `client` 缺失是合法场景（ASGI 直连 / 某些测试客户端），
      所以 client_key 必须有兜底，不能 `request.client.host` 裸写。
    """
    scope: dict = {
        "type": "http",
        "headers": headers or [],
        "method": "GET",
        "path": "/",
        "query_string": b"",
    }
    if host is not None:
        scope["client"] = (host, 12345)
    if state is not None:
        scope["app"] = SimpleNamespace(state=state)
    return Request(scope)


def test_client_key_uses_peer_ip_and_ignores_xff() -> None:
    """限流 key 取 TCP 层 IP，**不采信** X-Forwarded-For。

    ★ 为什么专门测「忽略 XFF」：XFF 是客户端可伪造的。信它等于把限流开关
      交给攻击者 —— 每次请求换一个假 IP，就永远打不到限额。
    """
    req = make_request("10.0.0.9", headers=[(b"x-forwarded-for", b"1.2.3.4")])

    assert client_key(req) == "10.0.0.9"
    assert client_key(make_request(None)) == "unknown"


def test_rate_limit_dependency_raises_with_retry_after() -> None:
    """依赖超限时抛 RateLimited，并且 body/header 两处都带等待秒数。"""
    state = SimpleNamespace(ratelimit=KeyedLimiter(limit=1, window_s=60))
    req = make_request("10.0.0.9", state=state)

    rate_limit(req)  # 第 1 次放行

    with pytest.raises(RateLimited) as ei:
        rate_limit(req)

    exc = ei.value
    assert exc.status == 429
    assert exc.data == {"retry_after_s": 60}
    assert exc.headers["Retry-After"] == "60"
    assert exc.headers["X-RateLimit-Limit"] == "1"
    assert "60 秒" in exc.msg
