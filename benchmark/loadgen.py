#!/usr/bin/env python
"""CodeLoom 压测脚本：三组口径，一个文件。

★ 为什么不用 wrk2：
  1. `/chat` 的瓶颈在上游（一次生成约 2 秒），wrk2 只给「请求完成时间」，
     拆不出首字节（TTFB）与 delta 间隔；
  2. `/chat/stream` 要逐帧读 SSE，wrk2 不会解析 `event:` 行；
  3. 项目里已经有 httpx，不引入新依赖。
  纯 HTTP 吞吐的极限值仍建议用 wrk2 复核（C 实现，能打满几万 RPS）：
      wrk2 -t4 -c200 -d30s -R2000 http://127.0.0.1:8001/healthz
  本脚本负责「与 LLM 相关的三组」。

★ 三组口径：
  probe   GET /healthz 或 /sessions   测框架与 SQLite，不受上游影响
  chat    POST /chat                  测端到端（真模型），记录 token 与 429
  stream  POST /chat/stream           测 TTFB、delta 数、delta 间隔（真流式证据）

★ 计费警告：
  chat / stream 每一发都真调模型。默认 `--requests 10`，别随手调大。
  先用 `--requests 2` 验通路，再放大。

★ 并发模型：
  关环（closed-loop）：每个 worker 串行发请求，N 个 worker = N 并发。
  这是诚实的做法 —— LLM 场景下「固定 QPS 开环」会让排队被隐藏在客户端，
  看起来数字很好看，实际服务端已经堆了。
  需要开环（固定 RPS）时用 wrk2 打 probe 组。

用法：
  uv run python benchmark/loadgen.py probe  --url http://127.0.0.1:8001 --duration 10
  uv run python benchmark/loadgen.py chat   --url http://127.0.0.1:8001 --requests 10
  uv run python benchmark/loadgen.py stream --url http://127.0.0.1:8001 --requests 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

# ══════════════════════ 一、采样与统计 ══════════════════════


@dataclass
class Sample:
    """一次请求的观测值。字段宁可多，事后不用重跑。"""

    ok: bool
    status: int
    latency_ms: float
    ttfb_ms: float | None = None  # 仅 stream：第一个 delta 的时间
    deltas: int = 0  # 仅 stream
    gap_p95_ms: float | None = None  # 仅 stream：相邻 delta 间隔的 P95
    prompt_tokens: int = 0  # 仅 chat
    completion_tokens: int = 0  # 仅 chat
    retry_after: str | None = None  # 429 才有
    error: str = ""


def pct(values: list[float], p: float) -> float:
    """最近秩百分位（nearest-rank）。样本少时不要用插值，会给出不存在的数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[k]


@dataclass
class Summary:
    group: str
    label: str
    concurrency: int
    requests: int
    wall_s: float
    rps: float
    success: int
    http_429: int
    errors: int
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_max: float
    ttfb_p50: float = 0.0
    ttfb_p95: float = 0.0
    deltas_avg: float = 0.0
    gap_p95_avg: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retry_after_seen: list[str] = field(default_factory=list)

    def md_row(self) -> str:
        """一行 Markdown，直接贴进报告。"""
        name = self.label.split("@")[0] if self.label else self.group
        extra = ""
        if self.group == "stream":
            extra = (
                f" TTFB P50={self.ttfb_p50:.0f}ms P95={self.ttfb_p95:.0f}ms"
                f" | delta 均值={self.deltas_avg:.0f} 个"
            )
        if self.group == "chat":
            extra = f" token={self.prompt_tokens}+{self.completion_tokens}"
        return (
            f"| {name} | {self.concurrency} | {self.requests} | {self.rps:.1f} "
            f"| {self.success} | {self.http_429} | {self.errors} "
            f"| {self.latency_p50:.0f} | {self.latency_p95:.0f} | {self.latency_p99:.0f} "
            f"| {self.latency_max:.0f} |{extra} |"
        )


def summarize(group: str, label: str, samples: list[Sample], wall_s: float, conc: int) -> Summary:
    lat = [s.latency_ms for s in samples]
    ok = [s for s in samples if s.ok]
    gaps = [s.gap_p95_ms for s in ok if s.gap_p95_ms is not None]
    return Summary(
        group=group,
        label=label,
        concurrency=conc,
        requests=len(samples),
        wall_s=round(wall_s, 2),
        rps=round(len(samples) / wall_s, 2) if wall_s > 0 else 0.0,
        success=len(ok),
        http_429=sum(1 for s in samples if s.status == 429),
        errors=sum(1 for s in samples if not s.ok and s.status != 429),
        latency_p50=round(pct(lat, 50), 1),
        latency_p95=round(pct(lat, 95), 1),
        latency_p99=round(pct(lat, 99), 1),
        latency_max=round(max(lat) if lat else 0.0, 1),
        ttfb_p50=round(pct([s.ttfb_ms for s in ok if s.ttfb_ms], 50), 1),
        ttfb_p95=round(pct([s.ttfb_ms for s in ok if s.ttfb_ms], 95), 1),
        deltas_avg=round(sum(s.deltas for s in ok) / len(ok), 1) if ok else 0.0,
        gap_p95_avg=round(sum(gaps) / len(gaps), 1) if gaps else 0.0,
        prompt_tokens=sum(s.prompt_tokens for s in ok),
        completion_tokens=sum(s.completion_tokens for s in ok),
        retry_after_seen=sorted({s.retry_after for s in samples if s.retry_after}),
    )


# ══════════════════════ 二、三种请求 ══════════════════════


@dataclass
class Config:
    url: str
    path: str = "/healthz"
    message: str = "用一句话解释什么是 I/O 多路复用"
    max_tokens: int = 100
    timeout_s: float = 90.0


def normalize_path(raw: str) -> tuple[str, bool]:
    """把 `--path` 归一成以 `/` 开头的 URL 路径。

    ★ 为什么需要这个：Git Bash（MSYS2）会把「看起来像路径」的参数改写成 Windows 路径，
      而且是**静默**的。实测 `--path /sessions` 变成了
          <盘符>:/.../PortableGit/sessions
      因为那个目录恰好存在；`/healthz` 不存在所以没被改 —— 于是「有时对有时错」。

      兜底规则：一旦识别出盘符或反斜杠，就取最后一段并补回前导 `/`。
      正常调用建议再叠一层 `MSYS_NO_PATHCONV=1`（run.sh 里已经设了）。
    """
    p = raw.strip()
    if re.match(r"^[A-Za-z]:[/\\]", p) or "\\" in p:
        tail = re.split(r"[/\\]", p.rstrip("/\\"))[-1]
        return f"/{tail}", True
    return (p if p.startswith("/") else f"/{p}"), False


async def one_probe(client: httpx.AsyncClient, cfg: Config) -> Sample:
    t0 = time.perf_counter()
    try:
        r = await client.get(cfg.url + cfg.path)
        dt = (time.perf_counter() - t0) * 1000
        return Sample(r.status_code < 400, r.status_code, dt)
    except Exception as exc:  # noqa: BLE001 —— 压测要记录任何失败，不能让一个请求终止整组
        dt = (time.perf_counter() - t0) * 1000
        return Sample(False, 0, dt, error=type(exc).__name__)


async def one_chat(client: httpx.AsyncClient, cfg: Config) -> Sample:
    payload = {"message": cfg.message, "max_tokens": cfg.max_tokens}
    t0 = time.perf_counter()
    try:
        r = await client.post(cfg.url + "/chat", json=payload)
        dt = (time.perf_counter() - t0) * 1000
        if r.status_code == 429:
            return Sample(False, 429, dt, retry_after=r.headers.get("retry-after"))
        if r.status_code != 200:
            return Sample(False, r.status_code, dt, error=f"http_{r.status_code}")

        body = r.json()
        if body.get("code") != "OK":
            return Sample(False, r.status_code, dt, error=str(body.get("code")))

        data = body.get("data") or {}
        usage = data.get("usage") or {}
        return Sample(
            True,
            r.status_code,
            dt,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )
    except Exception as exc:  # noqa: BLE001
        dt = (time.perf_counter() - t0) * 1000
        return Sample(False, 0, dt, error=type(exc).__name__)


async def one_stream(client: httpx.AsyncClient, cfg: Config) -> Sample:
    """逐帧读 SSE。

    ★ 三个时间点必须分开记：
      t0            发请求
      第一个 delta    = TTFB（用户「看到第一个字」的时间）
      最后一个 delta  = 生成结束（总时长）
      delta 间隔     = 真流式的直接证据（攒完再发的实现，间隔会接近 0）

    ★ `event: delta` 行先到、`data:` 行后到，所以在这里打时间戳，
      不要等解析完 data 再记 —— 那会把 JSON 解析时间算进 TTFB。
    """
    payload = {"message": cfg.message, "max_tokens": cfg.max_tokens}
    t0 = time.perf_counter()
    first: float | None = None
    last: float | None = None
    gaps: list[float] = []
    deltas = 0
    status = 0
    saw_error = ""

    try:
        async with client.stream("POST", cfg.url + "/chat/stream", json=payload) as r:
            status = r.status_code
            if status == 429:
                await r.aread()
                return Sample(
                    False,
                    429,
                    (time.perf_counter() - t0) * 1000,
                    retry_after=r.headers.get("retry-after"),
                )
            if status != 200:
                await r.aread()
                return Sample(
                    False, status, (time.perf_counter() - t0) * 1000, error=f"http_{status}"
                )

            async for line in r.aiter_lines():
                if line.startswith("event: delta"):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    if last is not None:
                        gaps.append((now - last) * 1000)
                    last = now
                    deltas += 1
                elif line.startswith("event: error"):
                    saw_error = "sse_error"

        dt = (time.perf_counter() - t0) * 1000
        if deltas == 0:
            return Sample(False, status, dt, error=saw_error or "no_delta")
        return Sample(
            ok=not saw_error,
            status=status,
            latency_ms=dt,
            ttfb_ms=(first - t0) * 1000 if first else None,
            deltas=deltas,
            gap_p95_ms=pct(gaps, 95) if gaps else None,
            error=saw_error,
        )
    except Exception as exc:  # noqa: BLE001
        dt = (time.perf_counter() - t0) * 1000
        return Sample(False, 0, dt, error=type(exc).__name__)


# ══════════════════════ 三、驱动 ══════════════════════


async def _repeat_n(
    one: Callable[[httpx.AsyncClient, Config], Awaitable[Sample]],
    client: httpx.AsyncClient,
    cfg: Config,
    n: int,
) -> list[Sample]:
    out: list[Sample] = []
    for _ in range(n):
        out.append(await one(client, cfg))
    return out


async def _repeat_until(
    one: Callable[[httpx.AsyncClient, Config], Awaitable[Sample]],
    client: httpx.AsyncClient,
    cfg: Config,
    deadline: float,
) -> list[Sample]:
    out: list[Sample] = []
    while time.perf_counter() < deadline:
        out.append(await one(client, cfg))
    return out


async def drive(
    one: Callable[[httpx.AsyncClient, Config], Awaitable[Sample]],
    cfg: Config,
    *,
    concurrency: int,
    requests: int,
    duration: float,
    timeout_s: float,
    trust_env: bool,
) -> tuple[list[Sample], float]:
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(connect=10.0, read=timeout_s, write=30.0, pool=timeout_s)

    # ★ trust_env=False：这台机器配了 HTTP 代理，走 127.0.0.1 也会被代理拦一道，
    #   会把「本地 0.3ms」测成「代理 200ms」。需要打外网时再显式 --trust-env。
    async with httpx.AsyncClient(limits=limits, timeout=timeout, trust_env=trust_env) as client:
        start = time.perf_counter()
        if requests > 0:
            base, extra = divmod(requests, concurrency)
            counts = [base + (1 if i < extra else 0) for i in range(concurrency)]
            tasks = [asyncio.create_task(_repeat_n(one, client, cfg, n)) for n in counts if n > 0]
        else:
            deadline = start + duration
            tasks = [
                asyncio.create_task(_repeat_until(one, client, cfg, deadline))
                for _ in range(concurrency)
            ]
        groups = await asyncio.gather(*tasks)
        wall = time.perf_counter() - start
    return [s for group in groups for s in group], wall


# ══════════════════════ 四、入口 ══════════════════════

_ONE = {"probe": one_probe, "chat": one_chat, "stream": one_stream}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CodeLoom 压测（probe / chat / stream）")
    p.add_argument("group", choices=sorted(_ONE))
    p.add_argument("--url", default="http://127.0.0.1:8001", help="服务根地址")
    p.add_argument("--path", default="/healthz", help="probe 组打哪个路径（/healthz 或 /sessions）")
    p.add_argument("--concurrency", type=int, default=4, help="并发数（关环）")
    p.add_argument("--requests", type=int, default=0, help="总请求数；>0 时忽略 --duration")
    p.add_argument("--duration", type=float, default=10.0, help="按时间跑（秒），probe 用")
    p.add_argument("--warmup", type=int, default=1, help="正式计时前的预热请求数（不计入结果）")
    p.add_argument("--message", default="用一句话解释什么是 I/O 多路复用")
    p.add_argument("--max-tokens", type=int, default=100, help="chat/stream 的输出上限（控成本）")
    p.add_argument("--timeout", type=float, default=90.0)
    p.add_argument("--label", default="", help="给这轮结果起个名字，写进报告")
    p.add_argument("--out-json", default="", help="结果 JSON 路径")
    p.add_argument("--out-md", default="", help="Markdown 片段路径")
    p.add_argument("--trust-env", action="store_true", help="允许读取 HTTP(S)_PROXY（默认关）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path, rewritten = normalize_path(args.path)
    if rewritten:
        print(
            f"[warn] --path 被 Git Bash 改写成了 Windows 路径，已归一为 {path}；"
            f"直接调用时建议加 MSYS_NO_PATHCONV=1",
            file=sys.stderr,
        )
    cfg = Config(
        url=args.url.rstrip("/"),
        path=path,
        message=args.message,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
    )
    one = _ONE[args.group]
    label = args.label or f"{args.group}@{time.strftime('%H:%M:%S')}"

    if args.group != "probe" and args.requests == 0:
        # 真模型按时间跑会失控（钱），强制要求给次数
        print("chat/stream 必须显式给 --requests（按时间跑会一直计费）。", file=sys.stderr)
        return 2

    target = f"{cfg.url}{cfg.path}" if args.group == "probe" else cfg.url
    print(
        f"[{args.group}] url={target} "
        f"c={args.concurrency} "
        f"{'n=' + str(args.requests) if args.requests else 't=' + str(args.duration) + 's'} "
        f"max_tokens={cfg.max_tokens if args.group != 'probe' else '-'}",
        file=sys.stderr,
    )

    # 预热：建连 + 触发首次导入，不计入采样
    if args.warmup > 0:
        asyncio.run(
            drive(
                one,
                cfg,
                concurrency=1,
                requests=args.warmup,
                duration=0,
                timeout_s=cfg.timeout_s,
                trust_env=args.trust_env,
            )
        )

    samples, wall = asyncio.run(
        drive(
            one,
            cfg,
            concurrency=args.concurrency,
            requests=args.requests,
            duration=args.duration,
            timeout_s=cfg.timeout_s,
            trust_env=args.trust_env,
        )
    )

    summary = summarize(args.group, label, samples, wall, args.concurrency)
    header = (
        "| 组 | 并发 | 请求 | RPS | 成功 | 429 | 其他错误 | P50(ms) | P95(ms) "
        "| P99(ms) | Max(ms) | 备注 |"
    )
    print(header)
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    print(summary.md_row())

    payload: dict[str, Any] = {
        "config": asdict(cfg)
        | {
            "concurrency": args.concurrency,
            "requests": args.requests,
            "duration": args.duration,
            "warmup": args.warmup,
            "trust_env": args.trust_env,
        },
        "summary": asdict(summary),
        "samples": [asdict(s) for s in samples[:500]],
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(summary.md_row() + "\n", "utf-8")

    # 失败时给出可执行的下一步，而不是只报个数字
    if summary.errors:
        print(
            f"\n⚠ 有 {summary.errors} 个非 429 失败，先看 JSON 里的 samples[].error；"
            f"常见原因：服务没起、上游 502、超时。",
            file=sys.stderr,
        )
    if summary.http_429:
        print(
            f"\n⚠ 有 {summary.http_429} 个 429（Retry-After={summary.retry_after_seen}）："
            f"要么调大 APP_RATE_LIMIT 再测吞吐，要么这就是限流组的结论。",
            file=sys.stderr,
        )
    return 0 if summary.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
