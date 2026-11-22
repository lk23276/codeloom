"""本地 trace：一行一个 span 的 JSONL 兜底。

为什么必须有它（而不是只接 Langfuse）：

    Langfuse 需要账号与网络。这两样东西一旦缺，服务就不可观测 —— 更糟的是，
    别人 clone 下来跑这个项目时，会卡在「请先注册」。

所以顺序是：**本地永远可用，远程有则加上**，而不是反过来。

★ 这里是**手工 span**，与 LangChain 的 CallbackHandler 覆盖范围不同：
  CallbackHandler 只能自动埋 chain / tool / retriever 这类框架组件；
  我们 RAG 的五步是手写函数，框架看不见，必须自己开 span。

★ 一行一个 span、扁平结构，靠 `trace_id` 关联：
  嵌套关系在查询时由 `trace_id` + 时间序重建，写入端不做树形拼接 ——
  追加写是最省事的，也最不容易因为进程被杀而丢半棵树。
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Tracer(Protocol):
    """span 记录器。业务层只依赖这个协议，换实现（Langfuse / OTel）不用改调用点。"""

    def span(self, name: str, **attrs: Any) -> Any: ...


class NullTracer:
    """什么都不记。测试与「显式关闭」用。"""

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        sink: dict[str, Any] = {}
        yield sink


class LocalTracer:
    """把 span 追加进 JSONL。

    用法：

        with tracer.span("retrieve", k=20) as span:
            docs = ...
            span["hits"] = len(docs)      # 退出时才会落盘
    """

    def __init__(self, path: Path, *, trace_id_getter: Any = None) -> None:
        self._path = path
        self._trace_id = trace_id_getter or _default_trace_id

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        sink: dict[str, Any] = dict(attrs)
        started = time.perf_counter()
        try:
            yield sink
        finally:
            # ★ 用 finally：span 里抛异常时也要留下记录，
            #   否则「出错的请求」恰好是没有 trace 的那些。
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            self._write(name, duration_ms, sink)

    def _write(self, name: str, duration_ms: float, attrs: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(UTC).astimezone().isoformat(timespec="milliseconds"),
            "trace_id": self._trace_id(),
            "span": name,
            "ms": duration_ms,
            "attrs": attrs,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:  # 磁盘满 / 权限不足：绝不能让可观测把主流程拖垮
            logger.warning("trace_write_failed path=%s err=%s", self._path, type(exc).__name__)


def _default_trace_id() -> str:
    """优先复用请求的 request_id，这样日志与 trace 是同一条线。"""
    try:
        from app.core.logging import request_id_var

        return request_id_var.get() or "-"
    except Exception:  # noqa: BLE001 —— 脱离请求上下文时（脚本 / 测试）不该报错
        return "-"


def build_tracer(settings: Any) -> Tracer:
    """按配置构造 tracer。默认本地 JSONL，关闭时退化为 NullTracer。"""
    if not getattr(settings, "trace_local", True):
        return NullTracer()

    raw = getattr(settings, "trace_path", "./data/traces.jsonl")
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return LocalTracer(path)
