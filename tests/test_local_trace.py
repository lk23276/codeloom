"""本地 trace 的契约测试：写什么、什么时候写、写失败怎么办。

四条值得钉住的：

1. span 退出时落一行，且 attrs 里后塞的值也在；
2. **span 内抛异常也要落一行** —— 出错的请求恰好是最需要 trace 的；
3. trace_id 与请求号一致（日志与 trace 是同一条线）；
4. 写盘失败不能让主流程失败（可观测不能拖垮业务）。
"""

from __future__ import annotations

import json

import pytest

from app.core.logging import request_id_var
from app.obs.local_trace import LocalTracer, NullTracer, build_tracer


@pytest.fixture
def trace_file(tmp_path):
    return tmp_path / "traces.jsonl"


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_span_writes_one_line_with_attrs(trace_file) -> None:
    tracer = LocalTracer(trace_file)

    with tracer.span("retrieve", k=20) as span:
        span["hits"] = 7

    rows = read_rows(trace_file)
    assert len(rows) == 1
    assert rows[0]["span"] == "retrieve"
    assert rows[0]["attrs"] == {"k": 20, "hits": 7}
    assert rows[0]["ms"] >= 0


def test_span_records_even_when_body_raises(trace_file) -> None:
    tracer = LocalTracer(trace_file)

    with pytest.raises(RuntimeError):
        with tracer.span("rerank", candidates=3) as span:
            span["scored"] = 0
            raise RuntimeError("模拟失败")

    rows = read_rows(trace_file)
    assert rows[0]["span"] == "rerank"
    assert rows[0]["attrs"]["candidates"] == 3


def test_trace_id_follows_request_id(trace_file) -> None:
    tracer = LocalTracer(trace_file)
    token = request_id_var.set("rid-abc123")
    try:
        with tracer.span("llm"):
            pass
    finally:
        request_id_var.reset(token)

    assert read_rows(trace_file)[0]["trace_id"] == "rid-abc123"


def test_missing_request_id_falls_back(trace_file) -> None:
    tracer = LocalTracer(trace_file)

    with tracer.span("llm"):
        pass

    assert read_rows(trace_file)[0]["trace_id"] == "-"


def test_write_failure_does_not_raise(tmp_path) -> None:
    """把目录当文件：open 会抛 IsADirectoryError，但业务不该受影响。"""
    tracer = LocalTracer(tmp_path)

    with tracer.span("retrieve") as span:      # 不应抛异常
        span["hits"] = 1


def test_null_tracer_writes_nothing(trace_file) -> None:
    tracer = NullTracer()

    with tracer.span("retrieve") as span:
        span["hits"] = 3

    assert not trace_file.exists()


def test_build_tracer_respects_switch(trace_file) -> None:
    class Cfg:
        trace_local = False
        trace_path = str(trace_file)

    assert isinstance(build_tracer(Cfg()), NullTracer)

    class CfgOn:
        trace_local = True
        trace_path = str(trace_file)

    assert isinstance(build_tracer(CfgOn()), LocalTracer)
