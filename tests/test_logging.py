"""结构化日志测试。

★ 日志的 bug 有个共同点：**不影响功能，只影响排查**。
  所以它最容易被写成「打印了就算完」，等真出事才发现字段名不对/没有 trace id。
  这里用断言把三件事钉住：

      ① JSON 必须是**一行一个对象**（多行会让采集端按行切分时把一条日志切成几条）
      ② `request_id` 必须真的来自 contextvar，且不带进下一条不相干的日志
      ③ access 日志的字段名（method/path/status/duration_ms）不能改
         —— 看板和告警是按字段名写的，改了会静默失效

用法：
    uv run pytest tests/test_logging.py -q
"""

from __future__ import annotations

import json
import logging
import sys

import pytest
from fastapi.testclient import TestClient

from app.client.llm import FakeClient
from app.core.logging import (
    JsonFormatter,
    RequestIdFilter,
    TextFormatter,
    get_request_id,
    request_id_var,
    setup_logging,
)
from app.main import create_app


def make_record(**extra: object) -> logging.LogRecord:
    """造一条 record；`extra` 模拟 logger.info(..., extra={...}) 的写法。"""
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


# ══════════════════════ ① JSON 形状 ══════════════════════


def test_json_formatter_is_one_line_with_required_fields() -> None:
    """一行一个 JSON，固定字段齐全，且不把 LogRecord 的内部字段倒出来。"""
    token = request_id_var.set("abc123")
    try:
        record = make_record(method="POST", path="/chat", status=200, duration_ms=12.5)
        RequestIdFilter().filter(record)
        line = JsonFormatter().format(record)
    finally:
        request_id_var.reset(token)

    assert "\n" not in line, "多行 JSON 会被采集端切成多条日志"

    data = json.loads(line)
    assert data["request_id"] == "abc123"
    assert data["level"] == "INFO"
    assert data["logger"] == "app.test"
    assert data["msg"] == "hello world", "msg 必须是格式化后的文本，不是模板"
    assert (data["method"], data["path"], data["status"]) == ("POST", "/chat", 200)
    assert data["duration_ms"] == 12.5
    assert "args" not in data and "exc_info" not in data, "内部字段不该出现在日志里"


def test_json_formatter_keeps_traceback_in_single_field() -> None:
    """异常堆栈进 `exc` 字段，仍然是一行 —— 否则 ELK 会把一条异常拆成几十条。"""
    try:
        _ = 1 / 0
    except ZeroDivisionError:
        record = logging.LogRecord(
            "app.test", logging.ERROR, __file__, 1, "炸了", (), sys.exc_info()
        )

    RequestIdFilter().filter(record)
    line = JsonFormatter().format(record)
    data = json.loads(line)

    assert "\n" not in line
    assert "ZeroDivisionError" in data["exc"]


def test_text_formatter_carries_trace_id() -> None:
    """本地开发用的文本格式也必须带 trace id，否则两种模式排查能力不对等。"""
    token = request_id_var.set("deadbeef")
    try:
        record = make_record()
        RequestIdFilter().filter(record)
        out = TextFormatter().format(record)
    finally:
        request_id_var.reset(token)

    assert "[deadbeef]" in out
    assert "hello world" in out


def test_request_id_defaults_to_dash_outside_request() -> None:
    """启动日志（不在任何请求里）应显示 `-`，不能抛异常。"""
    record = make_record()
    RequestIdFilter().filter(record)

    assert record.request_id == "-"
    assert json.loads(JsonFormatter().format(record))["request_id"] == "-"


def test_setup_logging_is_idempotent_and_accepts_both_formats() -> None:
    """重复调用不能把 handler 越堆越多（每条日志打印 N 遍）。

    ★ 这里临时清空 root.handlers 才能走到「首次安装」分支：
      pytest 在测试开始前就挂了采集 handler，正常路径下 `if not root.handlers`
      永远为 False —— 不这么做，安装逻辑一行都测不到。
      清空前存一份、finally 里原样放回，不影响其他用例。
    """
    root = logging.getLogger()
    before = list(root.handlers)
    access = logging.getLogger("uvicorn.access")
    access_was_disabled = access.disabled
    root.handlers.clear()
    try:
        setup_logging("DEBUG", "json")
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
        assert any(isinstance(f, RequestIdFilter) for f in root.handlers[0].filters)

        # 第二次调用（换格式）不能重复装 handler
        setup_logging("DEBUG", "text")
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)

        # ★ uvicorn 的访问日志必须彻底关掉：否则每条请求会打两行
        #   （一行我们的 JSON，一行 uvicorn 的文本），而且 uvicorn 那行没有 trace id
        assert access.disabled is True
        assert logging.getLogger("uvicorn.error").propagate is True
    finally:
        root.handlers[:] = before
        access.disabled = access_was_disabled


def test_get_request_id_reads_contextvar() -> None:
    """`get_request_id()` 与 formatter 读的是同一个来源，不会出现两套 id。"""
    token = request_id_var.set("xyz789")
    try:
        assert get_request_id() == "xyz789"
    finally:
        request_id_var.reset(token)


# ══════════════════════ ② 接线：access 日志 + 响应头 ══════════════════════


@pytest.fixture
def client(tmp_path) -> TestClient:
    app = create_app(db_path=str(tmp_path / "sessions.db"), llm=FakeClient(reply="假回复"))
    with TestClient(app) as c:
        yield c


def test_access_log_fields_match_response_header(client: TestClient, caplog) -> None:
    """一条请求一行 access：method / path / status / duration_ms / request_id 齐全。

    ★ `request_id` 由 formatter 的 filter 从 contextvar 取；测试里要把同一个
      filter 挂到 caplog 的 handler 上，否则看到的 record 没有这个属性
      —— 这本身就是个提醒：**字段是 filter 加的，不是中间件加的**。
    """
    caplog.handler.addFilter(RequestIdFilter())

    with caplog.at_level(logging.INFO, logger="app.main"):
        r = client.post("/chat", json={"message": "hi"})

    records = [x for x in caplog.records if x.getMessage() == "access"]
    assert len(records) == 1, "一个请求只应有一条 access，重复说明中间件挂了两层"

    rec = records[0]
    rid = r.headers["x-request-id"]
    assert rec.request_id == rid, "日志里的 id 和响应头必须是同一个"
    assert (rec.method, rec.path, rec.status) == ("POST", "/chat", 200)
    assert isinstance(rec.duration_ms, float)


def test_request_id_present_on_error_responses(client: TestClient) -> None:
    """404 也要带 X-Request-ID —— 客户端报「打不开」时最需要这个串。

    ★ 中间件放在路由之外才有这个效果；只在路由里生成的话，
      没进路由的请求（404/405/422）就没有 id 了。
    """
    r1 = client.get("/sessions/not-exist/messages")
    r2 = client.get("/nope")

    assert r1.status_code == 404 and r2.status_code == 404
    assert r1.headers["x-request-id"] and r2.headers["x-request-id"]
    assert r1.headers["x-request-id"] != r2.headers["x-request-id"], "每个请求都要新的 id"
    assert r1.json()["request_id"] == r1.headers["x-request-id"]
