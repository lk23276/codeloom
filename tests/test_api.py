"""HTTP 层测试：router 的职责边界。

★ 为什么 service 层的 10 条测试不够：
  它们把业务逻辑锁住了，但 `router/` 那 84 行一行都没跑到。
  而 router 的 bug 恰好是「service 全对、接口全错」这一类：

      - 探针套了统一 envelope      -> 编排系统读不懂（它只看状态码）
      - SSE 帧顺序错               -> 前端拿不到 session_id，没法续聊
      - 流式 404 变成 200 + error  -> HTTP 语义废了
      - 删除没级联                 -> 留下孤儿消息，而且不报错

  这些在 service 层都测不出来，因为它们只存在于「HTTP 契约」这一层。

★ 用 TestClient 而不是真起 uvicorn：
  它在进程内走完整的 ASGI 栈（lifespan / 中间件 / 异常处理器 / 路由），
  但不需要端口、不需要网络，跑一次不到 1 秒。
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.client.llm import FakeClient, LLMError
from app.core.config import Settings
from app.main import create_app

# ══════════════════════ 夹具 ══════════════════════


@pytest.fixture
def client(tmp_path) -> Iterator[TestClient]:
    """一个完整的 app：真实 SQLite（临时目录）+ 假 LLM（不联网）。"""
    app = create_app(
        db_path=str(tmp_path / "sessions.db"),
        llm=FakeClient(reply="假回复", chunks=["你", "好", "呀"]),
    )
    # with 才会触发 lifespan（建 store、注册日志），不加会拿到未初始化的 state
    with TestClient(app) as c:
        yield c


# ══════════════════════ ① 存活探针 ══════════════════════


def test_healthz_is_plain_and_uncacheable(client: TestClient) -> None:
    """`/healthz` 必须是裸 JSON，不能套 envelope，且不可缓存。

    ★ 两个断言各有来历：
      1. `"code" not in body` —— 探针消费者是 k8s/docker，它只看状态码；
         套了 envelope 等于多一个可能出错的环节（定下的规矩）
      2. `Cache-Control: no-store` —— 探针结果被代理缓存住的话，
         容器已经坏了但编排系统还看到 200
    """
    r = client.get("/healthz")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["uptime_s"], float)
    assert "code" not in body, "探针不套统一响应体"
    assert r.headers["cache-control"] == "no-store"


# ══════════════════════ ② 就绪探针：能接流量吗 ══════════════════════


def test_readyz_turns_503_when_db_breaks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """库坏了 -> `/readyz` 503，但 `/healthz` 仍然 200。

    ★ 这是真容器里实测过的分工，也是「为什么要有两个探针」的唯一证明：

        /healthz 200  -> 进程还活着，别重启（重启治不了数据库）
        /readyz  503  -> 摘流量，别再把请求送进来

      如果两个探针合成一个，就只能二选一：要么该重启时没重启，
      要么把好好的进程反复杀掉、丢掉在途请求。
    """
    assert client.get("/readyz").status_code == 200
    assert client.get("/readyz").json()["db"] == "ok"

    # 让 ping 抛错，模拟「库文件被删/连接断了」
    def boom() -> None:
        raise RuntimeError("模拟数据库不可用")

    monkeypatch.setattr(client.app.state.store, "ping", boom)

    r = client.get("/readyz")
    assert r.status_code == 503
    assert r.json()["status"] == "not_ready"

    # 存活探针不查依赖，所以不受影响 —— 这正是它该有的行为
    assert client.get("/healthz").status_code == 200


# ══════════════════════ ③ /chat 的统一响应体 ══════════════════════


def test_chat_returns_envelope_with_usage(client: TestClient) -> None:
    """`/chat` 必须回四字段 envelope，且 `request_id` 与响应头一致。

    ★ `request_id` 的双份存在不是冗余：
      body 里那份给客户端贴给客服/工单，响应头那份给网关和日志采集。
      两者必须同源（都来自中间件），否则排查时会发现「对不上」。
    """
    r = client.post("/chat", json={"message": "用一句话介绍 epoll"})

    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"code", "msg", "data", "request_id"}, body
    assert body["code"] == "OK"

    data = body["data"]
    assert data["reply"] == "假回复"
    assert data["session_id"], "不传 session_id 时应新建并回传"
    assert data["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }

    assert body["request_id"] == r.headers["x-request-id"]


# ══════════════════════ ④ 流式：帧序列 ══════════════════════


def test_chat_stream_emits_meta_delta_done(client: TestClient) -> None:
    """SSE 必须是 `meta -> delta... -> done`，且以空行结尾。

    ★ 三个判据缺一不可：
      1. `meta` 在第一位 —— 前端要拿 session_id 才能续聊
      2. `done` 在最后 —— 客户端靠它停止读；靠"连接关闭"分不清正常结束和断网
      3. 以 `\\n\\n` 结尾 —— SSE 帧的边界就是空行，少了最后一个空行，
         有些客户端会把最后一帧判成残帧丢弃
    """
    r = client.post("/chat/stream", json={"message": "hi"})

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-accel-buffering"] == "no", "不加 nginx 会攒着不发"

    body = r.text
    assert body.startswith("event: meta"), body[:80]
    assert body.endswith("\n\n"), "SSE 必须以空行结尾"

    # 顺序：meta -> delta -> done
    i_meta = body.index("event: meta")
    i_delta = body.index("event: delta")
    i_done = body.index("event: done")
    assert i_meta < i_delta < i_done

    # FakeClient 给三段增量，就应该有三个 delta 帧
    assert body.count("event: delta") == 3

    # meta 的载荷里必须有 session_id（否则前端没法续聊）
    meta_data = next(
        line[len("data: ") :]
        for line in body.splitlines()
        if line.startswith("data: ") and "session_id" in line
    )
    assert json.loads(meta_data)["session_id"]


# ══════════════════════ ⑤ 流式：404 不能变成 200 ══════════════════════


def test_chat_stream_unknown_session_is_404_not_sse(client: TestClient) -> None:
    """未知 session_id 必须是 404 + JSON，而不是「200 + 流内 error」。

    ★ 这是 SSE 最容易写错的一处：`StreamingResponse` 一旦开始发送，
      响应头 200 就已经出去了，改不了。所以存在性校验必须放在 `await` 阶段
      （`open_stream` 返回之前），不能拖进生成器。

      判据看两样：状态码 404，且 content-type 是 JSON 而不是 event-stream。
    """
    r = client.post("/chat/stream", json={"message": "hi", "session_id": "typo123"})

    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["code"] == "NOT_FOUND"


# ══════════════════════ ⑥ 会话 CRUD + 级联删除 ══════════════════════


def test_sessions_list_messages_and_cascade_delete(client: TestClient) -> None:
    """列表 -> 消息 -> 删除 -> 再查 404，且消息被级联删干净。

    ★ 级联那条断言（`store.count(sid) == 0`）是重点：
      `ON DELETE CASCADE` 依赖每条连接都设 `PRAGMA foreign_keys=ON`，
      SQLite 默认是 OFF —— 不设的话消息会变成孤儿，而且**不报错**。
      接口层面看不出来，只能查库。
    """
    assert client.get("/sessions").json()["data"]["sessions"] == []

    # 造两轮对话（4 条消息）
    sid = client.post("/chat", json={"message": "第一轮"}).json()["data"]["session_id"]
    client.post("/chat", json={"message": "第二轮", "session_id": sid})

    rows = client.get("/sessions").json()["data"]["sessions"]
    assert [x["id"] for x in rows] == [sid]
    assert set(rows[0]) == {"id", "created_at", "updated_at"}

    r = client.get(f"/sessions/{sid}/messages")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["count"] == 4
    assert [m["role"] for m in data["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert set(data["messages"][0]) == {"role", "content", "created_at"}

    # 删除：统一返回 200 + body（不用 204，因为 envelope 需要 body）
    r = client.delete(f"/sessions/{sid}")
    assert r.status_code == 200
    assert r.json()["data"] == {"session_id": sid, "deleted": True}

    # 消息级联删干净（接口看不出来，必须查库）
    assert client.app.state.store.count(sid) == 0, "外键级联没生效会留下孤儿消息"

    # 删完再查：两个端点都应 404
    assert client.get(f"/sessions/{sid}/messages").status_code == 404
    assert client.delete(f"/sessions/{sid}").status_code == 404


# ══════════════════════ ⑦ 流式：出错时必须是「流内 error + done」 ══════════════════════


class FailsMidStream(FakeClient):
    """吐一段之后失败，用来触发 router 的 `except AppError` 分支。"""

    async def astream(self, messages, **kw):  # type: ignore[no-untyped-def]
        yield "第一段"
        raise LLMError("模拟上游断开")


def test_chat_stream_emits_error_before_done(tmp_path) -> None:
    """上游中途失败：200 已发出，只能靠 `event: error` 告知，且必须在 done 之前。

    ★ 这条测的是「无法回头」的场景：
      响应头 200 在第一个字节发出时就定了，上游这时才失败，改不了状态码。
      所以契约变成两条：
        1. 必须有 `event: error`，带业务码（客户端据此提示"服务暂时不可用"）
        2. `done` 必须在 error 之后 —— 客户端靠 done 停止读；
           少了它，前端会一直挂着等更多数据，直到超时
    """
    app = create_app(
        db_path=str(tmp_path / "sessions.db"),
        llm=FailsMidStream(),
    )

    with TestClient(app) as c:
        r = c.post("/chat/stream", json={"message": "hi"})

    assert r.status_code == 200, "流已开始，状态码改不了"
    body = r.text
    assert "event: delta" in body, "失败前已产出的部分不能丢"
    assert "event: error" in body
    assert body.index("event: error") < body.index("event: done")
    assert "UPSTREAM_ERROR" in body


# ══════════════════════ ⑧ 限流：429 的语义 ══════════════════════


@pytest.fixture
def limited_client(tmp_path) -> Iterator[TestClient]:
    """额度只有 2 次的 app：用最小配置跑真实的「第 3 次被拒」。"""
    settings = Settings(_env_file=None, rate_limit=2, rate_window_s=60)
    app = create_app(
        settings=settings,
        db_path=str(tmp_path / "sessions.db"),
        llm=FakeClient(reply="假回复"),
    )
    with TestClient(app) as c:
        yield c


def test_chat_returns_429_with_retry_after(limited_client: TestClient) -> None:
    """超限 -> 429 + `RATE_LIMITED` + body/header 双份等待秒数。

    ★ 四个断言各管一件事：
      1. 状态码 429（不是 200 也不是 500）
      2. body 的 `data.retry_after_s` —— 前端要拿它显示倒计时
      3. `Retry-After` 头 —— 客户端库/网关读的是头，不解析 body
      4. `X-RateLimit-*` —— 让客户端知道额度是多少、还剩多少
    """
    assert limited_client.post("/chat", json={"message": "第一次"}).status_code == 200
    assert limited_client.post("/chat", json={"message": "第二次"}).status_code == 200

    r = limited_client.post("/chat", json={"message": "第三次"})

    assert r.status_code == 429
    body = r.json()
    assert body["code"] == "RATE_LIMITED"
    assert body["data"]["retry_after_s"] == 60
    assert r.headers["retry-after"] == "60"
    assert r.headers["x-ratelimit-limit"] == "2"
    assert r.headers["x-ratelimit-remaining"] == "0"

    # 流式与非流式共用一个桶：额度用完之后 /chat/stream 也应该被拒
    # （否则「便宜接口限流、贵接口不限」就成了一个绕过的口子）
    assert limited_client.post("/chat/stream", json={"message": "hi"}).status_code == 429


def test_probes_and_sessions_are_not_rate_limited(limited_client: TestClient) -> None:
    """探针与会话接口不参与限流。

    ★ 给探针限流是「自杀式配置」：它每 30 秒来一次，一旦被 429，
      容器编排会认为进程不健康 → 重启 → 正在处理的请求全丢。
      探针的消费者只看状态码，它不会为「限流」这个概念做退避。
    """
    for _ in range(10):
        assert limited_client.get("/healthz").status_code == 200
        assert limited_client.get("/readyz").status_code == 200
        assert limited_client.get("/sessions").status_code == 200


# ══════════════════════ ⑨ 未处理异常：500 也要可追溯 ══════════════════════


def test_unhandled_exception_returns_500_with_trace_id(tmp_path, caplog) -> None:
    """代码 bug -> 500 + 统一错误体 + trace id，堆栈只进日志。

    ★ 这条覆盖的是最外层：未处理异常由 `ServerErrorMiddleware` 接管，
      它在我们自己的中间件**之外**，所以 `response.headers[...] = rid`
      那段根本不会执行。500 的 X-Request-ID 必须由异常处理器自己补，
      否则「线上 500 了」这件事在日志里找不到对应请求。
    """
    import logging

    from app.core.logging import RequestIdFilter

    app = create_app(db_path=str(tmp_path / "sessions.db"), llm=FakeClient())

    @app.get("/boom")
    async def boom() -> None:  # type: ignore[return-value]
        raise RuntimeError("模拟未处理异常")

    caplog.handler.addFilter(RequestIdFilter())

    with caplog.at_level(logging.ERROR, logger="app.core.errors"):
        # 默认 TestClient 会重新抛出服务端异常；关掉才能拿到 500 响应
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/boom")

    assert r.status_code == 500
    body = r.json()
    assert body["code"] == "INTERNAL_ERROR"
    assert body["msg"] == "服务器内部错误", "堆栈不能暴露给客户端"
    assert body["request_id"]
    assert r.headers["x-request-id"] == body["request_id"]

    errors = [x for x in caplog.records if x.levelno == logging.ERROR]
    assert errors, "500 必须留下日志（否则线上无从查起）"


# ══════════════════════ ⑩ 演示页：同源 ══════════════════════


def test_demo_page_is_served_by_same_origin(client: TestClient) -> None:
    """演示页必须由**同一个 app** 提供 —— 跨端口 fetch 会被浏览器 CORS 拦掉。

    ★ 三个断言各管一件事：
      1. `/demo/` 返回 HTML（不是 404，也不是 JSON）
      2. 页面用**相对路径** `/chat/stream`（写成绝对 URL 就跨源了）
      3. 用 `getReader()` + `TextDecoder` 逐帧读（用 `resp.json()` 会退化成假流式）
    """
    r = client.get("/demo/")

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")

    html = r.text
    assert 'fetch("/chat/stream"' in html, "必须相对路径，否则跨源"
    assert "getReader" in html and "TextDecoder" in html, "要逐帧读，不能等整段 JSON"
    assert 'id="out"' in html, "正文容器不能少"
