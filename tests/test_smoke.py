"""环境自检：依赖装好了吗、FastAPI 能起吗、流式响应可用吗。

这不是业务测试，是「环境坏了」的第一道诊断。如果 CI 或本地突然挂了，
先跑这个文件，能快速区分「环境问题」和「业务 bug」。

★ 为什么值得留着一个"不测业务"的文件：
  每次新建项目最容易出问题的是环境（Python 版本、包没装、asyncio_mode 没配），
  而这些错误的表现往往是「业务测试报了一堆看不懂的 ImportError」。
  有一个只验证环境的最小集，定位成本会低很多。
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

import app


def test_package_importable() -> None:
    """src 布局 + module-name="app" 让 `import app` 指向 src/app。"""
    assert app.__name__ == "app"
    assert hasattr(app, "__path__"), "应该是个包"


async def test_asyncio_mode_auto() -> None:
    """没有 @pytest.mark.asyncio 也能跑 —— 验证 asyncio_mode="auto" 生效。

    如果这条报「async def functions are not natively supported」，
    说明 pyproject 里的 asyncio_mode 没生效。
    """
    assert True


def test_healthz_with_custom_header() -> None:
    """探针要用 JSONResponse 才能带自定义头（返回 dict 拿不到）。"""
    a = FastAPI()

    @a.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"}, headers={"Cache-Control": "no-store"})

    r = TestClient(a).get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
    assert r.headers["cache-control"] == "no-store"


def test_streaming_response_sse() -> None:
    """StreamingResponse + text/event-stream 是 v0.1 的流式基础。"""
    a = FastAPI()

    @a.get("/s")
    async def stream() -> StreamingResponse:
        async def gen():
            yield b"event: meta\ndata: {}\nid: 0\n\n"
            yield b"event: delta\ndata: hi\nid: 1\n\n"
            yield b"event: done\ndata: [DONE]\nid: end\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    r = TestClient(a).get("/s")
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    body = r.text
    assert "event: delta" in body
    assert body.endswith("\n\n"), "SSE 帧必须以空行结尾"


def test_path_param_and_query() -> None:
    """路径参数 + 查询参数的取值方式。"""
    a = FastAPI()

    @a.get("/sessions/{session_id}/messages")
    async def list_messages(session_id: str, request: Request, limit: int = 50) -> dict:
        return {"session_id": session_id, "limit": limit, "path": request.url.path}

    c = TestClient(a)
    r = c.get("/sessions/abc123/messages?limit=20")
    assert r.status_code == 200
    assert r.json() == {
        "session_id": "abc123",
        "limit": 20,
        "path": "/sessions/abc123/messages",
    }

    # 类型不对 → 422（不是 404）
    assert c.get("/sessions/abc/messages?limit=abc").status_code == 422
