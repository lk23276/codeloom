"""`/agent` 的 HTTP 契约测试。

只测这一层该管的事：

    - 响应体形状（envelope / data 字段）
    - 参数校验（message 与 approve 互斥、approve 必须带 thread_id）
    - 业务拒绝（approve 却没有待审批操作 -> 400 INVALID_INPUT）

★ 图的正确性与记忆行为在 `test_agent_service.py` 里测；这里用注入的图 + 假模型，
  不联网、不花钱，也不重复覆盖业务逻辑。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.agent.graph import build_agent_graph
from app.core.config import Settings
from app.main import create_app
from tests._fakes import ScriptedChatModel, tool_call


def _graphs(settings: Settings, checkpointer):
    model = ScriptedChatModel(
        responses=[
            tool_call("12*7"),
            AIMessage(content="12 × 7 = 84"),
            tool_call("3*5"),
            AIMessage(content="3 × 5 = 15"),
        ]
    )
    plain = build_agent_graph(settings, model=model, checkpointer=checkpointer)
    hitl = build_agent_graph(
        settings, model=model, checkpointer=checkpointer, interrupt_before=("tools",)
    )
    return plain, hitl


@pytest.fixture
def client(tmp_path) -> Iterator[TestClient]:
    from langgraph.checkpoint.memory import InMemorySaver

    settings = Settings(_env_file=None, agent_checkpoint="memory")
    plain, hitl = _graphs(settings, InMemorySaver())
    app = create_app(
        settings=settings,
        db_path=str(tmp_path / "sessions.db"),
        agent_graph=plain,
        agent_graph_hitl=hitl,
    )
    with TestClient(app) as c:
        yield c


# ══════════════════════ 一、正常路径 ══════════════════════


def test_agent_returns_answer_with_thread_id(client: TestClient) -> None:
    resp = client.post("/agent", json={"message": "12*7 等于几"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == "OK"
    data = body["data"]
    assert data["status"] == "answered"
    assert data["answer"] == "12 × 7 = 84"
    assert data["tool_calls"] == ["calculator"]
    assert len(data["thread_id"]) == 32
    assert body["request_id"]


def test_agent_reuses_thread(client: TestClient) -> None:
    first = client.post("/agent", json={"message": "12*7 等于几"}).json()["data"]
    second = client.post(
        "/agent", json={"message": "3*5 呢", "thread_id": first["thread_id"]}
    ).json()["data"]

    assert second["thread_id"] == first["thread_id"]
    assert second["status"] == "answered"


def test_agent_hitl_returns_awaiting_approval(client: TestClient) -> None:
    resp = client.post("/agent", json={"message": "12*7 等于几", "hitl": True})

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["status"] == "awaiting_approval"
    assert data["answer"] == ""
    assert data["pending"] == ['calculator({"expression": "12*7"})']


def test_agent_approve_finishes_paused_run(client: TestClient) -> None:
    paused = client.post("/agent", json={"message": "12*7 等于几", "hitl": True}).json()["data"]

    resp = client.post("/agent", json={"thread_id": paused["thread_id"], "approve": True})

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["status"] == "answered"
    assert data["answer"] == "12 × 7 = 84"


# ══════════════════════ 二、参数校验 ══════════════════════


def test_message_and_approve_are_mutually_exclusive(client: TestClient) -> None:
    resp = client.post("/agent", json={"message": "你好", "approve": True, "thread_id": "t"})

    assert resp.status_code == 422
    assert resp.json()["code"] == "VALIDATION_ERROR"


def test_approve_requires_thread_id(client: TestClient) -> None:
    resp = client.post("/agent", json={"approve": True})

    assert resp.status_code == 422


def test_empty_body_is_rejected(client: TestClient) -> None:
    resp = client.post("/agent", json={})

    assert resp.status_code == 422


# ══════════════════════ 三、历史 ══════════════════════


def test_agent_history_after_two_turns(client: TestClient) -> None:
    first = client.post("/agent", json={"message": "12*7 等于几"}).json()["data"]
    client.post("/agent", json={"message": "3*5 呢", "thread_id": first["thread_id"]})

    resp = client.get(f"/agent/{first['thread_id']}/messages")

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["status"] == "answered"
    assert data["count"] == 4
    assert [m["role"] for m in data["messages"]] == ["user", "assistant", "user", "assistant"]


def test_agent_history_unknown_thread_returns_404(client: TestClient) -> None:
    resp = client.get("/agent/never-started/messages")

    assert resp.status_code == 404
    assert resp.json()["code"] == "NOT_FOUND"


# ══════════════════════ 四、业务拒绝 ══════════════════════


def test_approve_without_pending_returns_400(client: TestClient) -> None:
    first = client.post("/agent", json={"message": "12*7 等于几"}).json()["data"]

    resp = client.post("/agent", json={"thread_id": first["thread_id"], "approve": True})

    assert resp.status_code == 400
    assert resp.json()["code"] == "INVALID_INPUT"
    assert resp.json()["request_id"]


def test_unknown_thread_returns_404(client: TestClient) -> None:
    """供了不存在的 thread_id：404，而不是静默新建 —— 与 /chat 的 session_id 口径一致。"""
    resp = client.post("/agent", json={"message": "1+1 等于几", "thread_id": "never-started"})

    assert resp.status_code == 404
    assert resp.json()["code"] == "NOT_FOUND"
