"""知识库问答的契约测试。

四条必须锁住的行为，都是「不报错但结果错」的类型：

1. **阈值拒答**：最高分低于阈值时不产生模型调用（费用与延迟都省了）
2. **重排顺序**：候选顺序被打乱后，引用按分数降序映射，而不是按召回顺序
3. **引用越界**：模型引用不存在的编号时只计数，不让整次请求失败
4. **候选为空**：知识库未建索引时给出可区分的原因，而不是「无相关依据」

全部使用替身：假检索函数 + 假重排器 + FakeClient。
不加载约 1 GB 的重排模型，也不需要向量库 —— 一次全跑不到 1 秒。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.client.llm import ChatResult, FakeClient, LLMError, Usage
from app.core.deps import get_rag_service
from app.core.errors import UpstreamError
from app.main import create_app
from app.rag.prompt import build_rag_messages, parse_citation_indices
from app.rag.rerank import FastEmbedReranker
from app.schema.rag import RagRequest
from app.service.rag_service import EMPTY_KB_REASON, NO_EVIDENCE_REASON, RagService

# ══════════════════════ 替身与夹具 ══════════════════════

EVENTFD = "eventfd 用于唤醒子反应堆"
HALFWRITE = "写半包续发的实现"
SIGTERM = "SIGTERM 之后优雅退出"


def make_doc(source: str, start: int, end: int, text: str) -> Document:
    """造一个与真实语料同构的块（metadata 字段与 vectorstore 写入的一致）。"""
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "kind": "code",
            "line_start": start,
            "line_end": end,
            "start_index": 0,
        },
    )


def candidates() -> list[tuple[Document, float]]:
    """三个候选，故意按「与重排结果相反」的顺序给出。

    距离字段（第二个值）在重排后不再使用，这里给常量即可 ——
    若实现误用了向量距离排序，这些断言会失败。
    """
    return [
        (make_doc("main.cpp", 88, 103, SIGTERM), 0.4),
        (make_doc("server2.cpp", 541, 552, HALFWRITE), 0.5),
        (make_doc("server2.cpp", 345, 362, EVENTFD), 0.6),
    ]


class FakeReranker:
    """按块正文查表打分，模拟 cross-encoder 的「越大越相关」。

    记录调用参数，便于断言「重排确实拿到了全部候选」。
    """

    def __init__(self, table: dict[str, float]) -> None:
        self.table = table
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        return [float(self.table.get(text, 0.0)) for text in documents]


class BrokenReranker:
    """少返回一个分数。用来验证长度校验 —— 错位不报错，只会答错文件。"""

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [1.0] * (len(documents) - 1)


def make_service(
    reply: str = "见 [1]",
    *,
    table: dict[str, float] | None = None,
    min_score: float = 0.2,
    recall: list[tuple[Document, float]] | None = None,
) -> tuple[RagService, FakeClient, FakeReranker]:
    llm = FakeClient(reply=reply)
    scores = {HALFWRITE: 3.2, SIGTERM: 1.5, EVENTFD: 0.9}
    reranker = FakeReranker(table if table is not None else scores)
    pool = candidates() if recall is None else recall
    svc = RagService(
        llm,
        reranker,
        recall_k=20,
        top_n=5,
        min_score=min_score,
        retrieve_fn=lambda _q, _k: pool,
    )
    return svc, llm, reranker


# ══════════════════════ 一、编排：拒答与作答 ══════════════════════


async def test_refuses_without_calling_model() -> None:
    """最高分低于阈值：直接拒答，模型一次都不该被调用。"""
    svc, llm, _ = make_service(min_score=10.0)

    answer = await svc.ask(RagRequest(question="A 股下周走势"))

    assert answer.refused is True
    assert answer.reason == NO_EVIDENCE_REASON
    assert answer.answer == ""
    assert answer.citations == []
    assert answer.recalled == 3
    assert answer.top_score == pytest.approx(3.2)
    assert llm.calls == [], "拒答路径不应该产生模型调用"


async def test_refuses_on_empty_corpus() -> None:
    """索引未建时给出可区分的原因，不能和「无相关依据」混成一个。"""
    svc, llm, reranker = make_service(recall=[])

    answer = await svc.ask(RagRequest(question="随便问问"))

    assert answer.refused is True
    assert answer.reason == EMPTY_KB_REASON
    assert answer.recalled == 0
    assert llm.calls == []
    assert reranker.calls == [], "空候选不该进入重排"


async def test_answers_with_citations_sorted_by_score() -> None:
    """引用按重排分数映射，而不是按召回顺序。"""
    svc, llm, reranker = make_service(reply="续发见 [1]，退出见 [2]")

    answer = await svc.ask(RagRequest(question="写半包续发在哪"))

    assert answer.refused is False
    assert answer.reranked == 3
    assert [c.line_start for c in answer.citations] == [541, 88]
    assert [c.score for c in answer.citations] == pytest.approx([3.2, 1.5])
    assert answer.citations[0].source == "server2.cpp"
    # 重排必须拿到全部候选（20 条召回里取 3 条，这里候选池就是 3 条）
    assert len(reranker.calls[0][1]) == 3
    assert len(llm.calls) == 1


async def test_top_n_limits_context() -> None:
    """`top_n` 截断在重排之后，`reranked` 反映真实进入上下文的条数。"""
    svc, _, _ = make_service(reply="见 [1]")

    answer = await svc.ask(RagRequest(question="q", top_n=2))

    assert answer.reranked == 2
    assert answer.recalled == 3


async def test_metric_fields_carry_usage() -> None:
    """usage 与耗时都要回填，否则成本与性能无法观测。"""
    llm = FakeClient(script=[ChatResult(text="见 [1]", usage=Usage(100, 20, 120))])
    svc = RagService(
        llm,
        FakeReranker({HALFWRITE: 3.2, SIGTERM: 1.5, EVENTFD: 0.9}),
        min_score=0.2,
        retrieve_fn=lambda _q, _k: candidates(),
    )

    answer = await svc.ask(RagRequest(question="q"))

    assert answer.usage.prompt_tokens == 100
    assert answer.usage.completion_tokens == 20
    assert answer.usage.total_tokens == 120
    assert answer.latency_ms >= 0


# ══════════════════════ 二、引用映射的边界 ══════════════════════


async def test_out_of_range_citation_is_counted_not_raised() -> None:
    """越界编号只计数：它是模型行为异常的信号，不该让请求 500。"""
    svc, _, _ = make_service(reply="见 [1] 与 [9]")

    answer = await svc.ask(RagRequest(question="q"))

    assert answer.refused is False
    assert len(answer.citations) == 1
    assert answer.citations[0].index == 1
    assert answer.invalid_citations == 1


async def test_duplicate_citations_are_deduped() -> None:
    """同一编号引用两次只算一处。"""
    svc, _, _ = make_service(reply="[1] 说了；再看 [1] 一次")

    answer = await svc.ask(RagRequest(question="q"))

    assert len(answer.citations) == 1
    assert answer.invalid_citations == 0


async def test_mismatched_scores_raise_upstream_error() -> None:
    """分数个数与候选不一致时必须报错：错位会静默答错文件。"""
    svc = RagService(
        FakeClient(reply="见 [1]"),
        BrokenReranker(),
        min_score=0.2,
        retrieve_fn=lambda _q, _k: candidates(),
    )

    with pytest.raises(UpstreamError):
        await svc.ask(RagRequest(question="q"))


async def test_llm_error_is_translated_to_upstream_error() -> None:
    """每一层只认识自己那一层的语言：LLMError 到本层必须翻成 UpstreamError。

    否则 router 要 import LLM 相关的异常类型才能处理 502 —— 分层即被破坏。
    """

    class FailingClient(FakeClient):
        async def achat(self, messages, **kw):  # type: ignore[no-untyped-def]
            raise LLMError("模拟上游失败")

    svc = RagService(
        FailingClient(),
        FakeReranker({HALFWRITE: 3.2, SIGTERM: 1.5, EVENTFD: 0.9}),
        min_score=0.2,
        retrieve_fn=lambda _q, _k: candidates(),
    )

    with pytest.raises(UpstreamError, match="模型服务暂时不可用"):
        await svc.ask(RagRequest(question="q"))


# ══════════════════════ 三、提示词与编号解析 ══════════════════════


def test_context_carries_index_and_location() -> None:
    """每块抬头必须是 `[n] 文件:起-止`，位置来自 metadata 而不是模型。"""
    messages = build_rag_messages("事件唤醒怎么实现", candidates())

    assert messages[0]["role"] == "system"
    context = messages[1]["content"]
    assert "[1] main.cpp:88-103" in context
    assert "[2] server2.cpp:541-552" in context
    assert "[3] server2.cpp:345-362" in context
    assert "问题：事件唤醒怎么实现" in context


def test_parse_citation_indices_keeps_order_and_dedupes() -> None:
    assert parse_citation_indices("[2] 与 [1]，再提 [2]") == [2, 1]
    assert parse_citation_indices("没有引用") == []


# ══════════════════════ 三之二、重排适配层的契约 ══════════════════════


class _StubEncoder:
    """只实现 rerank 的假编码器，用来验证适配层的转换规则。"""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def rerank(self, query: str, documents):  # type: ignore[no-untyped-def]
        self.calls.append(list(documents))
        return iter([1, 2.5])


def test_reranker_returns_plain_floats_in_input_order() -> None:
    """fastembed 返回的是迭代器，适配层必须转成 list[float] 且保持输入顺序。"""
    stub = _StubEncoder()
    reranker = FastEmbedReranker("dummy")
    reranker._encoder = stub

    scores = reranker.score("怎么唤醒", ["块一", "块二"])

    assert scores == [1.0, 2.5]
    assert all(isinstance(s, float) for s in scores)
    assert stub.calls == [["块一", "块二"]]


def test_reranker_short_circuits_on_empty_documents() -> None:
    """空候选不调用模型：这批调用点很多，短路能省掉无意义的批处理开销。"""
    stub = _StubEncoder()
    reranker = FastEmbedReranker("dummy")
    reranker._encoder = stub

    assert reranker.score("q", []) == []
    assert stub.calls == []


# ══════════════════════ 四、HTTP 层契约 ══════════════════════


@pytest.fixture
def client(tmp_path) -> Iterator[TestClient]:
    """完整的 app：真实 SQLite（临时目录）+ 假 LLM + 假重排器。

    替身服务按问题决定候选：含「A 股」的提问返回空候选，
    用来覆盖「知识库没有依据 → 200 + refused」这条 HTTP 契约。
    """

    def fake_retrieve(question: str, _k: int) -> list[tuple[Document, float]]:
        return [] if "A 股" in question else candidates()

    app = create_app(
        db_path=str(tmp_path / "sessions.db"),
        llm=FakeClient(reply="见 [1]"),
        reranker=FakeReranker({HALFWRITE: 3.2, SIGTERM: 1.5, EVENTFD: 0.9}),
    )
    service = RagService(
        FakeClient(reply="续发见 [1]"),
        FakeReranker({HALFWRITE: 3.2, SIGTERM: 1.5, EVENTFD: 0.9}),
        min_score=0.2,
        retrieve_fn=fake_retrieve,
    )
    app.dependency_overrides[get_rag_service] = lambda: service
    with TestClient(app) as c:
        yield c


def test_rag_ask_returns_envelope_with_citations(client: TestClient) -> None:
    resp = client.post("/rag/ask", json={"question": "写半包续发在哪"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == "OK"
    assert body["request_id"]
    data = body["data"]
    assert data["refused"] is False
    assert data["citations"][0]["source"] == "server2.cpp"
    assert data["citations"][0]["line_start"] == 541


def test_rag_refusal_is_200_not_4xx(client: TestClient) -> None:
    """拒答是正常结果：4xx 会让客户端当故障重试，而重试结果不会变。"""
    resp = client.post("/rag/ask", json={"question": "A 股下周走势"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["refused"] is True
    assert body["data"]["citations"] == []
