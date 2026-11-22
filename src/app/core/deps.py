"""装配层：谁提供什么，在哪创建。

★ 这里是全项目唯一「new 具体实现」的地方（除测试外）。
  业务层拿到的是 Protocol 类型，所以它永远不知道背后是 DeepSeek 还是 FakeClient。

★ 为什么用 `request.app.state` 而不是模块级全局：
  模块级全局会让「两个 app 用两个库」无法实现，测试之间互相污染；
  放进 app.state 后每个 app 实例各自独立。

★ 类型标注写 Protocol（`LLMClient`）而不是具体类（`OpenAICompatClient`）：
  这样测试里 `app.dependency_overrides[get_llm] = lambda: FakeClient()` 就能换掉，
  而且类型检查器也不允许业务层用到具体类的私有方法。
"""

from __future__ import annotations

from fastapi import Request

from app.client.llm import LLMClient
from app.core.config import Settings
from app.db.store import MessageStore
from app.service.agent_service import AgentService
from app.service.chat_service import ChatService
from app.service.rag_service import RagService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_llm(request: Request) -> LLMClient:
    """取 LLM 客户端（lifespan 里创建的）。

    ★ 注意它不检查 key 是否配置——那是 build_client() 的职责（启动时警告一次）。
      请求时才检查的话，每个请求都要多一次判断，而且日志会刷屏。
    """
    return request.app.state.llm


def get_store(request: Request) -> MessageStore:
    return request.app.state.store


def get_chat_service(request: Request) -> ChatService:
    """业务服务。每次请求新建一个实例是**刻意**的：

    ChatService 本身无状态（llm 和 store 都是共享的），
    轻量对象每次新建比维护单例更简单，也不用担心跨请求的状态污染。
    """
    return ChatService(llm=get_llm(request), store=get_store(request))


def get_rag_service(request: Request) -> RagService:
    """知识库问答服务。

    与 `ChatService` 的区别：它多一个重排器，而重排器**进程内复用**
    （模型约 1 GB，每次请求重建等于反复加载）。重排器本身是延迟加载的，
    所以这里只是取一个共享引用，不会触发模型下载。

    三个阈值/条数从 `Settings` 读，便于用环境变量调整而不改代码 ——
    拒答阈值本来就需要在标注集上反复标定。
    """
    cfg = get_settings(request)
    return RagService(
        llm=get_llm(request),
        reranker=request.app.state.reranker,
        recall_k=cfg.rag_recall_k,
        top_n=cfg.rag_top_k,
        min_score=cfg.rag_min_rerank_score,
        # 本地 JSONL tracer：不依赖账号，每个请求三个 span（retrieve / rerank / llm）
        tracer=request.app.state.tracer,
    )


def get_agent_service(request: Request) -> AgentService:
    """Agent 服务。**单例**，在 lifespan 里创建。

    ★ 与 `ChatService` / `RagService` 的最大区别：它有状态 —— 按 thread_id 分配的串行锁。
      每次请求新建实例，锁就各自一份，等于没有串行化（同一 thread 的并发请求照样互相覆盖）。
      这是「无状态服务可以随用随建，有状态服务必须复用」的一个具体例子。
    """
    return request.app.state.agent_service
