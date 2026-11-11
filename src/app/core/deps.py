"""装配层：谁提供什么，在哪创建。

★ 这里是全项目唯一「new 具体实现」的地方（除测试外）。
  业务层拿到的是 Protocol 类型，所以它永远不知道背后是 DeepSeek 还是 FakeClient。

★ 为什么用 `request.app.state` 而不是模块级全局：
  踩过的坑——模块级全局会让「两个 app 用两个库」变成不可能，
  测试之间互相污染。放进 app.state 后每个 app 实例各自独立。

★ 类型标注写 Protocol（`LLMClient`）而不是具体类（`OpenAICompatClient`）：
  这样测试里 `app.dependency_overrides[get_llm] = lambda: FakeClient()` 就能换掉，
  而且类型检查器也不允许业务层用到具体类的私有方法。
"""

from __future__ import annotations

from fastapi import Request

from app.client.llm import LLMClient
from app.core.config import Settings
from app.db.store import MessageStore
from app.service.chat_service import ChatService


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
