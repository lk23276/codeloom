"""应用装配：create_app() + lifespan + 中间件。

★ 装配顺序：
    中间件 → 异常处理器 → 路由
  异常处理器要在路由之前注册，否则路由里抛的异常不会被接住。
  （实际 FastAPI 允许任意顺序，但先注册更符合"分层包裹"的直觉：
    请求进来先过中间件，再进路由，出异常往上冒到处理器。）

★ 为什么把 store / llm 的创建放在 lifespan 而不是模块级：
  ① 模块级会在 **import 时**就建连接——测试只要 import 一次就落地一个真实数据库文件
  ② 进程退出时要能干净关闭（`store.close()`），模块级全局没有合适的时机
  ③ "两个 app 用两个库"在模块级下不可能，测试之间会互相污染（踩过）
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from app import VERSION
from app.client.llm import LLMClient, build_client
from app.core.config import Settings
from app.core.errors import register_exception_handlers
from app.core.logging import request_id_var, setup_logging
from app.core.ratelimit import KeyedLimiter, build_limiter
from app.db.store import MessageStore
from app.router import chat as chat_router
from app.router import health as health_router
from app.router import sessions as sessions_router

logger = logging.getLogger(__name__)

# 演示页目录。★ 放在 src/app/static 而不是 docs/：
#   docs 被 .dockerignore 排除，放这里镜像里也有一份，容器内同样能打开 /demo/。
_STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时建依赖，关闭时清理。"""
    settings: Settings = app.state.settings

    # ★ 再配一次日志：uvicorn 是在 import app **之后**才给自己装 handler 的，
    #   create_app() 那一次清不掉它（详见 core/logging.py 的说明）。
    setup_logging(settings.log_level, settings.log_format)

    store = MessageStore(settings.db_path)
    # 测试可能注入了 FakeClient —— 有就用，没有才建真实客户端
    llm = getattr(app.state, "llm", None) or build_client(settings)
    limiter: KeyedLimiter = build_limiter(settings)

    app.state.store = store
    app.state.llm = llm
    app.state.ratelimit = limiter
    logger.info(
        "启动完成 db=%s model=%s key=%s rate=%d/%ds",
        settings.db_path,
        settings.llm_model,
        settings.key_preview,
        settings.rate_limit,
        settings.rate_window_s,
    )

    try:
        yield
    finally:
        store.close()
        logger.info("已关闭数据库连接")


async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
    """给每个请求分配 request_id，并输出一条结构化 access 日志。

    ★ request_id 的价值：客户端报错时只要给这一个串，就能在日志里定位整条链路。
      没有它，只能靠时间戳猜 —— 并发下根本对不上。

    ★ 放在中间件而不是路由里：这样 404、422、429 也能带上 rid。
      只在路由里生成的话，没进路由的请求就没有 id 了。

    ★ 三个 `finally` 里的细节：
      - `token` 必须 reset，否则 context 会「粘」在复用它的任务上
      - rid 同时写 `request.state`（给异常处理器/路由用）和 ContextVar（给日志用）：
        两个消费方读的是不同的地方，少写一个就会出现「body 里有 id、日志里没有」
      - 流式响应的 `duration_ms` 只到**第一个字节**（`call_next` 在响应开始时返回），
        不是整个流的时长 —— 真正的流时长要在 SSE 的 done 帧里算
    """
    rid = uuid.uuid4().hex[:12]
    request.state.rid = rid
    token = request_id_var.set(rid)
    t0 = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.warning(
            "request_error",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": 500,
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
            },
        )
        raise
    else:
        response.headers["X-Request-ID"] = rid
        logger.info(
            "access",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
            },
        )
        return response
    finally:
        request_id_var.reset(token)


def create_app(
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    db_path: str | None = None,
) -> FastAPI:
    """装配一个 app。

    三个可选参数都是给测试用的：
      - `settings`：换一套配置（比如测试里的 model 名）
      - `llm`：注入 FakeClient，不联网、不花钱
      - `db_path`：指向 tmp_path，避免污染真实数据库
    """
    cfg = settings or Settings()
    if db_path is not None:
        cfg = cfg.model_copy(update={"db_path": db_path})

    setup_logging(cfg.log_level, cfg.log_format)

    app = FastAPI(
        title="CodeLoom API",
        version=VERSION,
        description="可溯源的多 Agent 工程知识助手（v0.1：流式问答 + 会话记忆）",
        lifespan=lifespan,
    )
    app.state.settings = cfg
    app.state.version = VERSION
    # 测试注入的 llm 先挂上；lifespan 里只在没有注入时才创建
    if llm is not None:
        app.state.llm = llm

    app.middleware("http")(request_context)
    register_exception_handlers(app)

    app.include_router(health_router.router)
    app.include_router(sessions_router.router)
    app.include_router(chat_router.router)

    # ★ 同源演示页：页面和接口在同一个 origin，浏览器 fetch('/chat/stream') 不触发 CORS。
    #   这正是「方案 B」能零前端构建的原因 —— 不需要 Vite/React，也不用开跨域。
    #   html=True：访问 /demo/ 直接返回 index.html。
    app.mount("/demo", StaticFiles(directory=_STATIC_DIR, html=True), name="demo")
    return app


app = create_app()
