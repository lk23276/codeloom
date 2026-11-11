"""统一错误码、统一响应体、四段异常处理器。

★ 为什么 code 和 HTTP 状态要维护在同一张表里（踩坑得来的）：
  如果让每个异常类自己带一个 `status` 属性，`code` 和 `status` 就成了两份独立事实，
  改一个忘一个就会出现「code 说 NOT_FOUND、HTTP 却是 400」的自相矛盾——
  这类 bug 不会报错，只会让前端逻辑莫名其妙。

★ 业务码用 StrEnum：
  既能被 json.dumps 直接序列化（不需要 .value），拼错时又是 AttributeError 而不是
  "能跑但错"的字符串。
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


# ══════════════════════ 一、业务码 ══════════════════════


class BizCode(StrEnum):
    OK = "OK"
    VALIDATION_ERROR = "VALIDATION_ERROR"  # 格式不对（Pydantic 拦下的）
    INVALID_INPUT = "INVALID_INPUT"  # 格式对但业务不允许
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    RATE_LIMITED = "RATE_LIMITED"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"  # 下游（LLM）出错
    INTERNAL_ERROR = "INTERNAL_ERROR"  # 我们自己的 bug


# ★ code → HTTP 状态，唯一事实源
_HTTP_STATUS: dict[BizCode, int] = {
    BizCode.OK: 200,
    BizCode.VALIDATION_ERROR: 422,
    BizCode.INVALID_INPUT: 400,
    BizCode.UNAUTHORIZED: 401,
    BizCode.FORBIDDEN: 403,
    BizCode.NOT_FOUND: 404,
    BizCode.METHOD_NOT_ALLOWED: 405,
    BizCode.RATE_LIMITED: 429,
    BizCode.UPSTREAM_ERROR: 502,
    BizCode.INTERNAL_ERROR: 500,
}

# 反查表：框架抛的 HTTPException 带的是状态码，要翻译回业务码
_STATUS_TO_CODE: dict[int, BizCode] = {v: k for k, v in _HTTP_STATUS.items()}


# ══════════════════════ 二、统一响应体 ══════════════════════


def ok(data: Any = None, *, request_id: str = "") -> dict[str, Any]:
    """成功响应。四个键永远都在，前端不用判空。"""
    return {"code": BizCode.OK.value, "msg": "ok", "data": data, "request_id": request_id}


def fail(code: BizCode, msg: str, *, data: Any = None, request_id: str = "") -> dict[str, Any]:
    return {"code": code.value, "msg": msg, "data": data, "request_id": request_id}


def _rid(request: Request) -> str:
    """取中间件塞进 state 的 request_id（没有就空串，不抛异常）。"""
    return getattr(request.state, "rid", "")


# ══════════════════════ 三、异常家族 ══════════════════════


class AppError(Exception):
    """所有业务异常的基类。

    ★ 只定义 code，不定义 status —— status 从 _HTTP_STATUS 查，保证两者永远一致。

    ★ `headers` 是给「响应本身要带语义」的异常用的，目前只有 429：
      `Retry-After` / `X-RateLimit-*` 必须出现在响应头里，
      客户端库（httpx-tenacity 之类）会直接读它做退避，而不是去解析 body。
      写在这里而不是让 handler 特判 RateLimited：handler 不需要认识具体异常类型。
    """

    code: BizCode = BizCode.INTERNAL_ERROR
    default_msg: str = "服务器内部错误"

    def __init__(
        self,
        msg: str | None = None,
        *,
        data: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.msg = msg or self.default_msg
        self.data = data
        self.headers = headers or {}
        super().__init__(self.msg)

    @property
    def status(self) -> int:
        return _HTTP_STATUS.get(self.code, 500)


class ValidationFailed(AppError):
    code = BizCode.VALIDATION_ERROR
    default_msg = "参数校验失败"


class InvalidInput(AppError):
    code = BizCode.INVALID_INPUT
    default_msg = "请求参数不合法"


class NotFound(AppError):
    code = BizCode.NOT_FOUND
    default_msg = "资源不存在"


class RateLimited(AppError):
    code = BizCode.RATE_LIMITED
    default_msg = "请求过于频繁，请稍后重试"


class UpstreamError(AppError):
    code = BizCode.UPSTREAM_ERROR
    default_msg = "模型服务暂时不可用"


# ══════════════════════ 四、异常处理器 ══════════════════════


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """业务异常 → 对应的 code/status。"""
    return JSONResponse(
        fail(exc.code, exc.msg, data=exc.data, request_id=_rid(request)),
        status_code=exc.status,
        # 异常自带的头优先（如 429 的 Retry-After），再兜一层 no-store
        headers={"Cache-Control": "no-store", **exc.headers},
    )


async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Pydantic 校验失败 → 422。

    ★ 必须单独注册：RequestValidationError 是 Exception 的子类，
      只注册 Exception 处理器的话会被它抢走，错误体结构就不对了。
    """
    fields = [
        {
            "loc": ".".join(str(x) for x in err.get("loc", ())),
            "msg": err.get("msg", ""),
            "type": err.get("type", ""),
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        fail(
            BizCode.VALIDATION_ERROR,
            "参数校验失败",
            data={"fields": fields},
            request_id=_rid(request),
        ),
        status_code=422,
        headers={"Cache-Control": "no-store"},
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """框架抛的 HTTPException（404/405 等）→ 翻译成业务码。

    ★ 必须注册 Starlette 的父类，不能注册 fastapi.HTTPException 子类（踩过的坑）：
      框架内部抛的是 Starlette 版本，注册子类命中不了，错误体就绕过统一格式了。
    """
    code = _STATUS_TO_CODE.get(exc.status_code, BizCode.INTERNAL_ERROR)
    detail = exc.detail if isinstance(exc.detail, str) else "请求处理失败"
    return JSONResponse(
        fail(code, detail, request_id=_rid(request)),
        status_code=exc.status_code,
        headers={"Cache-Control": "no-store"},
    )


async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底：未预料的异常。

    ★ 对外只给一句话 + request_id，细节写日志。
      把堆栈暴露给客户端既是安全问题，对排查也没帮助（有 rid 就能查日志）。
    """
    rid = _rid(request)
    logger.exception("未处理异常 rid=%s path=%s", rid, request.url.path)
    return JSONResponse(
        fail(BizCode.INTERNAL_ERROR, "服务器内部错误", request_id=rid),
        status_code=500,
        # ★ 500 也要带这个头：正常路径上中间件会补，但 500 是
        #   ServerErrorMiddleware（最外层）生成的，它在我们中间件之外，
        #   中间件那段代码根本不会执行 —— 不在这里补，500 响应就没有 trace id。
        headers={"Cache-Control": "no-store", "X-Request-ID": rid},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """四个处理器，注册顺序不重要，但『类型最具体的先匹配』由 FastAPI 保证。"""
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_handler)
