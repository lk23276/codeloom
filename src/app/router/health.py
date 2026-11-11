"""存活 / 就绪探针。

★ 两个探针的分工（实测过：DB 坏掉时 /healthz 仍 200、/readyz 变 503）：

    /healthz  存活（liveness）   进程还在跑吗？        失败 → 编排系统【重启】容器
    /readyz   就绪（readiness）  现在能接流量吗？      失败 → 编排系统【摘流量】但不重启

★ 为什么不套统一响应体：
  消费者是 k8s / docker 的探测器，它**只看 HTTP 状态码**，不看 body。
  套了 envelope 只是多一个可能出错的环节。

★ 为什么不用 `SELECT 1` 探数据库：
  它不碰任何表，库文件被删了照样"成功"。必须真查一张业务表。
"""

from __future__ import annotations

import os
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.async_utils import run_sync

router = APIRouter(tags=["health"])

_STARTED_AT = time.perf_counter()
_NO_STORE = {"Cache-Control": "no-store"}


def _uptime_s() -> float:
    return round(time.perf_counter() - _STARTED_AT, 3)


@router.get("/healthz", summary="存活探针（不碰任何依赖）")
async def healthz(request: Request) -> JSONResponse:
    """只证明「事件循环还在转」。

    ★ 这里"什么都不做"是刻意的：一旦它开始查 DB / 调模型，
      下游一抖就会被判成"进程死了"，编排系统重启容器——
      而重启治不了数据库故障，只是把好好的进程杀一遍，还丢掉在途请求。
    """
    return JSONResponse(
        {
            "status": "ok",
            "version": getattr(request.app.state, "version", "0.0.0"),
            "uptime_s": _uptime_s(),
        },
        headers=_NO_STORE,
    )


@router.get("/readyz", summary="就绪探针（查真实依赖）")
async def readyz(request: Request) -> JSONResponse:
    """查数据库能不能读、库目录能不能写。失败返 503（让上游摘流量）。"""
    store = getattr(request.app.state, "store", None)
    if store is None:
        return JSONResponse(
            {"status": "not_ready", "version": "", "db": "store 未初始化"},
            status_code=503,
            headers=_NO_STORE,
        )

    try:
        await run_sync(store.ping)
    except Exception as exc:  # noqa: BLE001 —— 探针只报告，不处理
        return JSONResponse(
            {"status": "not_ready", "db": f"读失败：{type(exc).__name__}"},
            status_code=503,
            headers=_NO_STORE,
        )

    # 目录可写：挂载卷权限不对时，读可能成功、写会失败
    db_path = str(getattr(store, "db_path", "") or "")
    if db_path:
        folder = os.path.dirname(os.path.abspath(db_path))
        if folder and not os.access(folder, os.W_OK):
            return JSONResponse(
                {"status": "not_ready", "db": "库目录不可写"},
                status_code=503,
                headers=_NO_STORE,
            )

    return JSONResponse(
        {"status": "ready", "version": getattr(request.app.state, "version", "0.0.0"), "db": "ok"},
        headers=_NO_STORE,
    )
