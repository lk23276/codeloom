"""Langfuse 接入：**未配置就返回 None**，让调用方不需要判空两次。

三条设计约束：

1. `CallbackHandler()` 挂到 `config={"callbacks": [h]}`，chain / tool / retriever
   会被自动切成嵌套 span，不用手写埋点；
2. 离线兜底优先：没有 `LANGFUSE_*` 时本模块返回 None，本地 JSONL 接管（见 `local_trace.py`）；
3. **不要让自己被一个 SaaS 账号卡住** —— 因此 `langfuse` 不做成硬依赖：
   装了就启用，没装就打一行 warning 返回 None，服务照常运行。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_langfuse_handler(settings: Any) -> Any | None:
    """构造 LangChain 回调处理器；未配置或未安装时返回 None。

    返回 None 的三种情况都要能跑通，不要抛异常：

        ① 没配 key（最常见：本地开发、别人 clone 之后）
        ② 装了 langfuse 但版本改了导入路径（v2 是 langfuse.callback，v3 是 langfuse.langchain）
        ③ 根本没装 langfuse
    """
    public_key = getattr(settings, "langfuse_public_key", "")
    secret_key = getattr(settings, "langfuse_secret_key", "")
    if not public_key or not secret_key:
        logger.debug("langfuse 未配置，使用本地 trace")
        return None

    try:
        # ★ 导入路径属于易腐信息：v2 用 langfuse.callback，v3 用 langfuse.langchain。
        #   这里按 v3 写，装到 v2 时下面的 ImportError 分支会给出可读提示。
        from langfuse.langchain import CallbackHandler
    except ModuleNotFoundError:
        logger.warning("已配置 LANGFUSE_* 但未安装 langfuse，继续使用本地 trace")
        return None
    except ImportError as exc:  # 版本不匹配的导入路径
        logger.warning("langfuse 导入失败（版本不符？）：%s；继续使用本地 trace", exc)
        return None

    host = getattr(settings, "langfuse_host", "") or None
    try:
        return CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            **({"host": host} if host else {}),
        )
    except Exception as exc:  # noqa: BLE001 —— 可观测初始化失败不该阻断服务启动
        logger.warning("langfuse 初始化失败：%s；继续使用本地 trace", exc)
        return None


def callbacks_of(handler: Any | None) -> dict[str, Any]:
    """把 handler 包成 `config` 片段；没有就返回空 dict。

    ★ 空 dict 而不是 `{"callbacks": [None]}`：后者会让框架在运行时报
      「NoneType 不可调用」，而错误点离配置处很远。
    """
    return {"callbacks": [handler]} if handler is not None else {}
