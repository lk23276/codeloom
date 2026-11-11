"""结构化日志：一行一个 JSON，`request_id` 贯穿整条链路。

★ 为什么不用默认的 `logging.basicConfig` 文本格式：
  文本日志是给人看的，一旦上了容器/采集（Loki / ELK / CloudWatch）就得靠正则去抓
  「哪个请求、多慢、多少 token」。结构化日志把这三件事变成**字段**，
  采集端直接按 `duration_ms > 1000` 或 `request_id == "abc"` 查，不需要解析字符串。

★ `request_id` 为什么用 ContextVar 而不是「层层传参」：
  它要出现在**每一行**日志里（包括 service、registry 这些深层模块），
  逐层加参数会污染所有函数签名。
  ContextVar 是「协程本地的全局变量」：中间件 `set()` 一次，同一请求内任何地方
  `get()` 都拿得到。而且 `asyncio.to_thread` 会**复制**当前 context
  （`contextvars.copy_context()`），所以 `run_sync()` 里跑的 SQLite 操作也带着同一个 id。

★ 两条纪律：
  1. 日志里**永不出现密钥**——只允许 `settings.key_preview`（前 6 位 + ***）
  2. 字段名保持稳定：`request_id` / `duration_ms` / `status` 改了会让看板静默失效
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

# 中间件 set，formatter / 各业务模块 get。默认 "-" 表示「不在请求上下文里」（如启动日志）
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# 允许进 JSON 的业务字段白名单。
# ★ 用白名单而不是「把 record.__dict__ 全倒出来」：
#   LogRecord 里有 args/exc_info/stack_info 等几十个内部字段，
#   全量输出既冗长又会把 `%(message)s` 的原始参数重复一遍。
_EXTRA_KEYS: tuple[str, ...] = (
    "method",
    "path",
    "status",
    "duration_ms",
    "model",
    "step",
    "prompt_tokens",
    "completion_tokens",
    "tool_calls",
    "tool",
    "success",
)


class RequestIdFilter(logging.Filter):
    """把当前 context 的 request_id 塞进 record，供两种 formatter 共用。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """一行一个 JSON 对象。异常堆栈放在 `exc` 字段里（不拆成多行）。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "msg": record.getMessage(),
        }
        for key in _EXTRA_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        # separators 去掉空格：日志体积能小 20% 左右，机器读不需要好看
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class TextFormatter(logging.Formatter):
    """本地开发用：带 trace id 的一行文本，比 JSON 好扫。"""

    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )


def _make_formatter(fmt: str) -> logging.Formatter:
    return JsonFormatter() if fmt == "json" else TextFormatter()


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """配置根日志器。

    ★ `if not root.handlers` 的用途：pytest 在测试开始前就挂了自己的 handler，
      这里再挂一个会让每条日志打印两遍。有这个判断，测试里保持 pytest 的配置，
      生产（无 handler）才装我们的。

    ★ uvicorn 的接管必须放在这里「再执行一次」：uvicorn 在 import app **之后**
      才 `configure_logging()`，给 `uvicorn.*` 装自己的 handler 并设 propagate=False。
      所以 create_app 时清一遍、lifespan 启动时再清一遍，才能保证启动日志也走 JSON。
      代价：`--log-config` 自定义配置会被覆盖 —— v0.1 不用它，写进 README 即可。
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.addFilter(RequestIdFilter())
        handler.setFormatter(_make_formatter(fmt))
        root.addHandler(handler)

    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # uvicorn 的启动/错误日志并进我们的 handler（否则它们还是文本格式，
    # 采集端一半 JSON 一半文本，等于没做结构化）。
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True

    # ★ 访问日志**直接关掉**，由 `main.request_context` 中间件输出。
    #   两个原因：
    #     1. 我们的那条带 request_id 和 duration_ms（JSON），uvicorn 的没有
    #     2. 实测：只传 `--no-access-log` 不够 —— 它只是不装默认 handler，
    #        记录仍然会发出来；一旦我们把 propagate 打开，它就冒出来了，
    #        结果是每条请求两行日志。`disabled=True` 才是真的不产生记录。
    logging.getLogger("uvicorn.access").disabled = True


def get_request_id() -> str:
    """给需要手动拼日志/响应的地方用（多数场景直接打日志即可，formatter 会自动带）。"""
    return request_id_var.get()
