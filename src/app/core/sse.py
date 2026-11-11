"""SSE（Server-Sent Events）帧构造。

★ 帧格式的硬性要求（少一个就会"看不到流"）：
    event: delta\\n
    data: 正文\\n
    id: 3\\n
    \\n              ← 必须有一个空行结尾，否则客户端不认为这帧结束

★ 三条最容易踩的坑（实测）：
  1. 忘了结尾的空行 → 客户端一直等，看着像"卡住"
  2. 响应的 Content-Type 不是 `text/event-stream` → 浏览器不按 SSE 解析
  3. 中间层（nginx）缓冲 → 攒够一批才发，看着像"最后一次性蹦出来"。
     靠 `X-Accel-Buffering: no` 关掉 nginx 的缓冲。

★ 心跳为什么需要：
  nginx / 负载均衡 / 云网关通常在 30~60 秒无数据时掐连接。
  长思考的请求中间可能十几秒没有 token，不发 `: ping` 就会"莫名断流"。
"""

from __future__ import annotations

import json
from typing import Any

SSE_MEDIA_TYPE = "text/event-stream"

SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache",
    # 关掉 nginx 的响应缓冲，否则流会被攒起来一次性发
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def sse_frame(data: str, *, event: str | None = None, id: str | None = None) -> str:
    """构造一帧。

    data 里的换行要拆成多个 `data:` 行（SSE 规范），不能直接塞进一行。
    """
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    for line in data.split("\n"):
        lines.append(f"data: {line}")
    if id:
        lines.append(f"id: {id}")
    return "\n".join(lines) + "\n\n"


def event_frame(event: str, data: str, *, id: str | None = None) -> str:
    return sse_frame(data, event=event, id=id)


def json_frame(event: str, payload: Any, *, id: str | None = None) -> str:
    """把 dict 序列化成 JSON 再发（ensure_ascii=False，中文不转义）。"""
    return sse_frame(json.dumps(payload, ensure_ascii=False), event=event, id=id)


def comment_frame(text: str = "ping") -> str:
    """注释帧：以 `:` 开头，客户端会忽略，但能保活连接。"""
    return f": {text}\n\n"


def done_frame() -> str:
    return event_frame("done", "[DONE]", id="end")
