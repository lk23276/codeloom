"""MCP Server：把项目的**只读**能力暴露给外部 MCP 客户端（IDE 插件 / CLI 工具）。

三个工具 + 一个资源，全部只读：

    search_code(query, top_k)     语义检索，返回 文件 / 行号区间 / 片段
    read_file(path, max_chars)    读取白名单目录内的文本文件
    list_files()                  列出可读文件
    corpus://{name}               按文件名取原文（Resource，不是 Tool）

★ 为什么用 `MCPServer` 而不是 `FastMCP`：
  mcp 1.x 的 `mcp.server.fastmcp.FastMCP` 在 2.x 改名为 `mcp.server.mcpserver.MCPServer`；
  旧路径留了一个会抛 ModuleNotFoundError 的桩模块，照着 1.x 写法会直接失败。
  装饰器（`@server.tool()` / `@server.resource(uri)`）与 `run()` 的形状没变。

★ stdio 传输下 **stdout 是 JSON-RPC 帧通道**，任何 `print` 都会破坏协议，
  现象是「客户端偶发解析失败」，而且往往加了调试输出才出错。日志一律走 stderr。

★ 工具全部只读，所以不需要 HITL 审批（对比 `/agent` 的写类工具才需要）。

★ 检索是延迟导入的：`search_code` 第一次被调用时才加载 embedding 模型（约 90 MB）。
  放在模块顶部会让「启动即加载」，客户端每次握手都要等一次模型初始化。
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from mcp.server.mcpserver import MCPServer

from app import VERSION
from app.mcp_server.security import PathGuard

logger = logging.getLogger(__name__)

# 语料目录以本文件位置为锚点：MCP 客户端的工作目录不一定是项目根
CORPUS_ROOT = Path(__file__).resolve().parents[3] / "data" / "corpus" / "raw"
_guard = PathGuard(CORPUS_ROOT)

server = MCPServer(
    name="codeloom",
    version=VERSION,
    instructions=(
        "CodeLoom 的语料检索服务。search_code 用自然语言查服务器实现与设计文档，"
        "返回文件名与行号区间；read_file 按文件名取原文。所有工具只读。"
    ),
)


def _hit(document: Document, distance: float) -> dict[str, Any]:
    """把一条检索结果压成工具输出。

    ★ 字段名用 `distance` 而不是 `score`：Chroma 的 cosine 空间里它是**距离**，
      越小越近。叫 score 会让模型按「越大越好」解读，直接把排序读反。
    """
    meta = document.metadata or {}
    text = document.page_content or ""
    return {
        "source": meta.get("source", "?"),
        "kind": meta.get("kind", "?"),
        "line_start": meta.get("line_start"),
        "line_end": meta.get("line_end"),
        "distance": round(float(distance), 4),
        "snippet": text[:400],
    }


@server.tool()
def search_code(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """在 CodeLoom 语料中按语义检索，返回文件名、行号区间与片段（只读）。

    适合回答「某个机制在哪里实现」「某个设计为什么这样选」这类问题。
    `distance` 越小越相关。不知道具体文件名时先用本工具，再用 read_file 取全文。
    """
    if not query.strip():
        return []

    # 延迟导入：避免进程启动即加载 embedding 模型
    from app.rag.retriever import retrieve_with_score

    k = max(1, min(int(top_k), 10))
    started = time.perf_counter()
    rows = retrieve_with_score(query, k=k)
    logger.info(
        "tool=search_code k=%d hits=%d ms=%d",
        k,
        len(rows),
        int((time.perf_counter() - started) * 1000),
    )
    return [_hit(document, distance) for document, distance in rows]


@server.tool()
def read_file(path: str, max_chars: int = 2000) -> str:
    """读取语料文件原文（仅限白名单目录内的文本文件，只读）。

    `path` 是相对语料目录的文件名，例如 "demo_notes.md"。
    越界路径（含 `../` 或绝对路径）、不存在的文件、非文本后缀都会被拒绝。
    """
    target = _guard.resolve(path)
    text = target.read_text(encoding="utf-8", errors="ignore")
    limit = max(1, min(int(max_chars), 8000))
    logger.info("tool=read_file path=%s chars=%d", target.name, min(len(text), limit))
    return text[:limit]


@server.tool()
def list_files() -> list[dict[str, Any]]:
    """列出语料目录下可读的文件及其行数、字符数（只读）。"""
    rows = _guard.catalog()
    logger.info("tool=list_files files=%d", len(rows))
    return rows


@server.resource("corpus://{name}")
def corpus_file(name: str) -> str:
    """按文件名读取语料原文（Resource：客户端按需取，不经模型决策）。"""
    return _guard.resolve(name).read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    # ★ 日志必须走 stderr：stdout 是 JSON-RPC 帧通道
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info("codeloom MCP 启动 root=%s version=%s", CORPUS_ROOT, VERSION)
    server.run()  # 默认 stdio；远程部署用 transport="streamable-http"


if __name__ == "__main__":
    main()
