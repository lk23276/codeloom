"""MCP Server 的脚本自测：用官方 client 走一遍 stdio。

★ 为什么必须有这个脚本，而不是只在客户端里点：
  GUI 点一次只能证明「今天能跑」，脚本能证明「改完之后还能跑」。
  这也是回归门禁的入口。

跑法：

    uv run python scripts/mcp_smoke.py

退出码：0 全部通过；1 有断言失败（便于挂 CI）。

★ 子进程用 `sys.executable` 而不是 `uv run`：
  脚本本身就跑在项目虚拟环境里，直接复用同一个解释器，少一层包装、少一处路径依赖。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f" | {detail}" if detail else ""))
    if not ok:
        failures.append(name)


async def run() -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server.server"],
        cwd=str(PROJECT_ROOT),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("== 1) 握手与工具列表 ==")
            listed = await session.list_tools()
            names = sorted(tool.name for tool in listed.tools)
            check("三个工具已注册", names == ["list_files", "read_file", "search_code"], str(names))

            print("== 2) 检索返回带行号的片段 ==")
            result = await session.call_tool(
                "search_code", {"query": "为什么要用 eventfd 唤醒", "top_k": 3}
            )
            rows = result.structured_content or {}
            hits = rows.get("result") if isinstance(rows, dict) else None
            # 不同版本对结构化输出的包法不同，退化为读文本内容也接受
            if hits is None:
                hits = []
            check("检索有结果", len(hits) > 0, f"{len(hits)} 条")
            if hits:
                first = hits[0]
                check(
                    "首条带文件与行号",
                    bool(first.get("source")) and first.get("line_start") is not None,
                    f"{first.get('source')}:{first.get('line_start')}-{first.get('line_end')}",
                )

            print("== 3) 路径穿越被拒 ==")
            escape = await session.call_tool("read_file", {"path": "../../etc/passwd"})
            text = " ".join(
                getattr(item, "text", "") for item in (escape.content or [])
            )
            check("越界路径被拒", bool(escape.is_error), text[:80] or "(无错误文本)")

            print("== 4) 合法文件可读 ==")
            ok_read = await session.call_tool(
                "read_file", {"path": "demo_notes.md", "max_chars": 60}
            )
            body = " ".join(getattr(item, "text", "") for item in (ok_read.content or []))
            check("读取白名单文件成功", not ok_read.is_error and len(body) > 0, body[:40])

            print("== 5) 资源模板已注册 ==")
            templates = await session.list_resource_templates()
            uris = [t.uri_template for t in templates.resource_templates]
            check("corpus:// 模板存在", any("corpus://" in uri for uri in uris), str(uris))

    return 0 if not failures else 1


def main() -> int:
    print(f"项目根：{PROJECT_ROOT}")
    code = asyncio.run(run())
    print()
    if failures:
        print(f"失败 {len(failures)} 项：{', '.join(failures)}")
    else:
        print("全部通过")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
