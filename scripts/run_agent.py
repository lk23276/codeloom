"""命令行跑 ReAct 图：逐步打印每个节点的增量，不接 HTTP。

用法：

    uv run python scripts/run_agent.py
    uv run python scripts/run_agent.py -q "2 的 40 次方是多少"
    uv run python scripts/run_agent.py -q "123456789*987654321 等于多少" --max-steps 10

★ 为什么先做命令行而不是直接接接口：
  图的结构是否正确与 HTTP 无关。先用最小入口证明「能触发工具、能收敛到答案」，
  再接路由；否则「连不通」与「图跑不通」两类问题会混在一起，排查成本翻倍。

★ `--max-steps` 映射到 LangGraph 的 `recursion_limit`（超步数，不是轮数）：
  一轮「agent 调工具 → tools 返回」占 2 个超步，默认 8 约等于 3 轮工具调用，
  与手写实现的 `MAX_TOOL_STEPS = 3` 对齐。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from app.agent.graph import build_agent_graph
from app.core.config import Settings


def _text_of(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def _print_node(node: str, payload: dict[str, Any], index: int) -> str | None:
    """打印一个节点的增量；返回该节点产出的最终答案（若有）。"""
    for message in payload.get("messages", []):
        if isinstance(message, AIMessage):
            calls = message.tool_calls or []
            if calls:
                for call in calls:
                    args = json.dumps(call.get("args", {}), ensure_ascii=False)
                    print(f"[{index}] {node:<6} 调用工具 {call.get('name')}({args})")
                return None
            answer = _text_of(message)
            print(f"[{index}] {node:<6} 最终答案")
            return answer

        # ToolMessage：工具执行结果
        origin = getattr(message, "name", "tool")
        body = _text_of(message).replace("\n", " ")
        print(f"[{index}] {node:<6} {origin} -> {body[:120]}")
    return None


async def run(args: argparse.Namespace) -> int:
    settings = Settings()
    graph = build_agent_graph(settings)

    print(f"问题：{args.question}\n")
    started = time.perf_counter()
    frames: list[str] = []
    answer: str | None = None
    prompt_tokens = completion_tokens = 0

    async for event in graph.astream(
        {"messages": [("user", args.question)]},
        stream_mode="updates",
        config={"recursion_limit": args.max_steps},
    ):
        for node, payload in event.items():
            frames.append(node)
            text = _print_node(node, payload, len(frames))
            if text is not None:
                answer = text
            for message in payload.get("messages", []):
                usage = getattr(message, "usage_metadata", None) or {}
                prompt_tokens += int(usage.get("input_tokens") or 0)
                completion_tokens += int(usage.get("output_tokens") or 0)

    elapsed = time.perf_counter() - started
    print()
    if answer is None:
        print(f"未得到最终答案（帧序列 {' -> '.join(frames)}），可提高 --max-steps")
        return 1

    print(f"答案：{answer}\n")
    print(
        f"{len(frames)} 帧（{' -> '.join(frames)}）"
        f"｜耗时 {elapsed:.1f}s"
        f"｜tokens {prompt_tokens} / {completion_tokens}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 CodeLoom 的 ReAct 状态图")
    parser.add_argument("-q", "--question", default="12*7 等于几", help="要问的问题")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="recursion_limit：超步上限，一轮工具调用占 2 步（默认 8 ≈ 3 轮）",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(run(args))
    except Exception as exc:  # noqa: BLE001 —— 命令行入口，失败要给出可读原因
        print(f"运行失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
