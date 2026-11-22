"""Agent 评测：工具调用与多轮记忆，走真实的图（不经过 HTTP）。

两套金标：

    eval/gold/tool.jsonl    5 条「必须触发 calculator」的题 + 期望的工具输出
    eval/gold/memory.jsonl  3 条「第二轮依赖第一轮」的对话

用法：

    uv run python -m eval.agent_eval --out eval/runs/agent.json

★ 为什么单独一个脚本而不是塞进 eval/run.py：
  `run.py` 测的是检索链路（向量召回 + 重排），不调用 LLM；
  这里测的是图（模型决策 + 工具执行 + checkpointer），两者的成本与失败模式都不同。
  混在一起会让「检索退化」与「模型今天不听话」分不开。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.agent.checkpoint import open_checkpointer
from app.agent.graph import build_agent_graph
from app.client.lc_client import build_chat_model
from app.core.config import Settings
from app.schema.agent import AgentRequest
from app.service.agent_service import AgentService

DIGITS = re.compile(r"[^0-9.]")


def _normalize(text: str) -> str:
    """去掉千分位、空格、加粗符号，只留数字与小数点，避免格式差异误判。"""
    return DIGITS.sub("", text.replace("**", ""))


def _load(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


async def eval_tools(service: AgentService, cases: list[dict[str, Any]]) -> dict[str, Any]:
    hits_tool = hits_answer = 0
    rows: list[dict[str, Any]] = []

    for case in cases:
        answer = await service.ask(AgentRequest(message=case["question"]))
        invoked = "calculator" in answer.tool_calls
        digits_ok = _normalize(case["expect_tool_output"]) in _normalize(answer.answer)
        hits_tool += int(invoked)
        hits_answer += int(digits_ok)
        rows.append(
            {
                "id": case["id"],
                "tool_invoked": invoked,
                "answer_has_value": digits_ok,
                "answer": answer.answer[:60],
            }
        )

    total = max(len(cases), 1)
    return {
        "cases": rows,
        "metrics": {
            "tool_trigger_rate": round(hits_tool / total, 4),
            "answer_value_rate": round(hits_answer / total, 4),
        },
    }


async def eval_memory(service: AgentService, cases: list[dict[str, Any]]) -> dict[str, Any]:
    hits = 0
    rows: list[dict[str, Any]] = []

    for case in cases:
        first, second = case["turns"][0], case["turns"][1]
        started = await service.ask(AgentRequest(message=first))
        # 第二轮必须带上第一轮的 thread_id，否则测的就不是记忆
        answer = await service.ask(AgentRequest(message=second, thread_id=started.thread_id))
        ok = case["expect_contains"].lower() in answer.answer.lower()
        hits += int(ok)
        rows.append(
            {"id": case["id"], "remembered": ok, "answer": answer.answer[:60]}
        )

    total = max(len(cases), 1)
    return {"cases": rows, "metrics": {"memory_hit_rate": round(hits / total, 4)}}


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else Path.cwd() / path


async def run(args: argparse.Namespace) -> int:
    settings = Settings()
    gold = _resolve(args.gold)
    tool_cases = _load(gold / "tool.jsonl")
    memory_cases = _load(gold / "memory.jsonl")

    print(f"金标：tool {len(tool_cases)} 条 / memory {len(memory_cases)} 条")

    # 用内存 checkpointer：评测不需要跨进程持久化，也避免污染 data/agent_memory.db
    async with open_checkpointer(settings.model_copy(update={"agent_checkpoint": "memory"})) as cp:
        model = build_chat_model(settings)
        graph = build_agent_graph(settings, model=model, checkpointer=cp)
        graph_hitl = build_agent_graph(
            settings, model=model, checkpointer=cp, interrupt_before=("tools",)
        )
        service = AgentService(graph=graph, graph_hitl=graph_hitl)

        tool_result = await eval_tools(service, tool_cases)
        memory_result = await eval_memory(service, memory_cases)

    for row in tool_result["cases"]:
        mark = "OK  " if (row["tool_invoked"] and row["answer_has_value"]) else "FAIL"
        print(f"  {mark} {row['id']} tool={row['tool_invoked']} value={row['answer_has_value']}")
    for row in memory_result["cases"]:
        mark = "OK  " if row["remembered"] else "FAIL"
        print(f"  {mark} {row['id']} remembered={row['remembered']}")

    metrics = {**tool_result["metrics"], **memory_result["metrics"]}
    print("\n指标:", json.dumps(metrics, ensure_ascii=False))

    out = {
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "model": settings.llm_model,
        "metrics": metrics,
        "cases": {"tool": tool_result["cases"], "memory": memory_result["cases"]},
    }
    target = _resolve(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写入 {target}")

    # 退出码语义与 compare 对齐：0 全过 / 1 有失败
    return 0 if all(value == 1.0 for value in metrics.values()) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent 评测：工具与记忆")
    parser.add_argument("--gold", default="eval/gold", help="金标目录")
    parser.add_argument("--out", default="eval/runs/agent.json", help="结果输出路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(run(args))
    except Exception as exc:  # noqa: BLE001
        print(f"评测失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
