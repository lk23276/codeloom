"""对比两套 LLM 后端在同一组问题上的回答一致率。

用法：
    uv run python scripts/compare_backends.py
    uv run python scripts/compare_backends.py --gold eval/gold/chat_smoke.jsonl --max-tokens 120
    uv run python scripts/compare_backends.py --show-answers

为什么需要这个脚本
────────────────
ADR-7 决定「手写实现为默认、LangChain 作第二实现」。双实现最大的风险不是报错，
而是其中一套出现静默偏差：`usage` 恒为 0、`tool_calls` 恒为空、回答变得又短又空。
这些都不会抛异常，只有拿同一组问题跑两遍并逐条比对，才是可观测的。

判定口径（刻意保守：只验证契约，不评文采）
────────────────
    - 关键词命中：`must_contain` 里的词是否都出现（空列表视为恒真）
    - 长度比：`min(len) / max(len)`，低于阈值说明"答非所问或只答了一半"
    - 一致 = 两边的关键词命中结果相同 **且** 长度比达标

成本：用例数 × 2 次真实调用。默认 5 条、`max_tokens=120`，单轮成本可忽略。
      运行前应确认 `.env` 中的 key 可用（`GET /models` 不消耗 token）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time
import unicodedata
from dataclasses import dataclass

from app.client.llm import LLMClient, LLMError, build_client
from app.core.config import Settings

DEFAULT_GOLD = pathlib.Path(__file__).resolve().parents[1] / "eval" / "gold" / "chat_smoke.jsonl"
BACKENDS = ("openai", "langchain")


# ══════════════════════ 数据结构 ══════════════════════


@dataclass
class Case:
    id: str
    question: str
    must_contain: list[str]
    note: str = ""


@dataclass
class Answer:
    ok: bool
    text: str = ""
    error: str = ""
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def hits(self, keywords: list[str]) -> bool:
        """关键词是否全部命中。空列表恒真（用于无关问题）。"""
        return self.ok and all(k.lower() in self.text.lower() for k in keywords)


# ══════════════════════ 加载与调用 ══════════════════════


def load_cases(path: pathlib.Path) -> list[Case]:
    """读取 JSONL 金标集。

    ★ 空行跳过、注释行（`#` 开头）跳过：手写 jsonl 时常用这两种方式分组。
    """
    cases: list[Case] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
        cases.append(
            Case(
                id=obj["id"],
                question=obj["question"],
                must_contain=list(obj.get("must_contain") or []),
                note=obj.get("note", ""),
            )
        )
    if not cases:
        raise SystemExit(f"{path} 里没有任何用例")
    return cases


async def ask(client: LLMClient, case: Case, max_tokens: int) -> Answer:
    """单条用例调用一次；异常收敛成 Answer，不让整轮中断。"""
    t0 = time.perf_counter()
    try:
        result = await client.achat(
            [{"role": "user", "content": case.question}], max_tokens=max_tokens
        )
    except LLMError as exc:
        return Answer(ok=False, error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 —— 脚本要跑完整轮，不能因单条失败而中断
        return Answer(ok=False, error=f"{type(exc).__name__}: {exc}")

    return Answer(
        ok=True,
        text=result.text,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
    )


# ══════════════════════ 判定与输出 ══════════════════════


def _width(text: str) -> int:
    """按终端显示宽度算长度：中文/全角占 2 列。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def judge(case: Case, left: Answer, right: Answer, min_ratio: float) -> tuple[bool, str]:
    """返回（是否一致，理由）。"""
    if not left.ok or not right.ok:
        return False, "有一侧调用失败"

    left_hit = left.hits(case.must_contain)
    right_hit = right.hits(case.must_contain)
    if left_hit != right_hit:
        return False, f"关键词命中不一致（{left_hit}/{right_hit}）"

    a, b = len(left.text), len(right.text)
    ratio = min(a, b) / max(a, b) if max(a, b) else 1.0
    if ratio < min_ratio:
        return False, f"长度比 {ratio:.2f} 低于阈值 {min_ratio:.2f}"
    return True, f"关键词 {left_hit}/{right_hit}，长度比 {ratio:.2f}"


async def run(gold: pathlib.Path, max_tokens: int, min_ratio: float, show: bool) -> int:
    cases = load_cases(gold)
    print(f"金标集：{gold}（{len(cases)} 条）  max_tokens={max_tokens}  长度比阈值={min_ratio}")

    clients: dict[str, LLMClient] = {}
    for backend in BACKENDS:
        settings = Settings(llm_backend=backend)  # type: ignore[arg-type]
        try:
            clients[backend] = build_client(settings)
        except LLMError as exc:
            print(f"[跳过] {backend} 后端不可用：{exc}")
    if not clients:
        print("两个后端都不可用，无法对比。检查 .env 里的 key 与依赖安装。")
        return 2

    print()

    passed = 0
    for case in cases:
        answers: dict[str, Answer] = {}
        for backend, client in clients.items():
            answers[backend] = await ask(client, case, max_tokens)

        available = [answers[b] for b in clients]
        if len(available) == 2:
            ok, reason = judge(case, available[0], available[1], min_ratio)
        else:
            only = available[0]
            ok, reason = (only.ok, "仅一个后端可用，无法对比" if only.ok else only.error)
        passed += int(ok)

        print(f"[{case.id}] {_pad('一致' if ok else '不一致', 8)} {reason}")
        print(f"      {case.question}")
        for backend in clients:
            ans = answers[backend]
            if ans.ok:
                body = ans.text if show else ans.text.replace("\n", " ")[:60] + "…"
                print(
                    f"      {_pad(backend, 10)} {ans.latency_ms:>5}ms  "
                    f"tok={ans.prompt_tokens}+{ans.completion_tokens}  {body}"
                )
            else:
                print(f"      {_pad(backend, 10)} ERROR  {ans.error}")

        print()

    total = len(cases)
    print(f"一致率：{passed}/{total} = {passed / total:.0%}")
    return 0 if passed == total else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对比两套 LLM 后端的回答一致率")
    parser.add_argument("--gold", type=pathlib.Path, default=DEFAULT_GOLD, help="金标集路径")
    parser.add_argument("--max-tokens", type=int, default=120, help="单次生成上限")
    parser.add_argument("--min-ratio", type=float, default=0.5, help="长度比阈值，低于即判为不一致")
    parser.add_argument("--show-answers", action="store_true", help="打印完整回答")
    args = parser.parse_args(argv)

    return asyncio.run(run(args.gold, args.max_tokens, args.min_ratio, args.show_answers))


if __name__ == "__main__":
    sys.exit(main())
