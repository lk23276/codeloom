"""RAG 提示词、上下文拼装、引用编号解析。

两条硬规则写进 system，各有各的兜底作用：

1. **只能依据 `<context>`** —— 与重排阈值构成两道防线。阈值负责「检索不到就不答」，
   提示词负责「检索到了但不足以回答时不要补全」。只靠其中一道都会漏。
2. **只写编号，不写文件名或行号** —— 模型没有原文的偏移信息，它写出的
   `server2.cpp:1234` 没有任何依据。编号是它唯一能可靠输出的东西，映射由服务端完成。

★ 引用为什么必须由程序生成：这是本项目「可溯源」的基础。
  若允许模型自由书写出处，引用就退化成一个看起来可信的字符串，
  既无法校验，也无法在文件改动后保持一致。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document

# 只认 1-3 位数字：`[2026]` 这种不可能是块编号（上下文最多 10 块）。
CITATION_RE = re.compile(r"\[(\d{1,3})\]")

RAG_SYSTEM_PROMPT = """你是 CodeLoom（码络）的知识库问答助手。

<职责>
- 只依据 <context> 中提供的资料回答
- 每条结论之后用 [编号] 标注来源，编号必须出现在 <context> 里
</职责>

<约束>
- 不得自行写出文件名、路径或行号。引用一律用编号，服务端会映射成位置
- <context> 中没有依据时，只回答「知识库中无相关依据」，不要推测，也不要用常识补全
- 中文回答，技术名词保留英文原文
- 涉及代码时给出可运行的最小片段，不要给半截伪代码
</约束>"""


def format_context(blocks: Sequence[tuple[Document, float]]) -> str:
    """把块拼成 `<context>` 的内容。

    每块的抬头形如 `[1] server2.cpp:345-362`：编号供模型引用，
    文件名与行区间供人核对。两者都来自 metadata，不经过模型。
    """
    parts: list[str] = []
    for index, (doc, _score) in enumerate(blocks, start=1):
        meta = doc.metadata or {}
        where = (
            f"{meta.get('source', '未知文件')}:"
            f"{meta.get('line_start', '?')}-{meta.get('line_end', '?')}"
        )
        parts.append(f"[{index}] {where}\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


def build_rag_messages(
    question: str,
    blocks: Sequence[tuple[Document, float]],
) -> list[dict[str, Any]]:
    """组出一次问答的 messages。

    问题放在 `<context>` **之后**：模型的注意力更容易落在末尾，
    而末尾正是它需要回答的东西。反过来的写法在长上下文里更容易答偏。
    """
    user = f"<context>\n{format_context(blocks)}\n</context>\n\n问题：{question}"
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_citation_indices(text: str) -> list[int]:
    """按出现顺序提取引用编号，去重。

    只做提取，不判断编号是否有效 —— 有效性依赖当次候选集的大小，
    那是调用方（`RagService`）才知道的信息。
    """
    seen: set[int] = set()
    ordered: list[int] = []
    for raw in CITATION_RE.findall(text):
        index = int(raw)
        if index not in seen:
            seen.add(index)
            ordered.append(index)
    return ordered
