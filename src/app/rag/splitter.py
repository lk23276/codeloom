"""文本切分：两套结构感知策略，以及「块 → 行区间」的换算。

切分的目标不是把文本切小，而是让每一块都能回到原文的确定位置。
前者只是手段，后者才是引用功能的前提 —— 一旦切分时丢掉偏移，
原文与块之间的对应关系就无法恢复。

两套策略的差别只在分隔符优先级：

- 代码：class / struct → 函数签名 → 空行 → 行 → 词，逐级降级。
  字符盲切会切断函数体；检索即使命中，交给模型的上下文也是残缺的，
  表现是答案看似合理但结论错误，比检索失败更难发现。
- Markdown：三级标题 → 二级 → 一级 → 空行 → 行。

★ 不使用 `MarkdownHeaderTextSplitter` 做两级切分。实测在
  `split_documents()` 之后，子块的 `start_index` 是相对父块的偏移，
  全部为 0，行号引用随之失效。把标题当作普通分隔符、只切一次，
  才能保住相对原文件的全局偏移。代价是丢掉标题层级信息，
  这个取舍是明确的：行号是硬需求，标题路径只是附加信息。
"""

from __future__ import annotations

from pathlib import PurePosixPath

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 代码：从粗到细。前面的切得动就不降级，所以「函数签名」必须排在「空行」之前，
# 否则函数体内部的空行会先切断函数。
#
# ★ 不应加入 "\n}"。切分器默认保留分隔符，以它为切点时 `}` 会留在后一块的开头，
#   结果是「上一块的函数没有收尾、下一块以一个孤立的 } 起头」—— 函数被切断，
#   与按结构切分的目标相反。实测加入该分隔符后，73 块里有 29 块以 } 开头。
CODE_SEPARATORS: tuple[str, ...] = (
    "\nclass ",
    "\nstruct ",
    "\nvoid ",
    "\nint ",
    "\nbool ",
    "\nstatic ",
    "\nstd::",
    "\n\n",
    "\n",
    " ",
)

# Markdown：三级优先于二级，否则以 ## 开头的小节会被 # 的规则优先切分。
NOTE_SEPARATORS: tuple[str, ...] = (
    "\n### ",
    "\n## ",
    "\n# ",
    "\n\n",
    "\n",
    " ",
)

CODE_SUFFIXES: frozenset[str] = frozenset({".cpp", ".cc", ".c", ".h", ".hpp"})
NOTE_SUFFIXES: frozenset[str] = frozenset({".md", ".markdown"})

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

_KINDS = ("code", "note")


class UnsupportedSuffixError(ValueError):
    """文件后缀不属于任何一类语料。"""


class MissingStartIndexError(ValueError):
    """切分结果缺少 start_index，无法换算行号。

    ★ 单独定义而不复用 ValueError：缺少偏移意味着切分配置错了
      （忘记 `add_start_index=True`），需要在构建期立刻中断，
      不能降级成「行号记为 1」把问题带进索引。
    """


def kind_of(name: str) -> str:
    """按后缀判断语料类型，返回 `code` 或 `note`。"""
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in CODE_SUFFIXES:
        return "code"
    if suffix in NOTE_SUFFIXES:
        return "note"
    raise UnsupportedSuffixError(f"不支持的后缀 {suffix!r}；文件 {name!r}")


def separators_for(kind: str) -> tuple[str, ...]:
    """取对应类型的分隔符序列。"""
    if kind == "code":
        return CODE_SEPARATORS
    if kind == "note":
        return NOTE_SEPARATORS
    raise ValueError(f"未知语料类型 {kind!r}；可选：{'/'.join(_KINDS)}")


def _has_content(text: str) -> bool:
    """是否含有实际内容。

    切分器在纯符号边界会留下碎片（如单独的 `}` 或 Markdown 的 `---`）。
    这类块不会命中任何有意义的查询，只会占索引空间与上下文预算。
    判据用「是否含字母或数字」，而不是长度阈值：短函数、单行注释仍需保留。
    """
    return any(ch.isalnum() for ch in text)


def split_text(
    text: str,
    source: str,
    *,
    kind: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """把一段原文切成带 `source` / `kind` / `start_index` 的块。

    `kind` 省略时按 `source` 的后缀推断。
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})，否则切分不会前进"
        )

    resolver = kind or kind_of(source)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators_for(resolver)),
        add_start_index=True,
    )
    documents = splitter.create_documents([text], metadatas=[{"source": source, "kind": resolver}])
    return [d for d in documents if _has_content(d.page_content)]


def line_range_of(doc: Document, text: str) -> tuple[int, int]:
    """把 `start_index` 换算成闭区间的行号范围。

    `text` 必须与该块同源，否则换算结果无意义。

    ★ 结尾先 `rstrip("\\n")` 再数换行：块尾若带换行符，直接计数会多出一行，
      引用展示时就会出现「第 87-103 行」这种多一行的偏差。
    """
    start_index = doc.metadata.get("start_index")
    if not isinstance(start_index, int):
        raise MissingStartIndexError(
            f"缺少 start_index，无法换算行号：source={doc.metadata.get('source')!r}"
        )

    # count 带起止参数，避免为一个大文件复制出等长的前缀字符串
    start_line = text.count("\n", 0, start_index) + 1
    body = doc.page_content.rstrip("\n")
    end_line = start_line + body.count("\n")
    return start_line, end_line
