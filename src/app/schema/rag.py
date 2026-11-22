"""知识库问答的请求 / 响应契约。

与 `schema/chat.py` 分开而不复用 `ChatAnswer`：RAG 的响应多出「引用」与「拒答」
两个维度。塞进同一个模型会让对话接口永远带一个恒为 `false` 的 `refused` 字段，
也会让「没有引用」与「引用为空列表」混成同一件事。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.schema.chat import UsageInfo


class RagRequest(BaseModel):
    """一次知识库问答的请求。"""

    question: str = Field(min_length=1, max_length=2000, description="要查的问题")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """纯空白视同空问题：检索层会返回空列表，最终表现成「知识库为空」，容易误导。"""
        text = value.strip()
        if not text:
            raise ValueError("question 不能只包含空白字符")
        return text

    top_n: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="进入上下文的块数；不传则用服务端配置值",
    )

    # 默认 800：检索增强的回答带引用与小段代码，实测「解释一个机制」这类问题
    # 在 500 处会停在代码块中间。上限只限制长度，不产生额外调用。
    # ★ 上限 5000：讲解类问题（表格 + 代码块）实测 1000 处会在半句截断，2000 也只够中等篇幅。
    #   成本闸门在 default（800），不在上限 —— 不传就用默认值，不会因为上限高而多花钱。
    max_tokens: int = Field(default=800, ge=1, le=5000, description="输出 token 上限")


class Citation(BaseModel):
    """一处引用：回答正文里的 `[n]` 映射回的原文位置。"""

    index: int = Field(description="回答正文中出现的编号")
    source: str = Field(description="文件名")
    line_start: int = Field(description="块在原文中的起始行（从 1 开始）")
    line_end: int = Field(description="块在原文中的结束行（含）")
    score: float = Field(description="重排分数，原始 logits，越大越相关")


class RagAnswer(BaseModel):
    """一次知识库问答的结果。

    ★ 拒答是**正常结果**，不是错误：`refused=true` 时 HTTP 仍为 200。
      用 4xx 表达「知识库没有依据」会让客户端把它当成故障并重试，
      而重试不会改变检索结果 —— 问题出在语料，不在请求。
    """

    question: str = Field(description="原样带回的提问")
    refused: bool = Field(default=False, description="是否拒答")
    reason: str = Field(default="", description="拒答原因；未拒答时为空串")
    answer: str = Field(default="", description="回答正文，含 `[n]` 形式的引用标记")

    citations: list[Citation] = Field(default_factory=list, description="已解析出的引用")
    invalid_citations: int = Field(
        default=0,
        description="越界或重复的引用编号个数。非零说明模型引用了不存在的块",
    )

    top_score: float | None = Field(default=None, description="重排最高分（原始 logits）")
    recalled: int = Field(default=0, description="粗召回条数")
    reranked: int = Field(default=0, description="实际进入上下文的条数")

    usage: UsageInfo = Field(default_factory=UsageInfo)
    latency_ms: int = Field(default=0, description="端到端耗时，含检索与重排")
