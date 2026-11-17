"""LangChain 工具封装：把已有工具接到 `@tool` 上。

★ 只做包装，不重新实现。核心逻辑仍在 `tools/registry.py` 的 `_calculator`：

      手写路径   registry.TOOLS（JSON Schema） → dispatch() → _calculator()
      图路径     @tool 自动生成 Schema        → ToolNode  → _calculator()

  两条路径共用同一个实现，所以「图版与手写版同题答案一致」是结构上的必然，
  不是靠两边各写一遍再对齐。

★ `@tool` 的三件事（缺一个模型就用不对）：
  1. 函数名 = 工具名；
  2. **docstring = description** —— 这是发给模型的文案，不是给人看的注释；
  3. 类型注解 → 自动生成 JSON Schema（底层仍是 Pydantic）。

  因此这里的 docstring 与 `registry.TOOLS["description"]` 必须表达同一件事：
  能力边界（什么时候该用）比功能描述更重要。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from app.tools.registry import _calculator


@tool
def calculator(expression: str) -> str:
    """计算只含四则运算与括号的表达式，例如 "(12+5)*3"。

    仅用于算术求值。不要用它处理日期、单位换算或字符串操作。
    参数 expression 必须是合法的算术表达式字符串。
    """
    # 异常不在这里捕获：ToolNode 会把异常转成 ToolMessage 文本回给模型，
    # 与手写路径 dispatch() 的 `_fail()` 行为一致。
    return _calculator(expression)


# 图与 ToolNode 用这个列表注册工具；新增工具时只改这里。
LC_TOOLS: list[BaseTool] = [calculator]
