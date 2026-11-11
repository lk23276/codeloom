"""本地工具注册表（Function Calling）。

职责边界（违反就是分层破了，而且不会报错）：
  - 这里只做三件事：**声明 schema**、**按名字派发**、**把异常转成给模型看的文本**
  - 不许 `import openai` —— SDK 细节全在 `client/llm.py`
  - 不许 `import fastapi` —— 这里不认识 HTTP
  - 不做网络请求 —— v0.1 的工具全是本地计算：可测、可离线、不花钱

★ 为什么工具要单独成一层：
  模型不认识代码里的函数，它只认识一份 **JSON Schema 说明书**。
  「注册」= 把说明书交给模型；「派发」= 按模型给的名字和参数真正执行。
  两件事分开写，才不会把「给模型看的描述」和「实际干的活」混成一团。

★ 一个反直觉的点（实测）：
  `tools` 是请求的**顶层参数**，不在 `messages` 里，所以**每一轮都要传**。
  它不像 system 提示那样"发一次就记住" —— 少传一轮，那一轮模型就像忘了自己会工具。

═══════════════════════════════════════════════════════════════════
  一、client 层的前提：少了这三样，循环会永远在第一轮静默结束
═══════════════════════════════════════════════════════════════════

FC 要求 `ChatResult` 能带回「模型想调哪个工具、参数是什么」。现在它只有
`text / usage / finish_reason / latency_ms`，**没有 `tool_calls`**。
需要补四处（都在 `client/llm.py`）：

  ① 加一个数据类型（放在 `Usage` 附近）：

        @dataclass
        class ToolCall:
            id: str            # ★ 必须保留：tool 消息靠它配对
            name: str
            arguments: str     # ★ 是 JSON **字符串**，不是 dict（SDK 原样给的就是字符串）

  ② `ChatResult` 加字段（要有默认值，否则所有旧构造点全崩）：

        tool_calls: list[ToolCall] = field(default_factory=list)

  ③ `OpenAICompatClient.achat` 里解析（就在构造 ChatResult 那里）：

        tool_calls=[
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            for tc in (choice.message.tool_calls or [])
        ],

  ④ `FakeClient` 要能"脚本化"地返回 tool_calls，否则这条链路根本没法测：
     没有它，只能靠真模型 + 真 key 验证，一次几毛钱且不可复现。

  ⑤ 顺带把消息类型放宽：现在协议写的是 `list[dict[str, str]]`，
     而带 tool_calls 的 assistant 消息里，`tool_calls` 的值是**列表**，
     `tool` 消息还要多一个 `tool_call_id`。所以类型应改成 `list[dict[str, Any]]`
     （`service/chat_service.py` 的 `_build_messages` 返回值标注也要跟着改）。

  ★★ 最危险的失败形态：**④③ 都不做，`tool_calls` 永远是空列表**。
     于是业务层的循环第一轮就看到"模型没调工具"，直接返回普通文本 ——
     接口 200、答案也像模像样，只是**工具从来没被调用过**。
     不报错、不告警，和 `Usage=` 大写写错是同一类问题。

═══════════════════════════════════════════════════════════════════
  二、工具实现
═══════════════════════════════════════════════════════════════════

v0.1 只做一个计算器：参数简单、结果可验证、不需要网络。

★ 安全红线：**绝不要用 `eval()`**。
  模型给的 `arguments` 是不可信输入，`eval("__import__('os').system('...')")`
  能直接在当前进程里执行任意代码。正确做法是用 `ast.parse` 解析成语法树，
  只放行白名单节点（`BinOp` / `UnaryOp` / `Constant` / 括号），其余一律拒绝。
"""

from __future__ import annotations

import ast
import json
import logging
import operator
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)
# 工具结果最长多少字符，超过截断。
# ★ 为什么要管：工具返回几千字，下一轮的 prompt 就跟着涨，
#   而 prompt 是**每轮重发**的 —— 一次疏忽会让后面每一轮都变贵。
MAX_RESULT_CHARS = 2000


MAX_EXPRESSION_CHARS = 200
MAX_ABS_RESULT = 1e18
MAX_EXPONENT = 100
MAX_DEPTH = 20

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node: ast.AST, depth: int = 0) -> int | float:
    if depth > MAX_DEPTH:
        raise ValueError("表达式嵌套过深")
    if isinstance(node, ast.Expression):
        return _eval(node.body, depth + 1)
    if isinstance(node, ast.Constant):
        v = node.value
        # ★ 先排 bool：True/False 也是 int 实例
        if isinstance(v, bool) or not isinstance(v, int | float):
            raise ValueError("只允许数字常量")
        return v
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符：{type(node.op).__name__}")
        left, right = _eval(node.left, depth + 1), _eval(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError("指数过大")
        result = op(left, right)
        if isinstance(result, complex):
            raise ValueError("不支持复数结果")
        if abs(result) > MAX_ABS_RESULT:
            raise ValueError("结果过大")
        return result
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的一元运算符：{type(node.op).__name__}")
        return op(_eval(node.operand, depth + 1))
    raise ValueError(f"不支持的语法：{type(node).__name__}")


def _calculator(expression: str) -> str:
    """计算一个只含四则运算与括号的表达式。

    三条约束（已按此实现）：

      1. **不用 eval** —— 原因见模块 docstring 的安全红线
      2. 只支持 `+ - * / // % **` 与括号；变量名、函数调用、属性访问一律拒绝
      3. 结果转成**字符串**返回（`role: "tool"` 的 content 必须是字符串）

    实现路径：`ast.parse(mode="eval")` 建树 → `_eval()` 递归，
    只放行 `Expression` / `Constant` / `BinOp` / `UnaryOp`，其余节点默认 raise。

    实测验收：
        _calculator("(12+5)*3")  -> "51"
        _calculator("1/0")       -> ZeroDivisionError（由 dispatch 转成错误文本）
        _calculator("__import__('os')")  -> ValueError: 不支持的语法：Call，且不执行
    """
    if not expression or len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError("表达式为空或过长")
    value = _eval(ast.parse(expression, mode="eval"))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return format(value, ".10g")


# ══════════════════════ 三、schema 声明 ══════════════════════
#
# 这份列表就是「给模型看的说明书」，每轮随请求发出去。
# 三个字段的写法会直接影响模型用不用、用对不用对：
#
#   name        模型返回 tool_calls 时用的就是这个字符串，必须和 _DISPATCH 的键一致
#   description 决定"什么时候该用"。要写清**边界**（什么时候不该用），
#               否则模型会拿计算器算"(2024+3)年"这种不该算的东西
#   parameters  标准 JSON Schema。`arguments` 会按它生成，再由调用方校验

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "计算只含四则运算与括号的算术表达式，返回精确数值。"
                "需要精确计算时使用；不要用它做单位换算、解方程或日期推算。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的表达式，例如 (12+5)*3/2",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
]
#
# 加第二个工具（时间 / 天气 mock）时，**三处**都要加：
#   ① 这个列表里加一条 schema（给模型看）
#   ② 参数模型（Pydantic，和 schema 的 properties 一一对应）+ `_ARGS` 注册表
#   ③ `_DISPATCH` 里加一条映射
# 只加一部分的后果：
#   漏 ① -> 模型不知道有这个工具，永远调不到
#   漏 ② -> 派发时无从校验参数，模型给错字段会直接进函数
#   漏 ③ -> 调用时报"未知工具"，哪怕 schema 已经发出去了
# 三种都不报错，只是行为不对。
#
# ★ 一个实测坑：**不能拿 `BaseModel` 本身当参数模型**（无参工具的偷懒写法），
#   Pydantic 会抛 `PydanticUserError: BaseModel cannot be instantiated directly`。
#   每个工具都要有自己的 Args 类，哪怕它是空的。


# ══════════════════════ 四、派发 ══════════════════════


class CalculatorArgs(BaseModel):
    # ★ 工具参数用 extra="forbid"（和 HTTP 请求体的宽松策略相反）：
    #   调用方是模型，严格报错能让它下一轮改对；宽松忽略只会让它一直错下去
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(min_length=1, max_length=200)


# 三张表一一对应：schema（给模型看）→ 参数模型（校验）→ 实现（执行）
_ARGS: dict[str, type[BaseModel]] = {"calculator": CalculatorArgs}
_DISPATCH: dict[str, Callable[..., str]] = {"calculator": _calculator}


def _ok(result: str) -> str:
    return json.dumps({"ok": True, "result": result}, ensure_ascii=False)


def _fail(error: str) -> str:
    return json.dumps({"ok": False, "error": error}, ensure_ascii=False)


def _log_call(name: str, success: bool, t0: float) -> None:
    """一次工具调用的结构化日志（tool / success / duration_ms）。

    ★ 为什么连失败也记：工具失败是「模型会不会用工具」的最直接信号。
      线上看到某个 tool 的 success 一直是 False，八成是 schema 的 description
      写得不清楚，模型一直在传错参数。
    """
    logger.info(
        "tool_call",
        extra={
            "tool": name,
            "success": success,
            "duration_ms": round((time.perf_counter() - t0) * 1000, 3),
        },
    )


def dispatch(name: str, arguments: str) -> str:
    """按名字执行工具，返回**字符串**（会原样塞进 `role: "tool"` 的 content）。

    五步实现。核心原则：**任何失败都变成文本，绝不往上抛**。

      ① 查表
         name 不在 `_DISPATCH` -> 返回 `"error: 未知工具 xxx"`（不要抛）
         模型看到这句会自己改口，用户得到的是"我算不了"，而不是 500

      ② 解析参数
         `arguments` 是 JSON **字符串**（例如 `'{"expression": "(1+2)*3"}'`），
         先 `json.loads`。解析失败 -> 返回 `"error: 参数不是合法 JSON"`

      ③ 校验参数
         用 Pydantic 模型校验（字段名和 schema 对齐，`expression: str`）。
         失败 -> 返回字段级错误文本。**不要**在这里相信模型给的字段一定对：
         它偶尔会多给一个 `expr` 或者漏掉必填项

      ④ 执行
         `try: return fn(**validated.model_dump())`
         `except Exception as exc: return f"error: {type(exc).__name__}: {exc}"`
         ★ 用 `except Exception` 是**故意的**：工具是"不可信输入的执行现场"，
           除零、溢出、格式错都可能发生，全都该由模型转述，而不是让请求崩掉

      ⑤ 截断
         结果超过 `MAX_RESULT_CHARS` 就截断并注明 `... (已截断)`

    ★ 每条分支都 `_log_call` 一次：工具调用是**花钱且不可控**的一步，
      没有日志就无法回答「模型到底有没有调工具、调了几次、成没成」。
    """
    t0 = time.perf_counter()

    fn = _DISPATCH.get(name)
    if fn is None:
        _log_call(name, False, t0)
        return _fail(f"未知工具 {name!r}；可用：{', '.join(sorted(_DISPATCH))}")

    try:
        raw = json.loads(arguments or "{}")  # ★ 空串要当 {} 处理
    except json.JSONDecodeError as exc:
        _log_call(name, False, t0)
        return _fail(f"参数不是合法 JSON（{exc.msg}，位置 {exc.pos}）")

    if not isinstance(raw, dict):
        _log_call(name, False, t0)
        return _fail('参数必须是 JSON 对象，例如 {"expression": "1+2"}')

    try:
        args = _ARGS[name].model_validate(raw)
    except ValidationError as exc:
        _log_call(name, False, t0)
        detail = "; ".join(
            f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return _fail(f"参数校验失败：{detail}")

    try:
        out = fn(**args.model_dump())
    except Exception as exc:  # ★ Exception，不是 BaseException
        logger.warning("工具执行失败 name=%s: %s", name, exc)
        _log_call(name, False, t0)
        return _fail(f"{type(exc).__name__}: {exc}")

    _log_call(name, True, t0)
    if len(out) > MAX_RESULT_CHARS:
        out = out[:MAX_RESULT_CHARS] + "... (已截断)"
    return _ok(out)


# ══════════════════════ 五、服务层的循环怎么写 ══════════════════════
#
# 循环放在 `ChatService.chat()` 里（不新增端点，design.md 只有 7 个）。
# 形状和目标状态：
#
#     messages = 历史 + 本轮 user            # 现有逻辑不变
#     for step in range(MAX_TOOL_STEPS):     # 上限 3 轮足够
#         result = await self._llm.achat(messages, tools=TOOLS, tool_choice="auto", ...)
#         if not result.tool_calls:          # ← 终止条件：模型不再要工具
#             break / return
#         messages.append({                  # ← assistant 的"工具申请单"，必须原样带回
#             "role": "assistant",
#             "content": result.text or "",
#             "tool_calls": [
#                 {"id": tc.id, "type": "function",
#                  "function": {"name": tc.name, "arguments": tc.arguments}}
#                 for tc in result.tool_calls
#             ],
#         })
#         for tc in result.tool_calls:       # ← 同一轮可能多个工具，逐个执行
#             out = dispatch(tc.name, tc.arguments)
#             messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
#     # 循环结束：最后再要一次**不带 tools** 的纯文本答案，或返回兜底文案
#
# ★ 三个坑：
#
#   1. `tools` 每轮都要传。它不在 messages 里，少传一轮模型就"忘了会工具"。
#   2. assistant 那条消息里的 `tool_calls` **必须原样带回**，`content` 即使为空也要留。
#      少了 `tool_calls`，下一条 tool 消息就没有归属（服务端会 400 或静默丢弃）；
#      少了 `content`，部分兼容层直接报 400。
#   3. `tool_call_id` 要**一一对应**。同一轮返回两个工具时，每个结果的 id 必须
#      是自己的那个 —— 错位不报错，但答案会串（典型的"看着对、其实错")。
#
# ★ 步数用尽（stopped_reason = "max_steps"）要怎么处理，两个选择：
#   ① 再发一次**不带 tools** 的请求，逼模型用现有信息作答（多一次调用、但答案完整）
#   ② 直接返回兜底文案："这个问题需要多步工具调用，当前最多 3 步"
#   v0.1 选 ②：诚实、可预期、不额外花钱；① 记入 README 的「后续改进」。
