"""工具注册表的关键路径测试（Function Calling）。

★ 这个文件守的不是「覆盖率数字」，是**工具层的四条契约**。
  它们共同的特点：破了也不崩，只是模型拿到的东西不对，最后答案默默变差。

      ① 正常路径      -> 表达式算对、格式稳定（`4/2` 是 "2" 不是 "2.0"）
      ② 危险输入      -> 绝不 `eval`、绝不执行，只回错误文本
      ③ 坏参数        -> 未知工具 / 坏 JSON / 缺字段 / 多字段，一律 `ok: false`
      ④ 异常不外抛    -> 除零、表达式过大都交给模型转述，请求不能 500

★ 为什么错误要返回 JSON 文本而不是 raise：
  这些函数的调用方是**模型**，不是人。抛异常会让整个请求失败；
  返回 `{"ok": false, "error": ...}` 模型下一轮能自己改口。

用法：
    uv run pytest tests/test_tools_registry.py -q
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from app.tools import registry


def payload(out: str) -> dict[str, Any]:
    """把 dispatch 的返回值解析成 dict，顺带守住「返回值必须是 JSON 字符串」。"""
    data = json.loads(out)
    assert isinstance(data, dict)
    assert set(data) <= {"ok", "result", "error"}, data
    assert isinstance(data["ok"], bool)
    return data


# ══════════════════════ ① 正常路径：算得对 + 格式稳定 ══════════════════════


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(12+5)*3", "51"),
        ("7//2", "3"),  # 整除
        ("7%4", "3"),  # 取模
        ("2**10", "1024"),  # 幂
        ("-(3+4)", "-7"),  # 一元负号
        ("+5", "5"),  # 一元正号
        ("4/2", "2"),  # ★ 整数值的 float 不能输出 "2.0"
        ("1e3", "1000"),  # 科学计数法常量
        ("1e17", "100000000000000000"),  # ★ 大整数形态的 float 不能输出 "1e+17"
        ("1/3", "0.3333333333"),  # 浮点按 .10g 截断，避免 0.3333333333333333
    ],
)
def test_dispatch_calculator_happy_path(expression: str, expected: str) -> None:
    """正常表达式要算出正确结果，且**文本格式固定**。

    ★ 为什么连格式一起测：工具结果是喂回模型的字符串。`4/2` 输出 "2.0"
      不会报错，但模型转述时容易把整数说成小数；`.10g` 不加则会出现
      十七位小数，白烧 token。格式错了「看起来也算对了」，最难发现。
    """
    out = registry.dispatch("calculator", json.dumps({"expression": expression}))

    data = payload(out)
    assert data == {"ok": True, "result": expected}


# ══════════════════════ ② 危险 / 非法表达式：只拒绝，不执行 ══════════════════════


@pytest.mark.parametrize(
    ("expression", "match"),
    [
        # 空 / 超长（在 ast.parse 之前就被挡掉）
        ("", "表达式为空或过长"),
        ("1+" * 100 + "1", "表达式为空或过长"),
        # 任意代码执行：eval 的经典入口，必须停在 Call 节点
        ("__import__('os').system('echo hi')", "不支持的语法"),
        ("().__class__.__bases__", "不支持的语法"),
        ("x + 1", "不支持的语法"),
        ("1 if 1 else 2", "不支持的语法"),
        # 非数字常量：字符串 / 布尔（★ True 也是 int 实例，要先排掉）
        ("'a' + 'b'", "只允许数字常量"),
        ("True + 1", "只允许数字常量"),
        # 不支持的运算符
        ("1 << 2", "不支持的运算符"),
        ("~5", "不支持的一元运算符"),  # 按位取反也是一元运算，但不在白名单里
        # 资源上限：指数、结果绝对值、嵌套深度
        ("2 ** 101", "指数过大"),
        ("10 ** 19", "结果过大"),
        ("(-1) ** 0.5", "不支持复数结果"),
        ("1+" * 25 + "1", "嵌套过深"),
    ],
)
def test_calculator_rejects_unsafe_or_invalid_input(expression: str, match: str) -> None:
    """白名单之外的一切语法都抛 ValueError —— **特别是绝不能执行**。

    ★ 安全红线的回归测试：一旦有人把 `_eval` 换成 `eval(expression)`，
      上面第一条 `__import__('os').system(...)` 就会真的在当前进程里执行。
      这条测试的价值不在于「报错了」，而在于**它没有执行**：
      表达式里写的是 `echo hi`，真被执行也只是多一行输出；
      换成 `rm -rf` 就是事故。所以这里断言的是「停在语法检查」。
    """
    with pytest.raises(ValueError, match=match):
        registry._calculator(expression)


# ══════════════════════ ③ 派发：坏参数一律变成文本 ══════════════════════


def test_dispatch_unknown_tool_returns_error() -> None:
    """模型幻觉出一个不存在的工具名 -> `ok: false`，并告诉它有哪些可用。

    ★ 不能抛异常：模型偶尔会编工具名（尤其上下文里提过"天气""搜索"时）。
      抛异常 = 整个请求 500；返回文本 = 模型下一轮改用别的说法。
    """
    out = registry.dispatch("get_weather", '{"city": "北京"}')

    data = payload(out)
    assert data["ok"] is False
    assert "未知工具" in data["error"]
    assert "calculator" in data["error"], "错误里要列出可用工具，方便模型改口"


@pytest.mark.parametrize(
    ("arguments", "needle"),
    [
        ("not-json", "参数不是合法 JSON"),
        ("[1, 2]", "必须是 JSON 对象"),  # 合法 JSON 但不是对象
        ("", "expression"),  # 空串当 {}，走到"缺必填字段"
        ('{"expr": "1+1"}', "expression"),  # 字段名写错：多一个 expr、少一个 expression
        ('{"expression": "1+1", "unit": "cm"}', "unit"),  # ★ 多余字段必须被 forbid 拦下
        ('{"expression": 123}', "expression"),  # 类型不对
        ('{"expression": ""}', "expression"),  # 违反 min_length=1
    ],
)
def test_dispatch_rejects_bad_arguments(arguments: str, needle: str) -> None:
    """坏 JSON / 非对象 / 字段错 / 类型错 / 空值，全部变成字段级错误文本。

    ★ 这里体现的是 `CalculatorArgs` 的 `extra="forbid"`：
      模型给错字段名时必须**报错**，不能像 HTTP 请求体那样宽松忽略 ——
      宽松只会让它一直错下去，而报错能让它下一轮改对。
    """
    data = payload(registry.dispatch("calculator", arguments))

    assert data["ok"] is False
    assert needle in data["error"], data["error"]


# ══════════════════════ ④ 执行期异常：翻译成文本，绝不外抛 ══════════════════════


@pytest.mark.parametrize(
    ("expression", "needle"),
    [
        ("1/0", "ZeroDivisionError"),
        ("5 % 0", "ZeroDivisionError"),
        ("__import__('os')", "ValueError"),  # 危险表达式在派发层也只回文本
    ],
)
def test_dispatch_turns_exceptions_into_text(expression: str, needle: str) -> None:
    """工具执行失败 -> `ok: false` + 异常类名，**请求不崩**。

    ★ `except Exception` 在 dispatch 里是**故意的**：工具是"不可信输入的
      执行现场"，除零、溢出、格式错都可能发生，全都该由模型转述给用户，
      而不是让一次计算失误把整个 /chat 变成 500。
    """
    data = payload(registry.dispatch("calculator", json.dumps({"expression": expression})))

    assert data["ok"] is False
    assert needle in data["error"], data["error"]


# ══════════════════════ ⑤ 结果截断：控制 prompt 成本 ══════════════════════


class _EmptyArgs(BaseModel):
    """临时工具的零参数模型（★ 不能直接拿 BaseModel 当参数模型）。"""

    model_config = ConfigDict(extra="forbid")


def test_dispatch_truncates_long_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """工具结果超过 `MAX_RESULT_CHARS` 必须截断并注明。

    ★ 为什么值得单独测：工具结果会进 messages，而 messages **每轮重发**。
      一个返回几千字的工具会让后面每一轮都变贵 —— 这是「不报错的成本泄漏」。
      计算器的结果最长也就 19 个字符，永远碰不到这条分支，
      所以这里临时注册一个"话痨工具"来验证契约本身。

      `monkeypatch.setitem` 会在测试结束后自动还原两张表，不会污染其他用例。
    """
    long_text = "x" * (registry.MAX_RESULT_CHARS + 500)
    monkeypatch.setitem(registry._ARGS, "chatty", _EmptyArgs)
    monkeypatch.setitem(registry._DISPATCH, "chatty", lambda: long_text)

    data = payload(registry.dispatch("chatty", "{}"))

    assert data["ok"] is True
    assert data["result"] == "x" * registry.MAX_RESULT_CHARS + "... (已截断)"
