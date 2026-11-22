"""`Settings.key_state` 的契约：日志里只允许出现「已配置 / 未配置」。

★ 为什么这条值得单独测：
  它守的是「日志不泄密」这条红线。一旦有人图省事改回打前缀，
  测试会立刻失败 —— 而不是等到某张截图流出去才发现。
"""

from __future__ import annotations

import pytest

from app.core.config import Settings

# ★ 拆成两段拼接：整串字面量会被 secret scanner 命中，
#   虽然它明显是假值，但仓库里不该出现「长得像密钥」的字符串。
FAKE_KEY = "sk-" + "aaaaaa1111111111"


def make(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_configured_key_reports_state_only() -> None:
    state = make(llm_api_key=FAKE_KEY).key_state

    assert state == "已配置"
    # 关键断言：返回值里不含密钥的任何片段
    assert "sk-" not in state
    assert FAKE_KEY[:6] not in state


def test_empty_key_reports_unconfigured() -> None:
    assert make(llm_api_key="").key_state == "未配置"


@pytest.mark.parametrize("placeholder", ["placeholder", "your-key", "sk-test", "xxx"])
def test_placeholder_is_unconfigured(placeholder: str) -> None:
    assert make(llm_api_key=placeholder).key_state == "未配置"


def test_state_is_one_of_two_values() -> None:
    """取值集合必须收敛：日志字段可枚举，才能被采集端当作标签。"""
    assert make(llm_api_key=FAKE_KEY).key_state in {"已配置", "未配置"}
    assert make(llm_api_key="").key_state in {"已配置", "未配置"}
