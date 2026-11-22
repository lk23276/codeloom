"""路径白名单的单元测试：不起 MCP 进程、不需要网络。

为什么这批用例值得单独写：
  路径穿越是**唯一一条**「写错了也能跑通」的安全边界 —— 功能测试全绿，
  只因为没人试过 `../`。所以这里把六种绕过方式逐条钉住。

★ 六种输入：合法文件 / 上跳 / 绝对路径 / 空字节 / 不存在 / 后缀不对。
"""

from __future__ import annotations

import pytest

from app.mcp_server.security import PathGuard


@pytest.fixture
def guard(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "demo.md").write_text("# 标题\n正文\n", encoding="utf-8")
    (root / "code.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    (root / "secret.db").write_text("not allowed", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "nested.md").write_text("嵌套文件\n", encoding="utf-8")
    return PathGuard(root)


# ══════════════════════ 一、正常路径 ══════════════════════


def test_reads_file_inside_root(guard) -> None:
    assert guard.resolve("demo.md").name == "demo.md"


def test_reads_nested_file(guard) -> None:
    assert guard.resolve("sub/nested.md").parent.name == "sub"


# ══════════════════════ 二、六种绕过方式 ══════════════════════


def test_rejects_parent_traversal(guard) -> None:
    with pytest.raises(ValueError, match="越界"):
        guard.resolve("../outside.md")


def test_rejects_absolute_path(guard) -> None:
    # 绝对路径与 root 拼接时会直接替换掉 root，is_relative_to 会拦住
    with pytest.raises(ValueError, match="越界"):
        guard.resolve("/etc/passwd")


def test_rejects_null_byte(guard) -> None:
    with pytest.raises(ValueError, match="非法字符"):
        guard.resolve("demo.md\x00.png")


def test_rejects_empty_path(guard) -> None:
    with pytest.raises(ValueError, match="为空"):
        guard.resolve("   ")


def test_rejects_missing_file(guard) -> None:
    with pytest.raises(ValueError, match="不存在"):
        guard.resolve("not-there.md")


def test_rejects_disallowed_suffix(guard) -> None:
    with pytest.raises(ValueError, match="不支持"):
        guard.resolve("secret.db")


# ══════════════════════ 三、目录清单 ══════════════════════


def test_catalog_only_lists_allowed_suffixes(guard) -> None:
    names = {row["name"] for row in guard.catalog()}

    assert names == {"demo.md", "code.cpp"}       # 不含 secret.db，也不递归 sub/
    assert all("lines" in row and "chars" in row for row in guard.catalog())


def test_catalog_on_missing_root_is_empty(tmp_path) -> None:
    assert PathGuard(tmp_path / "nope").catalog() == []
