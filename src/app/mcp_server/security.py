"""路径白名单：MCP 工具读文件前的唯一关口。

为什么单独成文件而不是写在 `server.py` 里：
  这是**可以被单测直接覆盖**的纯逻辑。放在 `server.py` 里就得先起 MCP 进程才能测，
  而路径穿越恰恰是最该被反复测的一类代码。

★ 两个必须的步骤，缺一不可：

      candidate = (root / relative).resolve()      # 展开 .. / 软链接 / 相对路径
      candidate.is_relative_to(root)               # 展开后仍在白名单内

  只做字符串检查（`if ".." in path`）挡不住：软链接、`..\\`、URL 编码、大小写差异都能绕。

★ 为什么还要限制后缀：白名单目录里将来可能混进 .db / .pyc / 密钥文件，
  按后缀放行比「先读再判断」更早拦住。
"""

from __future__ import annotations

from pathlib import Path

# 与 `rag/splitter.py` 的语料后缀保持一致：能被切分入库的，才允许被读出来
ALLOWED_SUFFIXES: frozenset[str] = frozenset(
    {".cpp", ".cc", ".c", ".h", ".hpp", ".md", ".markdown", ".txt"}
)


class PathGuard:
    """把「相对路径」解析成「白名单内的绝对路径」，否则抛 ValueError。"""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, relative: str) -> Path:
        if not relative or not relative.strip():
            raise ValueError("路径为空")
        # NUL 字节会在部分系统调用处截断字符串，先拒掉
        if "\x00" in relative:
            raise ValueError("路径包含非法字符")

        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("路径越界，拒绝访问")
        if not candidate.is_file():
            raise ValueError("文件不存在或不是普通文件")
        if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError(f"不支持的文件类型：{candidate.suffix}")
        return candidate

    def catalog(self) -> list[dict[str, object]]:
        """列出白名单目录下的可读文件（只看一层，不递归）。"""
        if not self.root.is_dir():
            return []
        rows: list[dict[str, object]] = []
        for path in sorted(self.root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            rows.append(
                {
                    "name": path.name,
                    "lines": text.count("\n") + (0 if text.endswith("\n") else 1),
                    "chars": len(text),
                }
            )
        return rows
