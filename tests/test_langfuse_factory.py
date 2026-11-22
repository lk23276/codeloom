"""Langfuse 工厂的契约测试：三种「拿不到」都必须安静返回 None。

★ 为什么这个工厂值得测：
  它决定了「没配账号时服务还能不能起来」。写错一个分支的代价不是报错，
  而是**启动即失败**——而恰好是最不该失败的场景（别人 clone 之后第一次跑）。
"""

from __future__ import annotations

from app.obs.langfuse import build_langfuse_handler, callbacks_of


class Cfg:
    def __init__(self, public: str = "", secret: str = "", host: str = "") -> None:
        self.langfuse_public_key = public
        self.langfuse_secret_key = secret
        self.langfuse_host = host


def test_returns_none_without_keys() -> None:
    assert build_langfuse_handler(Cfg()) is None


def test_returns_none_with_only_public_key() -> None:
    assert build_langfuse_handler(Cfg(public="pk-lf-test")) is None


def test_returns_none_when_package_missing(monkeypatch) -> None:
    """配了 key 但没装 langfuse：返回 None，不抛异常。

    ★ 用 monkeypatch 造「模块不存在」比依赖真实环境更稳：
      开发机装了 langfuse 时，这条用例仍然测的是同一个分支。
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("langfuse"):
            raise ModuleNotFoundError("No module named 'langfuse'", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert build_langfuse_handler(Cfg(public="pk", secret="sk")) is None


def test_returns_none_on_import_error(monkeypatch) -> None:
    """装的是 v2（导入路径不同）时，也不该让服务起不来。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("langfuse"):
            raise ImportError("cannot import name 'CallbackHandler' from 'langfuse.langchain'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert build_langfuse_handler(Cfg(public="pk", secret="sk")) is None


def test_callbacks_of_empty_when_handler_is_none() -> None:
    assert callbacks_of(None) == {}


def test_callbacks_of_wraps_handler() -> None:
    sentinel = object()
    assert callbacks_of(sentinel) == {"callbacks": [sentinel]}
