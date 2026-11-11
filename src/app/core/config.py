"""配置集中地。

密钥从哪里来：
    .env（项目根，已被 .gitignore 忽略）→ 环境变量 → 这里的 Settings 字段

命名映射（env_prefix="APP_"）：
    APP_LLM_API_KEY  →  llm_api_key
    APP_DB_PATH      →  db_path

★ 密钥【绝不】写进代码，日志里只用 key_preview 打前 6 位。

★ 为什么 llm_api_key 默认给空字符串而不是必填：
  必填会让「没配 key」变成「服务起不来」——连 /healthz 都通不了，
  骨架也就无法验证。这里改成「允许为空 + 启动时警告 + 调用时返回 502」，
  生产环境应该在 CI 里加一条 `test -n "$APP_LLM_API_KEY"` 把关。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/app/core/config.py → parents[3] 才是项目根（config.py 上四级）
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """所有配置集中在这里；类型即校验。"""

    model_config = SettingsConfigDict(
        # 绝对路径指向项目根的 .env，不依赖运行时 cwd（D5 踩过的坑）
        env_file=str(_PROJECT_ROOT / ".env"),
        env_prefix="APP_",
        extra="ignore",
    )

    # ── LLM ──
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    # 用正名：截至 2026-11，DeepSeek 官方只列 deepseek-flash / deepseek-v4-pro；
    # 旧别名 deepseek-chat / deepseek-reasoner 会被静默路由，不要用。
    llm_model: str = "deepseek-flash"
    # 思考模式默认关：实测同一任务输出 token 77 → 12（正文一样）
    llm_thinking: bool = False
    timeout_s: float = 30.0
    # ★ SDK 自带重试必须关掉：它的 2 次 × 我们的 2 次 = 4 次真实调用
    max_retries: int = 2

    # ── 存储 ──
    db_path: str = "./data/sessions.db"

    # ── 限流 ──
    # ★ 用 Field(gt=0) 而不是「靠构造函数里 raise」：
    #   配置错误应该在**启动时**炸（ValidationError 一眼看到哪个变量错了），
    #   而不是等到第一个请求进来才抛 ValueError。
    rate_limit: int = Field(default=30, gt=0)
    rate_window_s: int = Field(default=60, gt=0)

    # ── 日志 ──
    log_level: str = "INFO"
    # json：生产/容器（一行一个对象，采集端直接解析）
    # text：本地开发（带 trace id 的一行文本，好扫）
    log_format: str = Field(default="json", pattern="^(json|text)$")

    @property
    def key_preview(self) -> str:
        """脱敏预览：只露前 6 位。日志里用这个，不要用原文。"""
        k = self.llm_api_key
        return f"{k[:6]}***" if len(k) > 6 else "***"

    @property
    def has_llm_key(self) -> bool:
        """是否配了看起来可用的 key（排除占位符）。"""
        k = self.llm_api_key.strip()
        if not k:
            return False
        low = k.lower()
        return not any(w in low for w in ("placeholder", "your-key", "xxx", "sk-test"))
