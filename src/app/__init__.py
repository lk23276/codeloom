"""CodeLoom（码络）· 可溯源的多 Agent 工程知识助手。

v0.1 的包结构：
    app.core     配置 / 装配 / 错误契约 / 限流 / 结构化日志 / SSE 帧
    app.schema   Pydantic 契约
    app.db       SQLite 存储
    app.client   LLM 客户端（唯一 import openai 的地方）
    app.service  业务逻辑
    app.router   HTTP 路由
    app.prompts  提示词
    app.tools    Function Calling 工具注册表
"""

VERSION = "0.1.0"
