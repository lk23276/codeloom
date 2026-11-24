# CodeLoom v1.0

> 可溯源的 Agent 工程知识问答服务。这一版把 v0.5 的「能查」扩展成「能规划、能对外、能验收」。

## 这一版做了什么

四块工作，每块都以「可验证」收尾：

| 阶段 | 能力 | 验收方式 |
|---|---|---|
| 状态图迁移 | ReAct 循环改状态图 | 与手写版对同一问题输出**同一字符串** |
| 记忆与暂停 | checkpointer 记忆 + HITL + `POST /agent` | **杀进程重启**后仍记得上一轮 |
| 工具外露 | MCP Server（stdio，3 个只读工具） | 脚本走官方 client，6 项断言 |
| 可观测与收口 | trace + 回归门禁 + 文档定稿 | 三层 span 可见；退化时退出码非 0 |

## 新增接口

| 接口 | 说明 |
|---|---|
| `POST /agent` | 多步工具调用，带记忆（`thread_id` 分桶），可暂停审批 |
| MCP Server | `search_code` / `read_file` / `list_files` 三个只读工具 + `corpus://{name}` 资源 |

三个 HTTP 入口的边界是明确的，不互相替代：

```
/chat       轻问答，一轮或短多轮，不需要工具
/rag/ask    文档问答，无状态，带引用与拒答
/agent      工具 / 多步，有记忆，可停在审批点
```

`/chat` 与 `/agent` **不共享历史**：前者用 SQLite `sessions`，后者用 checkpointer。
这是刻意的——双写两份历史必然分叉，而且两边都不报错。

## 指标

| 版本 | 能力 | 关键数字 |
|---|---|---|
| v0.1 | 流式问答 + FC + 限流 + 容器化 | 89 测试 / 94% 覆盖 |
| v0.5 | RAG（切分 / 检索 / 重排 / 引用 / 拒答） | Hit@3 87.5%、MRR 0.729、无关问题 7/7 拒答 |
| v1.0 | + 状态图 Agent + MCP + trace + 门禁 | 242 测试 / 90% 覆盖 |
四套金标（v1.0 跑完）：

| 套件 | 结果 |
|---|---|
| `rag_locate` + `rag_refuse`（8 + 7） | Hit@3 87.5%、MRR 0.729、拒答 7/7 |
| `tool`（5） | 5/5 触发 calculator 且数值正确 |
| `memory`（3） | 3/3 第二轮答出第一轮信息 |
| `redteam`（10） | 服务层拒答 9/10，**可疑输出 0** |

`redteam` 的第 10 条（要求打包语料）越过了相关性阈值，但模型在正文里写明「知识库中无相关依据」——
拒绝有两种形态，看 `refused` 字段会低估实际防护。


检索指标口径：8 条定位题 + 7 条无关题，语料 7 文件 / 2237 行 / 139 块，`k=3`。
**重排不是净收益**：它把某条从第 9 提到第 1，也把另外三条挤出原位，MRR 由纯向量的 0.792 降到 0.729。
样本量小，不据此调参。

## 升级注意

- 新增依赖：`langgraph`、`langgraph-checkpoint-sqlite`、`mcp>=2.3.0`
- 新配置项：`APP_AGENT_CHECKPOINT` / `APP_AGENT_MEMORY_PATH` / `APP_AGENT_RECURSION_LIMIT`、
  `APP_TRACE_LOCAL` / `APP_TRACE_PATH`、`APP_LANGFUSE_*`
- **修掉一个已发布缺陷**：计算器的整数结果误走 `.10g` 格式化，`2**40` 曾输出 `1.099511628e+12`；
  现在精确返回 `1099511627776`
- `mcp` 2.x 把 `FastMCP` 改名 `MCPServer`，照 1.x 的导入写法会直接 `ModuleNotFoundError`

## 快速开始

```bash
uv sync
cp .env.example .env                     # 填 APP_LLM_API_KEY
uv run python scripts/build_corpus.py    # 切分示例语料（13 块）
uv run python scripts/index_corpus.py    # 建索引
uv run uvicorn app.main:app --port 8000
```

打开 `http://127.0.0.1:8000/demo/`（流式）或 `/demo/rag.html`（知识库问答）。

MCP 客户端配置在仓库根的 `.mcp.json`，从项目根启动即可；
没有客户端时用 `uv run python scripts/mcp_smoke.py` 走同一套 stdio 协议。

## 容器

镜像自带 `scripts/` 与示例语料，`docker run` 后一条 `exec` 即可建索引：

```bash
docker build -t codeloom:1.0.0 .
docker run -d --name codeloom -p 8000:8000 -v codeloom-data:/data \
  -e APP_LLM_API_KEY=... -e HF_ENDPOINT=https://hf-mirror.com codeloom:1.0.0
docker exec codeloom python scripts/index_corpus.py          # 13 块
docker exec codeloom python scripts/prefetch_models.py       # 两个模型
```

| 约定 | 内容 |
|---|---|
| 可写状态 | 全部在 `/data`：会话、图状态、trace、索引、模型（`/data/fastembed`） |
| 镜像内容 | 源码 + `scripts/` + **只有 `demo_*` 的示例语料** + 构建期切好的 13 块 |
| 不进镜像 | 真实语料、索引、数据库、`.env`（`.dockerignore` 用白名单控制） |
| 模型不可用 | 返回 **502** 且文案含「设置 HF_ENDPOINT 或预下载」，不再是通用 500 |

重建镜像时若复用同一个 `codeloom-data` 卷，**不需要重新下载模型，也不需要重建索引**。

## 已知限制

1. 语料需自备；仓库与镜像都只带示例语料（13 块），评测基线来自另一套完整语料，重跑不复现
2. 重排模型 1.1 GB，首次使用需下载；国内网络先设 `HF_ENDPOINT`
3. `/agent` 的串行锁是进程内的，多实例部署会退化成互相覆盖
4. `/agent` 无流式；MCP 只有 stdio 传输，未做鉴权
5. trace 是单机 JSONL，没有轮转与保留策略
6. 评测样本小（8 + 7 + 5 + 3），指标只支持看趋势

## 校验

```
ruff check                 All checks passed
pytest                     242 passed
ruff / pytest 之外        eval.compare --current eval/runs/v1.json  -> exit 0
```
