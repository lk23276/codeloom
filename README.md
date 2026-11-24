# CodeLoom（码络）· 可溯源的 Agent 工程知识问答服务

> 面向**自有工程语料**（源码 + 项目文档）的问答服务：答案能落到 `server2.cpp:609-625` 这样的文件行号。
> 重点是**用了服务端工程的方法把它做稳**：分层边界、流式与背压、限流、
> 结构化日志、双探针、记忆与暂停、可观测与回归门禁。

| 版本 | 范围 | 状态 |
|---|---|---|
| v0.1 | `/chat` + SSE 流式 + SQLite 多轮 + 1 个本地工具 + 限流 + 日志 + Docker | ✅ |
| v0.5 | 切分 → bge Embedding → Chroma → 重排 → 行号引用 → 评测基线 | ✅ |
| v1.0 | 状态图 Agent + checkpointer 记忆 + HITL + MCP + trace + 回归门禁 + 演示台 + 部署脚本 | ✅ |

## 与 lkserver2 的关系

lkserver2 是 C++ 写的 Reactor 服务器：**在回调里绝不做阻塞 IO**、按连接管理生命周期。
CodeLoom 把同一条纪律搬到 Python：`async def` 里不碰同步 IO、请求上下文用 `ContextVar` 贯穿、
探针区分「活着」与「能接流量」。一个是底座，一个是长在上面的应用。

## 架构

```
router/    接参数、调 service、包响应体      ← 只做协议转换
service/   编排流程，不认识 HTTP / SDK      ← 可被单测直接调用
client/    唯一 import openai 的地方
rag/       唯一 import chromadb 的地方
agent/     唯一 import langgraph 的地方
core / schema / db / obs / tools           ← 配置、契约、存储、观测、工具
```

依赖规则可被命令校验，不是口头约定：

```bash
git grep -nE "^\s*(from|import)\s+(fastapi|openai|chromadb)" src/app/service/   # 应无输出
git grep -ln "^import openai\|^from openai" src/app/                            # 只有 client/llm.py
```

唯一例外是 `agent_service.py` 为类型标注 import 了 `CompiledStateGraph`（不是运行时依赖）。

三个入口互不替代，**也不共享历史**：

| 入口 | 职责 | 记忆 |
|---|---|---|
| `/chat` | 轻问答，可调本地工具 | SQLite `session_id` |
| `/rag/ask` | 文档问答，带引用与拒答 | 无（无状态） |
| `/agent` | 多步工具，可暂停审批 | checkpointer `thread_id` |

## 核心能力与判据

| 能力 | 入口 / 产物 | 判据 |
|---|---|---|
| 流式问答 | `/chat/stream`（`meta → delta… → done`） | `curl -N` 可见逐 token；三响应头齐全 |
| 多轮记忆 | `session_id` | 未知 id 返回 404，**不静默新建** |
| Function Calling | `ast` 白名单计算器，最多 3 步 | `2**40` 精确输出 `1099511627776` |
| 限流 | 滑动窗口 + 按 IP 分桶 + LRU | 120/min 要打满 **121 次**才 429 |
| 可溯源问答 | `/rag/ask` 引用映射 | 引用来自服务端，不是模型写的 |
| 状态图 Agent | `/agent`、`/agent/{id}/messages` | 同一 thread 第二轮记得第一轮 |
| HITL | `hitl=true` → 暂停 → `approve` | 暂停时 `tool_calls` 为空，工具未执行 |
| MCP | stdio，3 工具 + 1 资源 | `mcp_smoke.py` 6 项断言；越界路径被拒 |
| 可观测 | `data/traces.jsonl`、Langfuse（可选） | 三层 span，`trace_id == X-Request-ID` |
| 回归门禁 | `eval/compare.py` | 退出码 0/1/2（2 = 口径不符） |
| 部署 | `scripts/setup.*`、`run.*` | 新机器一条命令起；重复执行跳过已完成步骤 |
| 容器 | `Dockerfile` | 镜像内一条 `exec` 建 13 块；复用卷不重下模型 |

实测数字：

| 项 | 结果 |
|---|---|
| 测试 | **242 passed**，覆盖 **90%** |
| 检索 | Hit@recall 100%、Hit@3 87.5%、MRR@3 0.729（8 条定位题，`k=3`） |
| 工具 / 记忆 | tool 5/5、memory 3/3 |
| 红队 | 服务层拒答 9/10，**可疑输出 0** |
| 门禁 | 正常 0；MRR 降 0.10 → 1；`k` 不符 → 2 |
| 压测 | probe P50 14ms；chat P50 1304ms（自研开销占端到端 1~2%） |

## 快速开始

```bash
# Linux / macOS / Git Bash
scripts/setup.sh            # 装依赖 → 准备 .env → 建索引 → 自检
scripts/run.sh              # 启动，打印四个入口
scripts/run.sh --verify     # 启动并冒烟（含 4 次模型调用）
```

Windows 用 `scripts\setup.ps1` / `run.ps1`，参数相同。首次会提示填 `APP_LLM_API_KEY`
（脚本只判断是否为空，不读取、不打印），填完再跑一次即可。

| 命令 | 作用 | 幂等行为 |
|---|---|---|
| `setup` | 磁盘/uv 检查 → `uv sync` → `.env` → key 校验 → 建索引 → 自检 | `.env` 不覆盖；索引健康则跳过 |
| `run` | 选空闲端口 → 起服务 → 等就绪 → 打印入口 | 端口占用时自动往后找（显式 `--port` 则报错） |
| `verify` | 6 项冒烟 | 只读 |
| `doctor` | 环境体检 | 只读 |

退出码：**0** 成功 / **1** 环境问题（uv、依赖、端口、索引）/ **2** 配置问题（key 为空）。

不想用脚本时的手动三步：

```bash
uv sync
uv run python scripts/build_corpus.py    # 示例语料 → 13 块
uv run python scripts/index_corpus.py --verify   # 建索引，并用独立进程确认能从磁盘加载
uv run uvicorn app.main:app --port 8000 --workers 1
```

三个演示页由同一个 FastAPI 提供，同源、单文件、零构建：

| 页面 | 演示什么 |
|---|---|
| `/demo/v1.html` | 一页打通三个入口；Agent 面板带 thread 记忆与审批；顶部是实测指标 |
| `/demo/` | 逐字渲染，显示首字耗时 / 总耗时 / delta 帧数 |
| `/demo/rag.html` | 带行号的引用、按编号跳转、拒答状态与重排分数 |

![流式问答演示：同一次请求的逐帧推送](docs/demo.gif)

一次真实调用的帧序列（中间省略）：

```
event: meta
data: {"session_id":"...","model":"deepseek-flash"}

event: delta
data: epoll

event: done
data: [DONE]
id: end
```

> `--workers 1` 不能省：SQLite 锁、限流器、Agent 线程锁都是进程内的。

## Docker

镜像自带 `scripts/` 与示例语料，容器内可直接建索引，不需要挂宿主机目录。

```bash
docker build -t codeloom:1.0.0 .

docker volume create codeloom-data
read -rsp "API key: " K; echo; docker run -d --name codeloom --restart unless-stopped -p 8000:8000 -v codeloom-data:/data -e APP_LLM_API_KEY="$K" -e HF_ENDPOINT=https://hf-mirror.com codeloom:1.0.0
unset K

docker exec codeloom python scripts/index_corpus.py        # 13 块
docker exec codeloom python scripts/prefetch_models.py     # 预热两个模型（推荐）
```

`/data` 是唯一需要挂卷的位置：

| 文件 | 来源 | 删了会怎样 |
|---|---|---|
| `sessions.db` | `/chat` 会话 | 多轮历史丢失 |
| `agent_memory.db` | checkpointer | `/agent` 记忆丢失 |
| `traces.jsonl` | 本地 trace | 只影响可观测 |
| `chroma/` | 索引 | 重跑 `index_corpus.py` |
| `fastembed/` | 两个模型（≈1.2 GB） | 下次重新下载 |

重建镜像时卷不动，模型与索引全部复用。换成自己的语料：

```bash
docker run -d --name codeloom -p 8000:8000 -v codeloom-data:/data \
  -v /path/to/corpus:/corpus:ro -e APP_LLM_API_KEY=... -e HF_ENDPOINT=https://hf-mirror.com codeloom:1.0.0
docker exec codeloom python scripts/build_corpus.py --root /corpus --out /data/chunks.jsonl
docker exec codeloom python scripts/index_corpus.py --chunks /data/chunks.jsonl --rebuild
```

容器排错（均为实测踩到）：

| 现象 | 根因 |
|---|---|
| `can't open file '/app/scripts/...'` 但 `COPY scripts` 成功 | 空目录也能 COPY 成功；镜像里已加 `test -s` 守卫 |
| `Permission denied (os error 13)` 下载模型失败 | 命名卷挂到不存在的路径会以 root 创建；已预建并 chown |
| 索引能建，首个 `/rag/ask` 卡 70 秒后 502 | `HF_ENDPOINT` 只给了脚本，服务进程没给 |
| 粘多行命令时 key 变成下一行命令 | `read -rsp` 会消费后续输入，须与 `run` 同一行 |

> 用 `--env-file .env` 时，`.env` 里的 `APP_DB_PATH` 等四项是本地相对路径，会覆盖镜像内置的 `/data/...`；
> 要么只传 key，要么把这四项显式 `-e` 覆盖。文件必须是 LF，CRLF 会把 `\r` 带进值里（表现为 401）。

## 检索增强

链路：139 块 → 向量召回 top-20 → cross-encoder 精排 top-5 → 阈值判定 → 生成带 `[n]` → 服务端映射行号。

```bash
curl -s -X POST localhost:8000/rag/ask -H "Content-Type: application/json" \
  -d '{"question":"为什么要用 eventfd 唤醒"}' | head -c 200
```

```json
{"question":"为什么要用 eventfd 唤醒","refused":false,"answer":"…[1]…",
 "citations":[{"index":1,"source":"demo_echo.cpp","line_start":41,"line_end":58}],
 "recalled":13,"reranked":5,"top_score":5.52}
```

| 指标 | 值 | 说明 |
|---|---|---|
| Hit@recall | **100%** | 答案是否落在粗召回 top-20（链路天花板） |
| Hit@3 | **87.5%** | 8 条定位题中 7 条进前三 |
| MRR@3 | **0.729** | 纯向量基线 0.792 |
| 拒答 | 7/7 无关题被挡，0/8 误拒 | 阈值 0.2，标定区间 (−2.59, 1.06) |

**重排不是净收益**：它把一条从第 9 提到第 1，也把另外三条挤出原位，MRR 由 0.792 降到 0.729。
8 条题不足以调参，结论是「留着、可替换、不据此定阈值」。

两条容易误读的语义：拒答返回 **200**（正常业务结果，`usage` 全 0 证明没调模型）；
`citations` 的位置来自块 metadata，模型只写编号——它没有原文偏移，写出的路径没有依据。

## Agent（状态图）

```
START → agent ──(有 tool_calls)──→ tools ──→ agent
            └──(无 tool_calls)──→ END
```

```bash
# 第一轮：拿 thread_id
curl -s -X POST localhost:8000/agent -H "Content-Type: application/json" \
  -d '{"message":"12*7 等于几"}' | head -c 160

# 暂停：hitl=true → 工具不执行
curl -s -X POST localhost:8000/agent -H "Content-Type: application/json" \
  -d '{"message":"3*5 等于几","hitl":true}'

# 恢复：approve（与 message 互斥）
curl -s -X POST localhost:8000/agent -H "Content-Type: application/json" \
  -d '{"thread_id":"<上一步的 id>","approve":true}'

# 历史：读 checkpointer，不另存一份
curl -s localhost:8000/agent/<thread_id>/messages
```

- 记忆 = **快照 + 分桶键**：每个超步写一份 state，按 `thread_id` 分桶；不传 `thread_id` 会直接报错，没有隐式默认桶。
- 同 thread 按 `asyncio.Lock` 串行：并发写同一个桶会互相覆盖（实测期望 8 条得 6 条，两路都返回 200）。
- `approve` 先校验：不存在 → 404，「存在但无待审批」→ 400。混在一起会让「id 拼错」和「重复审批」无法区分。
- `/agent` 无流式（工具轮次里没有稳定 token 流），也无跨实例记忆（锁在进程内）。

## MCP Server

| 原语 | 名字 | 用途 |
|---|---|---|
| Tool | `search_code` / `read_file` / `list_files` | 模型主动调用，全部只读 |
| Resource | `corpus://{name}` | 客户端按 URI 取原文 |

两条实现约束：**stdout 是 JSON-RPC 帧通道**，日志必须走 stderr（`print` 会污染帧通道，且错误出现在若干帧之后）；
**路径白名单两步且不能反序**——先 `resolve()` 再 `is_relative_to(root)`，字符串查 `..` 挡不住软链接与 `%2e%2e%2f`。

自测不依赖 GUI 客户端：

```bash
uv run python scripts/mcp_smoke.py     # 握手 → list_tools → call_tool → list_resource_templates
```

配置在仓库根的 `.mcp.json`，从项目根启动即可。

## 可观测与回归门禁

trace-id 是「一根线」，span 树多出层级与耗时归因。一次 `/rag/ask` 的真实记录：

```
retrieve   24.8ms   k=20 hits=13
rerank   2000.0ms   candidates=13 scored=13
llm      2357.4ms   prompt=1491 completion=383
```

三者 `trace_id` 相同，且与响应头 `X-Request-ID` 一致——排查时不必在两个系统之间二次搜索。
本地 JSONL 永远可用；Langfuse 没配 `LANGFUSE_*` 时返回 `None`，不阻断启动。

门禁的判据是**退出码**，不是输出好看：

| 码 | 含义 | 典型原因 |
|---:|---|---|
| 0 | 通过 | 全部指标在阈值内 |
| 1 | 退化 | MRR 下降超过阈值 |
| 2 | 口径不符 | `k` / `with_llm` / 语料块数跟基线不同 |

2 单独给：它不是模型问题，归到 1 会把人带去错方向。阈值有两套口径——比率按百分点、计数按绝对条数；
8 条题里 1 条翻转就是 12.5pp，小样本下不要用百分比表达变化。

## 接口

| # | 方法 | 路径 | 作用 | 限流 |
|---:|---|---|---|---|
| 1 | POST | `/chat` | 一次性问答 | ✅ |
| 2 | POST | `/chat/stream` | SSE 流式问答 | ✅ |
| 3 | GET | `/sessions` | 会话列表 | ❌ |
| 4 | GET | `/sessions/{id}/messages` | 会话消息 | ❌ |
| 5 | DELETE | `/sessions/{id}` | 删除会话（级联消息） | ❌ |
| 6 | GET | `/healthz` | 存活（不碰依赖） | ❌ |
| 7 | GET | `/readyz` | 就绪（查 DB） | ❌ |
| 8 | POST | `/rag/ask` | 语料问答 + 引用 + 拒答 | ✅ |
| 9 | POST | `/agent` | 多步工具 + 记忆 + 审批 | ✅ |
| 10 | GET | `/agent/{thread_id}/messages` | thread 历史 | ❌ |

另有 `GET /demo/`：同源静态演示页，不计入 API 端点，也不出现在 `/docs` 的 OpenAPI 里。

统一响应体永远四键：`{code, msg, data, request_id}`。SSE 帧：

| 帧 | data | 说明 |
|---|---|---|
| `meta` | `{"session_id","model"}`（JSON） | 首帧，先于任何 delta |
| `delta` | 纯文本增量 | 可能几十上百帧 |
| `error` | `{"code","msg"}`（JSON） | 状态码已是 200，只能事件告知 |
| `done` | `[DONE]`（`id: end`） | 明确结束；不能靠连接关闭表示结束 |
| `: ping` | — | 注释帧，长时间无 token 时保活 |

## 配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `APP_LLM_API_KEY` | 空 | 为空时服务能起、`/chat` 返回 502 |
| `APP_LLM_BASE_URL` / `APP_LLM_MODEL` | `api.deepseek.com` / `deepseek-flash` | OpenAI 兼容端点 |
| `APP_LLM_THINKING` | `false` | 思考默认关（成本差 5~14 倍） |
| `APP_LLM_BACKEND` | `openai` | 可切 `langchain`（第二实现，用于对比） |
| `APP_DB_PATH` | `./data/sessions.db` | SQLite（WAL） |
| `APP_AGENT_CHECKPOINT` | `sqlite` | `memory` 仅用于测试 |
| `APP_AGENT_MEMORY_PATH` | `./data/agent_memory.db` | 图状态 |
| `APP_RAG_PERSIST_DIR` / `APP_RAG_EMBED_MODEL` | `./data/chroma` / `bge-small-zh-v1.5` | 索引与向量模型 |
| `APP_RAG_RECALL_K` / `APP_RAG_TOP_K` | 20 / 5 | 粗召回 / 进上下文的块数 |
| `APP_RAG_MIN_RERANK_SCORE` | 0.2 | 低于此值拒答 |
| `APP_RATE_LIMIT` / `APP_RATE_WINDOW_S` | 30 / 60 | 滑动窗口 |
| `APP_TRACE_LOCAL` / `APP_TRACE_PATH` | `true` / `./data/traces.jsonl` | 本地 trace |
| `APP_LANGFUSE_*` | 空 | 配了才启用 |

`max_tokens` 默认 `/chat` 200、`/rag/ask` 800，上限 5000。**默认值是成本闸门，上限只是上限**。
截断判据是 `completion_tokens == max_tokens`，不是「看起来短」。

## 设计决策

| 决策 | 代价 |
|---|---|
| 手写 SDK 客户端为默认，LangChain 作第二实现 | 多维护一层适配；换来异常可控、切后端不影响业务 |
| 三入口不共享历史（SQLite / checkpointer / 无状态） | 不能跨入口续聊；换来无双写分叉 |
| SQLite + `--workers 1` | 不能水平扩容；换来零外部依赖 |
| RAG 无状态，多轮需另做 query rewrite | 少了「追问」体验；换来评测可复现、请求可重放 |
| 拒答用阈值（区间标定） | 有漏答风险；用误拒率盯住 |
| 重排保留但不据此调参 | 多 1 GB 模型与约 2 秒；等样本足够再判断 |

## 已知限制

1. 单机：锁 / 限流 / SQLite 都是进程内，只能 `--workers 1`。
2. 评测样本小（8 + 7 + 5 + 3），指标只支持看趋势，不支持「提升 2%」这类结论。
3. `/rag/ask` 无记忆，多轮追问需要 query rewrite（未做）。
4. `/agent` 无流式；MCP 只有 stdio、无鉴权。
5. trace 是单机 JSONL，无轮转与保留策略。
6. 语料需自备：仓库与镜像只带示例语料（13 块），基线来自另一套完整语料，重跑不复现。
7. 重排模型 1.1 GB，首次使用需下载，国内先设 `HF_ENDPOINT`。

## 测试与压测

```bash
uv run pytest -q --cov          # 242 passed / 90%
uv run ruff check .             # 分层与格式
uv run python -m eval.run --k 3 --out eval/runs/v1.json
uv run python -m eval.compare --current eval/runs/v1.json   # 0/1/2
```

| 层 | 覆盖什么 |
|---|---|
| 契约（`test_api` / `test_rag_ask` / `test_agent_api`） | 状态码、错误码、响应体四键 |
| 服务（`test_chat_service` / `test_agent_service`） | 工具循环、记忆、HITL、并发串行 |
| 组件（`test_tools_registry` / `test_llm_client` / `test_lc_client`） | AST 白名单、异常翻译、参数映射 |
| RAG（`test_rag_splitter` / `test_rag_vectorstore` / `test_rag_corpus`） | 切分偏移、幂等写入、语料完整性 |
| 设施（`test_ratelimit` / `test_sse` / `test_store` / `test_logging`） | 窗口边界、帧格式、WAL、脱敏 |

压测（`benchmark/`，11-10 实测）：`probe` 928 RPS @c=20（P50 14ms / P99 104ms）；
`chat` P50 1304ms；流式 TTFB P50 609ms。**自研开销只占端到端 1~2%，瓶颈在上游**。

## 目录结构

```
src/app/
├── main.py              装配：lifespan / 中间件 / 静态页
├── core/                配置、依赖注入、错误契约、日志、限流、SSE 帧
├── schema/              Pydantic 契约（就是 /docs 的源码）
├── client/              LLM 适配（手写 + LangChain 两种实现）
├── service/             业务编排（chat / rag / agent）
├── router/              接口层，只做协议转换
├── rag/                 加载、切分、向量库、检索、重排、提示词
├── tools/               工具注册表（AST 白名单）
├── agent/               状态图与 checkpointer
├── mcp_server/          MCP Server 与路径白名单
├── obs/                 本地 trace 与 Langfuse 工厂
├── db/                  SQLite 会话存储
└── static/              演示页（v1.html / index.html / rag.html）
eval/                    四套金标与门禁
scripts/                 切分、建索引、预热、MCP 自测、部署（ops.py + 4 个封装）
tests/                   242 例
```

## License

MIT
