# CodeLoom（码络）· 可溯源的多 Agent 工程知识助手

> **一句话**：面向**自有工程语料**（源码 + 项目文档）的多 Agent 问答服务。
> 卖点不是「用了 LangChain」，而是**用服务器思维把 LLM 服务做稳**：分层边界、流式、限流、
> trace-id 日志、双探针、一条命令起容器 —— 以及答案可以溯源到文件行号。

当前状态：**v0.5 完成**（检索增强已可用）。多 Agent 在 v1.0。

| 版本 | 范围 | 状态 |
|---|---|---|
| v0.1 | `/chat` + `/chat/stream`(SSE) + SQLite 多轮 + 1 个本地工具 + 限流 + 结构化日志 + Docker | ✅ |
| v0.5 | 语料加载 → 切分 → bge Embedding → Chroma → 重排 → 文件行号引用 → 评测基线 | ✅ |
| v1.0 | LangGraph Supervisor + checkpointer + 自研 MCP Server + Langfuse trace + step limit | 计划中 |


---

## 与 server2 的关系

```
server2  = 自己造的 C++ 高并发传输层  -> 懂 epoll / Reactor / 背压 / 优雅关闭 / 压测
CodeLoom = 在这套底层认知上盖的 Python + LLM 应用层
```

| server2 的能力 | 在 CodeLoom 的落点 |
|---|---|
| 主从 Reactor / epoll ET | `asyncio` 事件循环（uvicorn） |
| EPOLLOUT 写半包 / 抽干 | SSE 生产快于消费时的背压与慢客户端 |
| 信号量 / 锁 | 滑动窗口限流 + `asyncio.to_thread` 串行化 SQLite |
| signalfd 优雅关闭 | `lifespan` 关闭时释放数据库连接 |
| MSG_NOSIGNAL / 对端断开 | 客户端断流 → `finally: await stream.aclose()` 关上游（省钱） |
| 压测方法论 | 后续压测表复用同一套（wrk2 / pidstat / fd） |

---

## 架构

```
                  +---------------------------------------------+
  curl / 前端 ---> | uvicorn (workers=1)                         |
                  |   middleware: request_id + 结构化 access 日志 |
                  +----------------------+----------------------+
                                         v
                  +---------------------------------------------+
                  | router/   /chat   /chat/stream              |
                  |           /sessions*   /healthz   /readyz   |
                  +----------------------+----------------------+
                                         v
                  +---------------------------------------------+
                  | service/  ChatService                       |
                  |   会话编排 / FC 循环(<=3 步) / 异常翻译        |
                  +---------+----------------------+------------+
                            v                      v
                  +-------------------+   +----------------------+
                  | client/llm.py     |   | tools/registry.py    |
                  | 唯一 import openai |   | ast 白名单计算器      |
                  +---------+---------+   +----------------------+
                            v
                      DeepSeek API

                  db/store.py   SQLite(WAL)：会话 + 消息
```

分层规则（被破坏时不会报错，只会逐步失效）：

- `service/` 不许 `import fastapi`、不许 `import openai`、不许写 SQL
- `client/` 是唯一 import `openai` 的地方
- `tools/` 不认识 HTTP，也不知道 SDK
- `rag/` 是唯一接触向量库与重排模型的地方，`service/` 只认识它的 Protocol
- 校验命令：`grep -rnE "^\s*(from|import)\s+(fastapi|openai|chromadb|fastembed)" src/app/service/` 应无输出
  （只匹配真实 import 行；docstring 里提到的名字不算）

> 检索链路（v0.5）在对话链路之外单独成一条：
> `router/rag.py` → `service/rag_service.py` → `rag/`（Chroma 检索 + cross-encoder 重排）。
> 它与对话链路共用 `client/llm.py`，不改变 `/chat` 的既有行为。

---

## 核心特性

- **流式问答**：`/chat/stream` 按 `meta -> delta... -> done` 推送；`X-Accel-Buffering: no` 防中间层缓冲
- **多轮记忆**：SQLite 落库，`session_id` 续聊；未知 id 返回 404，**不静默新建**
- **Function Calling**：本地计算器（`ast` 白名单，**禁用 eval**），最多 3 步，工具结果全部转成文本
- **限流**：滑动窗口（无边界突刺），按客户端 IP 分桶 + LRU 上限，超限 429 + `Retry-After`
- **结构化日志**：一行一个 JSON，`request_id` 用 `ContextVar` 贯穿中间件/业务/工具层
- **统一错误体**：`{code, msg, data, request_id}`，业务码与 HTTP 状态同源
- **双探针**：`/healthz` 不碰依赖（存活），`/readyz` 查 DB（就绪）
- **容器化**：多阶段构建、非 root、`/data` 卷、镜像内 HEALTHCHECK
- **可溯源问答**（v0.5）：`/rag/ask` 检索自有语料，回答带 `文件:行号` 引用；依据不足时直接拒答，不产生模型调用

---

## 检索增强（v0.5）

> 本节说明实现与指标。需要先跑通，见「快速开始」；只查接口字段，见「接口」。

把自有工程语料变成可检索、可引用的知识库。链路是三段式，顺序不可交换：

```
139 个块 ──向量粗召回 top-20──> 候选 ──cross-encoder 精排 top-5──> 上下文
                                                        │
                                      最高分 < 阈值 ────┴──> 直接拒答（不调用模型）
                                                        └──> 注入 [n] 文件:行号 ──> LLM ──> 回答 + 引用
```

粗召回负责不漏，精排负责排准。**精排只能重排候选集里已有的块** —— 候选里没有正确答案时，
再准的重排也无法补救，所以 `Hit@recall` 是整条链路的天花板。

### 建立索引

```bash
uv run python scripts/build_corpus.py    # 语料 -> 块（带 source 与行区间）
uv run python scripts/index_corpus.py    # 块 -> 向量库（按稳定 id 幂等写入）
```

实测：7 个文件 / 2237 行 → **139 块**（平均 495 字），入库 18.2 秒、索引 2.6 MB。
embedding 模型 91 MB，重排模型 1.1 GB，均缓存在 `data/fastembed`（`data/` 已被忽略，不入库）。

> 两个模型都通过 `huggingface_hub` 下载。`scripts/index_corpus.py` 内置镜像配置；
> 运行期加载重排模型前，同样需要设置 `HF_ENDPOINT` 与 `NO_PROXY`，
> 否则首次请求会卡在下载。建议在部署阶段预下载。

### 调用

```bash
curl -s -X POST localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"eventfd 是怎么用来唤醒事件循环的"}'
```

响应节选（真实调用）：

```json
{
  "code": "OK",
  "data": {
    "refused": false,
    "answer": "在 `EventLoop` 构造时创建 `eventfd`，包装成 Channel 注册到本反应堆 [2]……（正文节选）",
    "citations": [
      {"index": 2, "source": "server2.cpp", "line_start": 359, "line_end": 368, "score": 4.0964},
      {"index": 3, "source": "server2.cpp", "line_start": 345, "line_end": 362, "score": 3.6527}
    ],
    "invalid_citations": 0,
    "top_score": 5.5801,
    "recalled": 20,
    "reranked": 5,
    "latency_ms": 9876
  },
  "request_id": "……"
}
```

三个字段需要单独说明：

- `citations` 的位置由服务端从块 metadata 映射得到。模型只输出 `[n]`，**不写文件名或行号** ——
  它没有原文的偏移信息，写出来的路径无法校验。
- `invalid_citations` 统计模型引用了不存在的编号。非零表示提示词或块数异常，不会让请求失败。
- `refused=true` 时 HTTP 仍为 200，且 `usage` 全为 0：拒答发生在调用模型之前。

`max_tokens` 默认 800。上限只限制长度、不产生额外调用；判据是 `completion_tokens` 正好等于上限，
此时正文通常停在半句或代码块中间（300 与 500 都实测过截断）。

### 评测基线

```bash
uv run python -m eval.run                 # 不调用模型，同输入结果一致
uv run python -m eval.run --with-llm      # 追加引用精度（8 次真实调用）
```

基线文件：`eval/baseline.json` 是可复现基线（不含生成指标），回归对比以它为准；
`eval/baseline_with_llm.json` 是含引用精度的一次快照，受采样影响，只作参考。

| 指标 | 值 | 口径 |
|---|---|---|
| Hit@recall | **100%** | 金标块是否落在粗召回 top-20 内（链路天花板） |
| Hit@3 | **87.5%**（7/8） | 精排后 top-3 是否含金标块 |
| MRR@3 | **0.729** | 首个命中排名倒数的均值 |
| 拒答正确率 | **100%**（7/7） | 无关题被拒的比例 |
| 误拒率 | **0%**（0/8） | 相关题被误拒的比例，必须与上一行同看 |
| 引用精度 | **82.6%**（19/23） | 引用指向 `gold_sources` 的比例（来源级口径） |

`Hit@recall = 100%` 说明召回不是瓶颈；`Hit@3` 与 `MRR@3` 反映的是排序质量。

### 一个负面结果：重排不是净收益

引入 cross-encoder 后逐题对比（同一份金标、同一个阈值；排名为 1 起始，超过 5 记作「出前 5」）：

| 用例 | 纯向量排名（top-20 内） | 重排后排名 | 变化 |
|---|---:|---:|---|
| L03 边沿触发 | 9 | 1 | 修好 |
| L01 半包续发 | 3 | 出前 5 | 变差 |
| L04 零拷贝 | 1 | 3 | 变差 |
| L08 定量结论 | 1 | 2 | 变差 |

结果是 `Hit@3` 不变（失败用例从 L03 换成 L01），而 **MRR 由 0.792 降到 0.729**。
一个可解释的原因是：重排模型偏好字面重合，「续发」与 `handleWrite` 之间缺少直接对应；
而向量检索本就把该块排在前三。该解释尚未验证 —— 4 个样本不足以支撑因果结论。

决策：**不在这 8 条题上调整参数或更换模型**。小样本下调阈值、改权重、换重排模型，
得到的是对样本的拟合，换一批题即失效。下一步是把定位题扩充到 30 条以上再判断净收益；
候选方案是 `bge-reranker-v2-m3-int8`（0.57 GB，多语，体积为当前的一半）。

### 拒答阈值

重排输出是**原始 logits**（未过 sigmoid），可以为负，也可以远大于 1，因此阈值必须在标注集上标定：

```bash
uv run python scripts/calibrate_rerank.py
```

实测（8 条相关题 + 7 条无关题，其中 3 条是同领域但语料未覆盖的难负例）：

| 类别 | 最高分范围 |
|---|---|
| 相关 | 1.06 ～ 6.62 |
| 无关（易：行情 / 烹饪 / 宠物 / 影视） | −8.89 ～ −4.13 |
| 无关（难：Kafka / Redis / MySQL） | −3.52 ～ −2.59 |

可用区间为 **(−2.59, 1.06)**，默认值 0.2 落在区间内，误拒 0/8、挡掉 7/7。
阈值上调到 2.0 开始误拒（1/8），3.0 时误拒 3/8 —— 不应为了「拒得更干净」而上调，
拒答正确率必须与误拒率一起看。

---

## 快速开始

```bash
uv sync                                  # 安装依赖（Python >= 3.13）
cp .env.example .env                     # 填 APP_LLM_API_KEY
uv run uvicorn app.main:app --reload --port 8000 --no-access-log
```

不用 key 也能起：探针与会话接口正常，`/chat*` 返回 502；
`/rag/ask` 只有在需要生成时才返回 502，拒答路径（`refused=true`）不依赖模型。

```bash
# 非流式
curl -s -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"用一句话介绍 epoll"}'

# 流式（★ -N 不能省，否则 curl 会缓冲，看起来像"没有流"）
curl -N -X POST localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"你好"}'

# 探针
curl -s localhost:8000/healthz ; echo
curl -s localhost:8000/readyz  ; echo
```

> 默认 `max_tokens=200` 是**成本闸门**：长回答会被截断（正文停在半句或半个代码块中间，
> 日志里 `completion_tokens` 正好等于 200）。要完整答案就传 `"max_tokens": 600`。

### Windows PowerShell 验收

1. `curl` 是 `Invoke-WebRequest` 的别名，不支持 `-N` → 一律写 **`curl.exe`**
2. `Invoke-RestMethod` 在响应头没有 charset 时按 **ISO-8859-1** 解码，中文会变成 `æ¯ Linux`
   （JSON 按 RFC 8259 恒为 UTF-8，是 PS 5.1 的问题；服务端**不该**为此加非标准的 `charset`）
3. `-Body '中文字符串'` 会把非 ASCII 变成 `?` —— 实测入库是 `??A:?????`，
   模型只看到 `epoll` 这个 ASCII 词，回答看似合理，实际已丢掉原问题

正确姿势：请求体写 **UTF-8 无 BOM 文件**，响应落文件后按 UTF-8 读。

```powershell
[IO.File]::WriteAllText("$PWD\req.json", '{"message":"用一句话介绍 epoll","max_tokens":600}',
                        [Text.UTF8Encoding]::new($false))

curl.exe -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" `
  --data-binary "@req.json" -o resp.json
(Get-Content resp.json -Encoding UTF8 -Raw | ConvertFrom-Json).data.reply

# 流式：先切控制台到 UTF-8，否则中文 token 在 cmd 里是乱码
chcp 65001 > $null
curl.exe -N -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" `
  --data-binary "@req.json"
```

> 想一条命令看全流程（探针 / 非流式 / SSE / trace id），用 Git Bash 或 WSL 跑
> `bash scripts/smoke.sh` —— 它全部由 bash 实现，可避开上述三处问题。

### 浏览器演示页

服务起来后打开 `http://127.0.0.1:8000/demo/`（容器映射的 8000 也一样）：

- 输入一句话 → **逐字渲染**，页面上直接显示「首字耗时 / 总耗时 / delta 帧数」
- 页面和接口**同一个 origin**（`GET /demo/` 由同一个 FastAPI 提供），所以
  `fetch('/chat/stream')` 不会触发 CORS —— 这是不需要 Vite/React 的原因
- 单文件、零依赖：`src/app/static/index.html`；`html=True` 让 `/demo/` 直接返回它

![CodeLoom 流式演示](docs/demo.gif)

### 一次真实调用

```bash
$ curl -s -X POST localhost:8000/chat -H "Content-Type: application/json" \
       -d '{"message":"用一句话介绍 epoll"}' | jq '.data'
{
  "reply": "epoll 是 Linux 提供的一种 I/O 多路复用机制：内核维护一个事件就绪列表……",
  "session_id": "fb259cb188714172953e5cba6b735c5b",
  "usage": {"prompt_tokens": 444, "completion_tokens": 114, "total_tokens": 558}
}
```

```bash
$ curl -N -X POST localhost:8000/chat/stream -H "Content-Type: application/json" \
       -d '{"message":"你好"}'
event: meta
data: {"session_id": "4cd9cbfb44344d77a83c6d1dffee3721", "model": "deepseek-flash"}

event: delta
data: 你好

event: delta
data: ！

... （同一句共 179 个 delta 帧，此处省略）

event: done
data: [DONE]
id: end
```

> 179 这个数字比任何注释都能证明「真流式」：上游按小块推，我们是收一块转一块。
> 上方动图就是同一次请求的录屏：首字 597 ms、总耗时 1343 ms、77 个 delta 帧。

### Docker

```bash
docker build -t codeloom:latest .

# ★ 用 -d --name 而不是 --rm：容器退出时要能看到日志和退出码
# ★ --restart unless-stopped：WSL 空闲会关发行版、dockerd 跟着停，
#   没有重启策略的话容器就一直是 Exited，得手动 docker start
docker run -d --name codeloom --restart unless-stopped \
  -p 8000:8000 \
  -v codeloom-data:/data \
  -e APP_LLM_API_KEY=sk-... \
  codeloom:latest
```

> 镜像里 `APP_DB_PATH=/data/sessions.db`；`/data` 必须挂卷，否则容器一删会话就没了。
>
> 更稳的传 key 方式：`printf 'APP_LLM_API_KEY=%s\n' 'sk-...' > ~/.codeloom.env && chmod 600 ~/.codeloom.env`，
> 然后 `--env-file ~/.codeloom.env`。★ 文件必须是 **LF** 换行（`printf` 天然是），
> CRLF 会把 `\r` 带进值里，表现为 401。

---

## 接口

| # | 方法 | 路径 | 作用 | 限流 |
|---:|---|---|---|---|
| 1 | POST | `/chat` | 一次性问答 | ✅ |
| 2 | POST | `/chat/stream` | SSE 流式问答 | ✅ |
| 3 | GET | `/sessions` | 会话列表 | ❌ |
| 4 | GET | `/sessions/{id}/messages` | 会话全部消息 | ❌ |
| 5 | DELETE | `/sessions/{id}` | 删除会话（消息级联） | ❌ |
| 6 | GET | `/healthz` | 存活探针（不碰依赖） | ❌ |
| 7 | GET | `/readyz` | 就绪探针（查 DB） | ❌ |
| 8 | POST | `/rag/ask` | 基于本地语料回答，带引用与拒答 | ✅ |

> 另有 `GET /demo/`：同源静态演示页（单文件 HTML，不计入 API 端点，也不出现在 `/docs` 的 OpenAPI 里）。

`/rag/ask` 的字段、引用映射规则、拒答语义与评测口径见「检索增强（v0.5）」一节。
其中两条容易误读：拒答返回 200（它是正常结果，不是请求错误）；
`citations` 的位置来自块 metadata，不是模型输出。

请求体（`/chat` 与 `/chat/stream` 相同）：

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `message` | string | 1–2000 | 用户这一轮说的话 |
| `session_id` | string \| null | 1–64 | 不传新建；传了续聊；**不存在 → 404** |
| `history_limit` | int | 1–200，默认 20 | 带多少条历史（越少越省钱） |
| `max_tokens` | int | 1–1000，默认 200 | 输出上限 |

成功响应统一为 `{code, msg, data, request_id}`；错误码：

| code | HTTP | 场景 |
|---|---|---|
| `VALIDATION_ERROR` | 422 | 请求体格式不对 |
| `NOT_FOUND` | 404 | 会话不存在 |
| `RATE_LIMITED` | 429 | 超限，带 `Retry-After` |
| `UPSTREAM_ERROR` | 502 | 模型服务不可用 |
| `INTERNAL_ERROR` | 500 | 本项目自身缺陷（堆栈只进日志） |

### SSE 事件

```
event: meta       data: {"session_id":"...","model":"deepseek-flash"}
event: delta      data: 逐 token 正文（可能几十上百帧）
: ping            注释帧，长时间无 token 时保活
event: error      data: {"code":"UPSTREAM_ERROR","msg":"..."}   仅出错时
event: done       data: [DONE]
```

`done` 一定在最后：客户端靠它停止读取 —— 靠「连接关闭」表示结束，分不清正常结束和断网。

---

## 配置

全部来自环境变量（`.env` 不入库，`.env.example` 是唯一文档）。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `APP_LLM_API_KEY` | 空 | 为空时服务可启动，但 `/chat*` 返回 502 |
| `APP_LLM_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容端点 |
| `APP_LLM_MODEL` | `deepseek-flash` | 用官方正名，不用旧别名 |
| `APP_LLM_BACKEND` | `openai` | `openai` 手写适配层 / `langchain` LCEL 适配层，对外契约相同（见 [ADR-7](docs/adr/ADR-7-langchain-boundary.md)） |
| `APP_LLM_THINKING` | `false` | 思考模式默认关（省 5~14 倍输出 token） |
| `APP_TIMEOUT_S` | `30` | 单次调用超时 |
| `APP_MAX_RETRIES` | `2` | 预留（v0.1 未接重试，见已知限制） |
| `APP_DB_PATH` | `./data/sessions.db` | 容器内是 `/data/sessions.db` |
| `APP_RAG_PERSIST_DIR` | `./data/chroma` | 向量库目录，由 `scripts/index_corpus.py` 重建 |
| `APP_RAG_EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | 512 维、约 90 MB，走 ONNX，不需要 PyTorch |
| `APP_RAG_TOP_K` | `5` | 精排后进入上下文的块数 |
| `APP_RAG_RECALL_K` | `20` | 粗召回条数，必须大于 `TOP_K` |
| `APP_RAG_RERANK_MODEL` | `BAAI/bge-reranker-base` | cross-encoder，约 1.1 GB，首次使用下载 |
| `APP_RAG_MIN_RERANK_SCORE` | `0.2` | 拒答阈值（原始 logits），需在标注集上标定 |
| `APP_RATE_LIMIT` | `30` | 每窗口请求数 |
| `APP_RATE_WINDOW_S` | `60` | 窗口秒数 |
| `APP_LOG_LEVEL` | `INFO` | — |
| `APP_LOG_FORMAT` | `json` | `json` 生产 / `text` 本地开发 |

---

## 设计决策

| 决策 | 为什么 | 代价 |
|---|---|---|
| **SSE 而不是 WebSocket** | 单向推送足够；SSE 走普通 HTTP，代理/网关/鉴权链路不用改 | 不能客户端中途插话 |
| **流式不落库** | 半句话入库会污染历史；`astream` 也拿不到 usage | 流式聊完历史只有 user 半边；成本统计缺流式 |
| **未知 `session_id` → 404** | `INSERT OR IGNORE` 会把「拼错 id」伪装成 200 并新开会话 | 会话被清理后旧 id 不能自动续上 |
| **`--workers 1`** | SQLite 的锁与限流器都是进程内的 | 只能单机纵向扩容 |
| **滑动窗口限流** | 固定窗口在边界会放过 2 倍流量；本项目不想给突发额度 | 每 key 存 N 个时间戳 |
| **不信任 `X-Forwarded-For`** | 该头可伪造，信它等于关掉限流 | 部署在代理后需额外配置可信代理 |
| **JSON 日志 + ContextVar** | 采集端按字段查，不解析字符串；不污染函数签名 | 人工直接读不如文本友好（`APP_LOG_FORMAT=text` 兜底） |
| **FC 最多 3 步** | 防止模型无限循环调用工具，使调用成本不可控 | 复杂任务直接给兜底文案 |

---

## 已知限制（真实边界）

1. **流式不落库**：用 `/chat/stream` 聊完，历史里只有 user；下一轮非流式会带上这条「孤儿 user」。流式也没有 usage，成本统计只覆盖非流式。
2. **限流是进程内的**：`--workers N` 下额度变成 N 倍；扩容前要换 Redis。按 IP 计数，NAT 后的用户共享额度。
3. **单写者 SQLite**：并发写会排队（`busy_timeout=5s`），适合单机小流量。
4. **没有鉴权**：任何能访问端口的人都能调 `/chat` 与 `/rag/ask`；上公网前必须加 API Key / 网关门禁。
5. **重试未接**：`APP_MAX_RETRIES` 只是配置占位，`OpenAICompatClient` 显式关掉了 SDK 重试但没接 tenacity。
6. **工具只有一个**：计算器。多 Agent 与 MCP 在 v1.0。
7. **压测是单机下限**：每组只有 10 个样本、客户端与服务端同机，P99 实际等于最大值；不代表集群容量（见上文「压测」一节）。
8. **检索是单路的**：只有向量召回 + 单模型精排，没有 BM25 混合、query 改写、分块重切。
9. **引用精度按来源文件判定**：同一个文件的不同块都算命中，因此 82.6% 这个数字偏乐观。
10. **评测样本小**：8 条定位题 + 7 条无关题，只支持看趋势，不支持据此调参。
11. **重排模型常驻 1.1 GB**：`--workers 1` 下每进程一份；横向扩容前要重新估算内存。
12. **`/rag/ask` 无会话记忆**：每次独立检索，不带历史，也不落库。
13. **拒答会随语料变化**：阈值是在当前语料与当前重排模型上标定的，换模型或大改语料后需重新标定。

---

## 测试

```bash
uv run pytest -q                                   # 160 passed
uv run pytest --cov --cov-report=term-missing      # TOTAL 92%
uv run ruff check . && uv run ruff format --check .
```

覆盖重点不是行数，而是**不会报错的错误**：

- 五步顺序（先落库再调模型）、usage 逐轮累加、未知 id 不静默新建
- SSE 帧顺序、404 不能变 200、真流式（门控替身，假流式会超时）
- FC：`tool_call_id` 一一配对、步数上限、工具失败不打断循环
- 限流：边界不放过 2 倍突发、`Retry-After` 精确、LRU 淘汰
- 日志：一行一个 JSON、trace id 贯穿、500 也带 id
- RAG：阈值拒答不产生模型调用、引用按精排顺序映射、越界编号只计数、
  分数与候选数量不一致时立即失败（错位会静默答错文件）

> 关键用例做过**故障注入**验证：把实现改坏（id 写死 / usage 不累加 / 步数 +1 / `except` 收窄 / 去掉截断 / `extra=forbid` 改 `ignore`），对应用例都会失败。

---

## 压测

环境：Windows 本机 / `uvicorn --workers 1` / 关环并发 / 限流关闭 / 日志 WARNING。
完整口径、局限与复跑方式见 [`benchmark/report.md`](benchmark/report.md)。

| 组 | 并发 | 请求 | RPS | P50 | P95 | P99 | 关键指标 |
|---|---:|---:|---:|---:|---:|---:|---|
| `probe` `/healthz` | 20 | 9297 | **928** | 14 ms | 66 ms | 104 ms | 纯本地，不碰依赖 |
| `chat` `/chat` | 2 | 10 | 1.5 | **1304 ms** | 1569 ms | 1569 ms | token 4490 + 683 |
| `stream` `/chat/stream` | 2 | 10 | 1.8 | 1091 ms | 1286 ms | 1286 ms | **TTFB P50 609 ms**，delta 均值 51 |

三个结论：

1. **我们的开销只占端到端 1~2%**：`/healthz` P50 14 ms vs `/chat` 1304 ms、`/stream` TTFB 609 ms。
   优化业务代码不会让聊天变快，瓶颈在上游生成。
2. **流式把首字延迟砍半**：609 ms vs 1304 ms。这是产品指标，不是服务器指标。
3. **`/chat` 没有排队**：`RPS ≈ 并发 / 延迟`（1.5 ≈ 2/1.304）。
   等式成立说明服务端没有积压；低于这个值才说明有排队。

> 局限：每组只有 10 个样本（P99 等于最大值）、同机压测（客户端与服务端抢 CPU）、
> 单进程、Windows 定时器精度约 15 ms。数字是**单机下限**，不是集群容量。

---

## 目录结构

```
codeloom/
├── Dockerfile / .dockerignore
├── pyproject.toml / uv.lock
├── docs/demo.gif               # README 演示动图
├── src/app/
│   ├── main.py                 # create_app + lifespan + request_id 中间件
│   ├── core/                   # config / deps / errors / ratelimit / logging / sse
│   ├── router/                 # chat / sessions / health / rag
│   ├── service/                # chat_service（五步 + FC 循环 + 流式）/ rag_service（五步检索链路）
│   ├── client/                 # llm.py 手写适配层 / lc_client.py LCEL 适配层
│   ├── rag/                    # loader / splitter / vectorstore / retriever / rerank / prompt
│   ├── tools/registry.py       # 工具 schema + 派发 + ast 白名单计算器
│   ├── db/store.py             # SQLite(WAL) 会话与消息
│   ├── schema/                 # chat.py / rag.py（Pydantic 契约）
│   ├── prompts/system.py       # 对话系统提示词
│   ├── static/index.html       # 演示页（/demo/，同源，零构建）
│   └── agent|mcp_server|obs    # v1.0 预留
├── tests/                      # 160 例
├── benchmark/                  # 压测：loadgen.py（probe/chat/stream）+ run.sh
├── scripts/                    # smoke.sh / build_corpus / index_corpus / eval_retrieval
│                               # calibrate_rerank / compare_backends
└── eval/                       # run.py（六项指标）+ gold/（金标）+ baseline*.json
```

---

## License

MIT，见 [LICENSE](LICENSE)。
