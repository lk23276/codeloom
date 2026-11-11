# 压测说明（benchmark/）

> 目标不是「刷一个好看的 QPS」，而是**能回答三个问题**：
> ① 本地能力（框架 + SQLite）能扛多少；② 端到端延迟里，多少是我们的、多少是上游的；③ 流式是真流式还是攒完再发。

---

## 1. 快速开始

**第一步：起一个压测专用服务**（别打线上容器，也别用默认限流）

```bash
cd codeloom          # 换成克隆后的目录

APP_RATE_LIMIT=100000 \
APP_DB_PATH=./data/bench.db \
APP_LOG_LEVEL=WARNING \
uv run uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 1 --no-access-log
```

三个参数的理由：

| 参数 | 为什么 |
|---|---|
| `APP_RATE_LIMIT=100000` | 默认 30/分钟，压测会被自己的限流挡住，测出来全是 429 |
| `APP_DB_PATH=./data/bench.db` | 别把压测会话写进演示库 |
| `APP_LOG_LEVEL=WARNING` | 否则每条请求一行 JSON，日志本身成为瓶颈 |

**第二步：另开一个终端跑**

```bash
bash benchmark/run.sh                       # 默认 http://127.0.0.1:8001
```

或者单组跑：

```bash
uv run python benchmark/loadgen.py probe  --url http://127.0.0.1:8001 --duration 10 --concurrency 20
uv run python benchmark/loadgen.py chat   --url http://127.0.0.1:8001 --requests 10 --concurrency 2
uv run python benchmark/loadgen.py stream --url http://127.0.0.1:8001 --requests 10 --concurrency 2
```

产物：`benchmark/results/<时间戳>/{probe,sessions,chat,stream}.json` + `report.md`。

---

## 2. 三组口径

| 组 | 打什么 | 量什么 | 回答什么问题 |
|---|---|---|---|
| `probe` | `GET /healthz` | RPS / P50 / P95 / P99 | 框架本身能扛多少（不碰依赖） |
| `probe --path /sessions` | `GET /sessions` | 同上 | SQLite 读 + 序列化开销 |
| `chat` | `POST /chat` | 延迟 / 429 / token | 端到端里上游占多少；成本多少 |
| `stream` | `POST /chat/stream` | **TTFB** / delta 数 / delta 间隔 | 首字延迟；是不是真流式 |

**为什么 `chat` 的 P99 主要由上游决定**：一次非流式请求要等模型生成完（实测约 2.3 秒），
我们的代码只占其中几毫秒。所以这组数字的意义是「端到端体验」与「成本」，不是「我们的吞吐」。

**为什么流式要看 TTFB 而不是总时长**：用户感知的是「多久看到第一个字」，
总时长由生成长度决定，跟实现质量无关。

---

## 3. 成本（务必看）

| 组 | 默认量 | 估算 |
|---|---:|---|
| `chat` | 10 发 × `max_tokens=100` | 约 1~3 分钱（按 deepseek-flash 量级） |
| `stream` | 10 发 × `max_tokens=100` | 同上 |
| `probe` / `sessions` | 纯本地，不计费 | 0 |

- 想先验通路：`--requests 2 --concurrency 1`
- **不要**给 `chat/stream` 传 `--duration`：脚本会拒绝（时间模式会一直烧钱）
- 真实 token 数会写进 JSON 的 `summary.prompt_tokens / completion_tokens`，报告里直接可见

---

## 4. 结果怎么读

### 4.1 probe

```
RPS 高、P50 低、P99 略高   -> 正常（P99 是排队尾部）
P99 突然到几百 ms          -> 连接池不够 / 事件循环被阻塞（检查有没有同步 IO）
RPS 上不去且 CPU 很低      -> 客户端先到瓶颈（Python 压测器的极限，换 wrk2 复核）
```

### 4.2 chat

```
成功 10 / 429 0 / 错误 0   -> 通路正常
429 > 0                    -> 服务端限流没关（APP_RATE_LIMIT 没生效或不是 8001 那个进程）
502                        -> 上游不可用（key/网络），不是压测脚本的问题
P99 远大于 P50             -> 上游长尾（模型生成时间波动），不是我们的排队
```

### 4.3 stream

```
TTFB 200~800ms             -> 正常（首 token 时间）
delta 均值 1~3             -> 假流式（攒完再发），要查实现
delta 均值几十~几百        -> 真流式
delta 间隔 P95 很大        -> 上游按段推，或网络抖动
```

> 我们的实测基线：「你好」→ **179 个 delta**；`/chat` 约 2.3 秒、444/114 tokens。

---

## 5. 已知边界（写进报告，别藏着）

1. **单进程**：`--workers 1`，SQLite 锁与限流器都是进程内状态。数字代表单机，不是集群。
2. **压测器是 Python**：本机 probe 组大约在几千 RPS 量级会先撞到客户端瓶颈。
   要极限值用 `wrk2 -t4 -c200 -d30s -R2000 http://127.0.0.1:8001/healthz` 复核。
3. **关环而不是开环**：每个 worker 串行发，N 并发 = N 个在途请求。
   LLM 场景下开环（固定 RPS）会把排队藏在客户端，数字好看但失真。
4. **代理**：脚本默认 `trust_env=False`（不读 `HTTP_PROXY`）。
   这台机器配了代理，本地请求走代理会把 0.3ms 测成 200ms。
5. **限流组单独测**：想知道 429 行为，就别改 `APP_RATE_LIMIT`，
   用 `--requests 40 --concurrency 1` 打默认 30/分钟的服务，看 `retry_after_seen`。

---

## 6. 这组数据说明什么（一句话版）

> 压测分三组：本地能力、端到端、流式。
> 本地组用关环并发测框架与 SQLite；端到端组的瓶颈在上游生成，所以我只看 429 与 token 成本；
> 流式组看 TTFB 和 delta 间隔——delta 均值只有 1 就说明是假流式。
> 单进程是刻意的：SQLite 锁和限流器都是进程内状态，扩容要先换 Postgres 和 Redis。

---

## 7. 文件

| 文件 | 作用 |
|---|---|
| `loadgen.py` | 压测器（probe / chat / stream），输出 JSON + Markdown 行 |
| `run.sh` | 一键跑四组并汇总成 `report.md` |
| `results/` | 产物（JSON 不入库，报告可入库） |
