# syntax=docker/dockerfile:1
# ══════════════════════════════════════════════════════════════════════════
#  CodeLoom v0.1 运行镜像
#
#  构建：  docker build -t codeloom:latest .
#  运行：  docker run --rm -p 8000:8000 -v codeloom-data:/data \
#                       -e APP_LLM_API_KEY=sk-... codeloom:latest
#  验收：  curl -s localhost:8000/healthz
#          curl -s localhost:8000/readyz
#
#  四条约束来自真容器里的实测，不是抄模板：
#
#    ① 基础镜像必须是 3.13-slim —— pyproject 写着 requires-python >= 3.13，
#       用 3.12 会在 uv sync 那步直接失败。
#    ② 依赖层与源码层分开（先 COPY 清单、uv sync，再 COPY src）——
#       改一行业务代码不会触发重装依赖，构建从几分钟回到十几秒。
#    ③ 属主用 COPY --chown 一次写定，绝不「拷完再 chown -R」——
#       后者会把 venv 里每个文件都标记为「变了」，overlayfs 整层复制一遍：
#       实测多出 113MB / 多花 39 秒。
#    ④ --workers 1，两个原因：
#       · SQLite 的连接锁是 threading.RLock，只在单进程内有效
#       · 限流器是进程内 dict，多 worker 会变成 N 倍额度
#       要横向扩容，先把这两处换成共享存储（Postgres / Redis）。
# ══════════════════════════════════════════════════════════════════════════

# ── 阶段一：只负责把第三方依赖装进 /app/.venv ─────────────────────────────
FROM python:3.13-slim AS deps

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_HTTP_TIMEOUT=120

WORKDIR /app

# uv 版本锁死：uv.lock 由 0.12.5 生成，换版本解析结果可能不同。
# 走清华源：国内直连 pypi.org 会卡住；换源只改「从哪下」，不改版本（版本由 uv.lock 定）。
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple "uv==0.12.5"

# ★★ 顺序是全文件最重要的东西：先只拷依赖清单。
#    --no-install-project：这一步不装本项目（源码还没进来），只装依赖。
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ── 阶段二：运行镜像 ────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    APP_DB_PATH=/data/sessions.db \
    APP_LOG_FORMAT=json

WORKDIR /app

# ★ 用户必须先建：下面的 COPY --chown 按名字解析，用户不存在会直接构建失败
RUN useradd --create-home --uid 10001 appuser

# 依赖：属主在写入时就定好，避免二次 chown 复制整层（见文件头 ③）
COPY --from=deps --chown=appuser:appuser /app/.venv /app/.venv

# 源码：归 root、appuser 只读 —— 运行期不需要改代码，改代码应该重新构建镜像。
# ★ 用 PYTHONPATH=/app/src 而不是在镜像里再 build 一次 wheel：
#   · 少一步构建，也少一个「构建期需要 README.md」的隐藏依赖
#   · 目录结构和仓库里完全一致，容器里读日志/调试时不用换算路径
# ★ 这里拷的是**整个 src/app**，不是「用到哪几个文件拷哪几个」——
#   踩过的坑：运行时 import 闭包（函数内 import）静态查不出来，漏一个就启动即崩。
COPY src ./src

# 运行时要写的唯一位置。空目录，chown 是 O(1)，不会像 venv 那样遍历几万个文件。
RUN mkdir -p /data && chown appuser:appuser /data

USER appuser

EXPOSE 8000

# 探针用 /healthz（存活），不用 /readyz：
#   readyz 查 DB，DB 短暂被锁时不该让 Docker 把整个容器重启掉 —— 重启治不了锁。
#   摘流量的判断交给编排系统（k8s 的 readinessProbe 才该打 /readyz）。
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import sys,urllib.request; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

# --no-access-log：访问日志由我们的中间件输出（带 request_id 和耗时，JSON 格式），
#                  留着 uvicorn 自己的会每条打两遍。
# --workers 1：见文件头 ④
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--no-access-log"]
