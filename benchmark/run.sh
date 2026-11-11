#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
#  CodeLoom 压测一键跑（三组：本地能力 / 端到端 / 流式）
#
#  用法：
#      bash benchmark/run.sh                       # 默认 http://127.0.0.1:8001
#      bash benchmark/run.sh http://127.0.0.1:8010
#
#  前置：单独起一个【压测专用】服务，不要打生产容器：
#
#      APP_RATE_LIMIT=100000 APP_DB_PATH=./data/bench.db APP_LOG_LEVEL=WARNING \
#      uv run uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 1 --no-access-log
#
#  产物：benchmark/results/<时间戳>/{probe,chat,stream}.json + report.md
# ══════════════════════════════════════════════════════════════════════
set -uo pipefail

# ★ Git Bash（MSYS2）会把 `--path /sessions` 静默改写成 Windows 路径
#   （实测变成 D:/.../PortableGit/sessions，因为该目录存在），
#   结果是 5 万次 404。关掉路径转换，脚本自己保证以 `/` 开头。
export MSYS_NO_PATHCONV=1

BASE="${1:-http://127.0.0.1:8001}"
UV="${UV:-uv}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="benchmark/results/$STAMP"
mkdir -p "$OUT"

echo "== 服务可达性 =="
if ! curl -s --max-time 5 "$BASE/healthz" > /dev/null; then
    echo "FAIL: $BASE/healthz 不可达。先按文件头的命令起服务。"
    exit 1
fi
echo "OK: $BASE"

run_group() {
    local name="$1"; shift
    echo
    echo "== $name =="
    # 单组失败不让整轮中止：压测的价值就在于「哪组红了」
    "$UV" run python benchmark/loadgen.py "$@" \
        --url "$BASE" --label "$name" \
        --out-json "$OUT/$name.json" --out-md "$OUT/$name.md" \
        || echo "（$name 组有失败项，见上方输出）"
}

# ① 本地能力：10 秒、20 并发。看框架与 SQLite，不受上游影响
run_group probe probe --path /healthz --duration 10 --concurrency 20 --warmup 5

# ② 本地能力（带存储）：会话列表，验证 SQLite 读
run_group sessions probe --path /sessions --duration 5 --concurrency 10 --warmup 3

# ③ 端到端：真模型，默认 10 发（约 1~3 分钱），并发 2
run_group chat chat --requests 10 --concurrency 2 --max-tokens 100 --warmup 1

# ④ 流式：真模型，10 发，量 TTFB 与 delta 间隔
run_group stream stream --requests 10 --concurrency 2 --max-tokens 100 --warmup 1

{
    echo "# CodeLoom 压测报告"
    echo
    echo "| 项 | 值 |"
    echo "|---|---|"
    echo "| 时间 | $(date '+%Y-%m-%d %H:%M:%S %Z') |"
    echo "| 服务 | $BASE |"
    echo "| 代码 | $(git rev-parse --short HEAD 2>/dev/null || echo '未提交') |"
    echo "| 机器 | $(uname -srm 2>/dev/null || echo 'unknown') |"
    echo
    echo "| 组 | 并发 | 请求 | RPS | 成功 | 429 | 其他错误 | P50(ms) | P95(ms) | P99(ms) | Max(ms) | 备注 |"
    echo "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    for f in "$OUT"/probe.md "$OUT"/sessions.md "$OUT"/chat.md "$OUT"/stream.md; do
        [ -f "$f" ] && cat "$f"
    done
    echo
    echo "## 怎么读"
    echo
    echo "- **probe / sessions**：RPS 与 P99 反映框架和 SQLite；P99 远高于 P50 说明有排队（连接池/单写锁）。"
    echo "- **chat**：延迟主要是上游生成时间，不是我们的代码；看 429 与错误数，看 token 总量估算成本。"
    echo "- **stream**：TTFB 是「用户看到第一个字」的时间；delta 均值为 1 说明是假流式；"
    echo "  delta 均值几十到几百、且间隔 P95 稳定，才是真流式。"
    echo
    echo '> 本机是单进程 `--workers 1`（SQLite 锁与限流器都是进程内的），'
    echo "> 所以数字代表「单机能力」，不是集群吞吐。"
} > "$OUT/report.md"

echo
echo "产物：$OUT/report.md"
cat "$OUT/report.md"
