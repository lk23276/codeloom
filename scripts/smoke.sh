#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
#  CodeLoom 冒烟验收：探针 -> 非流式 -> 流式
#
#  用法：
#      bash scripts/smoke.sh                      # 默认 http://127.0.0.1:8000
#      bash scripts/smoke.sh http://127.0.0.1:8010
#
#  ★ 为什么要有这个脚本：手敲 curl 最容易死在两处 ——
#    ① 流式忘了 -N，看到的是「一次性蹦出来」，误判成没流式
#    ② Windows PowerShell 会吃掉 JSON 里的双引号，报 422 但看不出原因
#    脚本里全是 bash，两个问题都不存在。
#
#  退出码：0 = 全过；1 = 有失败（会打印是哪一步）
# ══════════════════════════════════════════════════════════════════════
set -uo pipefail

BASE="${1:-http://127.0.0.1:8000}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAILED=0

ok()   { printf '  OK   %s\n' "$1"; }
bad()  { printf '  FAIL %s\n' "$1"; FAILED=1; }
step() { printf '\n== %s ==\n' "$1"; }

step "0) 服务可达性（$BASE）"
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE/healthz" || true)"
if [ "$code" = "200" ]; then
    ok "/healthz 200"
else
    bad "/healthz -> ${code:-连接失败}（服务没起 / 端口不对）"
    echo "  提示：先确认服务在跑，再执行本脚本。"
    exit 1
fi

step "1) 两个探针"
curl -s "$BASE/healthz" -o "$TMP/h.json" -w '  /healthz %{http_code} ' ; cat "$TMP/h.json"; echo
curl -s "$BASE/readyz"  -o "$TMP/r.json" -w '  /readyz  %{http_code} ' ; cat "$TMP/r.json"; echo
grep -q '"db":"ok"' "$TMP/r.json" && ok "readyz 的 db=ok" || bad "readyz 没返回 db=ok（DB 不可用）"

step "2) 会话接口"
curl -s "$BASE/sessions" -o "$TMP/s.json" -w '  /sessions %{http_code} ' ; cat "$TMP/s.json"; echo

step "3) POST /chat（非流式，真调模型）"
code="$(curl -s -X POST "$BASE/chat" \
    -H 'Content-Type: application/json' \
    -d '{"message":"用一句话介绍 epoll"}' \
    -o "$TMP/chat.json" -w '%{http_code}' --max-time 120)"
echo "  HTTP $code"
cat "$TMP/chat.json"; echo
case "$code" in
    200) grep -q '"reply"' "$TMP/chat.json" \
             && ok "/chat 200 且有 reply" \
             || bad "200 但 body 里没有 reply" ;;
    429) bad "429 被限流（等窗口过去再试，或调大 APP_RATE_LIMIT）" ;;
    502) bad "502 上游不可用（多半是 APP_LLM_API_KEY 没配 / 网络不通）" ;;
    *)   bad "/chat 返回 $code" ;;
esac

step "4) POST /chat/stream（SSE）"
curl -sN -X POST "$BASE/chat/stream" \
    -H 'Content-Type: application/json' \
    -d '{"message":"你好"}' \
    -o "$TMP/stream.txt" --max-time 120
head -8 "$TMP/stream.txt" | sed 's/^/  /'
deltas="$(grep -c '^event: delta' "$TMP/stream.txt" || true)"
echo "  ...（delta 帧 $deltas 个）"
grep -q '^event: meta' "$TMP/stream.txt" && ok "meta 在" || bad "没有 meta 帧（前端拿不到 session_id）"
[ "${deltas:-0}" -gt 0 ] && ok "有 $deltas 个 delta 帧" || bad "一个 delta 都没有（不是真流式，或上游失败）"
if grep -q '^event: error' "$TMP/stream.txt"; then
    # 取 error 帧紧跟的那一行 data（不能用 tail，最后一行是 done 的 [DONE]）
    err="$(grep -A1 '^event: error' "$TMP/stream.txt" | grep '^data:' | head -1 | sed 's/^data: //')"
    bad "流内出现 error 事件：$err"
else
    ok "无 error 事件"
fi
grep -q '^event: done' "$TMP/stream.txt" && ok "done 在（客户端有明确结束信号）" || bad "没有 done 帧"

step "5) 响应头里的 trace id"
curl -s -D - -o /dev/null "$BASE/healthz" | grep -i '^x-request-id' | sed 's/^/  /' \
    || bad "没有 X-Request-ID 头"

printf '\n'
if [ "$FAILED" -eq 0 ]; then
    echo "全部通过。"
else
    echo "有失败项 —— 先看服务端日志（docker logs codeloom 或 uvicorn 终端）。"
fi
exit "$FAILED"
