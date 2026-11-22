#!/usr/bin/env sh
# CodeLoom 部署入口（Linux / macOS / Git Bash）
#
# 这里只做一件事：找到能跑 Python 的解释器，把活交给 scripts/ops.py。
# 真正的逻辑只有一份实现，避免两个平台各写一套导致行为漂移。
set -eu
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] 未找到 uv。安装后重开终端再试：" >&2
  echo "        curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  echo "        装好后确认：uv --version" >&2
  exit 1
fi

for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    exec "$candidate" scripts/ops.py deploy "$@"
  fi
done

# 系统里没有 Python：让 uv 临时提供一个来跑脚本（不影响项目环境）
exec uv run --no-project python scripts/ops.py deploy "$@"
