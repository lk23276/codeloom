#!/usr/bin/env sh
# CodeLoom 启动入口（Linux / macOS / Git Bash）
set -eu
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "[run] 未找到 uv。先运行 scripts/setup.sh" >&2
  exit 1
fi

for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    exec "$candidate" scripts/ops.py run "$@"
  fi
done

exec uv run --no-project python scripts/ops.py run "$@"
