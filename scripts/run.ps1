# CodeLoom 启动入口（Windows PowerShell 5.1+）
#
# 用法：powershell -ExecutionPolicy Bypass -File scripts\run.ps1
#        powershell -ExecutionPolicy Bypass -File scripts\run.ps1 --verify
# 这里只负责找到解释器，真正的逻辑在 scripts/ops.py。

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[run] 未找到 uv。先运行 scripts\setup.ps1" -ForegroundColor Red
    exit 1
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    & python scripts\ops.py run @args
} else {
    & uv run --no-project python scripts\ops.py run @args
}
exit $LASTEXITCODE
