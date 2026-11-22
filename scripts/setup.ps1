# CodeLoom 部署入口（Windows PowerShell 5.1+）
#
# 用法：powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# 这里只负责找到解释器，真正的逻辑在 scripts/ops.py。

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "[setup] 未找到 uv。安装后重开终端再试：" -ForegroundColor Red
    Write-Host '        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
    Write-Host "        装好后确认：uv --version"
    exit 1
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    & python scripts\ops.py deploy @args
} else {
    & uv run --no-project python scripts\ops.py deploy @args
}
exit $LASTEXITCODE
