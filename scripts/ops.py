#!/usr/bin/env python3
"""CodeLoom 的部署与启动助手（跨平台，仅用标准库）。

用法（通过封装脚本调用，也可直接跑）：

    scripts/setup.sh          # 部署：装依赖 -> 准备 .env -> 建索引 -> 自检
    scripts/run.sh            # 启动：找空闲端口 -> 起服务 -> 打印入口
    scripts/run.sh --verify   # 启动后立刻跑一遍冒烟

★ 为什么逻辑放在 Python 而不是两份 shell：
  bash 与 PowerShell 的差异（引号、变量、错误处理）足以让两个版本行为漂移。
  这里只把「自举 uv」交给 shell，其余全部落在同一份实现上。

★ 关于 uv：
  默认只检测 + 提示安装命令，**不自动安装**（改动机器上的工具链应显式同意）。
  装好后如果当前终端还找不到，提示重开终端即可。

★ 关于密钥：
  只判断「是否为空 / 是否是占位符」，不打印任何片段，也不复制到别处。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
ENV_EXAMPLE = ROOT / ".env.example"
CORPUS_DIR = ROOT / "data" / "corpus" / "raw"
CHROMA_DIR = ROOT / "data" / "chroma"

EXIT_OK = 0
EXIT_ENV = 1
EXIT_CONFIG = 2

IS_WINDOWS = os.name == "nt"


# ══════════════════════ 基础设施 ══════════════════════


def say(msg: str = "") -> None:
    print(msg, flush=True)


def step(index: int, total: int, msg: str) -> None:
    say(f"[{index}/{total}] {msg}")


def ok(msg: str) -> None:
    say(f"      OK   {msg}")


def warn(msg: str) -> None:
    say(f"      WARN {msg}")


def fail(msg: str) -> None:
    say(f"      FAIL {msg}")


def find_uv() -> str | None:
    """找 uv。先在 PATH 里找，再看它的默认安装位置（刚装完可能还没进 PATH）。"""
    found = shutil.which("uv")
    if found:
        return found
    candidates = [
        Path.home() / ".local" / "bin" / ("uv.exe" if IS_WINDOWS else "uv"),
        Path.home() / ".cargo" / "bin" / ("uv.exe" if IS_WINDOWS else "uv"),
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def uv_install_hint() -> str:
    if IS_WINDOWS:
        return "powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""
    return "curl -LsSf https://astral.sh/uv/install.sh | sh"


def run(
    cmd: list[str],
    *,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        check=False,
        text=True,
        capture_output=capture,
        env={**os.environ, **(env or {})},
    )


def py(code: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """在项目环境里跑一小段 Python（用于自检与索引探活）。"""
    return run(["uv", "run", "python", "-c", code], capture=True, env=env)


# ══════════════════════ .env 处理 ══════════════════════

PLACEHOLDERS = {"", "sk-xxx", "sk-not-configured", "your-key", "changeme", "todo"}


def read_key_state(env_file: Path) -> tuple[str, bool]:
    """返回 (状态, 是否可用)。状态只有几种固定描述，**不带任何密钥片段**。"""
    if not env_file.exists():
        return "文件不存在", False

    text = env_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^APP_LLM_API_KEY=(.*)$", text, re.MULTILINE)
    if not match:
        return "未设置 APP_LLM_API_KEY", False

    value = match.group(1).strip().strip('"').strip("'")
    if value.lower() in PLACEHOLDERS:
        return "APP_LLM_API_KEY 为空或仍是占位符", False
    return "APP_LLM_API_KEY 已配置", True


def ensure_env_file(env_file: Path) -> bool:
    """不存在则从模板复制；已存在**绝不覆盖**。返回是否新建。"""
    if env_file.exists():
        return False
    if not ENV_EXAMPLE.exists():
        return False
    shutil.copyfile(ENV_EXAMPLE, env_file)
    return True


# ══════════════════════ 索引探活 ══════════════════════

PROBE = """
import sys
try:
    from app.core.config import Settings
    from app.rag.retriever import get_store
    store = get_store(Settings())
    n = store._collection.count()
    hits = len(store.similarity_search_with_score("eventfd 唤醒", k=1))
    print(f"{n} {hits}")
except Exception as exc:
    print(f"ERR {type(exc).__name__}: {exc}")
    sys.exit(3)
"""


def index_state() -> tuple[str, int, int]:
    """返回 (状态, 条数, 命中数)。状态：ok / empty / broken / missing。"""
    if not CHROMA_DIR.exists():
        return "missing", 0, 0

    result = py(PROBE)
    if result.returncode != 0:
        return "broken", 0, 0

    try:
        count, hits = (int(x) for x in result.stdout.strip().split()[-2:])
    except ValueError:
        return "broken", 0, 0

    if count == 0:
        return "empty", 0, 0
    if hits == 0:
        # 有数据但检索不出来：今天遇到过（建索引中途失败），不能只看 count
        return "broken", count, 0
    return "ok", count, hits


def build_index(rebuild: bool) -> int:
    args = ["uv", "run", "python", "scripts/build_corpus.py"]
    code = run(args).returncode
    if code != 0:
        return code

    index_args = ["uv", "run", "python", "scripts/index_corpus.py"]
    if rebuild:
        index_args.append("--rebuild")
    return run(index_args).returncode


# ══════════════════════ HTTP ══════════════════════


def http_json(url: str, payload: dict | None = None, timeout: float = 30.0) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"raw": body[:200]}
    except urllib.error.URLError as exc:
        return 0, {"error": str(exc.reason)}


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_ready(port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, _ = http_json(f"http://127.0.0.1:{port}/healthz", timeout=2)
        if status == 200:
            return True
        time.sleep(0.5)
    return False


# ══════════════════════ 子命令：deploy ══════════════════════


def cmd_deploy(args: argparse.Namespace) -> int:
    total = 8
    env_file = Path(args.env_file).resolve() if args.env_file else ENV_FILE
    child_env = {"HF_ENDPOINT": "https://hf-mirror.com"} if args.mirror else {}

    say("CodeLoom 部署")
    say(f"  项目目录 {ROOT}")
    say()

    # 1 磁盘
    step(1, total, "检查磁盘空间")
    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    if free_gb < 2:
        warn(f"剩余 {free_gb:.1f} GB，建议 ≥ 2 GB（重排模型约 1.1 GB）")
    else:
        ok(f"剩余 {free_gb:.1f} GB")

    # 2 uv
    step(2, total, "检查 uv")
    uv = find_uv()
    if not uv:
        fail("未找到 uv")
        say("      安装后重开终端，再运行一次本脚本：")
        say(f"        {uv_install_hint()}")
        return EXIT_ENV
    version = run([uv, "--version"], capture=True).stdout.strip()
    ok(version or "uv 可用")

    # 3 依赖
    step(3, total, "安装依赖（uv sync）")
    sync = run(["uv", "sync"], env=child_env)
    if sync.returncode != 0:
        fail("uv sync 失败；若锁文件与 pyproject 不一致，先执行 uv lock")
        return EXIT_ENV
    ok("依赖已就绪")

    # 4 .env
    step(4, total, "准备环境变量文件")
    created = ensure_env_file(env_file)
    if created:
        ok(f"已从模板生成 {env_file.name}")
    else:
        ok(f"沿用已有 {env_file.name}（不覆盖）")

    # 5 key
    step(5, total, "检查模型 key")
    key_state, key_ok = read_key_state(env_file)
    if not key_ok:
        fail(key_state)
        say(f"      编辑 {env_file.name} 填入 APP_LLM_API_KEY，然后重新运行本脚本")
        say("      （依赖与索引的步骤已完成，重跑会直接跳过）")
        return EXIT_CONFIG
    ok(key_state)

    # 6 语料
    step(6, total, "检查语料")
    files = [p for p in CORPUS_DIR.glob("*") if p.is_file()] if CORPUS_DIR.exists() else []
    if not files:
        fail(f"{CORPUS_DIR.relative_to(ROOT)} 为空")
        say("      把源码/文档放进该目录，或重新 clone 以获取示例语料")
        return EXIT_ENV
    ok(f"{len(files)} 个文件")

    # 7 索引
    step(7, total, "检查向量索引")
    state, count, hits = index_state()
    if state == "ok" and not args.rebuild_index:
        ok(f"{count} 块，检索可用")
    else:
        if state == "broken":
            warn(f"{count} 块但检索返回 0 —— 索引不可用，重建")
        elif state == "missing":
            say("      索引不存在，开始构建（首次需下载约 90 MB 的向量模型）")
        else:
            say("      索引为空，开始构建")
        if build_index(rebuild=True) != 0:
            fail("索引构建失败")
            return EXIT_ENV
        state, count, hits = index_state()
        if state != "ok":
            fail(f"构建后仍不可用（state={state}, count={count}, hits={hits}）")
            return EXIT_ENV
        ok(f"{count} 块，检索可用")

    # 8 自检
    step(8, total, "自检（导入应用）")
    check = py("import app; print(app.VERSION)", env=child_env)
    if check.returncode != 0:
        fail("导入失败：" + (check.stderr or "").strip().splitlines()[-1][:120])
        return EXIT_ENV
    ok(f"app 可导入，版本 {check.stdout.strip()}")

    say()
    say("部署完成。下一步：")
    say(f"  {Path('scripts') / ('run.ps1' if IS_WINDOWS else 'run.sh')}          启动服务")
    say(f"  {Path('scripts') / ('run.ps1' if IS_WINDOWS else 'run.sh')} --verify 启动并冒烟")
    say("  页面：http://127.0.0.1:8000/demo/v1.html")
    return EXIT_OK


# ══════════════════════ 子命令：run ══════════════════════


def pick_port(preferred: int, explicit: bool) -> int | None:
    if not port_in_use(preferred):
        return preferred
    if explicit:
        return None
    for candidate in range(preferred + 1, preferred + 11):
        if not port_in_use(candidate):
            return candidate
    return None


def cmd_run(args: argparse.Namespace) -> int:
    uv = find_uv()
    if not uv:
        fail("未找到 uv；先运行部署脚本")
        return EXIT_ENV

    env_file = Path(args.env_file).resolve() if args.env_file else ENV_FILE
    key_state, key_ok = read_key_state(env_file)
    if not key_ok:
        warn(f"{key_state} —— 探针与拒答可用，/chat 会返回 502")

    port = pick_port(args.port, args.port_explicit)
    if port is None:
        if args.port_explicit:
            fail(f"端口 {args.port} 已被占用")
            say(f"      换一个：--port {args.port + 1}（不传 --port 时会自动往后找）")
        else:
            fail(f"端口 {args.port}~{args.port + 10} 均被占用")
        return EXIT_ENV
    if port != args.port:
        say(f"端口 {args.port} 被占用，改用 {port}")

    child_env = {"HF_ENDPOINT": "https://hf-mirror.com"} if args.mirror else {}
    cmd = [
        uv, "run", "uvicorn", "app.main:app",
        "--host", "127.0.0.1", "--port", str(port),
        "--workers", "1", "--no-access-log",
    ]
    say(f"启动：127.0.0.1:{port}（Ctrl+C 停止）")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env={**os.environ, **child_env})
    try:
        if not wait_ready(port, timeout=args.timeout):
            fail("服务在超时内没有就绪；请查看上面的启动日志")
            return EXIT_ENV

        base = f"http://127.0.0.1:{port}"
        say()
        say("就绪。入口：")
        say(f"  演示台      {base}/demo/v1.html")
        say(f"  流式对话    {base}/demo/")
        say(f"  知识库      {base}/demo/rag.html")
        say(f"  接口文档    {base}/docs")
        say(f"  探针        {base}/healthz  {base}/readyz")
        say()

        if args.open:
            webbrowser.open(f"{base}/demo/v1.html")
        if args.verify:
            code = run_verify(base, with_rag=not args.no_rag)
            if code != EXIT_OK:
                warn("冒烟未全部通过（服务仍在运行）")

        say("按 Ctrl+C 停止服务")
        proc.wait()
        return EXIT_OK
    except KeyboardInterrupt:
        say()
        say("正在停止…")
        return EXIT_OK
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


# ══════════════════════ 子命令：verify ══════════════════════


def run_verify(base: str, *, with_rag: bool = True, timeout: float = 300.0) -> int:
    rows: list[tuple[str, bool, str]] = []

    status, body = http_json(f"{base}/healthz", timeout=5)
    rows.append(("/healthz", status == 200, f"HTTP {status}"))

    status, body = http_json(f"{base}/readyz", timeout=30)
    rows.append(("/readyz", status == 200, f"HTTP {status}"))

    status, body = http_json(
        f"{base}/chat", {"message": "只回复两个字：可用", "max_tokens": 50}, timeout=timeout
    )
    reply = (body.get("data") or {}).get("reply", "")
    rows.append(("/chat", status == 200 and bool(reply), f"HTTP {status} reply={reply[:12]!r}"))

    if with_rag:
        status, body = http_json(
            f"{base}/rag/ask", {"question": "为什么要用 eventfd 唤醒"}, timeout=timeout
        )
        data = body.get("data") or {}
        rows.append(
            (
                "/rag/ask",
                status == 200 and "refused" in data,
                f"HTTP {status} refused={data.get('refused')} "
                f"cites={len(data.get('citations') or [])}",
            )
        )

    status, body = http_json(f"{base}/agent", {"message": "12*7 等于几"}, timeout=timeout)
    data = body.get("data") or {}
    answered = status == 200 and data.get("status") == "answered"
    rows.append(("/agent", answered, f"HTTP {status} status={data.get('status')}"))

    status, body = http_json(
        f"{base}/agent", {"message": "3*5 等于几", "hitl": True}, timeout=timeout
    )
    data = body.get("data") or {}
    paused = data.get("status") == "awaiting_approval" and data.get("tool_calls") == []
    rows.append(
        ("/agent (hitl)",
         status == 200 and paused, f"HTTP {status} status={data.get('status')}"))

    say("冒烟结果")
    say(f"  {'检查项':<14}{'结果':<8}明细")
    for name, passed, detail in rows:
        say(f"  {name:<14}{'OK' if passed else 'FAIL':<8}{detail}")

    failed = [name for name, passed, _ in rows if not passed]
    say()
    if failed:
        fail(f"{len(failed)} 项未通过：{', '.join(failed)}")
        return EXIT_ENV
    ok("全部通过")
    return EXIT_OK


def cmd_verify(args: argparse.Namespace) -> int:
    base = args.base_url.rstrip("/")
    if not base.startswith("http"):
        base = f"http://127.0.0.1:{args.port}"
    return run_verify(base, with_rag=not args.no_rag)


# ══════════════════════ 子命令：doctor ══════════════════════


def cmd_doctor(args: argparse.Namespace) -> int:
    env_file = Path(args.env_file).resolve() if args.env_file else ENV_FILE
    uv = find_uv()
    state, count, hits = index_state() if uv else ("unknown", 0, 0)
    key_state, key_ok = read_key_state(env_file)

    rows = [
        ("项目目录", str(ROOT)),
        ("Python", sys.version.split()[0]),
        ("uv", run([uv, "--version"], capture=True).stdout.strip() if uv else "未找到"),
        ("依赖环境", "已就绪" if (ROOT / ".venv").exists() else "缺失（运行 setup）"),
        ("环境文件", f"{env_file.name} " + ("存在" if env_file.exists() else "缺失")),
        ("模型 key", key_state),
        ("向量索引", f"{state}（{count} 块，探活命中 {hits}）"),
        ("服务状态", f"端口 8000 {'被占用' if port_in_use(8000) else '空闲'}"),
    ]
    for name, value in rows:
        say(f"  {name:<10}{value}")

    if not key_ok:
        say()
        warn("key 未配置：/chat、/rag/ask 的生成步骤会返回 502；拒答路径仍可用")
    return EXIT_OK


# ══════════════════════ CLI ══════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ops.py",
        description="CodeLoom 部署与启动助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/ops.py deploy --mirror\n"
            "  python scripts/ops.py run --port 8000 --open\n"
            "  python scripts/ops.py verify --port 8000\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    deploy = sub.add_parser("deploy", help="安装依赖、准备 .env、建索引、自检")
    deploy.add_argument("--mirror", action="store_true", help="使用 HF 镜像（国内网络建议开启）")
    deploy.add_argument("--rebuild-index", action="store_true", help="强制重建向量索引")
    deploy.add_argument("--env-file", default="", help="自定义 .env 路径（默认项目根 .env）")
    deploy.set_defaults(func=cmd_deploy)

    run_p = sub.add_parser("run", help="启动服务并打印入口")
    run_p.add_argument("--port", type=int, default=8000, help="端口，默认 8000")
    run_p.add_argument("--open", action="store_true", help="就绪后打开浏览器")
    run_p.add_argument("--verify", action="store_true", help="就绪后跑一遍冒烟")
    run_p.add_argument("--no-rag", action="store_true", help="冒烟时跳过 /rag/ask（省一次生成）")
    run_p.add_argument("--mirror", action="store_true", help="使用 HF 镜像")
    run_p.add_argument("--timeout", type=float, default=60.0, help="等待就绪的秒数")
    run_p.add_argument("--env-file", default="", help="自定义 .env 路径")
    run_p.set_defaults(func=cmd_run, port_explicit=False)

    verify = sub.add_parser("verify", help="对已运行的服务跑冒烟")
    verify.add_argument("--port", type=int, default=8000)
    verify.add_argument("--base-url", default="", help="完整基址，例如 http://127.0.0.1:8000")
    verify.add_argument("--no-rag", action="store_true", help="跳过 /rag/ask")
    verify.set_defaults(func=cmd_verify)

    doctor = sub.add_parser("doctor", help="打印环境体检结果")
    doctor.add_argument("--env-file", default="")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        # 用户显式给了 --port 时，端口占用就直接报错，不静默换端口
        args.port_explicit = "--port" in (argv if argv is not None else sys.argv[1:])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
