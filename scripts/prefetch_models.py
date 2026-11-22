#!/usr/bin/env python3
"""预下载两个本地模型：embedding（约 90 MB）与重排（约 1.1 GB）。

用途：先把模型拉进缓存，避免「首个请求卡 70 秒然后 500」这种体验。

    python scripts/prefetch_models.py              # 默认走 HF 镜像
    python scripts/prefetch_models.py --no-mirror  # 直连 HuggingFace
    python scripts/prefetch_models.py --embedding-only

★ 为什么单独一个脚本而不是让服务启动时加载：
  加载 1.1 GB 模型会把启动时间从 1 秒拉到几十秒，探针期间会被编排系统判成「没起来」。
  预热应该是显式动作，不该藏在启动路径里。

★ 容器里的用法：
  docker exec codeloom python scripts/prefetch_models.py
  缓存落在 /home/appuser/.cache（挂卷后重建容器不用重下）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _size_mb(path: Path) -> str:
    if not path.exists():
        return "无"
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return f"{size / 1024**2:.0f}MB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="预下载本地模型")
    parser.add_argument("--no-mirror", action="store_true", help="不设置 HF 镜像")
    parser.add_argument("--embedding-only", action="store_true", help="只下向量模型")
    args = parser.parse_args(argv)

    if not args.no_mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        print(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}（--no-mirror 可关闭）")

    from app.core.config import Settings
    from app.rag.vectorstore import build_embeddings

    settings = Settings()

    started = time.perf_counter()
    print(f"[1/2] 向量模型 {settings.rag_embed_model}")
    try:
        embeddings = build_embeddings(settings)
        vector = embeddings.embed_query("预热")
    except Exception as exc:  # noqa: BLE001
        print(f"      失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        print("      检查网络，或设置 HF_ENDPOINT 后重试", file=sys.stderr)
        return 1
    print(f"      就绪：{len(vector)} 维，{time.perf_counter() - started:.1f}s")

    if not args.embedding_only:
        from app.rag.rerank import build_reranker

        started = time.perf_counter()
        print(f"[2/2] 重排模型 {settings.rag_rerank_model}")
        try:
            reranker = build_reranker(settings)
            scores = reranker.score("预热", ["预热文本"])
        except Exception as exc:  # noqa: BLE001
            print(f"      失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            print("      重排模型约 1.1 GB，网络不稳时建议先设 HF_ENDPOINT", file=sys.stderr)
            return 1
        print(f"      就绪：score={scores[0]:.3f}，{time.perf_counter() - started:.1f}s")

    # ★ 真实缓存位置由代码决定：`rag_persist_dir` 的同级 `fastembed` 目录
    #   （容器里是 /data/fastembed），不是 ~/.cache —— 打印错地方比不打印更误导。
    cache_dir = Path(settings.rag_persist_dir).parent / "fastembed"
    print(f"\n缓存目录：{cache_dir}（{_size_mb(cache_dir)}）")
    print(f"HF 辅助目录：{Path(os.environ.get('HOME', '.')) / '.cache'}")
    print("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
