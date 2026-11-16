"""构建向量索引：读 chunks.jsonl → 写入 Chroma → 校验。

用法：
    uv run python scripts/index_corpus.py                 # 增量（按 id 幂等）
    uv run python scripts/index_corpus.py --rebuild       # 删库重建
    uv run python scripts/index_corpus.py --fake          # 用假向量，跳过模型下载
    uv run python scripts/index_corpus.py --smoke "写半包续发在哪实现"

产物在 `data/chroma/`，已被 `.gitignore` 覆盖：索引是派生物，可由本脚本重建。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from app.core.config import Settings
from app.rag.vectorstore import (
    COLLECTION_NAME,
    build_embeddings,
    count_chunks,
    ensure_indexed,
    get_vectorstore,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_HF_MIRROR = "https://hf-mirror.com"


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_hf_endpoint(mirror: str) -> None:
    """配置模型下载地址。

    `huggingface.co` 在部分网络环境下无法直连，且本地代理对该域名常超时；
    镜像可以直连。`huggingface_hub` 在**导入时**读取 `HF_ENDPOINT`，
    因此这一步必须发生在加载模型之前 —— 本模块对 fastembed 是延迟导入，
    在 `main` 开头设置即可生效。

    `NO_PROXY` 也要带上镜像域名，否则请求仍会被代理接管。
    """
    if os.environ.get("HF_ENDPOINT"):
        print(f"      HF_ENDPOINT 已设置：{os.environ['HF_ENDPOINT']}")
        return

    os.environ["HF_ENDPOINT"] = mirror
    host = urlparse(mirror).hostname or ""
    if host:
        existing = os.environ.get("NO_PROXY", "")
        merged = f"{existing},{host}" if existing else host
        os.environ["NO_PROXY"] = merged
        os.environ["no_proxy"] = merged
    print(f"      HF_ENDPOINT={mirror}（可用 --no-mirror 关闭）")


def load_chunks(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建 RAG 向量索引")
    parser.add_argument("--chunks", default="data/corpus/chunks.jsonl", help="块文件路径")
    parser.add_argument("--rebuild", action="store_true", help="先删除已有索引再建")
    parser.add_argument("--fake", action="store_true", help="用假向量（不下载模型）")
    parser.add_argument("--no-mirror", action="store_true", help="不自动使用 HF 镜像")
    parser.add_argument("--smoke", default="", help="建库后跑一条检索验证")
    parser.add_argument("--k", type=int, default=3, help="冒烟检索返回条数")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Settings()
    chunks_path = _resolve(args.chunks)
    persist_dir = _resolve(cfg.rag_persist_dir)

    print("[1/4] 读取块")
    if not chunks_path.is_file():
        print(f"      失败：找不到 {chunks_path}（先跑 scripts/build_corpus.py）", file=sys.stderr)
        return 2
    chunks = load_chunks(chunks_path)
    print(f"      {chunks_path.name} -> {len(chunks)} 块")

    if args.rebuild and persist_dir.exists():
        print(f"      --rebuild：删除已有索引 {persist_dir}")
        shutil.rmtree(persist_dir)

    print("\n[2/4] 初始化 embedding")
    if args.fake:
        from langchain_core.embeddings import FakeEmbeddings

        embeddings = FakeEmbeddings(size=512)
        print("      FakeEmbeddings(size=512)：仅验证链路，检索结果无语义")
    else:
        if not args.no_mirror:
            configure_hf_endpoint(DEFAULT_HF_MIRROR)
        embeddings = build_embeddings(cfg)
        print(f"      {cfg.rag_embed_model}（首次使用会下载约 90 MB）")

    print("\n[3/4] 写入向量库")
    db = get_vectorstore(cfg, embeddings=embeddings, persist_dir=str(persist_dir))
    print(f"      {persist_dir} | collection={COLLECTION_NAME}")

    started = time.perf_counter()
    written = ensure_indexed(db, chunks)
    elapsed = time.perf_counter() - started
    print(f"      {written} 块写入完成，耗时 {elapsed:.1f}s")

    print("\n[4/4] 校验")
    count = count_chunks(db)
    print(f"      count={count}（块数 {len(chunks)}）")
    if count != len(chunks):
        print("      失败：数量不一致，可能混入了旧索引（用 --rebuild 重建）", file=sys.stderr)
        return 1

    ensure_indexed(db, chunks)
    again = count_chunks(db)
    print(f"      重跑一次后 count={again} -> {'幂等' if again == count else '出现重复'}")
    if again != count:
        print("      失败：未传稳定 id 或 id 冲突，索引已污染", file=sys.stderr)
        return 1

    if args.smoke:
        print(f"\n检索冒烟：{args.smoke}")
        # 直接用本次的 db，不经 retriever 单例：后者会按配置另建一个实例，
        # 用 --fake 建库时它仍会去加载真模型。
        for rank, (doc, distance) in enumerate(
            db.similarity_search_with_score(args.smoke, k=args.k), start=1
        ):
            meta = doc.metadata
            head = doc.page_content.splitlines()[0][:48] if doc.page_content else ""
            print(
                f"      {rank}. distance={distance:.4f}  "
                f"{meta['source']}:{meta['line_start']}-{meta['line_end']}"
            )
            print(f"         {head}")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
