"""v4 (polyglot-code-ko) 학습 데이터 일괄 다운로드.

HF 데이터셋:
- CornStack 6언어 (Python/Java/JS/Go/PHP/Ruby) — potion-code-16M 학습 데이터
- The Stack v2 Rust/HTML/CSS subset
- code_search_net (전통 baseline)
- bigcode/commitpackft (커밋 + 코드, 다양 언어)

저장: /Volumes/SAMSUNG/hf_cache_v2 (외장하드)
"""
from __future__ import annotations

import os
from pathlib import Path

HF_CACHE = Path("/Volumes/SAMSUNG/hf_cache_v2")
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE / "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")

from datasets import load_dataset  # noqa: E402

OUT_INFO = []


def try_load(name: str, *args, **kwargs):
    """try-except로 다운로드 + 정보 수집."""
    try:
        ds = load_dataset(name, *args, **kwargs, streaming=False)
        if hasattr(ds, "keys"):
            for split in ds.keys():
                n = len(ds[split])
                OUT_INFO.append((f"{name}[{split}]", n))
                print(f"  OK  {name}[{split}]: {n:,}")
        else:
            OUT_INFO.append((name, len(ds)))
            print(f"  OK  {name}: {len(ds):,}")
        return ds
    except Exception as e:
        print(f"  FAIL {name}: {str(e)[:120]}")
        return None


def main():
    print("=" * 60)
    print("[1] CornStack 6개 언어 (potion 학습 데이터)")
    print("=" * 60)
    for lang in ["python", "java", "javascript", "go", "php", "ruby"]:
        try_load(f"nomic-ai/CornStack-{lang}-v1")

    print()
    print("=" * 60)
    print("[2] code_search_net (전통 baseline, 6언어)")
    print("=" * 60)
    for lang in ["python", "java", "javascript", "go", "php", "ruby"]:
        try_load("code_search_net", lang)

    print()
    print("=" * 60)
    print("[3] The Stack v2 — Rust / HTML / CSS subset")
    print("=" * 60)
    for lang_dir in ["Rust", "HTML", "CSS"]:
        try_load(
            "bigcode/the-stack-v2-dedup",
            data_files=f"data/{lang_dir}/*.parquet",
            split="train",
        )

    print()
    print("=" * 60)
    print("[4] CommitPack (커밋 메시지 + 코드)")
    print("=" * 60)
    try_load("bigcode/commitpackft")

    print()
    print("=" * 60)
    print("[5] (선택) 다국어 NLI/STS 보강")
    print("=" * 60)
    try_load("MoritzLaurer/multilingual-NLI-26lang-2mil7")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    total = sum(n for _, n in OUT_INFO)
    print(f"total rows downloaded: {total:,}")
    for name, n in sorted(OUT_INFO, key=lambda x: -x[1])[:20]:
        print(f"  {n:>12,}  {name}")


if __name__ == "__main__":
    main()
