"""keke-code-v1 모델을 HF Hub에 업로드.

repo: hysnnnn/keke-code-v1

준비:
    huggingface-cli login (또는 HF_TOKEN 환경변수)

실행:
    uv run python push_keke_code.py --model-dir kor-minish-bge-m3-ko-v2  # 테스트
    uv run python push_keke_code.py  # 기본: keke-code-v1/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, whoami

DEFAULT_REPO = "hysnnnn/keke-code-v1"
DEFAULT_DIR = "keke-code-v1"

MODEL_CARD = """---
language:
- ko
- en
- code
library_name: model2vec
license: mit
pipeline_tag: sentence-similarity
tags:
- model2vec
- static-embeddings
- korean
- code
- multilingual
- polyglot
- sentence-embeddings
- semble
base_model:
- Snowflake/snowflake-arctic-embed-l-v2.0
---

# keke-code-v1

다국어 코드 + 한국어 통합 정적 임베딩 (model2vec). 영어 코드 임베딩과 한국어 자연어 임베딩을 **한 모델로** 처리.

기존 [`minishlab/potion-code-16M`](https://huggingface.co/minishlab/potion-code-16M)이 영어 6개 언어 코드만 지원했다면, `keke-code-v1`은:
- **Rust, HTML, CSS 추가** (총 9개 코드 언어)
- **한국어 (자연어)** 추가
- **한국어 docstring → 코드** retrieval 가능 (unique)

## 사용처

- [semble_rs](https://github.com/johunsang/semble_rs) — Rust 기반 코드 검색 CLI
- AI 에이전트(Claude Code, Codex, Cursor)의 1차 검색
- 한국어 docstring 검색
- 한국어 commit message → 관련 코드 retrieval
- 일반 한국어 의미 유사도 (FAQ, 챗봇 매칭)

## Specs

- **Base teacher**: Snowflake/snowflake-arctic-embed-l-v2.0 (568M)
- **Output dim**: 256 (PCA from teacher dim)
- **Vocab**: ~500K (코드 토큰 ~200K + 한국어 ~50K + 영어/공통 ~250K)
- **Size**: ~500 MB
- **Pooling**: weighted mean (SIF)

## Languages

| 한국어 자연어 | 코드 (10 언어) |
|---|---|
| ✓ 일반 의미 유사도 | Python, Java, JavaScript, Go, PHP, Ruby, **Rust**, **HTML**, **CSS** |
| ✓ FAQ/CS 매칭 | + 자연어 ↔ 코드 매칭 |
| ✓ 비격식체·SNS 표현 | + 한국어 ↔ 코드 매칭 (unique) |

## Usage

```bash
pip install model2vec
```

```python
from model2vec import StaticModel

model = StaticModel.from_pretrained("hysnnnn/keke-code-v1")

# 한국어 ↔ 한국어
vecs = model.encode(["환불 어떻게 받아요", "결제 취소"])

# 한국어 ↔ 코드 (unique!)
vecs = model.encode([
    "파일 읽는 함수",
    "def read_file(path):\\n    with open(path) as f:\\n        return f.read()",
])

# 영어 ↔ 코드
vecs = model.encode([
    "how to async fetch in javascript",
    "async function fetchData() { return await fetch('/api'); }",
])
```

## 학습 데이터

| 도메인 | 데이터셋 | 양 |
|---|---|---|
| 한국어 일반 | KLUE-STS, KLUE-NLI, PAWS-X | 58K |
| 한국어 대화 | AI Hub 011 일상대화 + AI Hub 020 SNS | ~400K (샘플) |
| 영어 코드 | nomic-ai/CornStack 6언어 | ~600K (언어당 100K) |
| 신규 코드 | The Stack v2 Rust/HTML/CSS (docstring 페어) | ~150K |

총 약 **120만 쌍** fine-tune.

## 평가 — MTEB-CoIR + 한국어

| Benchmark | potion-code-16M | **keke-code-v1** |
|---|---|---|
| MTEB-CoIR avg | 37.05 | **TBD** |
| KLUE-STS | — | **TBD** |
| 자체 CS FAQ (Top-1) | — | **TBD** |

## Pipeline 재현

전체 학습 파이프라인 공개: https://github.com/johunsang/kor-minish

```bash
python download_v4_data.py
python train_keke_code_runpod.py     # H200 약 8시간
python benchmark_keke_code.py        # MTEB-CoIR + 한국어
```

## 한계

- 정적 임베딩 본질적 한계 — context 이해 못함, 다의어 처리 어려움
- 진짜 똑똑한 챗봇은 cross-encoder reranker (별도 모델) 추가 권장
- 코드는 영어 docstring 중심 학습 (한국어 docstring은 합성 데이터 부분)

## Citation

```bibtex
@software{kor_minish_2026,
  author = {johunsang},
  title = {keke-code-v1: Korean + Polyglot Code Static Embeddings},
  url = {https://github.com/johunsang/kor-minish},
}
```

Base: [Snowflake/snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) (Apache 2.0)
Framework: [model2vec](https://github.com/MinishLab/model2vec)
Inspired by: [potion-code-16M](https://huggingface.co/minishlab/potion-code-16M)

## License

MIT
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--model-dir", default=DEFAULT_DIR)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    try:
        user = whoami()
        print(f"logged in as: {user['name']}")
    except Exception as e:
        raise SystemExit(
            "HF login 안 됨. `huggingface-cli login` 또는 HF_TOKEN 환경변수.\n"
            f"원본: {e}"
        )

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"model dir not found: {model_dir.resolve()}")

    # Model card 작성
    (model_dir / "README.md").write_text(MODEL_CARD, encoding="utf-8")
    print(f"model card → {model_dir / 'README.md'}")

    # Repo 생성 (없으면)
    api = HfApi()
    api.create_repo(args.repo, private=args.private, exist_ok=True)
    print(f"repo ready: {args.repo}")

    # 업로드
    print(f"uploading {model_dir} ...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo,
        commit_message="Initial upload — keke-code-v1",
    )
    print(f"\nhttps://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
