"""kor-minish 모델을 HuggingFace Hub에 업로드.

사전 준비:
    1) https://huggingface.co/settings/tokens 에서 "write" 권한 토큰 발급
    2) 터미널에서 `huggingface-cli login` 실행 후 토큰 붙여넣기
    3) `uv run python push_to_hf.py` 실행
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, whoami

DEFAULT_REPO = "hysnnnn/kor-minish-bge-m3-ko"
DEFAULT_DIR = "kor-minish-bge-m3-ko"

MODEL_CARD = """---
language:
- ko
library_name: model2vec
license: mit
pipeline_tag: sentence-similarity
tags:
- model2vec
- static-embeddings
- korean
- sentence-embeddings
- bge-m3
base_model:
- BAAI/bge-m3
---

# kor-minish-bge-m3-ko

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)를 [model2vec](https://github.com/MinishLab/model2vec) 기법으로 **한국어 정적 임베딩(static embedding)** 으로 추출(distill)한 모델입니다. 한국어 vocab을 추가 주입해 위키 기반 일반 도메인 한국어에 강합니다.

- **속도**: CPU에서 sentence-transformers/bge-m3 대비 **수백 배 빠름** (정적 lookup + 평균 풀링)
- **크기**: ~140 MB (양자화하면 35 MB까지)
- **차원**: 256 (PCA로 1024→256 압축)
- **GPU 불필요**: 추론 시 GPU 없어도 됨. 모바일·엣지·서버리스에 적합

---

## 1. 왜 만들었나 (Motivation)

**bge-m3는 강력하지만 무겁습니다.** 568M 파라미터, 추론 시 GPU 거의 필수, 모바일·엣지에는 부담. 반면 한국어 RAG·검색·클러스터링에서는 *수백만 문서를 빠르게 훑어야* 하는 경우가 많고, 그때 매번 dense encoder를 돌리면 비용·지연이 폭증합니다.

**model2vec**은 이 간극을 메웁니다. 사전학습된 sentence transformer의 *지식*을 정적 임베딩 테이블 하나로 압축합니다.

- 토큰별 임베딩을 한 번 추출 → 평균/SIF 풀링 → 끝
- 추론 시 forward pass 없음. 단순 lookup + 평균
- 정확도는 원본 대비 70~80% 수준이지만, **속도·메모리는 100배+ 이득**

이 모델은 그 흐름을 한국어로 가져왔습니다.

---

## 2. 어떻게 만들었나 (Pipeline)

### 2-1. Base 모델 선정: BAAI/bge-m3

다국어 retrieval에서 가장 강력한 모델 중 하나. 한국어 단독 모델(`jhgan/ko-sroberta-multitask` 등)도 후보였지만, **bge-m3의 250K subword vocab과 다국어 retrieval 성능**이 distill 후에도 유리할 것으로 판단.

### 2-2. 한국어 vocab 추가 주입

bge-m3 기본 vocab만으로는 한국어 신조어·고유명사 커버리지가 부족합니다. 그래서:

1. **Korean Wikipedia 20231101.ko** (HuggingFace `wikimedia/wikipedia`)에서 200,000개 문서를 스트리밍
2. **kiwipiepy**로 형태소 분석, **NNG / NNP / NNB / NR** 태그(보통/고유/의존/수 명사)만 추출
3. 빈도순 **top 30,000**개를 vocab 후보로 저장

→ 최종 vocab: **278,203 토큰** (bge-m3 250K + 한국어 30K, 중복 제외)

> 명사만 골라낸 이유: 정적 임베딩은 단어의 *의미*를 담아야 하므로, 조사·어미 같은 기능 형태소는 노이즈가 됩니다. 동사·형용사는 어간이 변화해서 별도 처리가 필요해 1차 버전에서는 제외.

### 2-3. Distillation

`model2vec.distill()`로 다음 과정을 자동 수행:

1. 각 vocab 토큰을 bge-m3 인코더에 통과시켜 1024차원 임베딩 추출
2. **PCA로 256차원 압축** (1024→256, 정보 손실 ~5% 이내)
3. **SIF (Smooth Inverse Frequency) 가중치** 적용 — 흔한 토큰(조사 등)에 페널티
4. 정적 임베딩 테이블로 저장 (safetensors)

Colab T4 GPU에서 약 6분, 결과물 약 140 MB.

### 2-4. 결과 모델 구조

```
StaticModel
├── tokenizer (bge-m3 XLM-RoBERTa SentencePiece + 한국어 vocab)
├── embedding_table  (278,203 × 256, float32)
└── pooling (weighted mean, SIF)
```

**문장 → 토큰 ID → lookup → SIF 가중평균 → L2 정규화**. 이게 전부입니다. forward pass 없음.

---

## 3. 평가 결과의 의미

10쌍 동의/무관, 16개 카테고리 문장, 3개 retrieval 쿼리로 정성·정량 평가:

| Test | Result | 의미 |
|---|---|---|
| Synonym pair avg cosine | **+0.546** | 의미 비슷한 한국어 문장쌍은 0.5 이상으로 잘 모임 |
| Unrelated pair avg cosine | **+0.073** | 무관한 쌍은 0.1 미만으로 깔끔하게 분리 |
| Margin (synonym − unrelated) | **+0.473** | 동의/무관 분리 폭이 매우 넓음 |
| Intra/inter category margin | +0.150 | 카테고리(음식/기술/스포츠/금융) 클러스터링 가능 |
| Retrieval Top-1 accuracy | **3/3 (100%)** | 쿼리당 후보 5개 중 의미상 가장 가까운 걸 1위로 잡음 |

**이 결과가 말해주는 것**:

- 일반 도메인 한국어 문장에서 **의미 유사도 측정**으로 즉시 쓸 수 있는 수준
- 의미 클러스터링이 잘 됨 → 분류·중복 감지에 적합
- 정답이 있는 retrieval에서 top-1을 잡음 → RAG 1차 후보 추출(coarse retriever)로 쓰면 dense reranker 부담을 크게 줄임

---

## 4. 사용법

```bash
pip install model2vec
```

```python
from model2vec import StaticModel

model = StaticModel.from_pretrained("hysnnnn/kor-minish-bge-m3-ko")
vectors = model.encode(["안녕하세요", "반갑습니다"])
print(vectors.shape)  # (2, 256)
```

### 4-1. 유사도 검색

```python
import numpy as np

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

vecs = model.encode([
    "김치찌개 레시피",
    "된장국 끓이는 법",
    "주식 매수 타이밍",
])
print(cos(vecs[0], vecs[1]))  # ~0.63  (음식끼리)
print(cos(vecs[0], vecs[2]))  # ~0.10  (무관)
```

### 4-2. 대량 인코딩

```python
# 백만 건 문서도 CPU에서 빠르게 인코딩
import numpy as np
docs = [...]  # 1M Korean docs
vectors = model.encode(docs, batch_size=4096)
# vectors: np.ndarray (1_000_000, 256)
```

### 4-3. FAISS 또는 ScaNN과 결합

```python
import faiss

index = faiss.IndexFlatIP(256)  # cosine = inner product on L2-normalized
index.add(vectors)
q = model.encode(["검색 쿼리"])
D, I = index.search(q, k=10)
```

---

## 5. 적합/부적합 사용처

### 잘 작동하는 경우

| 시나리오 | 적합도 |
|---|---|
| RAG의 1차(coarse) retriever — 수백만 문서에서 후보 100~1000개 추출 | ⭐⭐⭐⭐⭐ |
| 한국 뉴스/위키 도메인 토픽 분류 | ⭐⭐⭐⭐ |
| 중복 문서·이슈 감지 (모니터링·로그 클러스터링) | ⭐⭐⭐⭐ |
| 모바일·엣지·서버리스 임베딩 (GPU 없음) | ⭐⭐⭐⭐⭐ |
| FAQ 챗봇 1차 매칭 | ⭐⭐⭐⭐ |
| 검색어 유의어 확장, 자동완성 | ⭐⭐⭐⭐ |

### 부족한 경우

- **고난도 RAG의 메인 retriever** — bge-m3나 fine-tuned reranker가 필요
- **의료·법률 등 정밀도 critical** — 도메인 vocab 부재
- **장문 문서 임베딩** — 토큰 평균 풀링이라 정보가 흐려짐. 짧은 청크로 자르세요.
- **다의어 의존 도메인** — 동음이의어 구분 못함

---

## 6. 알려진 한계 (Limitations)

### 6-1. 한자어 ↔ 순우리말 동의어 매칭 약함

평가에서 발견된 약점입니다.

| 문장쌍 | 유사도 | 평가 |
|---|---|---|
| 월급이 인상되었다 ↔ 급여가 올랐다 | +0.307 | 동의어인데 낮음 |
| 이 책은 정말 흥미롭다 ↔ 이 도서는 매우 재미있다 | +0.430 | 약함 |
| 서울에서 부산까지 KTX ↔ 서울발 부산행 고속열차 | +0.443 | 약함 |

**원인**: bge-m3의 한국어 학습이 한자어/순우리말 alignment까지 깊지 않음. 정적 임베딩으로 압축되면서 더 멀어짐.

**완화 방법**:
- 동의어 사전으로 토큰 임베딩 후처리 (벡터 평균 정렬)
- KorSTS로 supervised fine-tuning (model2vec[training])

### 6-2. 다의어 (Polysemy)

"배" → 과일/선박/복부 모두 한 벡터. 정적 임베딩의 본질적 한계.

### 6-3. 비격식체·신조어

위키 vocab 기반이라 "ㄹㅇ", "갑분싸", "킹받네" 같은 신조어 약함. 도메인 corpus로 추가 vocab을 주입하면 개선됨.

### 6-4. 장문 임베딩

토큰 평균이라 문서가 길수록 의미가 흐려짐. **권장 청크 길이**: 한국어 기준 100~300자.

---

## 7. 재현 / 직접 만들기

이 모델을 만든 전체 파이프라인은 공개되어 있습니다:

- `build_vocab.py` — Wikipedia 코퍼스에서 한국어 vocab 추출
- `distill.py` — model2vec.distill 호출 래퍼
- `evaluate.py` — 평가 스크립트
- `server/server.py` — FastAPI 임베딩 서버 (JS/Java 클라이언트 예시 포함)

다른 도메인(법률·의료·게임·금융 등)에 맞춰 vocab을 바꿔 다시 distill하면 도메인 특화 버전을 30분 안에 만들 수 있습니다.

---

## 8. Citation

이 모델은 [model2vec](https://github.com/MinishLab/model2vec) 기법으로 만들어졌습니다.

```bibtex
@software{minishlab2024model2vec,
  author = {Tulkens, Stephan and {van Dongen}, Thomas},
  title = {Model2Vec: The Fastest State-of-the-Art Static Embeddings in the World},
  url = {https://github.com/MinishLab/model2vec},
}
```

Base model: BAAI/bge-m3 — [Multi-Linguality, Multi-Functionality, Multi-Granularity](https://arxiv.org/abs/2402.03216)

## License

MIT — 상업적 이용 포함 자유. 원 저작권 고지만 유지하세요.
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
            "HuggingFace에 로그인되지 않았습니다.\n"
            "터미널에서 'huggingface-cli login' 실행 후 토큰 입력하세요.\n"
            f"원본 오류: {e}"
        )

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"model dir not found: {model_dir.resolve()}")

    (model_dir / "README.md").write_text(MODEL_CARD, encoding="utf-8")
    print(f"updated model card: {model_dir / 'README.md'}")

    api = HfApi()
    api.create_repo(args.repo, private=args.private, exist_ok=True)
    print(f"repo ready: {args.repo} (private={args.private})")

    print("uploading model files...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo,
        commit_message="Initial upload — kor-minish-bge-m3-ko distilled model",
    )
    print(f"\nhttps://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
