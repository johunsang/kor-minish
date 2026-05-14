# kor-minish

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)를 [model2vec](https://github.com/MinishLab/model2vec)로 한국어 정적 임베딩(static embedding)으로 추출하는 파이프라인. CPU에서 sentence-transformers 대비 **수백 배 빠르고** 모델 크기는 약 140 MB입니다.

> **Model**: 🤗 [hysnnnn/kor-minish-bge-m3-ko](https://huggingface.co/hysnnnn/kor-minish-bge-m3-ko)

## 빠르게 써보기

```bash
pip install model2vec
```

```python
from model2vec import StaticModel

model = StaticModel.from_pretrained("hysnnnn/kor-minish-bge-m3-ko")
vectors = model.encode(["안녕하세요", "반갑습니다"])  # (2, 256)
```

## 파이프라인 개요

| 단계 | 스크립트 | 출력 |
|---|---|---|
| 1. 한국어 vocab 추출 | `build_vocab.py` | `vocab_ko.txt` (top 30K 명사) |
| 2. Distillation | `distill.py` (또는 `distill_colab.ipynb`) | `kor-minish-bge-m3-ko/` |
| 3. 평가 | `evaluate.py`, `test_embed.py` | 콘솔 출력 |
| 4. HTTP 배포 | `server/server.py` + `Dockerfile` | port 8000 API |
| 5. HF Hub 업로드 | `push_to_hf.py` | 공개 repo |

### 1. vocab 추출

Korean Wikipedia 20231101.ko 스트리밍 → kiwipiepy 형태소 분석 → NNG/NNP/NNB/NR 빈도 top 30,000.

```bash
uv sync
uv run python build_vocab.py
```

### 2. Distillation

vocab 30K 토큰을 bge-m3로 임베딩 → PCA 256차원 압축 → SIF 가중치 → 저장. **T4 GPU에서 약 6분.**

```bash
# 로컬 (CPU/MPS, 느림)
uv run python distill.py --vocab vocab_ko.txt --out kor-minish-bge-m3-ko

# 또는 Colab T4 사용 (권장)
# distill_colab.ipynb 를 Colab에 업로드
```

### 3. 평가

```bash
uv run python evaluate.py --model kor-minish-bge-m3-ko
```

10쌍 동의/무관, 16개 카테고리 문장, 3개 retrieval 쿼리로 정성 평가.

| Test | Result |
|---|---|
| Synonym pair avg cosine | **+0.546** |
| Unrelated pair avg cosine | **+0.073** |
| Margin (synonym − unrelated) | **+0.473** |
| Intra/inter category margin | +0.150 |
| Retrieval Top-1 | **3/3 (100%)** |

### 4. HTTP API 서버

다른 언어/서비스에서 쉽게 쓰도록 FastAPI 서버 제공.

```bash
uv run uvicorn server.server:app --host 0.0.0.0 --port 8000
```

또는 Docker:

```bash
docker build -t kor-minish .
docker run --rm -p 8000:8000 kor-minish
```

엔드포인트:
- `GET /health` — 상태/모델 정보
- `POST /encode` — `{texts, normalize}` → `{dim, embeddings}`
- `POST /similarity` — `{query, docs}` → `{scores, order}`

클라이언트 예시: [examples/](examples/) (curl, Node.js, Java).

### 5. HF Hub 업로드

```bash
huggingface-cli login   # write 권한 토큰
uv run python push_to_hf.py --repo your-username/kor-minish-bge-m3-ko
```

## 다른 도메인으로 변형하기

도메인 특화 모델은 **vocab만 바꾸고 재-distill**하면 됩니다. 약 30분.

1. 해당 도메인 코퍼스 준비 (법률/의료/금융 등)
2. `build_vocab.py`를 수정해 그 코퍼스에서 명사 추출
3. `distill.py --vocab <new_vocab>` 실행

## 알려진 한계

- **한자어 ↔ 순우리말 동의어 매칭이 약함** (e.g., 월급 ↔ 급여 cos≈0.31)
- **다의어 처리 불가** — 정적 임베딩의 본질적 한계
- **장문일수록 정보가 흐려짐** — 권장 청크: 한국어 100~300자
- **비격식체·신조어**는 위키 vocab 기반이라 약함

## 라이선스

[MIT](LICENSE)

Base model: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (MIT)
Distillation framework: [model2vec](https://github.com/MinishLab/model2vec) (MIT)
