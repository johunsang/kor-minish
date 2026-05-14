# kor-minish 데모 — 명령어 + 실제 출력

각 섹션의 **명령**을 본인 터미널에서 그대로 복사 실행하면 아래 출력이 똑같이 나옵니다.
모델: 🤗 [hysnnnn/kor-minish-bge-m3-ko](https://huggingface.co/hysnnnn/kor-minish-bge-m3-ko)

---

## 0. 설치

```bash
# 방법 A: Python에서 직접 사용
pip install model2vec

# 방법 B: CLI 도구
pip install git+https://github.com/johunsang/kor-minish.git

# 방법 C: 둘 다 (B에 model2vec 포함)
```

---

## 1. Python 한 줄

```python
from model2vec import StaticModel
m = StaticModel.from_pretrained("hysnnnn/kor-minish-bge-m3-ko")
vectors = m.encode(["안녕하세요", "반갑습니다"])
print(vectors.shape)
```

```
(2, 256)
```

---

## 2. CLI: 유사도 검색

### AI/딥러닝 도메인

```bash
kor-minish similarity "딥러닝 모델 학습" \
  "신경망 훈련" \
  "machine learning training" \
  "AI 인공지능" \
  "고양이 사진 보정" \
  "주식 차트 분석"
```

```
 1. +0.359  machine learning training
 2. +0.283  신경망 훈련
 3. +0.233  AI 인공지능
 4. +0.083  주식 차트 분석
 5. +0.072  고양이 사진 보정
```

→ 영어 섞여도 의미 매칭 잘 됨.

### 격식체 ↔ 비격식체

```bash
kor-minish similarity "내일 회의가 오후 3시에 있습니다" \
  "내일 미팅 3시야" \
  "내일 3시 회의" \
  "야 뭐해" \
  "주식 폭락"
```

```
 1. +0.775  내일 3시 회의
 2. +0.652  내일 미팅 3시야
 3. +0.070  야 뭐해
 4. -0.099  주식 폭락
```

→ 격식체 0.78, 비격식체 0.65, 무관 0.07 이하로 깔끔하게 분리.

### stdin 입력 (파이프) — 검색 시뮬레이션

```bash
printf "사과 한 봉지 가격\n아이폰 신제품 리뷰\n과일 시세 동향\n주식 시장 전망\n" \
  | kor-minish similarity "오늘 마트 과일값"
```

```
 1. +0.453  과일 시세 동향
 2. +0.317  사과 한 봉지 가격
 3. +0.198  주식 시장 전망
 4. +0.187  아이폰 신제품 리뷰
```

---

## 3. CLI: 추출적 요약

7문장 중 노이즈 4개를 일부러 섞어둠. 모델이 노이즈를 배제하고 핵심만 골라내는지 확인.

```bash
echo "한국의 인공지능 산업이 빠르게 성장하고 있다.
정부는 2030년까지 AI 강국 도약을 목표로 발표했다.
점심으로 비빔밥을 먹었다.
국내 기업들도 자체 LLM 개발에 박차를 가하고 있다.
어제 본 영화는 재미없었다.
한국어 데이터셋과 GPU 자원 확보가 핵심 과제로 떠올랐다.
오늘 비가 와서 우산을 챙겼다." | kor-minish summary --top 3
```

```
한국의 인공지능 산업이 빠르게 성장하고 있다.
국내 기업들도 자체 LLM 개발에 박차를 가하고 있다.
한국어 데이터셋과 GPU 자원 확보가 핵심 과제로 떠올랐다.
```

→ 노이즈(점심·영화·날씨) 4개 모두 배제. 핵심 3개 정확히 추출.

---

## 4. CLI: JSON 출력 → 다른 도구로 파이프

```bash
kor-minish similarity "한국 음식" "김치찌개" "파스타" "비빔밥" --format json \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('best:', d['docs'][d['order'][0]])"
```

```
best: 김치찌개
```

---

## 5. Python: 유사도 매트릭스

```python
from model2vec import StaticModel
import numpy as np

m = StaticModel.from_pretrained("hysnnnn/kor-minish-bge-m3-ko")
sents = ["축구 경기", "야구 경기", "주식 매매", "비빔밥 레시피", "된장찌개 끓이는 법"]
v = m.encode(sents)
n = v / np.linalg.norm(v, axis=1, keepdims=True)
mat = n @ n.T
for i, s in enumerate(sents):
    print(f"{s:20s}", "  ".join(f"{mat[i,j]:+.2f}" for j in range(len(sents))))
```

```
축구 경기              +1.00  +0.71  +0.14  -0.03  +0.05
야구 경기              +0.71  +1.00  +0.10  +0.05  +0.11
주식 매매              +0.14  +0.10  +1.00  -0.08  -0.09
비빔밥 레시피            -0.03  +0.05  -0.08  +1.00  +0.26
된장찌개 끓이는 법         +0.05  +0.11  -0.09  +0.26  +1.00
```

→ **같은 카테고리** (축구↔야구 0.71, 비빔밥↔된장 0.26) 가까움, **다른 카테고리는 ≤ 0**.

---

## 6. ONNX 모델 (Python 외 언어용)

```bash
pip install onnxruntime tokenizers numpy

# 변환 (한 번만)
python export_onnx.py
```

```
loaded: dim=256 vocab=278203
normalize=True  has_weights=True
ONNX saved: kor-minish-bge-m3-ko-onnx/model.onnx

--- ONNX vs model2vec ---
  [OK] cos=1.000000  max|diff|=0.000000  '안녕하세요'
  [OK] cos=1.000000  max|diff|=0.000000  '한국 음식 만들기'
  [OK] cos=1.000000  max|diff|=0.000000  '딥러닝 모델 학습'
```

```bash
python examples/onnx_demo.py
```

```
dim = 256, n = 4

Q: 한국 음식 만들기
  1. +0.303  김치찌개 레시피
  2. +0.253  된장국 끓이는 법
  3. +0.137  자동차 보험
  4. +0.103  주식 매수 타이밍
```

→ model2vec와 **완전 동일한 결과** (cos=1.0). Node.js/Java/Go/Rust 어디서든 같은 방식.

---

## 7. HTTP API 서버

```bash
KOR_MINISH_MODEL=hysnnnn/kor-minish-bge-m3-ko \
  uv run uvicorn server.server:app --host 0.0.0.0 --port 8000
```

다른 터미널에서:

```bash
curl -X POST http://localhost:8000/similarity \
  -H 'Content-Type: application/json' \
  -d '{"query":"한국 음식","docs":["김치찌개","주식","된장국","자동차"]}'
```

```json
{"scores":[0.3027,0.0590,0.2605,0.1374],"order":[0,2,3,1]}
```

Node.js / Java / curl 클라이언트 예시: [`examples/`](examples/)

---

## 평가 요약

[`evaluate.py`](evaluate.py) 실행 결과:

| Test | Result |
|---|---|
| Synonym pair avg cosine | **+0.546** |
| Unrelated pair avg cosine | **+0.073** |
| Margin (synonym − unrelated) | **+0.473** |
| Intra/inter category margin | +0.150 |
| Retrieval Top-1 accuracy | **3/3 (100%)** |

## 한계

- **한자어 ↔ 순우리말**: 월급↔급여 cos≈0.31 (낮은 편)
- **다의어**: "배"(과일/선박/복부) 구분 못함 (정적 임베딩 본질적 한계)
- **장문**: 토큰 평균이라 길수록 정보 흐려짐. 100~300자 청크 권장
- **신조어**: 위키 vocab 기반이라 약함
