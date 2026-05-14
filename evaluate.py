"""한글 임베딩 정량/정성 평가."""
from __future__ import annotations

import argparse
import statistics

import numpy as np
from model2vec import StaticModel


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


SYNONYM_PAIRS = [
    ("강아지가 공원에서 뛰어논다", "개가 공원에서 놀고 있다"),
    ("인공지능 모델 학습", "AI 모델을 훈련시킨다"),
    ("커피 한 잔 마시고 싶다", "따뜻한 커피가 마시고 싶어"),
    ("어제 영화 봤는데 재밌었어", "어제 본 영화 진짜 재밌더라"),
    ("주말에 등산을 갔다", "토요일에 산에 다녀왔어"),
    ("아침에 일찍 일어났다", "오늘 새벽에 기상했다"),
    ("서울에서 부산까지 KTX", "서울발 부산행 고속열차"),
    ("이 책은 정말 흥미롭다", "이 도서는 매우 재미있다"),
    ("냉장고가 고장났다", "냉장고에 문제가 생겼어"),
    ("월급이 인상되었다", "급여가 올랐다"),
]

UNRELATED_PAIRS = [
    ("김치찌개 레시피", "자동차 엔진 정비"),
    ("주식 시장 폭락", "강아지 산책 코스"),
    ("축구 경기 결과", "양자역학 기초"),
    ("화장품 추천", "건축 도면 작성"),
    ("재택근무 팁", "야생 동물 다큐멘터리"),
    ("프로그래밍 입문", "조선시대 왕실 복식"),
    ("드라마 OST", "세금 신고 방법"),
    ("다이어트 식단", "암호화폐 시세"),
    ("육아 일기", "기계공학 논문"),
    ("여행 가방 추천", "회계 결산"),
]


CATEGORIES = {
    "음식": [
        "김치찌개 끓이는 법",
        "된장국 레시피",
        "비빔밥에 들어가는 재료",
        "삼겹살 굽는 방법",
    ],
    "기술": [
        "딥러닝 모델 학습",
        "파이썬 비동기 프로그래밍",
        "쿠버네티스 클러스터 운영",
        "리액트 컴포넌트 설계",
    ],
    "스포츠": [
        "손흥민 골 모음",
        "프리미어리그 순위",
        "야구 한국시리즈",
        "올림픽 양궁 결승",
    ],
    "금융": [
        "주식 매수 타이밍",
        "코스피 시장 동향",
        "비트코인 시세 분석",
        "연말정산 절세 팁",
    ],
}


RETRIEVAL = [
    {
        "query": "딥러닝으로 이미지를 분류하는 방법",
        "candidates": [
            "CNN 모델로 사진 인식하기",
            "이미지넷 데이터셋 학습",
            "고양이 사진 잘 찍는 팁",
            "디지털카메라 추천",
            "주식 차트 분석",
        ],
        "expected_top": {0, 1},
    },
    {
        "query": "한국 전통 음식 만들기",
        "candidates": [
            "김치 담그는 법",
            "한식 요리 레시피",
            "이탈리아 파스타 만드는 법",
            "프로그래밍 입문 책",
            "운동화 사이즈 고르기",
        ],
        "expected_top": {0, 1},
    },
    {
        "query": "전세 계약 시 주의사항",
        "candidates": [
            "전세금 보호 방법",
            "임대차 계약 체크리스트",
            "맛집 추천 리스트",
            "여행 짐 싸는 법",
            "고양이 사료 추천",
        ],
        "expected_top": {0, 1},
    },
]


def report_pair_separation(model: StaticModel) -> None:
    print("\n=== 1. 동의 vs 무관 분리 ===")
    syn_sims, unr_sims = [], []
    for a, b in SYNONYM_PAIRS:
        v = model.encode([a, b])
        syn_sims.append(cos(v[0], v[1]))
    for a, b in UNRELATED_PAIRS:
        v = model.encode([a, b])
        unr_sims.append(cos(v[0], v[1]))

    print(f"  동의 쌍 ({len(SYNONYM_PAIRS)}):   "
          f"평균 {statistics.mean(syn_sims):+.3f}  "
          f"min {min(syn_sims):+.3f}  max {max(syn_sims):+.3f}")
    print(f"  무관 쌍 ({len(UNRELATED_PAIRS)}):   "
          f"평균 {statistics.mean(unr_sims):+.3f}  "
          f"min {min(unr_sims):+.3f}  max {max(unr_sims):+.3f}")
    gap = statistics.mean(syn_sims) - statistics.mean(unr_sims)
    print(f"  margin (동의 평균 - 무관 평균): {gap:+.3f}  {'OK' if gap > 0.2 else 'WEAK'}")

    print("\n  worst synonym pairs (낮은 동의):")
    for sim, (a, b) in sorted(zip(syn_sims, SYNONYM_PAIRS))[:3]:
        print(f"    {sim:+.3f}  {a}  <->  {b}")
    print("  worst unrelated pairs (높은 무관 — false positive):")
    for sim, (a, b) in sorted(zip(unr_sims, UNRELATED_PAIRS), reverse=True)[:3]:
        print(f"    {sim:+.3f}  {a}  <->  {b}")


def report_clustering(model: StaticModel) -> None:
    print("\n=== 2. 카테고리 클러스터링 (intra vs inter) ===")
    cat_vecs = {c: model.encode(s) for c, s in CATEGORIES.items()}

    intra, inter = [], []
    for cat, vecs in cat_vecs.items():
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                intra.append(cos(vecs[i], vecs[j]))
    cats = list(cat_vecs.keys())
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            for a in cat_vecs[cats[i]]:
                for b in cat_vecs[cats[j]]:
                    inter.append(cos(a, b))

    print(f"  intra-category (같은 주제, n={len(intra)}): 평균 {statistics.mean(intra):+.3f}")
    print(f"  inter-category (다른 주제, n={len(inter)}): 평균 {statistics.mean(inter):+.3f}")
    gap = statistics.mean(intra) - statistics.mean(inter)
    print(f"  margin: {gap:+.3f}  {'OK' if gap > 0.15 else 'WEAK'}")


def report_retrieval(model: StaticModel) -> None:
    print("\n=== 3. Retrieval ranking ===")
    hits_top1 = 0
    hits_top2 = 0
    for case in RETRIEVAL:
        q_vec = model.encode([case["query"]])[0]
        c_vecs = model.encode(case["candidates"])
        sims = [cos(q_vec, v) for v in c_vecs]
        order = sorted(range(len(sims)), key=lambda i: -sims[i])

        is_top1 = order[0] in case["expected_top"]
        is_top2 = set(order[:2]) & case["expected_top"]
        hits_top1 += int(is_top1)
        hits_top2 += int(bool(is_top2))

        print(f"\n  Q: {case['query']}")
        for rank, idx in enumerate(order):
            mark = " <" if idx in case["expected_top"] else "  "
            print(f"    {rank + 1}. {sims[idx]:+.3f}  {case['candidates'][idx]}{mark}")

    n = len(RETRIEVAL)
    print(f"\n  Top-1 accuracy: {hits_top1}/{n} = {hits_top1/n:.0%}")
    print(f"  Top-2 hit rate: {hits_top2}/{n} = {hits_top2/n:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="kor-minish-bge-m3-ko")
    args = parser.parse_args()

    model = StaticModel.from_pretrained(args.model)
    print(f"model: {args.model}  dim={model.embedding.shape[1]}  "
          f"vocab={model.embedding.shape[0]:,}")

    report_pair_separation(model)
    report_clustering(model)
    report_retrieval(model)


if __name__ == "__main__":
    main()
