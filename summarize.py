"""임베딩 기반 추출적 요약 (extractive summary).

원리:
    1) 글을 문장 단위로 분리
    2) 각 문장 임베딩 + 전체 문서 임베딩 계산
    3) "문서 중심"에 가까운 문장 top-N 선택
    4) 원문 순서대로 재정렬

이 방식은 새 문장을 만들지 못합니다 (extractive). 추상적 요약은 LLM이 필요합니다.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
from model2vec import StaticModel


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    parts = re.split(r"(?<=[.!?。…])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def summarize(model: StaticModel, text: str, top_n: int = 3) -> tuple[list[str], list[float]]:
    sents = split_sentences(text)
    if len(sents) <= top_n:
        return sents, [1.0] * len(sents)

    sent_vecs = model.encode(sents)
    doc_vec = sent_vecs.mean(axis=0)

    sent_norm = sent_vecs / np.linalg.norm(sent_vecs, axis=1, keepdims=True)
    doc_norm = doc_vec / np.linalg.norm(doc_vec)
    scores = sent_norm @ doc_norm

    top_idx = np.argsort(-scores)[:top_n]
    keep = sorted(top_idx.tolist())
    return [sents[i] for i in keep], [float(scores[i]) for i in keep]


SAMPLES = {
    "기술 뉴스": """
인공지능 기술이 빠르게 발전하면서 다양한 산업 분야에 도입되고 있다.
특히 자연어처리 분야에서는 대형 언어 모델이 상업적으로 활용되기 시작했다.
한국어 처리 기술도 함께 성장하고 있으며, 국내 기업들이 자체 모델을 개발하고 있다.
하지만 학습에 필요한 컴퓨팅 자원과 한국어 데이터셋이 여전히 부족한 상황이다.
정부는 인공지능 산업 육성을 위해 다양한 지원 정책을 발표했다.
오늘 점심에는 김치찌개를 먹었다.
업계에서는 향후 5년 안에 한국어 대형 모델 시장이 크게 확대될 것으로 전망하고 있다.
""",
    "역사 위키": """
조선은 1392년 이성계가 건국한 한반도의 마지막 왕조이다.
태조 이성계는 고려 말의 무신 출신으로, 위화도 회군을 통해 권력을 장악했다.
조선은 유교를 국가 이념으로 채택하고 한양을 수도로 정했다.
세종대왕은 1443년 훈민정음을 창제하여 한국어 표기의 토대를 마련했다.
임진왜란과 병자호란이라는 두 차례의 큰 전란을 겪으며 국력이 약화되었다.
19세기 말 외세의 압력 속에서 근대화에 어려움을 겪었다.
오늘 저녁 메뉴는 된장찌개로 정했다.
1910년 한일병합조약으로 조선왕조는 막을 내렸다.
""",
    "제품 리뷰": """
이 노트북을 한 달 정도 사용해보고 후기를 남긴다.
첫인상은 무게가 가볍고 디자인이 깔끔해서 만족스러웠다.
배터리는 일반 사무 작업 기준으로 8시간 정도 지속된다.
키보드는 키감이 부드럽고 오타가 거의 없어 편하다.
다만 발열이 심한 편이라 게임이나 영상 편집 같은 작업에는 부담스럽다.
스피커 음질은 가격대 대비 평범한 수준이다.
어제 밤에 라면을 끓여 먹었는데 야식은 역시 별로다.
가성비를 따진다면 추천할 만하지만 고성능을 원한다면 다른 선택지를 보길 권한다.
""",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hysnnnn/kor-minish-bge-m3-ko")
    parser.add_argument("--top", type=int, default=3, help="추출할 문장 수")
    args = parser.parse_args()

    model = StaticModel.from_pretrained(args.model)
    print(f"model: {args.model}  dim={model.embedding.shape[1]}\n")

    for title, text in SAMPLES.items():
        print(f"\n{'=' * 60}")
        print(f"[{title}]  원문 {len(split_sentences(text))}문장 → 요약 {args.top}문장")
        print("=" * 60)
        summary, scores = summarize(model, text, top_n=args.top)
        for s, score in zip(summary, scores):
            print(f"  ({score:+.3f}) {s}")


if __name__ == "__main__":
    main()
