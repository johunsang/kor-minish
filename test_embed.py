"""Distill된 모델 sanity check: 한국어 문장 유사도."""
from __future__ import annotations

import argparse

import numpy as np
from model2vec import StaticModel


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="kor-minish-bge-m3")
    args = parser.parse_args()

    m = StaticModel.from_pretrained(args.model)

    pairs = [
        ("강아지가 공원에서 뛰어논다", "개가 공원에서 놀고 있다"),
        ("강아지가 공원에서 뛰어논다", "주식 시장이 폭락했다"),
        ("인공지능 모델 학습", "AI 모델을 훈련시킨다"),
        ("김치찌개 레시피", "된장찌개 만드는 법"),
        ("김치찌개 레시피", "자동차 엔진 정비"),
    ]

    sentences = [s for pair in pairs for s in pair]
    vecs = m.encode(sentences)

    print(f"dim = {vecs.shape[1]}")
    for i, (a, b) in enumerate(pairs):
        sim = cos(vecs[2 * i], vecs[2 * i + 1])
        print(f"{sim:+.3f}  {a}  <->  {b}")


if __name__ == "__main__":
    main()
