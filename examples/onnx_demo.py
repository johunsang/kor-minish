"""ONNX 모델 사용 예시 (Python, model2vec 없이).

설치:
    pip install onnxruntime tokenizers numpy

실행:
    python examples/onnx_demo.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ONNX_DIR = Path(__file__).parent.parent / "kor-minish-bge-m3-ko-onnx"


def encode(
    texts: list[str],
    tokenizer: Tokenizer,
    session: ort.InferenceSession,
) -> np.ndarray:
    encs = tokenizer.encode_batch(texts, add_special_tokens=False)
    max_len = max(len(e.ids) for e in encs) or 1
    ids = np.zeros((len(encs), max_len), dtype=np.int64)
    mask = np.zeros((len(encs), max_len), dtype=np.float32)
    for i, e in enumerate(encs):
        ids[i, : len(e.ids)] = e.ids
        mask[i, : len(e.ids)] = 1.0
    out = session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
    return out


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    tok = Tokenizer.from_file(str(ONNX_DIR / "tokenizer.json"))
    sess = ort.InferenceSession(str(ONNX_DIR / "model.onnx"))

    sentences = [
        "김치찌개 레시피",
        "된장국 끓이는 법",
        "주식 매수 타이밍",
        "자동차 보험",
    ]
    vecs = encode(sentences, tok, sess)
    print(f"dim = {vecs.shape[1]}, n = {vecs.shape[0]}")

    query = "한국 음식 만들기"
    q_vec = encode([query], tok, sess)[0]
    scores = [cos(q_vec, v) for v in vecs]
    order = sorted(range(len(scores)), key=lambda i: -scores[i])

    print(f"\nQ: {query}")
    for rank, idx in enumerate(order):
        print(f"  {rank + 1}. {scores[idx]:+.3f}  {sentences[idx]}")


if __name__ == "__main__":
    main()
