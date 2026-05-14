"""model2vec StaticModel → ONNX export.

토크나이저는 별도(`tokenizer.json` + HF tokenizers 라이브러리)로 사용합니다.
ONNX 모델 입력/출력:
    input_ids:      (B, S) int64
    attention_mask: (B, S) float32 (0/1, padding 처리)
    embedding:      (B, D) float32 — L2 정규화된 문장 임베딩

forward는 model2vec의 encode와 동등:
    out = mean(embedding[input_ids] * weights[input_ids])  # 단순 mean, weights는 미리 곱해둠
    out = out / |out|
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from model2vec import StaticModel


class StaticEmbedder(nn.Module):
    """ONNX export용 모듈 — embedding lookup + masked mean + L2."""

    def __init__(self, embedding_weighted: np.ndarray, normalize: bool) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(embedding_weighted.astype(np.float32)),
            freeze=True,
        )
        self.normalize = normalize

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        masked = embeds * mask
        denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = masked.sum(dim=1) / denom
        if self.normalize:
            norm = pooled.norm(dim=-1, keepdim=True).clamp(min=1e-32)
            pooled = pooled / norm
        return pooled


def export(model_path: str, out_dir: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sm = StaticModel.from_pretrained(model_path)
    print(f"loaded: dim={sm.embedding.shape[1]} vocab={sm.embedding.shape[0]}")
    print(f"normalize={sm.normalize}  has_weights={sm.weights is not None}")

    # weights를 embedding에 미리 곱한다 → forward가 단순해짐
    embedding = np.asarray(sm.embedding, dtype=np.float32)
    if sm.weights is not None:
        embedding = embedding * np.asarray(sm.weights, dtype=np.float32)[:, None]

    module = StaticEmbedder(embedding, normalize=bool(sm.normalize)).eval()

    dummy_ids = torch.zeros(1, 16, dtype=torch.long)
    dummy_mask = torch.ones(1, 16, dtype=torch.float32)
    onnx_path = out / "model.onnx"

    torch.onnx.export(
        module,
        (dummy_ids, dummy_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "embedding": {0: "batch"},
        },
        opset_version=14,
    )
    print(f"ONNX saved: {onnx_path}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # 토크나이저 파일 함께 복사 (사용자가 같은 디렉토리에서 모두 로드)
    src = Path(model_path)
    for fname in ["tokenizer.json", "config.json", "modules.json"]:
        f = src / fname
        if f.exists():
            shutil.copy(f, out / fname)
            print(f"copied: {fname}")

    return onnx_path


def verify(model_path: str, onnx_dir: str) -> None:
    """model2vec.encode 와 onnxruntime 결과 비교."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    sm = StaticModel.from_pretrained(model_path)
    tok = Tokenizer.from_file(str(Path(onnx_dir) / "tokenizer.json"))
    sess = ort.InferenceSession(str(Path(onnx_dir) / "model.onnx"))

    sentences = [
        "안녕하세요",
        "한국 음식 만들기",
        "딥러닝 모델 학습",
        "주식 시장이 폭락했다",
        "강아지가 공원에서 뛰어논다",
    ]

    # ONNX 추론: 토크나이즈 → padding → ONNX
    encs = tok.encode_batch(sentences, add_special_tokens=False)
    max_len = max(len(e.ids) for e in encs)
    ids = np.zeros((len(encs), max_len), dtype=np.int64)
    mask = np.zeros((len(encs), max_len), dtype=np.float32)
    for i, e in enumerate(encs):
        ids[i, : len(e.ids)] = e.ids
        mask[i, : len(e.ids)] = 1.0
    onnx_out = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]

    # model2vec 추론
    ref_out = sm.encode(sentences)

    diffs = np.abs(onnx_out - ref_out).max(axis=1)
    cos = (onnx_out * ref_out).sum(axis=1) / (
        np.linalg.norm(onnx_out, axis=1) * np.linalg.norm(ref_out, axis=1)
    )

    print("\n--- ONNX vs model2vec ---")
    for s, d, c in zip(sentences, diffs, cos):
        flag = "OK" if c > 0.999 else "DIFF"
        print(f"  [{flag}] cos={c:.6f}  max|diff|={d:.6f}  '{s}'")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="kor-minish-bge-m3-ko")
    p.add_argument("--out", default="kor-minish-bge-m3-ko-onnx")
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    export(args.model, args.out)
    if not args.no_verify:
        verify(args.model, args.out)


if __name__ == "__main__":
    main()
