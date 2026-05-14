"""kor-minish train — 사용자 데이터로 도메인 특화 임베딩 만들기.

Pipeline:
    1) 데이터 로드 (paraphrase / Q&A / STS 자동 감지)
    2) train / eval 분리
    3) sentence-transformer fine-tune (CosineSimilarityLoss)
    4) model2vec 재-distill (정적 임베딩 압축)
    5) eval set으로 정확도 평가
    6) 저장 (+선택적 HF Hub 푸시)
"""
from __future__ import annotations

import argparse
import random
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np

from kor_minish.data_loader import load_training_data

DEFAULT_BASE = "BAAI/bge-m3"


def _pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _split(examples, eval_frac: float, seed: int = 42):
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_frac)) if eval_frac > 0 else 0
    return shuffled[n_eval:], shuffled[:n_eval]


def _evaluate(static_model_path: str, eval_examples) -> dict:
    """eval pair에 대한 cos 유사도를 label과 비교 (Spearman + accuracy)."""
    from model2vec import StaticModel
    m = StaticModel.from_pretrained(static_model_path)

    s1 = [e.texts[0] for e in eval_examples]
    s2 = [e.texts[1] for e in eval_examples]
    labels = np.array([e.label for e in eval_examples], dtype=np.float32)

    v1 = m.encode(s1)
    v2 = m.encode(s2)
    v1n = v1 / np.linalg.norm(v1, axis=1, keepdims=True).clip(min=1e-9)
    v2n = v2 / np.linalg.norm(v2, axis=1, keepdims=True).clip(min=1e-9)
    preds = (v1n * v2n).sum(axis=1)

    # threshold 0.5 기반 정확도 (label 0.5 이상이면 positive)
    correct = ((preds > 0.5) == (labels > 0.5)).sum()
    accuracy = float(correct) / len(labels)

    # Pearson 상관 (근사)
    from numpy import corrcoef
    pearson = float(corrcoef(preds, labels)[0, 1])

    return {
        "n": len(labels),
        "accuracy": accuracy,
        "pearson": pearson,
        "pred_mean": float(preds.mean()),
        "pred_min": float(preds.min()),
        "pred_max": float(preds.max()),
    }


def train(
    data_path: str,
    output_dir: str,
    base_model: str = DEFAULT_BASE,
    epochs: int = 1,
    batch_size: int = 16,
    eval_split: float = 0.1,
    pca_dims: int = 256,
    push_to_hub: str | None = None,
    skip_finetune: bool = False,
) -> dict:
    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader
    from model2vec.distill import distill

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] loading data: {data_path}")
    examples, fmt = load_training_data(data_path)
    print(f"      format = {fmt}, total {len(examples):,} pairs")

    train_ex, eval_ex = _split(examples, eval_split)
    print(f"      train {len(train_ex):,}  |  eval {len(eval_ex):,}")

    if not train_ex:
        raise SystemExit("no training examples")

    device = _pick_device()
    print(f"[2/5] base model = {base_model}, device = {device}")

    ft_dir = str(out / "_finetuned")
    if skip_finetune:
        print("      --skip-finetune: distilling base directly")
        ft_dir = base_model
    else:
        st = SentenceTransformer(base_model, device=device)
        loader = DataLoader(train_ex, shuffle=True, batch_size=batch_size)
        loss = losses.CosineSimilarityLoss(st)
        warmup = int(len(loader) * epochs * 0.1)

        print(f"[3/5] fine-tuning ({epochs} epoch, batch={batch_size}, warmup={warmup})")
        st.fit(
            train_objectives=[(loader, loss)],
            epochs=epochs,
            warmup_steps=warmup,
            output_path=ft_dir,
            show_progress_bar=True,
        )

    print(f"[4/5] distilling to static embeddings (pca_dims={pca_dims})")
    # model2vec은 PyTorch 2.10+에서 MPS 비활성화 → cuda 또는 cpu만 사용
    distill_device = "cuda" if device == "cuda" else "cpu"
    m2v = distill(
        model_name=ft_dir,
        vocabulary=None,
        pca_dims=pca_dims,
        device=distill_device,
    )
    final_dir = str(out)
    m2v.save_pretrained(final_dir)
    print(f"      saved → {final_dir}")

    if not skip_finetune and Path(ft_dir).exists():
        shutil.rmtree(ft_dir, ignore_errors=True)

    metrics = {}
    if eval_ex:
        print("[5/5] evaluating on held-out split")
        metrics = _evaluate(final_dir, eval_ex)
        print(f"      n={metrics['n']}  acc={metrics['accuracy']:.3f}  "
              f"pearson={metrics['pearson']:+.3f}  "
              f"pred[{metrics['pred_min']:.2f}, {metrics['pred_max']:.2f}]")

    if push_to_hub:
        print(f"[+] pushing to HF Hub: {push_to_hub}")
        m2v.push_to_hub(push_to_hub)

    return {"format": fmt, "n_train": len(train_ex), "n_eval": len(eval_ex), **metrics}


def cli_main(args: argparse.Namespace) -> None:
    result = train(
        data_path=args.data,
        output_dir=args.output,
        base_model=args.base,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_split=args.eval_split,
        pca_dims=args.pca,
        push_to_hub=args.push_to_hub,
        skip_finetune=args.skip_finetune,
    )
    print("\n=== result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


def add_subparser(sub) -> None:
    p = sub.add_parser("train", help="사용자 데이터로 도메인 특화 임베딩 학습")
    p.add_argument("--data", required=True, help="학습 데이터 (csv/jsonl)")
    p.add_argument("--output", required=True, help="저장 디렉토리")
    p.add_argument("--base", default="BAAI/bge-m3",
                   help="base sentence-transformer (default: BAAI/bge-m3). "
                        "model2vec 모델 X — fine-tune 가능한 ST만 가능")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-split", type=float, default=0.1)
    p.add_argument("--pca", type=int, default=256)
    p.add_argument("--push-to-hub", default=None, help="username/repo (HF Hub)")
    p.add_argument("--skip-finetune", action="store_true",
                   help="fine-tune 건너뛰고 base를 그대로 distill (테스트용)")
    p.set_defaults(func=cli_main, _needs_model=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_subparser(sub)
    args = parser.parse_args()
    args.func(args)
