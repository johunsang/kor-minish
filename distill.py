"""BAAI/bge-m3 → 한글용 정적 임베딩 distill (vocab 주입 지원)."""
from __future__ import annotations

import argparse
from pathlib import Path

from model2vec.distill import distill


def load_vocab(path: str | None) -> list[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    words = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]
    words = [w for w in words if w]
    print(f"loaded vocab: {len(words):,} tokens from {p}")
    return words


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="BAAI/bge-m3")
    parser.add_argument("--out", default="kor-minish-bge-m3")
    parser.add_argument("--pca", type=int, default=256, help="PCA target dim (0 = skip)")
    parser.add_argument("--no-zipf", action="store_true", help="apply_zipf=False")
    parser.add_argument("--vocab", default=None,
                        help="추가 한글 vocab 파일 (한 줄당 한 토큰)")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    vocabulary = load_vocab(args.vocab)

    m2v = distill(
        model_name=args.base,
        vocabulary=vocabulary,
        pca_dims=args.pca if args.pca > 0 else None,
        apply_zipf=not args.no_zipf,
    )
    m2v.save_pretrained(str(out))
    print(f"saved -> {out.resolve()}")


if __name__ == "__main__":
    main()
