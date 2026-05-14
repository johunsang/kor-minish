"""한국어 위키에서 명사/고유명사 빈도 어휘를 추출."""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from kiwipiepy import Kiwi
from tqdm import tqdm

NOUN_TAGS = {"NNG", "NNP", "NNB", "NR"}
HANGUL_RE = re.compile(r"[가-힣]")
MIN_LEN = 2
MAX_LEN = 20


def iter_articles(name: str, config: str, split: str, limit: int | None):
    ds = load_dataset(name, config, split=split, streaming=True)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield row["text"]


def is_valid_token(form: str) -> bool:
    if not (MIN_LEN <= len(form) <= MAX_LEN):
        return False
    if not HANGUL_RE.search(form):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wikimedia/wikipedia")
    parser.add_argument("--config", default="20231101.ko")
    parser.add_argument("--split", default="train")
    parser.add_argument("--articles", type=int, default=200_000,
                        help="processed article count (None = all)")
    parser.add_argument("--top", type=int, default=30_000, help="vocab size cap")
    parser.add_argument("--min-count", type=int, default=5,
                        help="drop tokens with frequency below this")
    parser.add_argument("--out", default="vocab_ko.txt")
    args = parser.parse_args()

    kiwi = Kiwi()
    counter: Counter[str] = Counter()

    pbar = tqdm(
        iter_articles(args.dataset, args.config, args.split, args.articles),
        total=args.articles,
        desc="tokenizing",
        unit="doc",
    )
    for text in pbar:
        if not text:
            continue
        for token in kiwi.tokenize(text):
            if token.tag not in NOUN_TAGS:
                continue
            form = token.form
            if is_valid_token(form):
                counter[form] += 1

    items = [(w, c) for w, c in counter.most_common() if c >= args.min_count]
    items = items[: args.top]

    out = Path(args.out)
    out.write_text(
        "\n".join(w for w, _ in items) + "\n",
        encoding="utf-8",
    )
    print(f"saved {len(items):,} tokens -> {out.resolve()}")
    print("top 20 preview:")
    for w, c in items[:20]:
        print(f"  {c:>8,}  {w}")


if __name__ == "__main__":
    main()
