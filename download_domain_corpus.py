"""한국어 도메인 corpus(법률/의료/금융/뉴스 등) 다운로드.

이 데이터는 STS 쌍이 아니라 raw 텍스트. vocab 강화용으로 사용:
    data/raw_corpus/<repo>/  ← 다운로드 받은 텍스트
    data/vocab_domain.txt    ← 명사 추출 후 빈도 top 30K

ko-roberta tokenizer 또는 kiwipiepy로 명사 추출.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from datasets import load_dataset
from kiwipiepy import Kiwi

OUT = Path("data/raw_corpus")
VOCAB_OUT = Path("data/vocab_domain.txt")
OUT.mkdir(parents=True, exist_ok=True)

NOUN_TAGS = {"NNG", "NNP", "NNB", "NR"}
MIN_LEN = 2
MAX_LEN = 20

# 가용한 한국어 도메인 corpus 후보 (try-except로 안전하게)
CANDIDATES = [
    # (repo, config, text_field, label)
    ("lcw99/wikipedia-korean-20240501", None, "text", "wiki"),
    ("daekeun-ml/naver-news-summarization-ko", None, "document", "news"),
    ("kakaobrain/kowiki", None, "text", "wiki2"),
    ("Bingsu/korean-laws-text", None, "text", "law"),
    ("lcw99/korean_legal_text", None, "text", "law2"),
    ("etrans/financial-news-korean", None, "text", "finance"),
    ("daekeun-ml/sec-news-summary-kr", None, "document", "finance2"),
    ("lcw99/medical-korean", None, "text", "medical"),
    ("KisanY/medical-korean", None, "text", "medical2"),
    ("heegyu/namuwiki", None, "text", "namuwiki"),
    ("HAERAE-HUB/HAE-RAE-BENCH", None, "text", "haerae"),
]


def download_all() -> list[tuple[str, Path, str]]:
    """가용한 데이터셋 다운로드. (label, path_to_dir, text_field) 리스트 반환."""
    successful = []
    for repo, config, text_field, label in CANDIDATES:
        target_dir = OUT / repo.replace("/", "__")
        if target_dir.exists():
            print(f"  SKIP {label} ({repo}) — already downloaded")
            successful.append((label, target_dir, text_field))
            continue
        try:
            ds = load_dataset(repo, config) if config else load_dataset(repo)
            split = "train" if "train" in ds else list(ds.keys())[0]
            sample = ds[split][0]
            # 실제 텍스트 필드 추정
            actual_field = text_field
            if actual_field not in sample:
                for cand in ["text", "document", "content", "body", "passage", "article"]:
                    if cand in sample:
                        actual_field = cand
                        break
                else:
                    print(f"  SKIP {label} ({repo}) — no text field in {list(sample.keys())[:5]}")
                    continue
            ds.save_to_disk(str(target_dir))
            n_rows = sum(len(ds[s]) for s in ds.keys())
            print(f"  OK   {label:10s} ({repo}): {n_rows:,} rows, field='{actual_field}'")
            successful.append((label, target_dir, actual_field))
        except Exception as e:
            err = str(e).split("\n")[0][:100]
            print(f"  FAIL {label:10s} ({repo}): {err}")
    return successful


def extract_vocab(sources: list[tuple[str, Path, str]], top_k: int = 30_000) -> None:
    """다운로드된 corpus에서 명사 추출, top-k 저장."""
    from datasets import load_from_disk

    kiwi = Kiwi()
    counter: Counter[str] = Counter()
    per_source: dict[str, int] = {}

    for label, path, field in sources:
        try:
            ds = load_from_disk(str(path))
            split = "train" if "train" in ds else list(ds.keys())[0]
            n_words = 0
            # 메모리 절약 위해 각 도메인에서 최대 50K 문서만
            limit = min(50_000, len(ds[split]))
            for i in range(limit):
                text = ds[split][i].get(field, "")
                if not isinstance(text, str) or not text:
                    continue
                # 매우 긴 문서는 앞부분만
                text = text[:5000]
                for tok in kiwi.tokenize(text):
                    if tok.tag in NOUN_TAGS:
                        form = tok.form
                        if MIN_LEN <= len(form) <= MAX_LEN:
                            counter[form] += 1
                            n_words += 1
            per_source[label] = n_words
            print(f"  {label:10s}: {limit:,} docs, {n_words:,} noun tokens")
        except Exception as e:
            print(f"  {label} extract FAIL: {e}")

    items = [(w, c) for w, c in counter.most_common() if c >= 3]
    items = items[:top_k]
    VOCAB_OUT.write_text("\n".join(w for w, _ in items) + "\n", encoding="utf-8")
    print(f"\n총 {len(items):,} 도메인 명사 → {VOCAB_OUT}")
    print("top 20:")
    for w, c in items[:20]:
        print(f"  {c:>6,}  {w}")


def main() -> None:
    print("=== 도메인 corpus 다운로드 ===")
    sources = download_all()
    if not sources:
        print("\n도메인 데이터를 하나도 못 받았습니다. vocab 강화 건너뜀.")
        return
    print(f"\n다운로드 성공: {len(sources)}개")
    print("\n=== 명사 추출 (도메인 vocab 빌드) ===")
    extract_vocab(sources)


if __name__ == "__main__":
    main()
