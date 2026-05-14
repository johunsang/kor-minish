"""keke-code-v1: polyglot 코드 + 한국어 정적 임베딩 학습 (RunPod H200).

다국어 코드 (Python/Java/JS/Go/PHP/Ruby/Rust/HTML/CSS) + 한국어를 한 모델로.

Teacher: Snowflake/snowflake-arctic-embed-l-v2.0 (568M, 다국어+코드 강력)
또는 nomic-ai/CodeRankEmbed (137M, potion 원본 teacher) — 가벼움

학습 데이터 (Pod의 /workspace/v4_data/):
- CornStack 6언어 + The Stack v2 Rust/HTML/CSS + commitpack
- 한국어: AI Hub 011/020 + KLUE/PAWS-X (이미 Pod에 있음)

Pipeline:
    1) 데이터 로드 + 통합
    2) Fine-tune (CosineSimilarityLoss + MultipleNegativesRankingLoss 혼합)
    3) Vocab 빌드 (코드 토큰 + 한국어 토큰 = 500K)
    4) Model2Vec distill
    5) zip 다운로드

출력: keke-code-v1/ (정적 임베딩, ~500MB)
"""
from __future__ import annotations

import csv
import json
import random
import shutil
from itertools import combinations
from pathlib import Path

random.seed(42)

DATA_DIR = Path("/workspace")
V4_DATA = DATA_DIR / "v4_data"
OUT_DIR = Path("/workspace/keke-code-v1")
FT_DIR = Path("/workspace/keke-code-v1-finetuned")

NLI = {0: 1.0, 1: 0.5, 2: 0.0}

# Teacher 선택
TEACHER = "Snowflake/snowflake-arctic-embed-l-v2.0"  # 568M, 다국어 + 코드
# 대안: TEACHER = "nomic-ai/CodeRankEmbed"  # 137M, potion 원본

# 샘플링 한도
MAX_CODE_PAIRS_PER_LANG = 200_000  # 언어당 20만
MAX_KO_AIHUB_QA = 300_000
MAX_KO_AIHUB_SUMMARY = 200_000


def load_corn_stack(lang: str, max_n: int) -> list:
    """nomic-ai/CornStack-{lang}-v1 → InputExample 리스트."""
    from datasets import load_dataset
    from sentence_transformers import InputExample

    out = []
    try:
        ds = load_dataset(f"nomic-ai/CornStack-{lang}-v1", split="train")
        rows = list(ds)
        if len(rows) > max_n:
            rows = random.sample(rows, max_n)
        for row in rows:
            # CornStack 형식: query, code 또는 anchor, positive
            q = row.get("query") or row.get("anchor") or ""
            c = row.get("code") or row.get("positive") or ""
            if q and c:
                out.append(InputExample(texts=[q, c], label=1.0))
    except Exception as e:
        print(f"  CornStack-{lang} load failed: {e}", flush=True)
    return out


def load_the_stack_lang(lang_dir: str, max_n: int) -> list:
    """The Stack v2의 Rust/HTML/CSS 추출 → docstring·comment 페어."""
    from datasets import load_dataset
    from sentence_transformers import InputExample
    import re

    out = []
    try:
        ds = load_dataset(
            "bigcode/the-stack-v2-dedup",
            data_files=f"data/{lang_dir}/*.parquet",
            split="train",
            streaming=True,
        )
        count = 0
        for row in ds:
            if count >= max_n:
                break
            content = row.get("content", "")
            if not content or len(content) < 100:
                continue
            # 주석 추출 — 언어별
            if lang_dir == "Rust":
                # /// doc comment 또는 // 일반 주석
                comments = re.findall(r"///\s*(.+)", content)
            elif lang_dir == "HTML":
                # <title> 태그, <meta description>
                comments = re.findall(r"<title>(.+?)</title>", content)
                comments += re.findall(r'description"\s*content="([^"]+)"', content)
            elif lang_dir == "CSS":
                # /* 주석 */
                comments = re.findall(r"/\*\s*(.+?)\s*\*/", content)
            else:
                comments = []

            comments = [c.strip() for c in comments if 10 <= len(c.strip()) <= 200]
            if not comments:
                continue

            for c in comments[:3]:  # 파일당 최대 3 pair
                # 주석 + 코드 chunk(주변 200자) 페어
                snippet = content[:500]
                out.append(InputExample(texts=[c, snippet], label=1.0))
                count += 1
                if count >= max_n:
                    break
    except Exception as e:
        print(f"  the-stack/{lang_dir} load failed: {e}", flush=True)
    return out


def load_ko_data() -> list:
    """Pod에 이미 있는 한국어 데이터 재사용."""
    from sentence_transformers import InputExample

    out = []
    # AI Hub Q&A
    try:
        rows = []
        with (DATA_DIR / "aihub_qa.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                q = row["question"].strip()
                a = row["answer"].strip()
                if q and a:
                    rows.append((q, a))
        if len(rows) > MAX_KO_AIHUB_QA:
            rows = random.sample(rows, MAX_KO_AIHUB_QA)
        for q, a in rows:
            out.append(InputExample(texts=[q, a], label=0.7))
        print(f"  aihub-011-qa: {len(rows):,}", flush=True)
    except Exception as e:
        print(f"  aihub-qa skip: {e}", flush=True)

    # AI Hub 020
    try:
        rows = []
        with (DATA_DIR / "020_qa.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                q = row["question"].strip()
                a = row["answer"].strip()
                if q and a:
                    rows.append((q, a))
        if len(rows) > MAX_KO_AIHUB_QA:
            rows = random.sample(rows, MAX_KO_AIHUB_QA)
        for q, a in rows:
            out.append(InputExample(texts=[q, a], label=0.7))
        print(f"  aihub-020-qa: {len(rows):,}", flush=True)
    except Exception as e:
        print(f"  020-qa skip: {e}", flush=True)

    # Summary paraphrase
    try:
        with (DATA_DIR / "aihub_summary_pairs.csv").open(encoding="utf-8") as f:
            n = 0
            for row in csv.DictReader(f):
                if n >= MAX_KO_AIHUB_SUMMARY:
                    break
                out.append(InputExample(
                    texts=[row["sentence1"], row["sentence2"]],
                    label=float(row["score"]),
                ))
                n += 1
        print(f"  aihub-summary: {n:,}", flush=True)
    except Exception:
        pass

    return out


def main():
    from datasets import load_dataset
    from sentence_transformers import (
        InputExample, SentenceTransformer, losses,
    )
    from torch.utils.data import DataLoader

    print("=" * 60, flush=True)
    print("[1/4] Loading data — code + Korean", flush=True)
    print("=" * 60, flush=True)

    examples = []
    stats = {}

    def add(name, count):
        stats[name] = count
        print(f"  + {name}: {count:,}", flush=True)

    # KLUE/PAWS (의미 유사도 baseline)
    try:
        ds = load_dataset("klue", "sts", split="train")
        n = 0
        for row in ds:
            examples.append(InputExample(
                texts=[row["sentence1"], row["sentence2"]],
                label=row["labels"]["label"] / 5.0,
            ))
            n += 1
        add("klue-sts", n)
    except Exception as e:
        print(f"  klue-sts: {e}", flush=True)

    try:
        ds = load_dataset("paws-x", "ko", split="train")
        n = 0
        for row in ds:
            if row["label"] == 1:
                examples.append(InputExample(
                    texts=[row["sentence1"], row["sentence2"]],
                    label=1.0,
                ))
                n += 1
        add("paws-x-ko", n)
    except Exception:
        pass

    # CornStack 6 언어
    for lang in ["python", "java", "javascript", "go", "php", "ruby"]:
        sub = load_corn_stack(lang, MAX_CODE_PAIRS_PER_LANG)
        examples.extend(sub)
        add(f"cornstack-{lang}", len(sub))

    # The Stack v2 Rust/HTML/CSS
    for lang_dir in ["Rust", "HTML", "CSS"]:
        sub = load_the_stack_lang(lang_dir, MAX_CODE_PAIRS_PER_LANG)
        examples.extend(sub)
        add(f"the-stack-{lang_dir}", len(sub))

    # 한국어 (Pod 로컬)
    ko = load_ko_data()
    examples.extend(ko)
    add("ko-aihub-total", len(ko))

    random.shuffle(examples)
    total = len(examples)
    print(f"\n총 학습 쌍: {total:,}", flush=True)
    print(f"\n소스별:", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v:,}", flush=True)

    # Fine-tune
    BATCH = 64  # 더 큰 teacher라 batch 작게
    EPOCHS = 1
    print("\n" + "=" * 60, flush=True)
    print(f"[2/4] Fine-tuning {TEACHER}", flush=True)
    print(f"      ({total:,} pairs, batch={BATCH}, H200)", flush=True)
    print("=" * 60, flush=True)

    model = SentenceTransformer(TEACHER, device="cuda", trust_remote_code=True)
    loader = DataLoader(examples, shuffle=True, batch_size=BATCH)
    loss = losses.CosineSimilarityLoss(model)
    warmup = int(len(loader) * EPOCHS * 0.1)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=EPOCHS,
        warmup_steps=warmup,
        output_path=str(FT_DIR),
        show_progress_bar=True,
    )
    print("fine-tune done", flush=True)

    # Vocab 빌드 — 코드 토큰 + 한국어
    print("\n" + "=" * 60, flush=True)
    print("[3/4] Building extended vocab (code + Korean)", flush=True)
    print("=" * 60, flush=True)

    vocab = []
    # 한국어 vocab (Pod의 vocab_ko.txt)
    try:
        ko_vocab = [
            line.strip()
            for line in (DATA_DIR / "vocab_ko.txt").open(encoding="utf-8")
            if line.strip()
        ]
        vocab.extend(ko_vocab)
        print(f"  + korean vocab: {len(ko_vocab):,}", flush=True)
    except Exception:
        pass

    # 코드 vocab — 학습 데이터의 코드 부분에서 자주 등장하는 토큰
    # (간단히: 영어 단어·snake_case·camelCase 추출)
    import re
    code_tokens = {}
    for ex in random.sample(examples, min(100_000, len(examples))):
        for text in ex.texts:
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,30}", text):
                code_tokens[tok] = code_tokens.get(tok, 0) + 1
    code_vocab = [t for t, c in sorted(code_tokens.items(), key=lambda x: -x[1])[:200_000] if c >= 3]
    vocab.extend(code_vocab)
    print(f"  + code vocab: {len(code_vocab):,}", flush=True)

    # dedupe
    seen = set()
    final_vocab = []
    for w in vocab:
        if w not in seen:
            seen.add(w)
            final_vocab.append(w)
    print(f"  final vocab: {len(final_vocab):,}", flush=True)

    # Distill
    print("\n" + "=" * 60, flush=True)
    print("[4/4] Distilling to keke-code-v1 (model2vec)", flush=True)
    print("=" * 60, flush=True)
    from model2vec.distill import distill

    m2v = distill(
        model_name=str(FT_DIR),
        vocabulary=final_vocab,
        pca_dims=256,
        device="cuda",
    )
    m2v.save_pretrained(str(OUT_DIR))
    print(f"saved → {OUT_DIR} (vocab: {m2v.embedding.shape[0]:,})", flush=True)

    # Zip
    zip_path = shutil.make_archive(str(OUT_DIR), "zip", str(OUT_DIR))
    print(f"zip → {zip_path}", flush=True)

    (DATA_DIR / "keke_code_stats.json").write_text(
        json.dumps({
            "teacher": TEACHER,
            "total_pairs": total,
            "sources": stats,
            "final_vocab_size": len(final_vocab),
            "model_dim": int(m2v.embedding.shape[1]),
            "model_vocab": int(m2v.embedding.shape[0]),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("done.")


if __name__ == "__main__":
    main()
