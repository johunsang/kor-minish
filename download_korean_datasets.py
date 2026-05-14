"""HuggingFace에서 한국어 STS/NLI/Paraphrase 데이터셋을 모두 외장하드에 저장.

저장 위치: data/raw/<repo>__<config>/
통합본:    data/train_pairs.jsonl  ({sentence1, sentence2, score, source} per line)

학습 데이터 통일 스키마:
    sentence1: str
    sentence2: str
    score:     float (0.0 ~ 1.0)
    source:    str
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

OUT = Path("data/raw")
TRAIN_JSONL = Path("data/train_pairs.jsonl")
OUT.mkdir(parents=True, exist_ok=True)
TRAIN_JSONL.parent.mkdir(parents=True, exist_ok=True)

NLI_SCORE = {
    0: 1.0, 1: 0.5, 2: 0.0,
    "entailment": 1.0, "neutral": 0.5, "contradiction": 0.0,
}


def _safe_save(ds, name: str) -> None:
    target = OUT / name
    if target.exists():
        return
    ds.save_to_disk(str(target))


def _emit(f, source: str, s1, s2, score) -> int:
    if not s1 or not s2:
        return 0
    score = max(0.0, min(1.0, float(score)))
    f.write(json.dumps({
        "sentence1": s1, "sentence2": s2, "score": score, "source": source,
    }, ensure_ascii=False) + "\n")
    return 1


def main() -> None:
    stats: dict[str, int] = {}
    with TRAIN_JSONL.open("w", encoding="utf-8") as f:

        # 1) KLUE-STS
        try:
            ds = load_dataset("klue", "sts")
            _safe_save(ds, "klue__sts")
            n = 0
            for row in ds["train"]:
                n += _emit(f, "klue-sts", row["sentence1"], row["sentence2"],
                          row["labels"]["label"] / 5.0)
            stats["klue-sts"] = n
            print(f"klue-sts: {n:,}")
        except Exception as e:
            print(f"klue-sts FAIL: {e}")

        # 2) KLUE-NLI
        try:
            ds = load_dataset("klue", "nli")
            _safe_save(ds, "klue__nli")
            n = 0
            for row in ds["train"]:
                n += _emit(f, "klue-nli", row["premise"], row["hypothesis"],
                          NLI_SCORE.get(row["label"], 0.5))
            stats["klue-nli"] = n
            print(f"klue-nli: {n:,}")
        except Exception as e:
            print(f"klue-nli FAIL: {e}")

        # 3) KorSTS
        for cfg in ["sts", "kor_sts"]:
            try:
                ds = load_dataset("kor_nlu", cfg)
                _safe_save(ds, f"kor_nlu__{cfg}")
                n = 0
                for row in ds["train"]:
                    raw = row.get("score", row.get("label", 0))
                    n += _emit(f, f"kor_nlu/{cfg}", row["sentence1"],
                              row["sentence2"], float(raw) / 5.0)
                stats[f"kor_nlu/{cfg}"] = n
                print(f"kor_nlu/{cfg}: {n:,}")
                break
            except Exception as e:
                print(f"kor_nlu/{cfg} FAIL: {e}")

        # 4) KorNLI
        for cfg in ["multi_nli", "kor_nli", "snli"]:
            try:
                ds = load_dataset("kor_nlu", cfg)
                _safe_save(ds, f"kor_nlu__{cfg}")
                n = 0
                for row in ds["train"]:
                    lbl = row.get("gold_label", row.get("label"))
                    n += _emit(f, f"kor_nlu/{cfg}", row["sentence1"],
                              row["sentence2"], NLI_SCORE.get(lbl, 0.5))
                stats[f"kor_nlu/{cfg}"] = n
                print(f"kor_nlu/{cfg}: {n:,}")
                break
            except Exception as e:
                print(f"kor_nlu/{cfg} FAIL: {e}")

        # 5) KorQPair
        try:
            ds = load_dataset("kor_qpair")
            _safe_save(ds, "kor_qpair")
            n = 0
            for row in ds["train"]:
                n += _emit(f, "kor_qpair", row["question1"], row["question2"],
                          1.0 if row["is_duplicate"] else 0.0)
            stats["kor_qpair"] = n
            print(f"kor_qpair: {n:,}")
        except Exception as e:
            print(f"kor_qpair FAIL: {e}")

        # 6) xnli 한국어 (다국어 NLI의 ko split)
        try:
            ds = load_dataset("xnli", "ko")
            _safe_save(ds, "xnli__ko")
            n = 0
            for row in ds["train"]:
                lbl = row.get("label")
                n += _emit(f, "xnli-ko", row["premise"], row["hypothesis"],
                          NLI_SCORE.get(lbl, 0.5))
            stats["xnli-ko"] = n
            print(f"xnli-ko: {n:,}")
        except Exception as e:
            print(f"xnli-ko FAIL: {e}")

        # 7) paws-x 한국어 (paraphrase)
        try:
            ds = load_dataset("paws-x", "ko")
            _safe_save(ds, "paws-x__ko")
            n = 0
            for row in ds["train"]:
                # paws-x: label 1=paraphrase, 0=not
                score = 1.0 if row["label"] == 1 else 0.0
                n += _emit(f, "paws-x-ko", row["sentence1"], row["sentence2"], score)
            stats["paws-x-ko"] = n
            print(f"paws-x-ko: {n:,}")
        except Exception as e:
            print(f"paws-x-ko FAIL: {e}")

        # 8) MoritzLaurer 다국어 NLI (한국어 포함)
        try:
            ds = load_dataset("MoritzLaurer/multilingual-NLI-26lang-2mil7", "ko_nli")
            split = "train" if "train" in ds else list(ds.keys())[0]
            _safe_save(ds, "multi-NLI-26lang__ko_nli")
            n = 0
            for row in ds[split]:
                lbl = row.get("label", row.get("gold_label"))
                n += _emit(f, "multi-NLI-26lang-ko", row["premise"], row["hypothesis"],
                          NLI_SCORE.get(lbl, 0.5))
            stats["multi-NLI-26lang-ko"] = n
            print(f"multi-NLI-26lang-ko: {n:,}")
        except Exception as e:
            print(f"multi-NLI-26lang-ko skip: {e}")

        # 9) 기타 후보 (없을 가능성 큼)
        for repo in [
            "nayohan/Korean-STS-NLI",
            "tabtoyou/KoCommonGEN",
            "jhgan/kor_para",
            "Lee-Soohyun/korean_paraphrase",
        ]:
            try:
                ds = load_dataset(repo)
                _safe_save(ds, repo.replace("/", "__"))
                split = "train" if "train" in ds else list(ds.keys())[0]
                sample = ds[split][0]
                fields = list(sample.keys())
                s1k = next((k for k in ["sentence1", "premise", "s1", "question1"] if k in fields), None)
                s2k = next((k for k in ["sentence2", "hypothesis", "s2", "question2"] if k in fields), None)
                sck = next((k for k in ["score", "label", "similarity"] if k in fields), None)
                if not (s1k and s2k):
                    print(f"{repo}: schema unknown {fields}, skipping")
                    continue
                n = 0
                for row in ds[split]:
                    raw = row.get(sck, 0.5) if sck else 0.5
                    if isinstance(raw, str):
                        raw = NLI_SCORE.get(raw, 0.5)
                    score = float(raw) / 5.0 if isinstance(raw, (int, float)) and raw > 1 else float(raw)
                    n += _emit(f, repo, row[s1k], row[s2k], score)
                stats[repo] = n
                print(f"{repo}: {n:,}")
            except Exception as e:
                print(f"{repo} skip: {e}")

    total = sum(stats.values())
    print(f"\n총 {total:,} pairs → {TRAIN_JSONL}")
    print("source별:")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
