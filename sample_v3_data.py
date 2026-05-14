"""v3 학습용 데이터 통합·샘플링.

AI Hub 데이터에서 학습에 효율적인 양만 선별 (GitHub commit 가능한 사이즈로):
- aihub_qa: 122만 중 50K (랜덤)
- aihub_summary_pairs: 42만 중 30K
- aihub_paraphrase: 45K 그룹 중 10K (그룹당 utt 최대 8개)

출력: data/v3/*.jsonl, *.csv (Colab notebook이 GitHub raw에서 받음)
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

random.seed(42)

SRC = Path("data")
OUT = Path("data/v3")
OUT.mkdir(parents=True, exist_ok=True)


def sample_csv(src: Path, dst: Path, n: int) -> None:
    with src.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []
    if len(rows) > n:
        rows = random.sample(rows, n)
    with dst.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"  {src.name}: {len(rows):,} → {dst}")


def sample_paraphrase(src: Path, dst: Path, max_groups: int, max_examples: int) -> None:
    groups = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if "examples" in obj and len(obj["examples"]) >= 2:
                    groups.append(obj)
    random.shuffle(groups)
    groups = groups[:max_groups]
    with dst.open("w", encoding="utf-8") as f:
        for g in groups:
            g["examples"] = g["examples"][:max_examples]
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"  {src.name}: {len(groups):,} groups → {dst}")


def main() -> None:
    print("샘플링 시작 (seed=42)...")
    sample_csv(SRC / "aihub_qa.csv", OUT / "aihub_qa_sample.csv", 50_000)
    sample_csv(SRC / "aihub_summary_pairs.csv", OUT / "aihub_summary_sample.csv", 30_000)
    sample_paraphrase(SRC / "aihub_paraphrase.jsonl", OUT / "aihub_paraphrase_sample.jsonl",
                      max_groups=10_000, max_examples=8)
    # vocab_ko.txt 복사 (Colab이 받기 쉽게)
    import shutil
    if Path("vocab_ko.txt").exists():
        shutil.copy("vocab_ko.txt", OUT / "vocab_ko.txt")
        print(f"  vocab_ko.txt → {OUT/'vocab_ko.txt'}")
    # 통계
    total_size = sum(p.stat().st_size for p in OUT.glob("*"))
    print(f"\n총 출력 사이즈: {total_size/1e6:.1f} MB ({total_size:,} bytes)")
    print(f"파일들 → {OUT.resolve()}")


if __name__ == "__main__":
    main()
