"""AI Hub 일상대화 한국어 멀티세션 데이터 → kor-minish 학습 형식.

입력:
    ~/Downloads/011.일상대화 한국어 멀티세션 데이터/3.개방데이터/1.데이터/
        {Training,Validation}/{01.원천데이터,02.라벨링데이터}/*.zip

출력 (data/ 디렉토리):
    aihub_qa.csv          — speaker1↔speaker2 인접 turn (Q&A 쌍)
    aihub_paraphrase.jsonl — 같은 sessionKeywords 묶음 (intent별 paraphrase)
    aihub_summary_pairs.jsonl — sessionSummary 문장들 paraphrase
    aihub_stats.json      — 변환 통계

JSON 구조 (AI Hub 데이터):
    {
      "FileInfo": {...},
      "participantsInfo": {speaker1, speaker2},
      "sessionInfo": [
        {
          "sessionKeywords": ["topic"],
          "dialog": [{"speaker": "speaker1|2", "utt": "..."}],
          "sessionSummary": {"apprentice": [...], "wizard": [...]},
          ...
        }
      ]
    }
"""
from __future__ import annotations

import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterator

DATA_ROOT = Path.home() / "Downloads" / "011.일상대화 한국어 멀티세션 데이터" / "3.개방데이터" / "1.데이터"
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_UTT_LEN = 5
MAX_UTT_LEN = 200
MIN_GROUP_SIZE = 2
MAX_GROUP_SIZE = 50


def iter_session_files(root: Path) -> Iterator[tuple[str, dict]]:
    """라벨링 + 원천 zip 전부에서 JSON yield. 0바이트(미완료) zip은 skip."""
    for split in ["Training", "Validation"]:
        for kind in ["02.라벨링데이터", "01.원천데이터"]:
            zip_dir = root / split / kind
            if not zip_dir.exists():
                continue
            for zip_path in sorted(zip_dir.glob("*.zip")):
                if zip_path.stat().st_size == 0:
                    continue  # placeholder (INNORIX 미완료)
                try:
                    with zipfile.ZipFile(zip_path) as zf:
                        for name in zf.namelist():
                            if not (name.endswith(".txt") or name.endswith(".json")):
                                continue
                            try:
                                with zf.open(name) as f:
                                    data = json.load(f)
                                yield f"{split}/{kind}/{zip_path.name}:{name}", data
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                continue
                except zipfile.BadZipFile:
                    print(f"  bad zip: {zip_path}")


def valid_utt(s: str) -> bool:
    s = s.strip() if isinstance(s, str) else ""
    return MIN_UTT_LEN <= len(s) <= MAX_UTT_LEN


def extract_qa_pairs(data: dict) -> list[tuple[str, str]]:
    """speaker1 ↔ speaker2 인접 turn → Q&A 쌍."""
    pairs: list[tuple[str, str]] = []
    for session in data.get("sessionInfo", []):
        dialog = session.get("dialog", [])
        for i in range(len(dialog) - 1):
            u1 = dialog[i]
            u2 = dialog[i + 1]
            if not (isinstance(u1, dict) and isinstance(u2, dict)):
                continue
            if u1.get("speaker") == u2.get("speaker"):
                continue  # 같은 화자 연속 발화 제외
            q = (u1.get("utterance") or "").strip()
            a = (u2.get("utterance") or "").strip()
            if valid_utt(q) and valid_utt(a):
                pairs.append((q, a))
    return pairs


def extract_topic_groups(data: dict, groups: dict[str, list[str]]) -> None:
    """같은 sessionKeywords의 utterances를 누적."""
    for session in data.get("sessionInfo", []):
        keywords = session.get("sessionKeywords", []) or []
        if not keywords:
            continue
        topic = keywords[0].strip("*").split(",")[0].strip()
        if not topic:
            continue
        for turn in session.get("dialog", []):
            if isinstance(turn, dict):
                utt = (turn.get("utterance") or "").strip()
                if valid_utt(utt):
                    groups[topic].append(utt)


def extract_summary_pairs(data: dict) -> list[tuple[str, str]]:
    """sessionSummary 안에서 apprentice 또는 wizard 문장끼리 paraphrase (같은 페르소나)."""
    pairs: list[tuple[str, str]] = []
    for session in data.get("sessionInfo", []):
        summary = session.get("sessionSummary", {}) or {}
        for role in ("apprentice", "wizard"):
            sents = [s.strip() for s in (summary.get(role) or []) if valid_utt(s)]
            for i in range(len(sents)):
                for j in range(i + 1, len(sents)):
                    pairs.append((sents[i], sents[j]))
    return pairs


def main() -> None:
    if not DATA_ROOT.exists():
        raise SystemExit(f"data root not found: {DATA_ROOT}")

    qa_pairs: list[tuple[str, str]] = []
    summary_pairs: list[tuple[str, str]] = []
    topic_groups: dict[str, list[str]] = defaultdict(list)

    n_files = 0
    n_sessions = 0
    print(f"scanning: {DATA_ROOT}")
    for src, data in iter_session_files(DATA_ROOT):
        n_files += 1
        n_sessions += len(data.get("sessionInfo", []))
        qa_pairs.extend(extract_qa_pairs(data))
        summary_pairs.extend(extract_summary_pairs(data))
        extract_topic_groups(data, topic_groups)
        if n_files % 1000 == 0:
            print(f"  processed {n_files:,} files, {len(qa_pairs):,} qa pairs")

    print(f"\ntotal: {n_files:,} files, {n_sessions:,} sessions")

    # 1) Q&A — kor-minish의 'qa' 형식 (csv)
    qa_csv = OUT_DIR / "aihub_qa.csv"
    seen_qa: set[tuple[str, str]] = set()
    with qa_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "answer"])
        for q, a in qa_pairs:
            key = (q, a)
            if key in seen_qa:
                continue
            seen_qa.add(key)
            w.writerow([q, a])
    print(f"Q&A pairs (deduped): {len(seen_qa):,} → {qa_csv}")

    # 2) Topic groups — paraphrase 형식 (jsonl)
    para_jsonl = OUT_DIR / "aihub_paraphrase.jsonl"
    n_groups = 0
    with para_jsonl.open("w", encoding="utf-8") as f:
        for topic, utts in topic_groups.items():
            unique = list(dict.fromkeys(utts))  # 순서 보존 dedupe
            if len(unique) < MIN_GROUP_SIZE:
                continue
            f.write(json.dumps({
                "intent": topic,
                "examples": unique[:MAX_GROUP_SIZE],
            }, ensure_ascii=False) + "\n")
            n_groups += 1
    print(f"Topic groups: {n_groups:,} → {para_jsonl}")

    # 3) Summary paraphrase pairs (STS 형식 csv)
    sum_csv = OUT_DIR / "aihub_summary_pairs.csv"
    seen_sum: set[tuple[str, str]] = set()
    with sum_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sentence1", "sentence2", "score"])
        for s1, s2 in summary_pairs:
            key = (s1, s2) if s1 < s2 else (s2, s1)
            if key in seen_sum:
                continue
            seen_sum.add(key)
            w.writerow([s1, s2, "1.0"])
    print(f"Summary paraphrase pairs: {len(seen_sum):,} → {sum_csv}")

    # 4) 통계
    stats = {
        "n_files": n_files,
        "n_sessions": n_sessions,
        "n_qa_pairs": len(seen_qa),
        "n_topic_groups": n_groups,
        "n_summary_pairs": len(seen_sum),
        "top_topics": sorted(
            ((t, len(utts)) for t, utts in topic_groups.items()),
            key=lambda x: -x[1],
        )[:20],
    }
    stats_path = OUT_DIR / "aihub_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nstats → {stats_path}")
    print("\ntop 10 topics by utterance count:")
    for t, c in stats["top_topics"][:10]:
        print(f"  {c:>6,}  {t}")


if __name__ == "__main__":
    main()
