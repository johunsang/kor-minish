"""사용자 학습 데이터를 InputExample 리스트로 변환.

자동 감지하는 세 가지 형식:

1) Paraphrase 그룹 (.jsonl): 같은 intent끼리 묶기 (FAQ/CS 추천)
   {"intent": "refund", "examples": ["환불", "돈 돌려주세요", "결제 취소"]}

2) Q&A 쌍 (.csv): 질문-답변. 같은 답변 → 같은 의도로 자동 묶음
   question,answer
   환불은 어떻게?,주문 페이지에서...

3) STS 점수 (.csv): 문장1-문장2-점수(0~1)
   sentence1,sentence2,score
   환불 신청,돈 돌려받기,1.0
"""
from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Iterator

from sentence_transformers import InputExample


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def detect_format(path: Path) -> str:
    """파일 확장자 + 첫 줄 스키마로 형식 자동 감지."""
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "examples" in obj and "intent" in obj:
                    return "paraphrase"
                if "sentence1" in obj and "sentence2" in obj:
                    return "sts"
                break
        raise ValueError(f"unsupported jsonl schema in {path}")

    if path.suffix.lower() == ".csv":
        fields, _ = _read_csv(path)
        if {"sentence1", "sentence2", "score"} <= set(fields):
            return "sts"
        if {"sentence1", "sentence2", "label"} <= set(fields):
            return "sts"
        if {"question", "answer"} <= set(fields):
            return "qa"
        raise ValueError(f"unsupported csv schema in {path}: {fields}")

    raise ValueError(f"unsupported file type: {path.suffix}")


def load_paraphrase(path: Path) -> list[InputExample]:
    """Paraphrase 그룹 → 그룹 내 모든 쌍 positive(1.0), 그룹 간 쌍은 학습에서 자동 negative."""
    examples: list[InputExample] = []
    for obj in _read_jsonl(path):
        ex_list = obj.get("examples", [])
        # 같은 그룹 내 모든 쌍 → positive
        for a, b in combinations(ex_list, 2):
            examples.append(InputExample(texts=[a, b], label=1.0))
    return examples


def load_qa(path: Path) -> list[InputExample]:
    """Q&A → 같은 답변을 공유하는 질문들을 paraphrase 그룹으로 자동 묶음."""
    _, rows = _read_csv(path)
    by_answer: dict[str, list[str]] = {}
    for row in rows:
        a = row["answer"].strip()
        q = row["question"].strip()
        if not a or not q:
            continue
        by_answer.setdefault(a, []).append(q)

    examples: list[InputExample] = []
    for questions in by_answer.values():
        for a, b in combinations(questions, 2):
            examples.append(InputExample(texts=[a, b], label=1.0))
        # 질문과 답변도 약한 양성으로 (선택)
        if questions:
            for q in questions:
                examples.append(InputExample(texts=[q, by_answer_key(by_answer, q)], label=0.7))
    return examples


def by_answer_key(by_answer: dict, q: str) -> str:
    for ans, qs in by_answer.items():
        if q in qs:
            return ans
    return ""


def load_sts(path: Path) -> list[InputExample]:
    """STS 점수 형식 (csv 또는 jsonl)."""
    examples: list[InputExample] = []
    if path.suffix.lower() == ".csv":
        _, rows = _read_csv(path)
        for row in rows:
            score = float(row.get("score", row.get("label", 0.5)))
            score = max(0.0, min(1.0, score))
            examples.append(InputExample(
                texts=[row["sentence1"].strip(), row["sentence2"].strip()],
                label=score,
            ))
    else:
        for obj in _read_jsonl(path):
            score = float(obj.get("score", obj.get("label", 0.5)))
            score = max(0.0, min(1.0, score))
            examples.append(InputExample(
                texts=[obj["sentence1"], obj["sentence2"]], label=score,
            ))
    return examples


def load_training_data(path: str | Path) -> tuple[list[InputExample], str]:
    """파일 경로 → (examples, detected_format)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    fmt = detect_format(p)
    loader = {"paraphrase": load_paraphrase, "qa": load_qa, "sts": load_sts}[fmt]
    return loader(p), fmt
