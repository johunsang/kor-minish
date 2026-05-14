"""kor-minish CLI — Ollama 스타일 한 줄 명령으로 임베딩 사용.

사용법:
    kor-minish encode "안녕하세요"
    kor-minish encode "문장1" "문장2" "문장3"
    echo "안녕하세요" | kor-minish encode
    kor-minish similarity "한국 음식" "김치찌개 레시피" "주식 매수" "된장국"
    kor-minish summary article.txt --top 3
    cat article.txt | kor-minish summary --top 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from model2vec import StaticModel

DEFAULT_MODEL = "hysnnnn/kor-minish-bge-m3-ko"


def _read_stdin_lines() -> list[str]:
    if sys.stdin.isatty():
        return []
    data = sys.stdin.read()
    return [line.strip() for line in data.splitlines() if line.strip()]


def _resolve_texts(positional: list[str]) -> list[str]:
    if positional:
        return positional
    lines = _read_stdin_lines()
    if not lines:
        sys.exit("no input — pass texts as args or pipe via stdin")
    return lines


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。…])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def cmd_encode(model: StaticModel, args: argparse.Namespace) -> None:
    texts = _resolve_texts(args.texts)
    vecs = model.encode(texts)
    if args.format == "json":
        print(json.dumps(
            {"dim": int(vecs.shape[1]), "embeddings": vecs.tolist()},
            ensure_ascii=False,
        ))
    else:
        for text, vec in zip(texts, vecs):
            print(f"# {text}")
            print(" ".join(f"{x:.6f}" for x in vec))


def cmd_similarity(model: StaticModel, args: argparse.Namespace) -> None:
    docs = args.docs if args.docs else _read_stdin_lines()
    if not docs:
        sys.exit("no documents — pass as args or pipe via stdin")

    vecs = model.encode([args.query, *docs])
    q = vecs[0] / max(np.linalg.norm(vecs[0]), 1e-9)
    d = vecs[1:] / np.linalg.norm(vecs[1:], axis=1, keepdims=True).clip(min=1e-9)
    scores = (d @ q).tolist()
    order = sorted(range(len(scores)), key=lambda i: -scores[i])

    if args.format == "json":
        print(json.dumps({
            "query": args.query,
            "scores": scores,
            "order": order,
            "docs": docs,
        }, ensure_ascii=False))
    else:
        for rank, idx in enumerate(order):
            print(f"{rank + 1:>2}. {scores[idx]:+.3f}  {docs[idx]}")


def cmd_summary(model: StaticModel, args: argparse.Namespace) -> None:
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        sys.exit("no text — pass a file path or pipe via stdin")

    sents = _split_sentences(text)
    if not sents:
        sys.exit("no sentences found")
    if len(sents) <= args.top:
        for s in sents:
            print(s)
        return

    vecs = model.encode(sents)
    norm = vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
    doc_centroid = norm.mean(axis=0)
    doc_centroid /= max(np.linalg.norm(doc_centroid), 1e-9)
    scores = norm @ doc_centroid

    top_idx = sorted(np.argsort(-scores)[: args.top].tolist())
    if args.format == "json":
        print(json.dumps({
            "summary": [sents[i] for i in top_idx],
            "scores": [float(scores[i]) for i in top_idx],
        }, ensure_ascii=False))
    else:
        for i in top_idx:
            print(sents[i])


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kor-minish",
        description="Korean static embedding CLI (model2vec-based)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"HF repo or local path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text",
        dest="global_format",
        help="output format (default: text)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    def add_common(p):
        p.add_argument("--format", choices=["text", "json"], help="output format (overrides global)")

    p_enc = sub.add_parser("encode", help="텍스트를 임베딩 벡터로 변환")
    p_enc.add_argument("texts", nargs="*", help="문장(들). 없으면 stdin")
    add_common(p_enc)
    p_enc.set_defaults(func=cmd_encode)

    p_sim = sub.add_parser("similarity", help="쿼리-문서 유사도 랭킹")
    p_sim.add_argument("query", help="검색 쿼리")
    p_sim.add_argument("docs", nargs="*", help="후보 문서들. 없으면 stdin")
    add_common(p_sim)
    p_sim.set_defaults(func=cmd_similarity)

    p_sum = sub.add_parser("summary", help="추출적 요약 (중요 문장 선택)")
    p_sum.add_argument("file", nargs="?", help="텍스트 파일. 없으면 stdin")
    p_sum.add_argument("--top", type=int, default=3, help="추출할 문장 수")
    add_common(p_sum)
    p_sum.set_defaults(func=cmd_summary)

    from kor_minish.train import add_subparser as add_train_subparser
    add_train_subparser(sub)

    args = parser.parse_args()
    if not hasattr(args, "format") or args.format is None:
        args.format = getattr(args, "global_format", "text")

    if getattr(args, "_needs_model", True):
        model = StaticModel.from_pretrained(args.model)
        args.func(model, args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
