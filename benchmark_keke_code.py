"""keke-code-v1 정식 벤치마크.

평가 대상:
1. MTEB-CoIR (Code Information Retrieval) — 영어 코드 검색 표준
   - CodeSearchNet (CSN) — 코드 검색 baseline
   - CosQA — natural language → code
   - CodeFeedback ST/MT — 코드 생성 검색
   - StackOverflow QA — 코드 답변 검색
   - APPS, CodeTrans-Contest

2. MTEB Korean — 한국어 STS/Retrieval
   - KLUE-STS-K
   - 그 외 Korean retrieval tasks

3. 자체 평가
   - 우리 CS FAQ 1500 (top-1/3/5)
   - 코드 vs 한국어 같은 모델로 처리되는지

비교: potion-code-16M, kor-minish-bge-m3-ko (v1), keke-code-v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_mteb(model_name: str, tasks: list[str], output_folder: str) -> dict:
    """MTEB 벤치마크 실행."""
    import mteb
    from sentence_transformers import SentenceTransformer

    # model2vec 정적 모델은 SentenceTransformer wrapper로 mteb에 줘야 함
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
    except Exception:
        # 정적 모델이면 어댑터 필요
        from model2vec import StaticModel
        m2v = StaticModel.from_pretrained(model_name)

        class M2VWrapper:
            def __init__(self, m):
                self.m = m

            def encode(self, sentences, batch_size=256, **kwargs):
                return self.m.encode(sentences, batch_size=batch_size)

        model = M2VWrapper(m2v)

    selected = mteb.get_tasks(tasks=tasks)
    evaluation = mteb.MTEB(tasks=selected)
    results = evaluation.run(
        model,
        output_folder=output_folder,
        verbosity=1,
        overwrite_results=False,
    )
    return {t.metadata.name: r for t, r in zip(selected, results)}


def coir_benchmark(model_name: str, output: Path) -> dict:
    """MTEB-CoIR (코드 검색)."""
    tasks = [
        "CodeSearchNetRetrieval",
        "CosQA",
        "CodeFeedbackST",
        "CodeFeedbackMT",
        "StackOverflowQA",
        "AppsRetrieval",
        "CodeTransOceanContest",
        "CodeTransOceanDL",
    ]
    return run_mteb(model_name, tasks, str(output / "coir"))


def korean_benchmark(model_name: str, output: Path) -> dict:
    """MTEB 한국어 (있는 task만)."""
    tasks = [
        "KLUE-STS",
        "MIRACLRetrieval",  # 다국어 포함, ko subset
    ]
    return run_mteb(model_name, tasks, str(output / "korean"))


def custom_cs_faq(model_name: str) -> dict:
    """우리 자체 CS FAQ 1500."""
    from model2vec import StaticModel
    import numpy as np

    faqs_path = Path("data/cs_faqs.jsonl")
    if not faqs_path.exists():
        return {"error": "data/cs_faqs.jsonl not found"}

    intents = {}
    with faqs_path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj.get("examples"):
                intents[obj["intent"]] = obj["examples"]

    try:
        m = StaticModel.from_pretrained(model_name)
    except Exception as e:
        return {"error": str(e)}

    # 각 intent의 첫 예시를 query, 나머지를 풀로
    flat_texts, flat_intents = [], []
    for intent, ex_list in intents.items():
        for ex in ex_list[1:]:
            flat_texts.append(ex)
            flat_intents.append(intent)
    db_vecs = m.encode(flat_texts)
    db_norm = db_vecs / np.linalg.norm(db_vecs, axis=1, keepdims=True).clip(min=1e-9)

    queries = [v[0] for v in intents.values()]
    truth = list(intents.keys())
    q_vecs = m.encode(queries)
    q_norm = q_vecs / np.linalg.norm(q_vecs, axis=1, keepdims=True).clip(min=1e-9)
    sims = q_norm @ db_norm.T

    top1 = top3 = top5 = 0
    for i, t in enumerate(truth):
        top_idx = np.argsort(-sims[i])[:5]
        top_intents = [flat_intents[j] for j in top_idx]
        if top_intents[0] == t:
            top1 += 1
        if t in top_intents[:3]:
            top3 += 1
        if t in top_intents[:5]:
            top5 += 1
    n = len(truth)
    return {
        "n_queries": n,
        "top1_acc": top1 / n,
        "top3_acc": top3 / n,
        "top5_acc": top5 / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[
        "hysnnnn/keke-code-v1",
        "minishlab/potion-code-16M",  # baseline
        "hysnnnn/kor-minish-bge-m3-ko",  # v1 (한국어만)
    ])
    parser.add_argument("--output", default="benchmark_results")
    parser.add_argument("--skip-mteb", action="store_true",
                        help="Skip MTEB (custom only)")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    summary = {}
    for m in args.models:
        print(f"\n{'='*60}\n  {m}\n{'='*60}")
        summary[m] = {}

        # Custom CS FAQ
        print("\n[1] Custom CS FAQ (Korean, 150 intents x 10 paraphrase)")
        cs = custom_cs_faq(m)
        summary[m]["cs_faq"] = cs
        for k, v in cs.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")

        if args.skip_mteb:
            continue

        # MTEB-CoIR
        print("\n[2] MTEB-CoIR (code retrieval)")
        try:
            coir = coir_benchmark(m, output / m.replace("/", "_"))
            summary[m]["coir"] = coir
        except Exception as e:
            print(f"  CoIR failed: {e}")

        # MTEB Korean
        print("\n[3] MTEB Korean")
        try:
            kor = korean_benchmark(m, output / m.replace("/", "_"))
            summary[m]["korean"] = kor
        except Exception as e:
            print(f"  Korean failed: {e}")

    # Summary
    sum_path = output / "summary.json"
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n\nsummary → {sum_path}")

    # Compare table
    print(f"\n{'='*60}")
    print("  Comparison")
    print('='*60)
    print(f"{'Model':45s}  Top-1  Top-3  Top-5")
    for m, r in summary.items():
        cs = r.get("cs_faq", {})
        if "top1_acc" in cs:
            print(f"{m[:45]:45s}  {cs['top1_acc']:.1%}  "
                  f"{cs['top3_acc']:.1%}  {cs['top5_acc']:.1%}")


if __name__ == "__main__":
    main()
