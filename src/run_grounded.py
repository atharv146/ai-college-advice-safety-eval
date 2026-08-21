#!/usr/bin/env python
"""
Runs the prompt bank through the grounded (retrieval + forced-citation)
condition and appends the results to results/raw_responses.jsonl with
condition="grounded" -- same file, same schema as run_harness.py's vanilla
and guarded rows, so judge.py and analyze.py pick this up automatically
as a third condition with no changes needed there.

This is the "propose and test a fix" half of the project: vanilla and
guarded measure whether a model gets it right on its own. Grounded asks
whether forcing the model to answer only from verified, cited source
material beats both, which is the actual proposal this project is making,
not just a problem report.

Usage:
  python src/run_grounded.py --sample-per-category 4
"""

import argparse
import concurrent.futures
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import MODELS_UNDER_TEST
from src.openrouter_client import complete
from src.rag import GROUNDED_SYSTEM_PROMPT, Retriever, build_user_message, load_corpus

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RAW_PATH = RESULTS_DIR / "raw_responses.jsonl"

CONDITION = "grounded"


def load_prompts(sample_per_category: int | None) -> list[dict]:
    prompts = json.loads((DATA_DIR / "prompts.json").read_text())
    if sample_per_category is None:
        return prompts
    rng = random.Random(42)  # same seed as run_harness.py, same sample
    by_cat: dict[str, list[dict]] = {}
    for p in prompts:
        by_cat.setdefault(p["category"], []).append(p)
    sampled = []
    for items in by_cat.values():
        rng.shuffle(items)
        sampled.extend(items[:sample_per_category])
    return sampled


def already_done() -> set[tuple[str, str, str]]:
    if not RAW_PATH.exists():
        return set()
    done = set()
    for line in RAW_PATH.read_text().splitlines():
        row = json.loads(line)
        if row.get("error") is None:
            done.add((row["prompt_id"], row["model"], row["condition"]))
    return done


def run_one(prompt: dict, model: str, retriever: Retriever) -> dict:
    retrieved = retriever.retrieve(prompt["text"], k=2)
    user_msg = build_user_message(prompt["text"], retrieved)
    t0 = time.time()
    try:
        text = complete(model, GROUNDED_SYSTEM_PROMPT, user_msg)
        error = None
    except Exception as e:  # noqa: BLE001
        text = None
        error = str(e)
    return {
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "prompt_text": prompt["text"],
        "model": model,
        "condition": CONDITION,
        "response": text,
        "error": error,
        "latency_s": round(time.time() - t0, 2),
        "retrieved_chunks": [r["id"] for r in retrieved],
        "retrieval_similarities": [r["similarity"] for r in retrieved],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-per-category", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    corpus = load_corpus()
    retriever = Retriever(corpus)

    prompts = load_prompts(args.sample_per_category)
    jobs = [(p, model) for p in prompts for model in MODELS_UNDER_TEST]

    if args.resume:
        done = already_done()
        jobs = [j for j in jobs if (j[0]["id"], j[1], CONDITION) not in done]

    print(f"{len(prompts)} prompts x {len(MODELS_UNDER_TEST)} models = {len(jobs)} grounded calls")

    completed = 0
    with open(RAW_PATH, "a") as f, concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = [pool.submit(run_one, p, m, retriever) for p, m in jobs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            completed += 1
            status = "ERROR" if row["error"] else "ok"
            print(f"  [{completed}/{len(jobs)}] {row['model']:35s} {row['prompt_id']:8s} {status}")

    print(f"\nDone. {completed} grounded rows appended to {RAW_PATH}")


if __name__ == "__main__":
    main()
