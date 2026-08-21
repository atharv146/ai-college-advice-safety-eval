#!/usr/bin/env python
"""
Runs every prompt in data/prompts.json through every model in
src/models.py, under both the vanilla and guarded system prompts, and
saves the raw responses to results/raw_responses.jsonl.

Usage:
  python src/run_harness.py                  # everything
  python src/run_harness.py --sample-per-category 4   # a stratified subset

Each row already contains its own metadata, so the harness can be re-run
incrementally and results just get appended (with a --resume flag to skip
rows already collected).
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RAW_PATH = RESULTS_DIR / "raw_responses.jsonl"

sys.path.insert(0, str(DATA_DIR))
from system_prompts import GUARDED_SYSTEM_PROMPT, VANILLA_SYSTEM_PROMPT  # noqa: E402

CONDITIONS = {
    "vanilla": VANILLA_SYSTEM_PROMPT,
    "guarded": GUARDED_SYSTEM_PROMPT,
}


def load_prompts(sample_per_category: int | None) -> list[dict]:
    prompts = json.loads((DATA_DIR / "prompts.json").read_text())
    if sample_per_category is None:
        return prompts

    rng = random.Random(42)
    by_cat: dict[str, list[dict]] = {}
    for p in prompts:
        by_cat.setdefault(p["category"], []).append(p)

    sampled = []
    for cat, items in by_cat.items():
        rng.shuffle(items)
        sampled.extend(items[:sample_per_category])
    return sampled


def already_done() -> set[tuple[str, str, str]]:
    """Only counts rows that actually succeeded — a --resume run should retry
    anything that errored (usually a free-tier rate limit), not skip it."""
    if not RAW_PATH.exists():
        return set()
    done = set()
    for line in RAW_PATH.read_text().splitlines():
        row = json.loads(line)
        if row.get("error") is None:
            done.add((row["prompt_id"], row["model"], row["condition"]))
    return done


def run_one(prompt: dict, model: str, condition: str) -> dict:
    system = CONDITIONS[condition]
    t0 = time.time()
    try:
        text = complete(model, system, prompt["text"])
        error = None
    except Exception as e:  # noqa: BLE001
        text = None
        error = str(e)
    return {
        "prompt_id": prompt["id"],
        "category": prompt["category"],
        "prompt_text": prompt["text"],
        "model": model,
        "condition": condition,
        "response": text,
        "error": error,
        "latency_s": round(time.time() - t0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-per-category", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    prompts = load_prompts(args.sample_per_category)
    jobs = [
        (p, model, condition)
        for p in prompts
        for model in MODELS_UNDER_TEST
        for condition in CONDITIONS
    ]

    if args.resume:
        done = already_done()
        jobs = [j for j in jobs if (j[0]["id"], j[1], j[2]) not in done]

    print(f"{len(prompts)} prompts x {len(MODELS_UNDER_TEST)} models x {len(CONDITIONS)} "
          f"conditions = {len(jobs)} calls to make")

    completed = 0
    with open(RAW_PATH, "a") as f, concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = [pool.submit(run_one, p, m, c) for p, m, c in jobs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            completed += 1
            status = "ERROR" if row["error"] else "ok"
            if completed % 10 == 0 or row["error"]:
                print(f"  [{completed}/{len(jobs)}] {row['model']:35s} {row['condition']:8s} "
                      f"{row['prompt_id']:8s} {status}")

    print(f"\nDone. {completed} rows written to {RAW_PATH}")


if __name__ == "__main__":
    main()
