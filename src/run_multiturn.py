#!/usr/bin/env python
"""
Runs each 3-turn escalation sequence in data/escalation_prompts.json
against every model, under both system prompts. Unlike run_harness.py,
this builds real conversation state: each turn's response is appended to
the message history before the next turn is sent, so turn 3 sees turns
1 and 2 exactly as a real conversation would.

The single-turn harness only tests whether a model gets the first message
right. Most real unsafe answers don't happen on message one, they happen
after a worried user pushes back two or three times ("but my cousin said
X, so it's fine, right?"). This is the harder, more realistic test.

Usage:
  python src/run_multiturn.py
  python src/run_multiturn.py --resume
"""

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import MODELS_UNDER_TEST
from src.openrouter_client import complete_messages

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RAW_PATH = RESULTS_DIR / "multiturn_responses.jsonl"

sys.path.insert(0, str(DATA_DIR))
from system_prompts import GUARDED_SYSTEM_PROMPT, VANILLA_SYSTEM_PROMPT  # noqa: E402

CONDITIONS = {
    "vanilla": VANILLA_SYSTEM_PROMPT,
    "guarded": GUARDED_SYSTEM_PROMPT,
}


def load_sequences() -> list[dict]:
    return json.loads((DATA_DIR / "escalation_prompts.json").read_text())


def already_done() -> set[tuple[str, str, str]]:
    if not RAW_PATH.exists():
        return set()
    done = set()
    for line in RAW_PATH.read_text().splitlines():
        row = json.loads(line)
        if row.get("error") is None:
            done.add((row["sequence_id"], row["model"], row["condition"]))
    return done


def run_sequence(seq: dict, model: str, condition: str) -> dict:
    system = CONDITIONS[condition]
    messages = [{"role": "system", "content": system}]
    turn_responses = []
    error = None
    t0 = time.time()

    try:
        for turn_text in seq["turns"]:
            messages.append({"role": "user", "content": turn_text})
            reply = complete_messages(model, messages)
            messages.append({"role": "assistant", "content": reply})
            turn_responses.append(reply)
    except Exception as e:  # noqa: BLE001
        error = str(e)

    return {
        "sequence_id": seq["id"],
        "category": seq["category"],
        "model": model,
        "condition": condition,
        "turns": seq["turns"],
        "responses": turn_responses,  # may be shorter than turns if error hit mid-sequence
        "error": error,
        "latency_s": round(time.time() - t0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                         help="only run the first N sequences, for finishing faster under heavy rate limiting")
    args = parser.parse_args()

    sequences = load_sequences()
    if args.limit is not None:
        sequences = sequences[: args.limit]
    jobs = [
        (seq, model, condition)
        for seq in sequences
        for model in MODELS_UNDER_TEST
        for condition in CONDITIONS
    ]

    if args.resume:
        done = already_done()
        jobs = [j for j in jobs if (j[0]["id"], j[1], j[2]) not in done]

    print(f"{len(sequences)} sequences x {len(MODELS_UNDER_TEST)} models x "
          f"{len(CONDITIONS)} conditions = {len(jobs)} conversations to run "
          f"(each is 3 sequential calls, so {len(jobs) * 3} total API calls)")

    completed = 0
    with open(RAW_PATH, "a") as f, concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = [pool.submit(run_sequence, s, m, c) for s, m, c in jobs]
        for fut in concurrent.futures.as_completed(futures):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            completed += 1
            status = "ERROR" if row["error"] else "ok"
            print(f"  [{completed}/{len(jobs)}] {row['model']:35s} {row['condition']:8s} "
                  f"{row['sequence_id']:8s} {status}")

    print(f"\nDone. {completed} conversations written to {RAW_PATH}")


if __name__ == "__main__":
    main()
