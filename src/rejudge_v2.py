#!/usr/bin/env python
"""
Re-grades responses under the CURRENT (v2) judge prompt in src/judge.py and
writes results/judged_v2.jsonl, leaving results/judged.jsonl untouched.

Why this exists
---------------
data/rubric.md was revised to v2 on 2026-08-22, but src/judge.py kept the v1
wording until 2026-08-24 -- so every number in results/judged.jsonl was graded
against the *superseded* rubric. That is a real internal inconsistency: the
paper cites rubric v2 while its results were produced under v1.

There are two honest fixes: re-grade everything under v2 (needs ~$2 of API
credit), or keep the v1 results and *quantify* how much the revision would
have changed them. This script serves both. With --sample it re-grades a
random subset cheaply, which is enough to estimate the v1->v2 shift with a
confidence interval; without it, it re-grades everything.

Usage:
  python src/rejudge_v2.py --sample 80      # cheap representative estimate
  python src/rejudge_v2.py                  # full re-grade (~$2 at gpt-4o)

Resumable: already-graded rows in judged_v2.jsonl are skipped, so an abort
(including a budget abort) loses nothing.
"""

import argparse
import concurrent.futures
import json
import random
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import budget
from src.judge import JUDGE_SYSTEM_PROMPT, build_user_message, parse_judgment
from src.models import JUDGE_MODEL
from src.openrouter_client import complete

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RAW_PATH = RESULTS_DIR / "raw_responses.jsonl"
OUT_PATH = RESULTS_DIR / "judged_v2.jsonl"

# Rough per-response cost at gpt-4o rates, measured from the existing corpus
# (~730 input tokens + ~60 output tokens per call). Used only for the
# pre-flight budget check, not for billing.
EST_COST_PER_RESPONSE = 0.0024


def load_latest_responses() -> dict:
    """One row per (prompt_id, model, condition), keeping the last successful
    attempt -- same dedup rule src/judge.py uses."""
    latest = {}
    for line in RAW_PATH.read_text().splitlines():
        r = json.loads(line)
        if r.get("response"):
            latest[(r["prompt_id"], r["model"], r["condition"])] = r
    return latest


def already_done() -> set:
    if not OUT_PATH.exists():
        return set()
    done = set()
    for line in OUT_PATH.read_text().splitlines():
        row = json.loads(line)
        if not row.get("judge_parse_failed", True):
            done.add((row["prompt_id"], row["model"], row["condition"]))
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="re-grade a random subset of this size instead of everything")
    parser.add_argument("--seed", type=int, default=11,
                        help="sampling seed, fixed so the subset is reproducible")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--floor", type=float, default=0.05,
                        help="abort if remaining balance falls below this")
    args = parser.parse_args()

    if not RAW_PATH.exists():
        raise SystemExit("No results/raw_responses.jsonl yet.")

    latest = load_latest_responses()
    keys = sorted(latest.keys())

    if args.sample:
        rng = random.Random(args.seed)
        keys = rng.sample(keys, min(args.sample, len(keys)))

    done = already_done()
    todo = [k for k in keys if k not in done]

    est = len(todo) * EST_COST_PER_RESPONSE
    budget.require_budget(est, reserve=args.floor)

    print(f"Re-grading {len(todo)} responses under the v2 judge prompt "
          f"(model={JUDGE_MODEL}).")
    if args.sample:
        print(f"Random subset of {len(keys)} (seed={args.seed}) -- for estimating "
              f"the v1->v2 shift, not for replacing the headline numbers.\n")

    def judge_one(key):
        row = latest[key]
        try:
            raw = complete(JUDGE_MODEL, JUDGE_SYSTEM_PROMPT,
                           build_user_message(row["prompt_text"], row["response"]))
            j = parse_judgment(raw)
        except Exception as e:  # noqa: BLE001
            raw, j = str(e), None
        out = {
            "prompt_id": row["prompt_id"], "category": row["category"],
            "model": row["model"], "condition": row["condition"],
            "judge_raw": raw, "rubric_version": "v2",
        }
        if j:
            out.update({
                "unsafe_reassurance": j["unsafe_reassurance"],
                "fabricated_policy_claim": j["fabricated_policy_claim"],
                "unhelpfully_evasive": j["unhelpfully_evasive"],
                "rationale": j.get("rationale", ""),
                "judge_parse_failed": False,
            })
        else:
            out.update({
                "unsafe_reassurance": None, "fabricated_policy_claim": None,
                "unhelpfully_evasive": None, "rationale": None,
                "judge_parse_failed": True,
            })
        return out

    write_lock = threading.Lock()
    completed = 0
    with open(OUT_PATH, "a") as f, concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = [pool.submit(judge_one, k) for k in todo]
        for fut in concurrent.futures.as_completed(futures):
            out = fut.result()
            with write_lock:
                f.write(json.dumps(out) + "\n")
                f.flush()
            completed += 1
            if completed % 20 == 0:
                print(f"  {completed}/{len(todo)}", flush=True)

    print(f"\nDone. See {OUT_PATH}")
    print(f"[budget] ${budget.remaining():.3f} remaining")
    print("Now run: python src/compare_rubric_versions.py")


if __name__ == "__main__":
    main()
