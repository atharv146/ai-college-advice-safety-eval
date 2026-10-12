#!/usr/bin/env python
"""
Builds the training dataset for the false-reassurance guard model from the
existing benchmark responses and their judge labels.

Output: guard/dataset.jsonl, one row per (prompt, model, condition), with the
three rubric labels and a category-holdout fold assignment.

Leakage control (EXEA_PROPOSAL.md section 3)
--------------------------------------------
Every prompt in the bank was answered by several models under several
conditions, so one prompt yields ~10 highly correlated responses. If those
siblings were split across train and eval, the guard could score well by
recognizing the *question* rather than judging the *answer*.

So folds are assigned by CATEGORY, and every response to a given prompt
inherits that prompt's category. That is strictly stronger than prompt-level
splitting: holding out a whole category also guarantees no prompt in the eval
fold was ever seen in training. It's also the generalization test the proposal
actually asks for -- does the guard learn "false reassurance" in general, or
just the vocabulary of FERPA questions?

Label source
------------
--labels points at a judged file. It defaults to results/judged.jsonl, which
holds rubric-v1 labels. Those are fine for verifying the pipeline runs, but
the guard should be trained on rubric-v3 labels: v1's unsafe_reassurance
definition is the one that produced human-human kappa 0.118. Re-label first,
then rebuild with --labels pointing at the output:

  python src/rejudge_v2.py --model openai/gpt-4o-mini --out guard/labels_v3_mini.jsonl
  python src/guard_build_dataset.py --labels guard/labels_v3_mini.jsonl

(~$0.13 for all 840. gpt-4o-mini is used for TRAINING labels only; the gold
set stays judged by gpt-4o, which is the teacher the guard is measured
against. Note gpt-4o-mini is also one of the generating models, so on its own
responses it is grading itself -- report label agreement split by generator.)

Usage:
  python src/guard_build_dataset.py                          # v1 labels
  python src/guard_build_dataset.py --labels guard/labels_v3_mini.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "results" / "raw_responses.jsonl"
OUT_DIR = ROOT / "guard"

DIMS = ["unsafe_reassurance", "fabricated_policy_claim", "unhelpfully_evasive"]


def load_latest(path: Path, require: str) -> dict:
    """One row per (prompt_id, model, condition), keeping the last row that
    satisfies `require` -- the same dedup rule src/judge.py uses, since the
    raw logs contain failed attempts followed by successful retries."""
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if require == "response" and not r.get("response"):
            continue
        if require == "labels" and r.get("judge_parse_failed", True):
            continue
        out[(r["prompt_id"], r["model"], r["condition"])] = r
    return out


def format_input(question: str, response: str) -> str:
    """The guard sees exactly what the judge saw: the question and the answer.
    Question first, so truncation at the encoder's length limit cuts the tail
    of the response rather than the question that gives it context."""
    return f"QUESTION: {question}\n\nRESPONSE: {response}"


def assign_category_folds(categories: list[str], n_folds: int) -> dict[str, int]:
    """Deterministic round-robin of sorted categories onto folds. Sorted so the
    assignment never depends on file order, and is identical on every machine."""
    return {cat: i % n_folds for i, cat in enumerate(sorted(categories))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / "results" / "judged.jsonl"),
                    help="judged file to take labels from")
    ap.add_argument("--folds", type=int, default=5,
                    help="number of category-holdout folds (proposal specifies 5)")
    ap.add_argument("--out", default=str(OUT_DIR / "dataset.jsonl"))
    args = ap.parse_args()

    labels_path = Path(args.labels)
    raw = load_latest(RAW_PATH, require="response")
    labels = load_latest(labels_path, require="labels")

    keys = sorted(set(raw) & set(labels))
    missing = len(raw) - len(keys)

    categories = sorted({raw[k]["category"] for k in keys})
    fold_of = assign_category_folds(categories, args.folds)

    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for k in keys:
        r, l = raw[k], labels[k]
        rows.append({
            "id": "|".join(k),
            "prompt_id": k[0],
            "model": k[1],
            "condition": k[2],
            "category": r["category"],
            "fold": fold_of[r["category"]],
            "text": format_input(r["prompt_text"], r["response"]),
            "labels": {d: bool(l[d]) for d in DIMS},
            "label_source": labels_path.name,
        })

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --- report, and prove the leakage guarantee rather than assert it ---
    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"Labels from: {labels_path.name}")
    if missing:
        print(f"  ({missing} responses had no usable label and were dropped)")

    print(f"\nCategory folds ({args.folds}):")
    for fold in range(args.folds):
        cats = [c for c, f in fold_of.items() if f == fold]
        n = sum(1 for r in rows if r["fold"] == fold)
        print(f"  fold {fold}: {n:4d} rows  {', '.join(cats)}")

    prompt_folds = {}
    for r in rows:
        prompt_folds.setdefault(r["prompt_id"], set()).add(r["fold"])
    leaked = [p for p, fs in prompt_folds.items() if len(fs) > 1]
    print(f"\nLeakage check: {len(leaked)} prompts appear in more than one fold "
          f"({'PASS' if not leaked else 'FAIL'})")
    if leaked:
        raise SystemExit(f"Leaking prompts: {leaked[:10]}")

    print("\nLabel balance (positive rate):")
    for d in DIMS:
        pos = sum(r["labels"][d] for r in rows)
        print(f"  {d:26s} {pos:4d}/{len(rows)}  ({pos/len(rows):.1%})")

    by_model = Counter(r["model"] for r in rows)
    print("\nRows by generating model:")
    for m, n in sorted(by_model.items()):
        print(f"  {m:32s} {n}")


if __name__ == "__main__":
    main()
