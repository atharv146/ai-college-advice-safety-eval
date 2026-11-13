#!/usr/bin/env python
"""
Which judge labels should the guard model train on?

The guard learns whatever its training labels say, so before any training run
those labels are scored against the human raters on the items both labeled.
A label set that disagrees with humans more than an alternative does should
not be trained on, however much cheaper it was to produce.

This exists because the first v3 relabel (gpt-4o-mini, rubric v3) disagreed
with the human raters MORE than the original labels (gpt-4o, rubric v1) on
unsafe_reassurance -- the dimension the guard is built to detect. That
comparison is confounded twice over: the judge model and the rubric version
changed together, and the human raters labeled under rubric v2. Re-run this
once human labels under rubric v3 exist; that comparison decides the question.

Usage:
  python src/guard_check_labels.py
  python src/guard_check_labels.py --labels results/judged.jsonl guard/labels_v3_mini.jsonl
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import agreement_stats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "labels"
DIMS = ["unsafe_reassurance", "fabricated_policy_claim", "unhelpfully_evasive"]
DEFAULT_LABELS = [ROOT / "results" / "judged.jsonl", ROOT / "guard" / "labels_v3_mini.jsonl"]


def load_judge(path: Path) -> tuple[dict, str]:
    rows, tag = {}, None
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("judge_parse_failed", True):
            continue
        rows[(r["prompt_id"], r["model"], r["condition"])] = r
        tag = tag or f"{r.get('rubric_version', 'v1')}/{r.get('judge_model', 'openai/gpt-4o').split('/')[-1]}"
    return rows, tag


def load_humans() -> dict:
    """All human label files with a full-size sample. The 20-item first pass is
    excluded: it overlaps the others on too few items to compare."""
    out = {}
    for path in sorted(LABELS_DIR.glob("human_labels_*.csv")):
        rows = {(r["prompt_id"], r["model"], r["condition"]): r for r in csv.DictReader(open(path))}
        if len(rows) >= 100:
            out[path.stem.removeprefix("human_labels_")] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", type=Path, default=DEFAULT_LABELS,
                    help="judge-label files to compare")
    args = ap.parse_args()

    judges = {}
    for p in args.labels:
        rows, tag = load_judge(p)
        judges[f"{tag} ({p.name})"] = rows
    humans = load_humans()
    if not humans:
        raise SystemExit("No human label files with 100+ rows in labels/.")

    keys = set.intersection(*(set(h) for h in humans.values()), *(set(j) for j in judges.values()))
    keys = sorted(keys)
    print(f"Comparing on {len(keys)} responses labeled by every human and every judge.")
    print(f"Humans: {', '.join(humans)}\n")

    for d in DIMS:
        print(d)
        for name, H in humans.items():
            rate = sum(H[k][d] == "True" for k in keys) / len(keys)
            print(f"  human  {name:22s} fires {rate:6.1%}")
        for jname, J in judges.items():
            j = [bool(J[k][d]) for k in keys]
            parts = []
            for hname, H in humans.items():
                s = agreement_stats([H[k][d] == "True" for k in keys], j)
                parts.append(f"vs {hname}: k={s['cohens_kappa']:.3f} pos={s['pos_agree'] or 0:.3f}")
            print(f"  judge  {jname[:40]:40s} fires {sum(j)/len(j):6.1%} | " + " | ".join(parts))
        print()

    print("Train on the label set whose agreement with the humans is highest, "
          "especially pos (agreement on which responses fail). If the humans "
          "used a different rubric version than a judge, that comparison is "
          "confounded -- prefer human labels made under the same version.")


if __name__ == "__main__":
    main()
