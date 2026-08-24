#!/usr/bin/env python
"""
Compares results/judged.jsonl (graded under rubric v1) against
results/judged_v2.jsonl (graded under v2) on whatever rows both contain, and
reports how much the rubric revision actually changed the grading.

This is what makes it defensible to keep reporting the v1-graded headline
numbers while citing rubric v2: instead of assuming the revision was
cosmetic, it measures the shift per dimension with a Wilson interval, and
flags any dimension where the two versions disagree enough that the v1
numbers should not be carried forward.

Usage:
  python src/compare_rubric_versions.py
"""

import json
import math
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
V1_PATH = RESULTS_DIR / "judged.jsonl"
V2_PATH = RESULTS_DIR / "judged_v2.jsonl"
OUT_PATH = RESULTS_DIR / "rubric_version_comparison.csv"

DIMS = ["unsafe_reassurance", "fabricated_policy_claim", "unhelpfully_evasive"]


def wilson(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if not r.get("judge_parse_failed", True):
            out[(r["prompt_id"], r["model"], r["condition"])] = r
    return out


def main() -> None:
    v1, v2 = load(V1_PATH), load(V2_PATH)
    if not v2:
        raise SystemExit(
            f"No {V2_PATH} yet. Run: python src/rejudge_v2.py --sample 80"
        )

    shared = sorted(set(v1) & set(v2))
    if not shared:
        raise SystemExit("No overlapping rows between v1 and v2 gradings.")

    print(f"Comparing {len(shared)} responses graded under both rubric versions.\n")

    rows = []
    for dim in DIMS:
        agree = sum(1 for k in shared if v1[k][dim] == v2[k][dim])
        v1_fires = sum(1 for k in shared if v1[k][dim])
        v2_fires = sum(1 for k in shared if v2[k][dim])
        v1_only = sum(1 for k in shared if v1[k][dim] and not v2[k][dim])
        v2_only = sum(1 for k in shared if v2[k][dim] and not v1[k][dim])
        p, lo, hi = wilson(agree, len(shared))
        rows.append({
            "dimension": dim, "n": len(shared),
            "agreement": round(p, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
            "v1_fire_rate": round(v1_fires / len(shared), 3),
            "v2_fire_rate": round(v2_fires / len(shared), 3),
            "v1_only": v1_only, "v2_only": v2_only,
        })

    # "fail_any" is what the headline number is built from, so it gets its own row.
    def fail_any(r):
        return any(r[d] for d in DIMS)

    agree = sum(1 for k in shared if fail_any(v1[k]) == fail_any(v2[k]))
    p, lo, hi = wilson(agree, len(shared))
    rows.append({
        "dimension": "fail_any (headline)", "n": len(shared),
        "agreement": round(p, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
        "v1_fire_rate": round(sum(1 for k in shared if fail_any(v1[k])) / len(shared), 3),
        "v2_fire_rate": round(sum(1 for k in shared if fail_any(v2[k])) / len(shared), 3),
        "v1_only": sum(1 for k in shared if fail_any(v1[k]) and not fail_any(v2[k])),
        "v2_only": sum(1 for k in shared if fail_any(v2[k]) and not fail_any(v1[k])),
    })

    hdr = f"{'dimension':26s} {'n':>4s} {'agree':>7s} {'95% CI':>16s} {'v1 rate':>8s} {'v2 rate':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ci = f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
        print(f"{r['dimension']:26s} {r['n']:4d} {r['agreement']:7.3f} {ci:>16s} "
              f"{r['v1_fire_rate']:8.3f} {r['v2_fire_rate']:8.3f}")

    import csv
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {OUT_PATH}")

    headline = rows[-1]
    print("\nInterpretation:")
    if headline["agreement"] >= 0.95:
        print(f"  The rubric revision changed the headline pass/fail verdict on only "
              f"{(1-headline['agreement'])*100:.1f}% of responses. The v1-graded numbers in "
              f"PAPER.md are defensible to report as-is, with this measurement cited.")
    elif headline["agreement"] >= 0.90:
        print(f"  Headline verdicts differ on {(1-headline['agreement'])*100:.1f}% of responses. "
              f"Report the v1 numbers with this shift disclosed explicitly, or re-grade in full "
              f"if budget allows.")
    else:
        print(f"  Headline verdicts differ on {(1-headline['agreement'])*100:.1f}% of responses -- "
              f"too much to carry the v1 numbers forward. A full re-grade "
              f"(python src/rejudge_v2.py) is needed before submission.")


if __name__ == "__main__":
    main()
