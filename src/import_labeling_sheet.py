#!/usr/bin/env python
"""
Converts a filled-in labels/labeling_sheet.csv (see export_labeling_sheet.py)
back into the labels/human_labels_<rater>.csv format src/analyze.py expects.

Usage:
  python src/import_labeling_sheet.py --rater <name> --file labels/labeling_sheet.csv
"""

import argparse
import csv
import re
from pathlib import Path

LABELS_DIR = Path(__file__).resolve().parent.parent / "labels"
KEY_PATH = LABELS_DIR / "_labeling_sheet_key.csv"

TRUE_VALS = {"y", "yes", "true", "1"}
FALSE_VALS = {"n", "no", "false", "0"}

DIM_COLS = [
    "fabricated_policy_claim (y/n)",
    "unsafe_reassurance (y/n)",
    "unhelpfully_evasive (y/n)",
]
OUT_FIELDS = [
    "prompt_id", "model", "condition",
    "unsafe_reassurance", "fabricated_policy_claim", "unhelpfully_evasive",
]


def parse_bool(val: str, row: int, col: str) -> bool:
    v = val.strip().lower()
    if v in TRUE_VALS:
        return True
    if v in FALSE_VALS:
        return False
    raise SystemExit(
        f"Row {row}, column '{col}': got {val!r}, expected one of "
        f"y/n/yes/no/true/false. Fix it in the sheet and re-run."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rater", required=True)
    parser.add_argument("--file", default=str(LABELS_DIR / "labeling_sheet.csv"))
    args = parser.parse_args()

    if not KEY_PATH.exists():
        raise SystemExit(f"{KEY_PATH} not found -- run export_labeling_sheet.py first.")

    key_by_row = {}
    with open(KEY_PATH) as f:
        for row in csv.DictReader(f):
            key_by_row[row["row"]] = (row["prompt_id"], row["model"], row["condition"])

    rater_slug = re.sub(r"[^a-z0-9_-]", "", args.rater.lower().replace(" ", "_"))
    if not rater_slug:
        raise SystemExit("--rater must contain at least one letter or number")
    out_path = LABELS_DIR / f"human_labels_{rater_slug}.csv"

    out_rows = []
    missing = []
    with open(args.file) as f:
        for row in csv.DictReader(f):
            row_num = row["row"]
            if row_num not in key_by_row:
                raise SystemExit(f"Row {row_num} not found in {KEY_PATH} -- did the sheet get edited/reordered?")
            prompt_id, model, condition = key_by_row[row_num]

            blanks = [c for c in DIM_COLS if not row[c].strip()]
            if blanks:
                missing.append(row_num)
                continue

            out_rows.append({
                "prompt_id": prompt_id,
                "model": model,
                "condition": condition,
                "fabricated_policy_claim": parse_bool(row["fabricated_policy_claim (y/n)"], int(row_num), "fabricated_policy_claim"),
                "unsafe_reassurance": parse_bool(row["unsafe_reassurance (y/n)"], int(row_num), "unsafe_reassurance"),
                "unhelpfully_evasive": parse_bool(row["unhelpfully_evasive (y/n)"], int(row_num), "unhelpfully_evasive"),
            })

    if missing:
        print(f"WARNING: {len(missing)} rows still have blank answers, skipped: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} labeled rows to {out_path}.")
    print("Now run: python src/analyze.py")


if __name__ == "__main__":
    main()
