#!/usr/bin/env python
"""
Faster alternative to src/label_sample.py: exports the exact same 150-item
sample already fixed by labels/human_labels_atharv_v2.csv into a CSV you can
open in Excel/Numbers/Google Sheets and fill in with TRUE/FALSE, instead of
answering 450 sequential terminal prompts.

Reuses atharv_v2's specific (prompt_id, model, condition) list rather than
re-deriving the sample from scratch, because results/raw_responses.jsonl has
grown since the original sample was drawn -- re-running the same random seed
today would silently pick a different set of rows (this already happened
once between the v1 and v2 rater passes; see PAPER.md/chat history).

Usage:
  python src/export_labeling_sheet.py

Writes:
  labels/labeling_sheet.csv        -- what the rater fills in
  labels/_labeling_sheet_key.csv   -- row -> (prompt_id, model, condition),
                                       not meant for the rater to open
"""

import csv
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LABELS_DIR = Path(__file__).resolve().parent.parent / "labels"
SOURCE_SAMPLE = LABELS_DIR / "human_labels_atharv_v2.csv"
SHEET_PATH = LABELS_DIR / "labeling_sheet.csv"
KEY_PATH = LABELS_DIR / "_labeling_sheet_key.csv"


def main() -> None:
    if not SOURCE_SAMPLE.exists():
        raise SystemExit(f"{SOURCE_SAMPLE} not found -- need an existing sample to copy.")

    sample_keys = []
    with open(SOURCE_SAMPLE) as f:
        for row in csv.DictReader(f):
            sample_keys.append((row["prompt_id"], row["model"], row["condition"]))

    raw = {}
    for line in open(RESULTS_DIR / "raw_responses.jsonl"):
        r = json.loads(line)
        raw[(r["prompt_id"], r["model"], r["condition"])] = r

    with open(SHEET_PATH, "w", newline="") as sheet_f, open(KEY_PATH, "w", newline="") as key_f:
        sheet_writer = csv.writer(sheet_f)
        key_writer = csv.writer(key_f)
        sheet_writer.writerow([
            "row", "question", "response",
            "fabricated_policy_claim (y/n)",
            "unsafe_reassurance (y/n)",
            "unhelpfully_evasive (y/n)",
        ])
        key_writer.writerow(["row", "prompt_id", "model", "condition"])

        for i, key in enumerate(sample_keys, 1):
            r = raw.get(key)
            if r is None:
                continue
            sheet_writer.writerow([i, r["prompt_text"], r["response"], "", "", ""])
            key_writer.writerow([i, *key])

    print(f"Wrote {len(sample_keys)} rows to {SHEET_PATH}")
    print("Open it in a spreadsheet app, fill the last three columns with y/n "
          "(or TRUE/FALSE) for every row, save as CSV, then run:")
    print("  python src/import_labeling_sheet.py --rater <name> --file "
          f"{SHEET_PATH}")
    print(f"(Don't need to open {KEY_PATH} -- that's just for the import step.)")


if __name__ == "__main__":
    main()
