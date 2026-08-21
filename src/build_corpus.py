#!/usr/bin/env python
"""
One-time build step: extracts the sections relevant to this eval's risk
categories (FERPA, residency, mixed-status FAFSA, DACA/state aid, status
disclosure) from PathFinder's guide-articles.json, a separately-maintained
companion project whose content has its own verification discipline (see
that repo's data/freshness.ts and CLAUDE.md content rules).

Run once to produce data/corpus.json, which IS committed to this repo
(a frozen snapshot, not a live dependency) so this project doesn't need
PathFinder's repo present to run. Re-run this script and re-commit the
output if PathFinder's guide content is updated and you want the grounded
condition to reflect that.

python src/build_corpus.py [path to pathfinder/app/src/data/guide-articles.json]
"""

import json
import sys
from pathlib import Path

DEFAULT_SOURCE = Path("/Users/atharv/pathfinder/app/src/data/guide-articles.json")
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "corpus.json"

# (article title, heading substring) pairs worth pulling in. Chosen because
# they're the sections this eval's prompt bank actually asks about, not the
# whole guide (a parent-support article's tips on paying for prom aren't
# relevant retrieval material for an immigration-safety benchmark).
WANTED = [
    ("Resources for Immigrant Families", "On the FAFSA specifically, for mixed-status families"),
    ("Resources for Immigrant Families", "State aid for undocumented students, including DACA recipients"),
    ("Resources for Immigrant Families", "In-state tuition and how residency is actually decided"),
    ("Resources for Immigrant Families", "On disclosing immigration status on college applications"),
    ("Resources for Immigrant Families", "A note on legal questions specifically"),
    ("How to Support Your Child's Applications", "What changes when your child turns 18"),
]


def main() -> None:
    source_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    if not source_path.exists():
        raise SystemExit(
            f"{source_path} not found. This corpus is built from PathFinder's guide "
            "content, a separate project — pass its guide-articles.json path as an "
            "argument, or skip this: data/corpus.json is already committed."
        )

    articles = json.loads(source_path.read_text())
    by_title = {a["title"]: a for a in articles}

    chunks = []
    for title, heading_substr in WANTED:
        article = by_title.get(title)
        if not article:
            print(f"WARNING: article '{title}' not found, skipping")
            continue
        match = next(
            (s for s in article["sections"] if heading_substr in s.get("heading", "")), None
        )
        if not match:
            print(f"WARNING: heading '{heading_substr}' not found in '{title}', skipping")
            continue
        chunks.append(
            {
                "id": f"{title}::{match['heading']}",
                "source_article": title,
                "heading": match["heading"],
                "text": " ".join(match["paragraphs"]),
            }
        )

    OUT_PATH.write_text(json.dumps(chunks, indent=2))
    print(f"Wrote {len(chunks)} chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()
