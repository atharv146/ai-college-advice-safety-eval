#!/usr/bin/env python
"""
Reads results/judged.jsonl (and labels/human_labels.csv if it exists yet)
and produces:
  - results/summary_by_model_condition.csv: failure rates with Wilson 95%
    confidence intervals, per model x condition (vanilla vs guarded)
  - results/summary_by_category.csv: failure rates per risk category
  - figures/failure_rate_by_model.png
  - figures/guarded_vs_vanilla.png
  - judge/human agreement stats printed to the console, and written to
    results/judge_human_agreement.csv, IF at least one labels/human_labels_*.csv
    exists
  - results/inter_rater_agreement.csv: human-vs-human agreement, IF 2+
    labels/human_labels_*.csv files exist (from 2+ people running
    src/label_sample.py). This is the standard inter-rater reliability
    check and is independent of whether the AI judge is any good at all --
    it asks whether the RUBRIC itself is being applied consistently by
    different people, which judge-vs-human agreement alone can't tell you.
  - results/mcnemar_vanilla_vs_guarded.csv: a proper paired significance
    test for whether the guarded system prompt actually changes failure
    rates, per model

Run this after src/run_harness.py and src/judge.py. The judge/human
agreement section only appears once you've run src/label_sample.py
yourself; without it, everything else still works, it's just unvalidated.

Why McNemar's test and not just comparing two Wilson intervals: the
vanilla and guarded rates aren't independent samples, they're the SAME
82 prompts run twice, once under each condition. Treating them as two
separate binomial samples (which two side-by-side confidence intervals
implicitly do) throws away that pairing and understates how much
evidence a consistent per-prompt shift actually is. McNemar's test looks
only at the prompts where the two conditions disagreed, which is the
correct way to ask "did the guarded prompt change the outcome."
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LABELS_DIR = Path(__file__).resolve().parent.parent / "labels"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

JUDGED_PATH = RESULTS_DIR / "judged.jsonl"


def find_rater_files() -> dict[str, Path]:
    """Maps rater name -> their labels CSV, from labels/human_labels_<rater>.csv."""
    raters = {}
    for path in sorted(LABELS_DIR.glob("human_labels_*.csv")):
        name = path.stem.removeprefix("human_labels_")
        raters[name] = path
    return raters

FAIL_COLS = ["unsafe_reassurance", "fabricated_policy_claim", "unhelpfully_evasive"]

# Cohen's kappa on a handful of rows is dominated by sampling noise (a single
# flipped cell can swing it from 0 to 1), so pairs with less overlap than this
# are reported as "not enough data" rather than given a number that looks like
# a result.
MIN_OVERLAP_FOR_KAPPA = 20

sns.set_theme(style="whitegrid")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (point, low, high)."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def load_judged() -> pd.DataFrame:
    if not JUDGED_PATH.exists():
        raise SystemExit("No results/judged.jsonl yet. Run src/judge.py first.")
    df = pd.read_json(JUDGED_PATH, lines=True)
    return df[~df["judge_parse_failed"]].copy()


def add_pass_col(df: pd.DataFrame) -> pd.DataFrame:
    df["fail_any"] = df[FAIL_COLS].any(axis=1)
    return df


def summarize_by_model_condition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, condition), group in df.groupby(["model", "condition"]):
        n = len(group)
        row = {"model": model, "condition": condition, "n": n}
        for col in FAIL_COLS + ["fail_any"]:
            successes = int(group[col].sum())
            p, lo, hi = wilson_interval(successes, n)
            row[f"{col}_rate"] = round(p, 3)
            row[f"{col}_ci_low"] = round(lo, 3)
            row[f"{col}_ci_high"] = round(hi, 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "condition"])


def summarize_by_category(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category, group in df.groupby("category"):
        n = len(group)
        successes = int(group["fail_any"].sum())
        p, lo, hi = wilson_interval(successes, n)
        rows.append(
            {"category": category, "n": n, "fail_any_rate": round(p, 3),
             "ci_low": round(lo, 3), "ci_high": round(hi, 3)}
        )
    return pd.DataFrame(rows).sort_values("fail_any_rate", ascending=False)


def plot_failure_by_model(summary: pd.DataFrame) -> None:
    pivot = summary.pivot(index="model", columns="condition", values="fail_any_rate")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"vanilla": "#C44E52", "guarded": "#55A868", "grounded": "#4C72B0"}
    pivot.plot(kind="barh", ax=ax, color=[colors.get(c, "#888888") for c in pivot.columns])
    ax.set_xlabel("Rate of at least one safety failure (unsafe reassurance, "
                  "fabricated policy claim, or unhelpful evasion)")
    ax.set_title("Failure rate by model: vanilla vs. guarded system prompt",
                  fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    path = FIGURES_DIR / "failure_rate_by_model.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_category_breakdown(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    order = summary.sort_values("fail_any_rate")
    ax.barh(order["category"], order["fail_any_rate"], color="#4C72B0")
    ax.set_xlabel("Failure rate (any dimension, across all models/conditions)")
    ax.set_title("Which risk categories fail most often", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    path = FIGURES_DIR / "failure_rate_by_category.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    p_a_true = sum(a) / n
    p_b_true = sum(b) / n
    pe = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def agreement_stats(a: list[bool], b: list[bool]) -> dict:
    """Cohen's kappa alongside the statistics needed to read it.

    Kappa is known to collapse when the category being labeled is rare (the
    "kappa paradox", Feinstein & Cicchetti 1990): if both raters say "no" to
    ~90% of items, chance agreement is already ~0.86, so even 88% raw
    agreement leaves kappa near zero. That is exactly this rubric's situation
    -- unsafe_reassurance fires on under 10% of responses. So kappa is
    reported together with:

      prevalence  mean share of "yes" labels across the two raters
      PABAK       prevalence- and bias-adjusted kappa, 2*Po - 1 (Byrt, Bishop
                  & Carlin 1993) -- what kappa would be with balanced classes
      Gwet's AC1  chance-corrected agreement whose chance term stays small
                  when prevalence is extreme (Gwet 2008)

      pos_agree   positive specific agreement, 2a / (2a + b + c), and
      neg_agree   negative specific agreement, 2d / (2d + b + c)
                  (Cicchetti & Feinstein 1990). These separate the two halves
                  of the agreement that PABAK and AC1 blend together.

    None of these replaces kappa. PABAK and AC1 in particular must not be read
    alone: with rare positives they are driven almost entirely by agreement on
    negatives, so they can be high even when the raters almost never flag the
    SAME failure. pos_agree is the number that answers "do the raters agree on
    which responses are unsafe" -- the question a safety label depends on.
    """
    n = len(a)
    if n == 0:
        return {}
    both_yes = sum(1 for x, y in zip(a, b) if x and y)
    both_no = sum(1 for x, y in zip(a, b) if not x and not y)
    disagree = n - both_yes - both_no
    po = (both_yes + both_no) / n
    pa, pb = sum(a) / n, sum(b) / n
    pi = (pa + pb) / 2
    pe_ac1 = 2 * pi * (1 - pi)
    ac1 = (po - pe_ac1) / (1 - pe_ac1) if pe_ac1 != 1 else 1.0
    pos_den = 2 * both_yes + disagree
    neg_den = 2 * both_no + disagree
    return {
        "raw_agreement": round(po, 3),
        "cohens_kappa": round(cohens_kappa(a, b), 3),
        "pabak": round(2 * po - 1, 3),
        "gwet_ac1": round(ac1, 3),
        "pos_agree": round(2 * both_yes / pos_den, 3) if pos_den else None,
        "neg_agree": round(2 * both_no / neg_den, 3) if neg_den else None,
        "prevalence": round(pi, 3),
        "n_both_yes": both_yes,
    }


def mcnemar_test(df: pd.DataFrame) -> pd.DataFrame:
    """Paired significance test: for each model, does the guarded system
    prompt actually change the fail_any outcome relative to vanilla, on
    the SAME prompts? Uses the exact binomial form of McNemar's test
    (appropriate at these sample sizes, no chi-square approximation)."""
    rows = []
    for model, group in df.groupby("model"):
        pivot = group.pivot_table(
            index="prompt_id", columns="condition", values="fail_any", aggfunc="first"
        )
        if "vanilla" not in pivot.columns or "guarded" not in pivot.columns:
            continue
        pivot = pivot.dropna(subset=["vanilla", "guarded"])

        # b: vanilla failed, guarded didn't (guarded improved this prompt)
        # c: guarded failed, vanilla didn't (guarded made this prompt worse)
        b = int(((pivot["vanilla"]) & (~pivot["guarded"])).sum())
        c = int(((~pivot["vanilla"]) & (pivot["guarded"])).sum())
        discordant = b + c

        if discordant == 0:
            p_value = float("nan")
        else:
            # under H0, b ~ Binomial(discordant, 0.5)
            p_value = binomtest(b, discordant, 0.5).pvalue

        rows.append(
            {
                "model": model,
                "n_prompts": len(pivot),
                "vanilla_fixed_by_guarded": b,
                "guarded_made_it_worse": c,
                "discordant_pairs": discordant,
                "p_value": round(p_value, 4) if not math.isnan(p_value) else None,
                "significant_at_0.05": (p_value < 0.05) if not math.isnan(p_value) else None,
            }
        )
    return pd.DataFrame(rows)


def judge_human_agreement(judged: pd.DataFrame, raters: dict[str, Path]) -> pd.DataFrame | None:
    if not raters:
        print("\nNo labels/human_labels_*.csv yet. Run `python src/label_sample.py "
              "--rater yourname` to validate the judge before trusting judge-only numbers.")
        return None

    all_rows = []
    for rater_name, path in raters.items():
        human = pd.read_csv(path)
        merged = human.merge(
            judged, on=["prompt_id", "model", "condition"], suffixes=("_human", "_judge")
        )
        if merged.empty:
            print(f"\n{path} exists but has no rows matching judged.jsonl.")
            continue
        for col in FAIL_COLS:
            h = merged[f"{col}_human"].astype(bool).tolist()
            j = merged[f"{col}_judge"].astype(bool).tolist()
            all_rows.append(
                {"rater": rater_name, "dimension": col, "n": len(h), **agreement_stats(h, j)}
            )

    if not all_rows:
        return None
    out = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / "judge_human_agreement.csv"
    out.to_csv(out_path, index=False)
    print(f"\nJudge/human agreement, per rater:")
    print(out.to_string(index=False))
    print(f"Saved {out_path}")
    return out


def inter_rater_agreement(raters: dict[str, Path]) -> pd.DataFrame | None:
    """Human-vs-human agreement: the standard inter-rater reliability check,
    independent of the AI judge entirely. Needs 2+ people to have run
    label_sample.py. Computed pairwise across every pair of raters, on
    whatever rows both of them happened to label in common."""
    if len(raters) < 2:
        print(f"\nOnly {len(raters)} rater(s) so far -- inter-rater agreement needs 2+. "
              "Have a second person run `python src/label_sample.py --rater theirname`.")
        return None

    rows = []
    names = sorted(raters.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]

            # Skip same-person re-labelling passes. A rater who relabels the
            # same sample under a revised rubric shows up as e.g. "atharv" and
            # "atharv_v2"; comparing those two measures how much ONE person
            # changed their mind between rubric versions, which is a real
            # thing but is NOT inter-rater reliability. Reporting it as such
            # would overstate the validation (it reads as near-perfect
            # agreement on the handful of rows that happen to overlap).
            if a_name.startswith(b_name) or b_name.startswith(a_name):
                print(f"\nSkipping {a_name} vs {b_name}: these look like the same "
                      f"rater's passes, not two independent raters. Inter-rater "
                      f"reliability needs a genuinely different person.")
                continue

            a = pd.read_csv(raters[a_name])
            b = pd.read_csv(raters[b_name])
            merged = a.merge(b, on=["prompt_id", "model", "condition"], suffixes=("_a", "_b"))
            if merged.empty:
                print(f"\n{a_name} and {b_name} have no overlapping labeled rows yet.")
                continue
            if len(merged) < MIN_OVERLAP_FOR_KAPPA:
                print(f"\n{a_name} vs {b_name}: only {len(merged)} overlapping rows "
                      f"(need >= {MIN_OVERLAP_FOR_KAPPA} for a meaningful kappa). Skipped -- "
                      f"kappa on a handful of rows swings wildly and would mislead. "
                      f"Both raters should label the SAME sample; see "
                      f"src/export_labeling_sheet.py.")
                continue
            for col in FAIL_COLS:
                x = merged[f"{col}_a"].astype(bool).tolist()
                y = merged[f"{col}_b"].astype(bool).tolist()
                rows.append(
                    {"rater_a": a_name, "rater_b": b_name, "dimension": col, "n": len(x),
                     **agreement_stats(x, y)}
                )

    if not rows:
        return None
    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "inter_rater_agreement.csv"
    out.to_csv(out_path, index=False)
    print(f"\nInter-rater agreement (human vs. human, independent of the AI judge):")
    print(out.to_string(index=False))
    print(f"Saved {out_path}")
    return out


def main() -> None:
    df = add_pass_col(load_judged())
    print(f"Loaded {len(df)} judged responses.\n")

    by_model = summarize_by_model_condition(df)
    by_model.to_csv(RESULTS_DIR / "summary_by_model_condition.csv", index=False)
    print(by_model.to_string(index=False))

    by_category = summarize_by_category(df)
    by_category.to_csv(RESULTS_DIR / "summary_by_category.csv", index=False)
    print("\n" + by_category.to_string(index=False))

    plot_failure_by_model(by_model)
    plot_category_breakdown(by_category)

    mcnemar = mcnemar_test(df)
    mcnemar.to_csv(RESULTS_DIR / "mcnemar_vanilla_vs_guarded.csv", index=False)
    print("\nMcNemar's test, vanilla vs. guarded (paired by prompt):")
    print(mcnemar.to_string(index=False))

    raters = find_rater_files()
    judge_human_agreement(df, raters)
    inter_rater_agreement(raters)


if __name__ == "__main__":
    main()
