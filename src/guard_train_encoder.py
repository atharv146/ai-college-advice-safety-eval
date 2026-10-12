#!/usr/bin/env python
"""
Arm A of the guard-model study (EXEA_PROPOSAL.md section 2): fine-tune a small
encoder to detect the three rubric failures, evaluated under category holdout.

This is deliberately the FIRST experiment. It's the cheapest, and the most
likely to embarrass the headline: if a sub-400M encoder already clears the
bar, the larger QLoRA arms have to justify their cost against it.

Protocol
--------
- Category-holdout cross-validation over the folds in guard/dataset.jsonl.
  No prompt ever appears in both train and test (enforced at build time).
- Inside each training fold, a validation slice is carved out BY PROMPT, not
  by row, so sibling responses to one question can't straddle it. Decision
  thresholds are chosen on that slice and never on the test fold -- tuning
  them on test data would quietly inflate every reported number.
- Out-of-fold predictions are POOLED across folds and scored once. Per-fold
  scores are also reported, but some held-out categories contain almost no
  positives (in_state_residency_claims has zero unsafe_reassurance examples),
  so per-fold recall can be undefined or rest on a handful of items.
- Repeated over several seeds; mean and spread are reported, because a single
  run on 840 examples is mostly noise.

Primary metric (pre-registered in the proposal): recall at a threshold whose
false-positive rate on validation data is <= 0.15. Also reported: AUROC and
average precision, which need no threshold at all.

Usage:
  # verify the pipeline end to end in a couple of minutes, on a laptop
  python src/guard_train_encoder.py --smoke

  # the real run, on the GPU
  python src/guard_train_encoder.py --model microsoft/deberta-v3-base --seeds 3
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "guard" / "dataset.jsonl"
RESULTS_DIR = ROOT / "guard" / "results"

DIMS = ["unsafe_reassurance", "fabricated_policy_claim", "unhelpfully_evasive"]
TARGET_FPR = 0.15   # the operating point fixed in the proposal


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_val_by_prompt(rows: list[dict], frac: float, seed: int):
    """Hold out whole prompts, so every response to a validation question is in
    validation. Splitting by row would leak a question's siblings into training."""
    prompts = sorted({r["prompt_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(prompts)
    n_val = max(1, int(len(prompts) * frac))
    val_prompts = set(prompts[:n_val])
    train = [r for r in rows if r["prompt_id"] not in val_prompts]
    val = [r for r in rows if r["prompt_id"] in val_prompts]
    return train, val


def labels_matrix(rows: list[dict]) -> np.ndarray:
    return np.array([[float(r["labels"][d]) for d in DIMS] for r in rows], dtype=np.float32)


def make_loader(rows, tok, max_len, batch_size, shuffle):
    enc = tok([r["text"] for r in rows], truncation=True, max_length=max_len,
              padding=True, return_tensors="pt")
    y = torch.tensor(labels_matrix(rows))
    data = list(zip(enc["input_ids"], enc["attention_mask"], y))
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def train_one(model_name, train_rows, args, device):
    tok = AutoTokenizer.from_pretrained(model_name)
    # dtype MUST be explicit. This transformers version loads checkpoints in
    # their stored dtype by default, which for DeBERTa-v3 is float16. Adam's
    # eps (1e-8) rounds to exactly 0.0 in float16, so every Adam update becomes
    # a division by zero and all weights go NaN after the first step -- found
    # by the smoke test, and it would have hit the T4 identically. The encoder
    # is small enough (~184M params) that fp32 fits a 16GB card with room.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(DIMS), problem_type="multi_label_classification",
        dtype=torch.float32,
    ).to(device)
    non_fp32 = {str(p.dtype) for p in model.parameters() if p.dtype != torch.float32}
    if non_fp32:
        raise RuntimeError(f"Model loaded with {non_fp32} params; Adam requires fp32 here")

    # Imbalance: positives are 4-15% depending on dimension. Weight them up so
    # the model can't win by predicting "no" everywhere. Capped, because an
    # uncapped weight on a 4% class makes the model fire on nearly everything.
    y = labels_matrix(train_rows)
    pos = y.sum(0)
    neg = len(y) - pos
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, args.max_pos_weight),
                              dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loader = make_loader(train_rows, tok, args.max_len, args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    model.train()
    for _ in range(args.epochs):
        for ids, mask, yb in loader:
            ids, mask, yb = ids.to(device), mask.to(device), yb.to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_fn(logits, yb)
            if not torch.isfinite(loss):
                # Fail loudly rather than silently training on garbage. This
                # check is what caught the fp16/Adam-eps bug above.
                raise RuntimeError("Non-finite loss -- check precision settings")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
    return model, tok


@torch.no_grad()
def predict(model, tok, rows, args, device) -> np.ndarray:
    model.eval()
    loader = make_loader(rows, tok, args.max_len, args.batch_size, shuffle=False)
    out = []
    for ids, mask, _ in loader:
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out)


def threshold_at_fpr(y_true: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Lowest threshold whose false-positive rate on these rows is <= max_fpr.
    Lowest, because among thresholds meeting the FPR budget it gives the most
    recall. With no negatives present, fall back to 0.5 rather than guessing."""
    neg = scores[y_true == 0]
    if len(neg) == 0:
        return 0.5
    # threshold such that at most floor(max_fpr * n_neg) negatives score above it
    k = int(np.floor(max_fpr * len(neg)))
    neg_sorted = np.sort(neg)[::-1]
    return float(neg_sorted[k]) + 1e-9 if k < len(neg_sorted) else 0.0


def score(y_true: np.ndarray, scores: np.ndarray, preds: np.ndarray) -> dict:
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    out = {"n": int(len(y_true)), "n_pos": n_pos}
    if n_pos == 0 or n_neg == 0:
        # Undefined, not zero. Reporting 0.0 here would read as a total failure
        # on a fold that simply contained nothing to detect.
        out.update({"auroc": None, "avg_precision": None, "recall": None,
                    "fpr": None, "precision": None})
        return out
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    out.update({
        "auroc": round(float(roc_auc_score(y_true, scores)), 4),
        "avg_precision": round(float(average_precision_score(y_true, scores)), 4),
        "recall": round(tp / n_pos, 4),
        "fpr": round(fp / n_neg, 4),
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
    })
    return out


def run_seed(rows, folds, args, device, seed):
    n = len(rows)
    oof_scores = np.zeros((n, len(DIMS)), dtype=np.float32)
    oof_preds = np.zeros((n, len(DIMS)), dtype=np.int64)
    idx_of = {r["id"]: i for i, r in enumerate(rows)}
    per_fold = []

    for fold in folds:
        test = [r for r in rows if r["fold"] == fold]
        train_all = [r for r in rows if r["fold"] != fold]
        train, val = split_val_by_prompt(train_all, args.val_frac, seed)

        set_seed(seed)
        t0 = time.time()
        model, tok = train_one(args.model, train, args, device)

        val_scores = predict(model, tok, val, args, device)
        test_scores = predict(model, tok, test, args, device)
        y_val = labels_matrix(val)
        y_test = labels_matrix(test)

        thresholds = [threshold_at_fpr(y_val[:, j], val_scores[:, j], TARGET_FPR)
                      for j in range(len(DIMS))]
        test_preds = np.stack([(test_scores[:, j] >= thresholds[j]).astype(np.int64)
                               for j in range(len(DIMS))], axis=1)

        for r, s, p in zip(test, test_scores, test_preds):
            oof_scores[idx_of[r["id"]]] = s
            oof_preds[idx_of[r["id"]]] = p

        fold_result = {
            "fold": fold,
            "categories": sorted({r["category"] for r in test}),
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "seconds": round(time.time() - t0, 1),
            "thresholds": {d: round(t, 4) for d, t in zip(DIMS, thresholds)},
            "metrics": {d: score(y_test[:, j], test_scores[:, j], test_preds[:, j])
                        for j, d in enumerate(DIMS)},
        }
        per_fold.append(fold_result)
        print(f"    fold {fold} ({', '.join(fold_result['categories'])}) "
              f"done in {fold_result['seconds']}s", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    y_all = labels_matrix(rows)
    pooled = {d: score(y_all[:, j], oof_scores[:, j], oof_preds[:, j])
              for j, d in enumerate(DIMS)}
    return {"seed": seed, "pooled": pooled, "per_fold": per_fold}


def summarize(seed_results):
    summary = {}
    for d in DIMS:
        summary[d] = {}
        for m in ["auroc", "avg_precision", "recall", "fpr", "precision"]:
            vals = [s["pooled"][d][m] for s in seed_results if s["pooled"][d][m] is not None]
            summary[d][m] = ({"mean": round(float(np.mean(vals)), 4),
                              "std": round(float(np.std(vals)), 4)} if vals else None)
        summary[d]["n_pos"] = seed_results[0]["pooled"][d]["n_pos"]
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--data", default=str(DATA_PATH))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-pos-weight", type=float, default=10.0)
    ap.add_argument("--smoke", action="store_true",
                    help="deberta-v3-xsmall, short inputs, 1 epoch, 1 seed -- checks the "
                         "pipeline runs, NOT a result")
    args = ap.parse_args()

    if args.smoke:
        # Same architecture AND tokenizer as the real target (deberta-v3-base),
        # just smaller -- so the smoke test exercises the exact tokenization
        # and model code path the GPU run will use. An unrelated tiny model
        # (e.g. prajjwal1/bert-tiny, which this originally used) can pass
        # while the real model fails, which defeats the point.
        args.model = "microsoft/deberta-v3-xsmall"
        args.seeds, args.epochs, args.max_len, args.lr = 1, 1, 128, 5e-5

    rows = [json.loads(l) for l in open(args.data)]
    folds = sorted({r["fold"] for r in rows})
    device = pick_device()
    label_source = rows[0].get("label_source", "?")

    print(f"Model:   {args.model}")
    print(f"Device:  {device}")
    print(f"Data:    {len(rows)} rows, {len(folds)} category folds, labels from {label_source}")
    if args.smoke:
        print("SMOKE TEST -- verifies the pipeline only. These numbers mean nothing.")
    if "judged.jsonl" == label_source:
        print("WARNING: rubric-v1 labels. Fine for checking the pipeline; the guard "
              "should be trained on v3 labels before any result is reported.")
    print()

    seed_results = []
    for s in range(args.seeds):
        print(f"  seed {s}:")
        seed_results.append(run_seed(rows, folds, args, device, seed=1000 + s))

    summary = summarize(seed_results)

    print(f"\nPOOLED OUT-OF-FOLD RESULTS  (mean ± std over {args.seeds} seed(s))")
    print(f"{'dimension':26s} {'n_pos':>6s} {'AUROC':>14s} {'AvgPrec':>14s} "
          f"{'recall@FPR<=.15':>16s}")
    for d in DIMS:
        s = summary[d]

        def fmt(m):
            return "      -       " if s[m] is None else f"{s[m]['mean']:.3f} ± {s[m]['std']:.3f}"
        print(f"{d:26s} {s['n_pos']:>6d} {fmt('auroc'):>14s} {fmt('avg_precision'):>14s} "
              f"{fmt('recall'):>16s}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else args.model.split("/")[-1]
    out_path = RESULTS_DIR / f"encoder_{tag}_{Path(label_source).stem}.json"
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "device": str(device), "label_source": label_source,
                   "summary": summary, "seeds": seed_results}, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
