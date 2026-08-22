*Draft reply to the second Exea review. Longer than it needs to be — cut freely, but keep the three answered questions and the gate.*

---

Thank you for this — it's the most useful feedback I've received on any project, and it changed the design rather than the wording. Revised proposal (v3) attached; it's a substantial rewrite of §2–§5 rather than a patch.

**One clarification that may matter:** the version you reviewed was my original submission. After the first round of feedback I'd already revised the objective away from BCE toward a generative head, and cut compute from 60–80 h to 12 h. Your review of §5 and the model tiers therefore lands on a document I'd partly superseded — but almost every point still applies, and several apply *more* forcefully to the revision than the original. The T4 provisioning in particular invalidates my revised plan too, so §5 is rebuilt from scratch against the actual hardware.

### Adopted

**Evaluation design (Major #1–#3) — adopted in full, and these were the real gap.** You're right that the detail budget was on the parts least likely to break. The gold set is now specified: ~200–250 items, stratified by judge label with Horvitz–Thompson weights, fully double-labeled, sequential with a pre-registered stopping rule, read once. The success criterion is restated as a **one-sided 95% lower bound** rather than a point estimate — I verified your table numerically and get 47/50, 91/100, 179/200 (you rounded conservatively). I've also written in the consequence you named: **a near-miss is reported as inconclusive, not as failure.**

**PPI — adopted as the primary estimator.** I hadn't encountered it; it's plainly the right tool for a large machine-labeled pool plus a small gold sample, and it also gives an independent reason to measure judge–human agreement carefully rather than treating it as a tripwire. Gold-only intervals reported as a sensitivity check.

**Seed-level splitting (Major #3) — adopted, and this would have quietly ruined the headline result.** With ~10 generators × 3 conditions per seed prompt, pair-level splitting puts ~30 correlated siblings across the boundary. Splits are now at seed level with IDs propagated through augmentation, resampling strictly after splitting, and near-duplicate overlap reported.

**Human–human agreement first (Major #2) — adopted, and it is now a hard gate on your compute.** This is the item I'd most want you to hold me to. My first validation pass came back at κ ≈ 0.70 on unhelpfully-evasive but ~0.15 on unsafe-reassurance and near zero on fabricated-policy-claim, with 60–87% raw agreement — agreeing on easy negatives. **I'm not going to spend cluster hours distilling a judge whose labels I can't defend.** The revised §7 gates GPU work on running the human–human pilot first and firing the rubric-revision trigger on whichever κ is lower.

**The distillation ceiling (Major #4) — adopted, and you're right that I had the headline pointed the wrong way.** GPT-4o is now explicitly a *retention* metric ("recovers X% of judge recall at Y× cost, Z ms latency"); accuracy comparisons run against regex and prompt-intervention baselines. The deployment-agnostic/auditable argument is promoted to lead §2.

**Deployment base rate (Major #6) — adopted.** Precision at 1–2% base rates now reported alongside the benchmark rate. The benchmark is engineered to elicit failures and I shouldn't let that number travel.

**Annotator qualification (Major #5) — adopted, with a caveat I want to be honest about.** Baseline is myself plus a second trained annotator on the written rubric. I'll seek adjudication time from someone in immigration legal aid or a financial-aid office for the disagreement subset. If I can't secure that, I'll say so in the paper rather than imply expertise I didn't have.

**Compute (§5) — rebuilt around your allocation, largely as you specified it.** Encoder baseline first, frozen 7B features as the high-capacity arm, 1B/3B QLoRA as the fine-tuned tier, folds > seeds > sweep, calibration run before committing. Also adopted: NF4 over int8, fp16 with fp32 adapters, SDPA/xformers instead of FlashAttention-2, length bucketing, thermal-throttle buffer.

The frozen-7B-features suggestion (#2 under *What this forces*) is the piece that made the redesign work. It resolves a genuine conflict between my two reviews — the first argued sub-7B models lack the reasoning for this task, you correctly note that on a T4 only sub-7B tiers are affordable to fine-tune. Frozen 7B representations plus a cheap probe gets 7B-quality features at ~1.5 h instead of ~20 h, and it happens to be the cheapest form of the adapter-plus-secondary-model pattern the first reviewer suggested.

**Encoder baseline (#10) — adopted and promoted to *first* run**, for the reason you give: it's the cheapest experiment and the most likely to embarrass the headline, and if a sub-400M encoder clears threshold that's a *better* outcome for the deployment argument. DeBERTa-v3-base over ModernBERT on Turing, noted.

**Generator holdout (#11) — adopted.** Nearly free and closer to the deployment case than category holdout alone. Both axes now reported.

**Also adopted:** head architecture disambiguated (#7 — generative for the QLoRA arm, discriminative for encoder/probe, stated per arm rather than blurred); calibration actually measured (#12, ECE/Brier + Platt layer); threshold tuning moved to a third split (#9); LoRA rank stated (#16); ≥3 seeds (#17); notebook/Space instead of a hosted endpoint (#18); contamination noted (#19); frontier framing for the three dimensions (#14).

### Answered with data

**#8, dimension independence — you were right, and I tested it.** The claim cost nothing to check against the existing 840 labels, and it's false:

| Pair | φ | p |
|---|---|---|
| unsafe reassurance × fabricated policy claim | **+0.238** | 4×10⁻¹¹ |
| unsafe reassurance × unhelpfully evasive | −0.134 | 2×10⁻⁴ |
| fabricated policy × unhelpfully evasive | −0.092 | 0.015 |

The positive association is your predicted mechanism — a confident wrong policy claim is often *how* reassurance is delivered. The negative ones have **zero co-occurrences**: reassurance and evasion are structurally mutually exclusive. That's an argument for structured output over three independent sigmoids, and it's now a design input rather than an assumption.

**#13, multi-turn — scoped out explicitly** rather than left ambiguous. The guard consumes one (question, response) pair; reassurance built across turns is a different input format and a different problem. The 60 conversations stay in the dataset release as future work and are no longer counted toward scope.

**#20, "Gemini 3.7 Flash"** — verified; it's the live model identifier I called (`google/gemini-3.7-flash`), not a display name I paraphrased.

**§1 figures corrected.** You reviewed 32.9% / 65.9%. Those were inflated by a silent truncation bug found during human labeling — a reasoning model's internal tokens were consuming its response budget, cutting off 64% of one model's outputs while still passing a non-empty check. Corrected figures are 24.8% baseline and 41.5% for that model. All affected data was regenerated rather than patched and the incident is written up in the repo.

### Your six questions

1. **Gold set size:** ~200–250, double-labeled, stratified ~50/50 on judge label rather than base rate. Per-category *n* ≈ 40 is acknowledged as underpowered, which is why generalization is reported as a single paired contrast with a category-clustered bootstrap rather than five per-category F1s.
2. **Who labels:** author + second trained annotator on the written rubric; seeking legal-aid/financial-aid adjudication for disagreements, with the limitation reported if not secured.
3. **Splits:** seed-prompt level, IDs propagated through augmentation, resampling strictly after splitting.
4. **Head:** generative (LM head, constrained decoding) for the QLoRA arm; discriminative for encoder and frozen-feature probe. Per arm, stated.
5. **Thresholds:** tuned on a judge-labeled validation split. The gold set is read once.
6. **Multi-turn:** scoped out — see above.

### Not adopted

Only one, and it's minor. I've kept a single 3B QLoRA tier at 3 folds rather than collapsing entirely to 1B + encoder + probe. If the encoder and the frozen-7B probe bracket performance from below and above, 3B is the datapoint that tells me whether *fine-tuning* at a deployable size adds anything over *probing* a larger frozen model — which is the practical question someone deploying this would ask. It costs 6 h of the 20 and it's the first thing I cut if calibration shows my throughput assumptions are optimistic.

Thanks again. The evaluation-design section in particular is the part I'd have gotten wrong and only discovered afterwards.
