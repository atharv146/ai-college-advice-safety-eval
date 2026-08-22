# FR-Guard: Detecting False Reassurance in High-Stakes Advice

**Applicant:** Atharv Sharma
**Preliminary work:** [github.com/atharv146/ai-college-advice-safety-eval](https://github.com/atharv146/ai-college-advice-safety-eval) · [dataset](https://huggingface.co/datasets/atharv146/college-advice-ai-safety-eval-prompts)
**Revision:** v3, revised against two rounds of reviewer feedback. Compute plan rebuilt for the provisioned hardware (~20 h, single NVIDIA T4 16 GB).

---

## 1 · Problem

When immigrant and mixed-status families ask AI chatbots about college financial aid, the dangerous failure is not a refusal, it is *false reassurance*: a confident "no, that's completely safe" about an immigration-enforcement risk the model cannot actually assess. In prior work I built an 82-prompt benchmark for this failure class and measured a **24.8% baseline failure rate** across three commercial models, with a near-3× spread between them (Gemini 3.7 Flash 41.5%, Claude Haiku 4.5 14.6%). Prompt-level mitigation works (24.8% → 4.5%) but is model-specific, invisible at inference time, and unauditable by anyone deploying the system. There is no detector for this failure class, so no deployment can verify it is not happening.

*(Note: an earlier draft reported 32.9% / 65.9%. Those figures were inflated by a silent truncation bug — a reasoning model's internal tokens consuming its response budget — found during human labeling. All affected data was regenerated and the incident is documented in the repository README. The figures above are the corrected ones.)*

## 2 · Approach

**The deployment claim is the contribution, and it now leads.** The value of a guard is that it is deployment-agnostic (sits in front of any model), auditable by whoever operates the system, and cheap enough to run on every response — which a frontier-model judge is not. Accuracy parity with GPT-4o is explicitly *not* the claim; see §4.

**Three arms, in increasing cost order.** The provisioned hardware makes full 7B fine-tuning infeasible (§5), so the design separates *representation quality* from *fine-tuning cost*:

| Arm | What it is | Why |
|---|---|---|
| **A. Encoder baseline** | DeBERTa-v3-base / RoBERTa-base (~125–400M), full fine-tune | Cheapest, run first. Establishes the achievable floor and validates the split protocol before any expensive run. If it clears threshold, that *strengthens* the deployment argument. |
| **B. Frozen 7B + probe** | One 4-bit forward pass over the dataset → pooled hidden states → logistic-regression / small-MLP probe | Retains 7B-scale representation quality without 7B fine-tuning cost. All folds, seeds and threshold sweeps become near-free once features are cached. |
| **C. 1B/3B QLoRA** | Generative fine-tune, verdict + rationale | The only tier where the category-holdout matrix is affordable *and* the model is genuinely adapted rather than probed. |

Arm B is also the cheapest realization of the "low-rank adapter plus small secondary model" pattern suggested in the first review, and it directly resolves the tension between the two reviews: reviewer 1 argued sub-7B models may lack the reasoning for this task; reviewer 2 correctly noted that on a T4 only sub-7B tiers are affordable to fine-tune. Frozen 7B features give 7B-quality representations at ~1.5 h rather than ~20 h.

**Head and objective — resolved, previously ambiguous.** An earlier draft said both "in the architectural style of Llama Guard" and "emits calibrated per-dimension scores," which point at different models. Resolved explicitly:

- **Arm C is generative**, in the true Llama Guard sense: verdict tokens emitted through the LM head under ordinary next-token cross-entropy, with per-dimension probabilities read from constrained-decoding logits, *not* three sigmoid heads. Rationale supervision accompanies the verdict (the 840 existing examples already carry judge-written rationales), providing semantic signal beyond the bare label. Verdict tokens carry label smoothing.
- **Arms A and B are discriminative** by construction (encoder head; probe on frozen features). This is deliberate and stated rather than blurred: the arms differ in objective, and comparing them is part of the point.

**The generalization test remains the research question**, now with a second axis added per reviewer feedback:
- **Category holdout** — train without whole risk categories (e.g. FERPA, residency), test on them. Does the guard learn transferable "false reassurance" or memorize topic surface features?
- **Generator holdout** — train on responses from 8 generating models, test on 2 held-out ones. Nearly free (the generator label is already in the data) and arguably closer to the deployment case, where the guard scores output from arbitrary systems.

**The three dimensions are a frontier, not three targets.** Pushing recall on false reassurance pushes generators toward evasion, which is itself dimension three. The deliverable is the trade-off curve, not a single number.

## 3 · Dataset

**In hand:** 840 (question, response) pairs graded on three dimensions by a held-out GPT-4o judge (5 generator models × 3 conditions × 82 prompts), plus 60 graded multi-turn conversations. Prompt bank and rubric published CC-BY-4.0.

**Expansion:** ~10 generators × 3+ conditions over the existing prompt bank, targeting **4,000–6,000 training pairs** (reduced from 8–12k; the compute budget binds before the data does). Systematic *system-prompt* variation is the primary augmentation lever rather than paraphrase, because varying the generating condition produces near-boundary examples — responses that hedge somewhat but not enough — while paraphrase mostly produces easy duplicates. **The judge-labeled pool is kept large even though the training set shrinks**, because the PPI estimator in §4 draws its precision from exactly that pool; discarding machine labels to save GPU time would forfeit statistical power for nothing.

**Splitting — leakage controls, specified.** Splits are made at the **seed-prompt level**, with seed IDs propagated through all augmentation, so no paraphrase or sibling response of a training prompt appears in evaluation. The 10-generator × 3-condition structure means one seed prompt yields ~30 highly correlated responses; pair-level splitting would put near-duplicates on both sides and the generalization number would partly measure memorization. Any class-imbalance resampling happens **after** splitting, never before. Near-duplicate overlap across splits (embedding cosine / minhash) is reported as a sanity check.

**Three splits, not two.** train / threshold-tuning validation / human-labeled test. Thresholds, early stopping and model selection all happen on judge-labeled validation data. **The gold set is read once.**

**Gold set protocol (~200–250 items):**

1. **Stratified by judge label, not base rate** — roughly half judge-positive, half judge-negative, which roughly doubles positives per labeling hour. Horvitz–Thompson inverse-probability weights are applied and reported for every prevalence-dependent quantity (precision, prevalence, calibration), and for recall/FPR as well, since the stratifying variable is the judge label rather than truth.
2. **Fully double-labeled** — 200 items × 2 annotators = 400 annotations, affordable precisely because the set is small. Yields human–human κ directly, permits adjudication of disagreements, and enables the more robust framing: guard–human agreement *relative to the human ceiling* rather than an absolute recall figure.
3. **Sequential labeling with a pre-registered stopping rule** — batches of 50; stop when the bound resolves or budget is exhausted. The rule is written down before labeling starts, since optional stopping without a spending rule inflates error rates.
4. **Marginal labels go to guard-vs-judge disagreements**, with a random subsample retained as an unbiased spine and the disagreement sample treated as a separate reweighted stratum.

**Annotator qualification — stated, because it is a validity threat rather than a quality one.** Whether "no, that's completely safe" constitutes false reassurance about a FERPA disclosure or a residency determination depends on the actual exposure, which general annotators cannot assess. Baseline annotation is by the author plus a second trained annotator against the written rubric. I will additionally seek adjudication time from someone in immigration legal aid or a financial-aid office for the disagreement subset specifically; per the review this is the highest-value labeling spend available and I am treating it as a real deliverable rather than a nicety. **If no qualified adjudicator is secured, that limitation is reported in the paper explicitly rather than elided.**

**Dimension independence — tested, and the earlier claim was wrong.** A previous draft asserted the three labels were "binary and independent." Tested on the existing 840 pairs, they are not:

| Pair | φ | p |
|---|---|---|
| unsafe reassurance × fabricated policy claim | **+0.238** | 4×10⁻¹¹ |
| unsafe reassurance × unhelpfully evasive | −0.134 | 2×10⁻⁴ |
| fabricated policy claim × unhelpfully evasive | −0.092 | 0.015 |

The positive association is exactly the predicted mechanism — a confident wrong policy claim is often *how* false reassurance is delivered. The two negative associations have **zero co-occurrences**, i.e. reassurance and evasion are structurally mutually exclusive. This is evidence for a structured output over three independent sigmoids, and it is now a stated design input rather than an assumption.

## 4 · Metrics

**Primary, pre-registered:** a **one-sided 95% lower confidence bound** on recall for false reassurance exceeding 0.85, at ≤0.15 FPR, on held-out categories. The point-estimate form of this criterion was unfalsifiable at achievable gold-set sizes; the lower-bound form is checkable at any *n*, requiring a larger margin when *n* is small (≈47/50, ≈91/100, ≈179/200 by Wilson; verified numerically).

**Stated in advance: a near-miss is inconclusive, not a failure.** If the guard lands at 0.88 recall on 60 gold positives, the correct report is "not demonstrated at this sample size." Committing to that now is what keeps the pre-registration meaningful.

**Estimator: prediction-powered inference (PPI)** as primary, with gold-only intervals reported as a sensitivity check. PPI is built for exactly this setting — a large machine-labeled pool plus a small gold sample — using the former for precision and the latter to debias, yielding valid intervals materially tighter than gold-only when the judge is reasonably accurate.
- Angelopoulos, Bates, Fannjiang, Jordan & Zrnic, "Prediction-powered inference," *Science* 382(6671):669–674, 2023. doi:10.1126/science.adi6000
- Angelopoulos, Duchi & Zrnic, "PPI++: Efficient prediction-powered inference," arXiv:2311.01453

**Paired comparison on identical gold items**, with McNemar's test rather than overlapping marginal CIs. Paired tests are far more powerful at small *n*: "guard beats regex baseline" and "guard retains X% of judge recall" remain defensible on 200 items where absolute recall is not.

**Generalization reported as one contrast, not five.** Held-out-categories minus seen-categories, single paired number with a category-clustered bootstrap CI. Per-category figures go to an appendix explicitly labelled underpowered, since per-category *n* ≈ 40 is noise.

**Baselines, reframed.** The GPT-4o comparison is a **retention** metric — "recovers X% of judge recall at Y× lower cost and Z ms latency" — not an accuracy contest. The guard is distilled from GPT-4o labels and structurally cannot beat its teacher on label agreement; framing that comparison as accuracy would manufacture an apparent failure. Accuracy comparisons run against the keyword/regex baseline and prompt-level intervention.

**Also reported:** calibration (ECE and Brier, with a Platt/isotonic layer fitted on validation — previously "calibrated scores" were promised and never measured); over-refusal cost when deployed as a gate; latency and memory at inference; and **precision at realistic deployment base rates (1–2%)** alongside the benchmark base rate, since the benchmark is engineered to elicit failures and a 0.15 FPR at a 1% base rate yields low-single-digit precision. That number belongs next to the threshold so no reader carries a benchmark figure into a deployment claim.

## 5 · Compute — ~20 hours, single T4 (16 GB, Turing)

Rebuilt entirely for the provisioned hardware. The previous budget assumed MI250X/MI300X-class throughput; the T4 is roughly a 30–40× reduction in effective compute, which is a redesign rather than a trim.

**First action: calibrate.** 50 optimizer steps per tier, measure real tokens/sec, rebuild the table below before committing. Half an hour spent measuring prevents discovering an overrun at hour 18.

| Item | Hours |
|---|---|
| Throughput calibration across tiers | 0.5 |
| 7B frozen-feature extraction (full dataset, once) | 1.5 |
| Probe + calibration sweep on frozen features (all folds, all seeds) | 0.5 |
| **Arm A** — encoder baseline: 5 category folds × 3 seeds | 4.0 |
| **Arm C** — 1B QLoRA: 5 category folds | 4.0 |
| **Arm C** — 3B QLoRA: 3 category folds | 6.0 |
| Held-out eval, latency and memory measurement | 1.5 |
| Buffer | 2.0 |
| **Total** | **20.0** |

**Priority order if the budget slips: folds > seeds > sweep.** The hyperparameter sweep is cut first and known-good LoRA settings used instead (r=16, α=32, lr 1e-4–2e-4, cosine). The folds are the result; the sweep is not. Training-set size (4–6k) is reduced before test-set integrity is touched, and the judge-labeled pool is never reduced (§3).

**T4-specific engineering constraints**, adopted from the review:
- **No bf16** (Turing/SM75). fp16 with gradient scaler, LoRA adapters in fp32, NF4 compute dtype set to fp16, watch loss-scale instability.
- **No FlashAttention-2** (Ampere+). PyTorch SDPA memory-efficient backend or xformers; attention memory scales worse than originally assumed.
- **NF4 over int8** — `LLM.int8()` is slow in training due to mixed-precision decomposition; 4-bit NF4 is smaller and faster here. (This supersedes the int8 recommendation from the first review, which assumed different hardware.)
- **Length bucketing** rather than fixed 1,024 padding, since padding waste is a real fraction of a scarce budget.
- Thermal throttling is common on cloud T4s and is budgeted into the buffer.

**Memory** is not the binding constraint — throughput is. 7B at NF4 with a low-rank adapter, gradient checkpointing and ~1k-token inputs is ~11–13 GB and fits the card; earlier 20–40 GB figures assumed bf16 and higher rank.

**Raised for consideration, not a request:** if the binding constraint is hours rather than this specific card, an L4 or A10G is roughly 1.5–2× the hourly rate for 3–4× the throughput plus bf16 and FlashAttention-2. Twenty L4-hours would be worth something like 60–80 T4-hours. If the T4 is fixed, the plan above stands as written and I am not blocked.

**Not requested:** all data generation and judging runs on inference APIs, self-funded (the completed prior study cost ≈ $4). No cluster time is spent on data collection.

## 6 · Deliverables

1. **Model weights** for the best-performing arm, on HuggingFace under a permissive license.
2. **A paper** reporting the three-arm comparison, the category- and generator-holdout generalization results, and the safety/over-refusal frontier. Target: an AI-safety or socially-responsible-NLP workshop.
3. **An expanded public dataset**: labeled (question, response, 3-dimension label, rationale) pairs — the first for this domain — plus the ~200-item double-labeled gold set with annotator agreement statistics.
4. **A reproducible notebook or HuggingFace Space** rather than a hosted endpoint. Per the review, a live endpoint carries hosting and abuse-surface cost for a vulnerable-population tool while adding little to the research claim.

## 7 · Risk

**What breaks first: label quality. Already partially realized, and it now gates GPU spend.** The first human-validation pass has been run and is not reassuring: on 20 jointly-labeled responses, judge–human Cohen's κ was **0.70 on unhelpfully-evasive but ~0.15 on unsafe-reassurance and near zero on fabricated-policy-claim**. Raw agreement is 60–87%, which is the classic signature of agreeing on easy negatives in an imbalanced class.

*Plan A, before any cluster time:* establish **human–human κ first**, on a 100–150 item pilot with 2–3 annotators, because inter-annotator agreement is the ceiling on everything downstream. If two qualified annotators cannot agree on "is this false reassurance," then no guard can meaningfully hit 0.85 recall, the judge–human κ is uninterpretable, and the task needs redefinition before anything is trained. The revision trigger fires on **whichever κ is lower**. I would rather report that gate honestly than distil a judge whose labels I cannot defend.

**Second risk: the task is not learnable at affordable scale.** *Plan B:* the three-arm design makes this diagnosable rather than fatal. If the encoder, the frozen-7B probe, and 1B/3B QLoRA all fail similarly, that points to task difficulty rather than a bad checkpoint, and "not learnable at ≤3B from ~5k distilled labels" is a reportable, useful negative result. Conversely, if the encoder clears the threshold, the scaling story changes shape in a way that *strengthens* the deployment argument.

**Third risk: generalization fails.** *Plan C:* this is the experiment, not an accident. A negative result would show that guards for high-stakes advice need per-domain training data rather than transferring — directly useful to anyone planning to deploy one. The completed prior study already reports a counterintuitive negative result (retrieval-grounding cut unsafe reassurance to 0.4% but drove over-refusal to 47.2%, a net-worse system) rather than burying it, and this would be handled the same way.

**Fourth risk: benchmark contamination.** The prompt bank is published CC-BY-4.0 and may enter future models' training data; the GPT-4o judge's prior exposure to it is unknown and unverifiable. Noted as a standing limitation on all judge-derived labels.

**Scope control:** benchmark, rubric, evaluation harness, statistical pipeline and multi-rater labeling tooling already exist and are published. This proposal adds training and evaluation only.

**Explicitly scoped out:** the 60 multi-turn conversations are **not** used for guard training or evaluation in this project. The guard consumes a single (question, response) pair; false reassurance constructed across turns is a genuinely different input format and a different problem. They remain in the dataset release as future work rather than being quietly counted toward scope.
