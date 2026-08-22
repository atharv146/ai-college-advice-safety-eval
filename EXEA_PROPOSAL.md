# FR-Guard: A Small Guard Model for False Reassurance in High-Stakes Advice

**Applicant:** Atharv Sharma
**Preliminary work:** [github.com/atharv146/ai-college-advice-safety-eval](https://github.com/atharv146/ai-college-advice-safety-eval) · [dataset](https://huggingface.co/datasets/atharv146/college-advice-ai-safety-eval-prompts)

---

## 1 · Problem

When immigrant and mixed-status families ask AI chatbots about college financial aid, the dangerous failure is not a refusal, it is *false reassurance*: a confident "no, that's completely safe" about an immigration-enforcement risk the model cannot actually assess. In prior work I built an 82-prompt benchmark for this failure class and measured a 32.9% baseline failure rate across three commercial models, with a 4.5× spread between them (Gemini 3.7 Flash 65.9%, Claude Haiku 4.5 14.6%). Current mitigations are prompt-level: they work (32.9% → 6.5%) but are model-specific, invisible at inference time, and unauditable by anyone deploying the system. There is no detector for this failure class, so no deployment can verify it is not happening.

## 2 · Approach

Fine-tune 7B-class open-weight models as a **generative** guard: given a (question, response) pair, the model emits a short structured verdict plus a one-sentence rationale, trained with the standard next-token objective rather than a binary classification head.

**On the objective (revised after reviewer feedback).** An earlier draft of this proposal described "binary classifiers" while simultaneously citing Llama Guard as the architectural model. That was internally inconsistent: Llama Guard is a *generative* classifier that produces verdict tokens through the LM head, not a discriminative head trained with BCE. The reviewer's objection to a pure BCE formulation is well taken and the design is changed accordingly:

- **Verdict tokens through the LM head**, so the training signal is ordinary token-level cross-entropy, the objective the base model was actually pretrained under, rather than a freshly-initialized classification head fighting it.
- **Rationale supervision alongside the verdict**, which is where the semantic component enters. The model must generate *why* a response is unsafe, not just emit a label. This is free to implement: the 840 already-graded examples each carry a judge-written rationale (`rationale` field in `results/judged.jsonl`), so the supervision signal already exists in the collected data.
- **Label smoothing on the verdict tokens**, because the rubric's own boundary is genuinely fuzzy (the first human-validation pass produced Cohen's kappa of only 0.15 on the unsafe-reassurance dimension), and hard 0/1 targets would train the model to be more confident than the labels justify.

**On architecture (revised after reviewer feedback).** The reviewer's suggestion to pair a lower-rank LoRA with a separate small component rather than embed factuality in the adapter maps unusually cleanly onto this task, because the three rubric dimensions decompose along exactly that line:

| Dimension | Nature | Handled by |
|---|---|---|
| Unsafe reassurance | pragmatic / stylistic — is the hedging real? | low-rank LoRA |
| Unhelpfully evasive | pragmatic / stylistic | low-rank LoRA |
| Fabricated policy claim | **factual** — is this claim actually state-varying? | separate lightweight checker |

Trying to bake "which policies vary by state" into adapter weights is exactly the kind of factual memorization LoRA is poorly suited to, and it would go stale the moment a state changes its law. Splitting it out means the factual component can be updated without retraining anything. Adopting this.

Why a guard model at all rather than more prompt engineering: prompt-level intervention already works in my preliminary results, so the open question is not *whether* the behavior can be suppressed but whether it can be **detected independently of the generator**. A guard is deployment-agnostic (sits in front of any model), auditable (emits a verdict and a reason), and cheap enough to run on every response, which a GPT-4o judge is not.

**Model selection (revised after reviewer feedback).** The original 1B/3B/7B scaling ladder is dropped. The reviewer is right that sub-7B models likely lack the reasoning to distinguish appropriate hedging from false reassurance, which would produce three failed runs and one usable one. The primary axis is now **7B-class across three model families** (Llama, Qwen, Mistral), which isolates a cleaner question: does this task transfer across pretraining lineages, or is it an artifact of one? One 3B run is retained as a single deployment-cost probe, explicitly expected to underperform; if it fails, "the task requires ≥7B" is a concrete, useful finding for anyone planning to deploy a guard cheaply, and it costs one run to establish.

**The generalization test remains the actual research question.** I hold out entire risk categories from training (e.g. train without FERPA and residency prompts, test on them) to measure whether the guard learns a transferable notion of unsafe reassurance or merely memorizes topic surface features. A guard that only works on categories it has seen is not useful, since the next unsafe question a family asks will not be in anyone's training set.

## 3 · Dataset

**Existing, in hand:** 840 (question, response) pairs already graded on all three dimensions by a held-out GPT-4o judge, spanning 5 models × 3 intervention conditions × 82 prompts, plus 65 graded multi-turn conversations. Prompts and rubric are already published under CC-BY-4.0.

**Expansion plan:** widen generation to ~10 models × 3 conditions across the existing prompt bank, targeting 8,000–12,000 labeled pairs. Labels come from the same GPT-4o judge (distillation), the standard construction for guard-model training data.

Per the reviewer's suggestion, **systematic prompt engineering is used as the primary augmentation lever** rather than paraphrase alone. Varying the *system* prompt across a spectrum (unguarded → partially hedged → fully hardened → grounded) is what actually produces the hard, near-boundary examples this task needs: responses that hedge somewhat but not enough. Paraphrasing user questions mostly produces easy duplicates; varying the generating conditions produces genuine label diversity along the decision boundary. Both are used, weighted toward the latter.

**Held-out evaluation set:** a human-labeled subset is the ground truth for final numbers, specifically because a guard distilled from an LLM judge will inherit that judge's blind spots, and only human labels can measure that. Multi-rater tooling (Cohen's kappa, judge-vs-human and human-vs-human) is already built in the preliminary repo.

**Preprocessing:** the three dimensions are scored independently; class imbalance is severe (false reassurance is ~1% under hardened prompts vs. ~21% under baseline) and is handled with condition-stratified sampling and threshold tuning rather than naive resampling. Verdict targets carry label smoothing, per §2.

## 4 · Metrics

- **Per-dimension precision / recall / F1 / AUROC** on the human-labeled held-out set. Recall on false reassurance is the primary metric: a missed unsafe response is the costly error.
- **Cross-category generalization:** F1 on held-out risk categories vs. seen categories. This is the headline result.
- **Baseline comparisons:** (a) GPT-4o-as-judge, the capability ceiling and ~100× the cost; (b) prompt-level intervention alone; (c) a keyword/regex baseline, to prove the task is not trivially solvable.
- **Over-refusal cost when deployed as a filter.** My preliminary work found that retrieval-grounding cut unsafe reassurance to 0.8% but drove over-refusal from 6.1% to 50.4%, a net-worse system. Any guard that gates responses must be measured on both axes or it will repeat that failure.
- **Latency and memory at inference**, since the deployment claim depends on it being cheap enough to run on every response.

**Success threshold, stated in advance:** ≥0.85 recall on false reassurance at ≤0.15 false-positive rate on held-out categories, from a model small enough to serve on commodity hardware.

## 5 · Compute

**Request: 12 GPU-hours**, single 16 GB-class GPU, no multi-node. Revised down from 18 after reviewer feedback (int8 + lower LoRA rank, and dropping the 1B/3B scaling ladder).

The estimate is derived from the measured shape of the existing labeled data. Across the 840 responses already collected, mean length is 16 words (question) + 147 words (response) ≈ **219 tokens per example**; adding rationale supervision brings the training sequence to ≈240 tokens. At a 10,000-example target that is **≈2.4M tokens per epoch**, which is small: sequences are short, so batches pack efficiently.

**15 training runs total:**

| Work | Runs |
|---|---|
| Hyperparameter sweep (LoRA rank, LR, label-smoothing ε) | 4 |
| **Leave-one-category-out, 8 folds on the primary family** (headline experiment) | 8 |
| Cross-family replication at 7B (Qwen, Mistral), full data | 2 |
| 3B deployment-cost probe | 1 |

At 3 epochs and a **conservative** 3,000 tok·s⁻¹ for 7B int8 + low-rank LoRA, a run is ~40 min → 10 h, plus ~1 h of evaluation and inference passes ≈ **11 h**. At the more optimistic 5,000 tok·s⁻¹ the same grid completes in ~7 h. **12 is requested to cover the conservative case with modest headroom; the realistic floor is ~7.**

**Memory:** the reviewer's figure is right and my original was inflated. 7B at int8 with a low-rank adapter is ≈7 GB of weights plus optimizer/activation overhead, comfortably inside a 16 GB card. The earlier 20–40 GB estimate assumed bf16 and a higher rank, neither of which the revised design needs.

**Verification commitment:** throughput figures are literature-standard, not measured on AMD hardware. I will benchmark actual tokens·s⁻¹ in the first 30 minutes of access and report a revised estimate before consuming the rest of the allocation. If real throughput is materially worse, I cut the cross-family replications and the 3B probe first, and protect the 8 holdout folds, which are the result the project exists to produce.

**Dependencies:** PyTorch (ROCm build), HuggingFace `transformers` / `peft` / `trl` / `datasets`, `scikit-learn`. No CUDA-only kernels are required for LoRA fine-tuning at this scale.

**Not requested:** all data generation, judging, and evaluation of commercial models runs on inference APIs and is self-funded (the completed prior study cost ≈ $2). No cluster time is spent on data collection.

## 6 · Deliverables

1. **Model weights** for the best-performing guard, released on HuggingFace under a permissive license.
2. **A paper** (draft structure already written) reporting the cross-family comparison, the cross-category generalization result, and the safety/over-refusal frontier. Target: an AI-safety or socially-responsible-NLP workshop.
3. **An expanded public dataset**: ~10k labeled (question, response, 3-dimension label) pairs, the first for this domain.
4. **A demo**: a small hosted endpoint that scores a pasted chatbot response, so the artifact is inspectable without running anything.

## 7 · Risk

**What breaks first: label quality. This is already partially realized and is the reason for a hard gate before any GPU time is used.** The guard is distilled from an LLM judge, so systematic judge bias becomes systematic guard bias. The first human-validation pass has now been run and is not reassuring: on 20 jointly-labeled responses, judge-human Cohen's kappa was **0.70 on unhelpfully-evasive but only ~0.15 on unsafe-reassurance and near zero on fabricated-policy-claim**. Raw agreement is high (60–87%) but kappa is low, the signature of a heavily imbalanced class where agreement is mostly agreement on the easy negatives.

*Plan A, before requesting the cluster:* diagnose whether that is rubric ambiguity (fixable by sharpening the wording, especially the boundary of "real hedging") or a genuine judge failure, add a second independent rater for the human-human reliability number the tooling already computes, and re-label. **I will not spend GPU hours distilling a judge whose labels I cannot yet defend**, and would rather report that gate honestly than train on top of it.

**Second risk: 7B is insufficient for the task.** Distinguishing appropriate hedging from false reassurance is a subtle pragmatic judgment. *Plan B:* the cross-family replication at 7B is the diagnostic. If all three families fail similarly, that points to task difficulty rather than a bad checkpoint, and "this task is not learnable at 7B from ~10k distilled labels" is a reportable, useful negative result. The retained 3B probe provides the lower bound of that curve at the cost of one run.

**Third risk: cross-category generalization fails.** The guard may learn topic keywords instead of the underlying failure mode. *Plan B:* this is precisely the experiment, not an accident. A negative result here is publishable and directly useful, it would show that guard models for high-stakes advice need per-domain training data rather than transferring, which is important for anyone planning to deploy one. My preliminary work already reports a counterintuitive negative result (the retrieval-grounding tradeoff) rather than burying it, and I would treat this the same way.

**Scope control:** the benchmark, rubric, evaluation harness, statistical pipeline, and human-labeling tooling already exist and are published. This proposal adds training and evaluation only, not infrastructure from scratch.
