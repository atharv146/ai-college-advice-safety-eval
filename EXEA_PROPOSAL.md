# FR-Guard: A Small Guard Model for False Reassurance in High-Stakes Advice

**Applicant:** Atharv Sharma
**Preliminary work:** [github.com/atharv146/ai-college-advice-safety-eval](https://github.com/atharv146/ai-college-advice-safety-eval) · [dataset](https://huggingface.co/datasets/atharv146/college-advice-ai-safety-eval-prompts)

---

## 1 · Problem

When immigrant and mixed-status families ask AI chatbots about college financial aid, the dangerous failure is not a refusal, it is *false reassurance*: a confident "no, that's completely safe" about an immigration-enforcement risk the model cannot actually assess. In prior work I built an 82-prompt benchmark for this failure class and measured a 32.9% baseline failure rate across three commercial models, with a 4.5× spread between them (Gemini 3.7 Flash 65.9%, Claude Haiku 4.5 14.6%). Current mitigations are prompt-level: they work (32.9% → 6.5%) but are model-specific, invisible at inference time, and unauditable by anyone deploying the system. There is no detector for this failure class, so no deployment can verify it is not happening.

## 2 · Approach

Fine-tune small open-weight models (Llama 3.2 1B / 3B, Qwen 2.5 7B) as dedicated binary classifiers over three failure dimensions (false reassurance, fabricated policy claim, unhelpful evasion), in the architectural style of Llama Guard: a separate model that reads a (question, response) pair and emits calibrated per-dimension scores, rather than an instruction embedded in the generator.

Why this rather than more prompt engineering: prompt-level intervention has already been shown to work in my preliminary results, so the open question is not *whether* the behavior can be suppressed but whether it can be **detected independently of the generator**. A guard model is deployment-agnostic (works in front of any model), auditable (produces a score, not a vibe), and cheap enough to run on every response, which a GPT-4o judge is not.

The scaling comparison across 1B/3B/7B is a core part of the experiment, not a convenience: the practical question for anyone deploying this is the smallest model that retains acceptable recall on a rare, high-cost failure.

**The generalization test is the actual research question.** I will hold out entire risk categories from training (e.g. train without FERPA and residency prompts, test on them) to measure whether the guard learns a transferable notion of unsafe reassurance or merely memorizes topic surface features. A guard that only works on categories it has seen is not useful, since the next unsafe question a family asks will not be in anyone's training set.

## 3 · Dataset

**Existing, in hand:** 840 (question, response) pairs already graded on all three dimensions by a held-out GPT-4o judge, spanning 5 models × 3 intervention conditions × 82 prompts, plus 65 graded multi-turn conversations. Prompts and rubric are already published under CC-BY-4.0.

**Expansion plan:** widen generation to ~10 models × 3 conditions across the existing prompt bank plus paraphrase augmentation of the prompts themselves, targeting 8,000–12,000 labeled pairs. Labels come from the same GPT-4o judge (distillation), which is the standard construction for guard-model training data.

**Held-out evaluation set:** a human-labeled subset is the ground truth for final numbers, specifically because a guard distilled from an LLM judge will inherit that judge's blind spots, and only human labels can measure that. Multi-rater tooling (Cohen's kappa, judge-vs-human and human-vs-human) is already built in the preliminary repo.

**Preprocessing:** dimension labels are binary and independent; class imbalance is severe (false reassurance is ~1% under hardened prompts, ~21% under baseline) and will be handled with condition-stratified sampling and threshold tuning rather than naive resampling.

## 4 · Metrics

- **Per-dimension precision / recall / F1 / AUROC** on the human-labeled held-out set. Recall on false reassurance is the primary metric: a missed unsafe response is the costly error.
- **Cross-category generalization:** F1 on held-out risk categories vs. seen categories. This is the headline result.
- **Baseline comparisons:** (a) GPT-4o-as-judge, the capability ceiling and ~100× the cost; (b) prompt-level intervention alone; (c) a keyword/regex baseline, to prove the task is not trivially solvable.
- **Over-refusal cost when deployed as a filter.** My preliminary work found that retrieval-grounding cut unsafe reassurance to 0.8% but drove over-refusal from 6.1% to 50.4%, a net-worse system. Any guard that gates responses must be measured on both axes or it will repeat that failure.
- **Latency and memory at inference**, since the deployment claim depends on it being cheap enough to run on every response.

**Success threshold, stated in advance:** ≥0.85 recall on false reassurance at ≤0.15 false-positive rate on held-out categories, from a model small enough to serve on commodity hardware.

## 5 · Compute

**Request: 18 GPU-hours** (11 core + 7 buffer), single GPU, no multi-node.

The estimate is derived from the measured shape of the existing labeled data rather than assumed. Across the 840 responses already collected, mean length is 16 words (question) + 147 words (response) ≈ **219 tokens per training example**. At a 10,000-example target that is **2.2M tokens per epoch**, which is small: sequences are short, so batches pack efficiently and a single fine-tune is minutes, not hours.

Assuming conservative LoRA throughput on one MI250X-class GPU (15k / 7k / 3.5k tok·s⁻¹ for 1B / 3B / 7B):

| Phase | Work | Hours |
|---|---|---|
| 1 | Hyperparameter sweep, 6 configs on 3B only | 1.6 |
| 2 | Best config trained at all three sizes | 0.9 |
| 3 | **Leave-one-category-out, 8 folds × 3 sizes** (the headline experiment) | 7.3 |
| 4 | Evaluation and inference passes | 1.5 |
| | **Core total** | **11.2** |
| | Buffer (60%): ROCm environment setup, failed runs, reruns | 6.7 |
| | **Requested** | **18.0** |

A single 3-epoch run is ~7 min (1B), ~16 min (3B), ~31 min (7B). Phase 3 dominates because leave-one-category-out is 24 separate training runs; that is the cost of the generalization result, and it is the part I would protect if the budget were cut.

**Memory:** ~20–40 GB VRAM for 7B bf16 + LoRA; fits a single MI250X or MI300X with headroom. QLoRA is available as a fallback if memory is tighter than expected.

**Verification commitment:** the throughput figures above are literature-standard, not measured on AMD hardware. I will benchmark actual tokens·s⁻¹ in the first 30 minutes of access and report the revised estimate before consuming the rest of the allocation. If real throughput is materially worse, I will cut the 1B tier first (it is the most likely to fail the task anyway) rather than reduce the number of holdout folds.

**Dependencies:** PyTorch (ROCm build), HuggingFace `transformers` / `peft` / `trl` / `datasets`, `scikit-learn`. No CUDA-only kernels are required for LoRA fine-tuning at this scale.

**Not requested:** all data generation, judging, and evaluation of commercial models runs on inference APIs and is self-funded (the completed prior study cost ≈ $2). No cluster time is spent on data collection.

## 6 · Deliverables

1. **Model weights** for the best-performing guard, released on HuggingFace under a permissive license.
2. **A paper** (draft structure already written) reporting the scaling comparison, the cross-category generalization result, and the safety/over-refusal frontier. Target: an AI-safety or socially-responsible-NLP workshop.
3. **An expanded public dataset**: ~10k labeled (question, response, 3-dimension label) pairs, the first for this domain.
4. **A demo**: a small hosted endpoint that scores a pasted chatbot response, so the artifact is inspectable without running anything.

## 7 · Risk

**What breaks first: label quality.** The guard is distilled from an LLM judge, so any systematic judge bias becomes a systematic guard bias. *Plan B:* the human-labeled held-out set is built before training, not after, so this is measured rather than discovered late. If judge-human kappa on the seed labels is poor (< 0.6), I revise the rubric and re-label before spending any GPU time on training.

**Second risk: the small models cannot learn the task.** Distinguishing appropriate hedging from false reassurance is subtle, and 1B may simply lack the capacity. *Plan B:* the scaling comparison is designed so this is a reportable result rather than a failure. "The task requires ≥XB parameters" is a useful finding for anyone deploying a guard, and the 7B tier is a realistic fallback if the small tiers fail.

**Third risk: cross-category generalization fails.** The guard may learn topic keywords instead of the underlying failure mode. *Plan B:* this is precisely the experiment, not an accident. A negative result here is publishable and directly useful, it would show that guard models for high-stakes advice need per-domain training data rather than transferring, which is important for anyone planning to deploy one. My preliminary work already reports a counterintuitive negative result (the retrieval-grounding tradeoff) rather than burying it, and I would treat this the same way.

**Scope control:** the benchmark, rubric, evaluation harness, statistical pipeline, and human-labeling tooling already exist and are published. This proposal adds training and evaluation only, not infrastructure from scratch.
