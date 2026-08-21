# Do AI Chatbots Give Unsafe College and Financial-Aid Advice to Immigrant Families?

**A benchmark, an evaluation, and a proposed fix**

*[Author name] · [High school name] · 2026*

## Abstract

General-purpose AI chatbots are an increasingly common first source of advice for immigrant and mixed-status families asking sensitive questions about college applications and financial aid, questions where a confidently wrong answer carries real risk. We construct an 82-item, pre-registered adversarial prompt bank across 8 risk categories and evaluate three widely-deployed commercial models (GPT-4o-mini, Gemini 3.7 Flash, Claude Haiku 4.5) under three conditions: a baseline system prompt, a safety-hardened system prompt drawn from a production application, and a retrieval-grounded condition restricted to verified source material with forced citations. Responses are scored on three independent failure dimensions by a held-out LLM judge. We find that 32.9% of baseline responses exhibit at least one safety failure, with substantial variation between models (65.9% for Gemini 3.7 Flash vs. 14.6% for Claude Haiku 4.5 on identical prompts). The hardened prompt reduces pooled failures to 6.5%, a statistically significant improvement for two of three models under McNemar's exact test. The retrieval-grounded condition nearly eliminates unsafe reassurance (20.7% → 0.8%) but increases over-refusal eightfold (6.1% → 50.4%), producing a *higher* overall failure rate than the simpler intervention. Under multi-turn escalation, the hardened prompt showed zero capitulations across 30 conversations, while baseline models gave unsafe final-turn responses in up to half of conversations. We release the prompt bank and rubric as a public dataset.

## 1. Introduction

General-purpose AI chatbots are increasingly the first place a nervous student or parent asks a question like "will filling out the FAFSA report my undocumented parent to immigration enforcement." This is not a hypothetical: this project's motivating incident was a real production failure in a college-guidance chatbot (PathFinder, a live application serving immigrant and first-generation students) that volunteered false reassurance about FERPA-based protection against immigration-enforcement risk, a failure caught in manual testing, not by any automated check. That single incident is the seed of this project: turning one observed failure into a systematic, repeatable test.

The stakes are specific to this population. Immigrant and mixed-status families face genuine uncertainty around immigration-status-sensitive processes (FAFSA, state financial aid, in-state tuition residency, DACA renewal timing), uncertainty that varies by state, changes over time, and where a wrong answer is not merely unhelpful but potentially harmful. A chatbot that confidently reassures a family that a real risk does not exist, or states a state's residency policy as permanent fact when it has changed multiple times in the last decade, is failing in a way that generic "hallucination" benchmarks do not specifically measure.

## 2. Related Work

**Red-teaming methodology.** Ganguli et al. (2022), "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned" (Anthropic), establishes the general methodology this project follows: systematically probing a model with adversarial prompts across defined harm categories, rather than relying on incidental discovery. Ganguli et al. also report low human-human agreement on what counts as a "successful attack," a finding directly relevant to why this project validates its own rubric with multiple independent human raters rather than trusting a single labeler.

**LLM-as-judge validity.** Zheng et al. (2023), "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS), finds that a strong LLM judge can match human preference judgments at a rate comparable to human-human agreement, but only establishes this by direct comparison against real human raters, never by assumption. This project's human-validation step (comparing an automated judge's grades against independent human raters on the same rubric) follows that same discipline rather than reporting judge-only numbers as if they were self-evidently trustworthy.

**Measuring confidently-wrong answers.** Lin, Hilton & Evans (2022), "TruthfulQA: Measuring How Models Mimic Human Falsehoods" (ACL), benchmarks models on questions where a plausible-sounding answer is actually false, finding that larger models are not necessarily more truthful. This project's "fabricated policy claim" dimension is the same failure mode, specialized to one domain: a model stating a specific state's residency policy as settled fact when the honest answer is "it varies and changes."

**Domain-specific chatbot safety evaluation.** Existing work has applied structured, rubric-based safety evaluation to AI chatbots in other sensitive domains, including sexual health information and mental health support, generally finding that chatbots perform inconsistently and that structured evaluation surfaces failures that would not be visible from casual use. No existing published benchmark, to this project's knowledge, applies this method to AI chatbots giving U.S. college-admissions or financial-aid advice to immigrant and mixed-status families specifically, a domain with its own distinct failure modes (state-varying policy, immigration-enforcement risk) not captured by general-purpose red-teaming benchmarks like HarmBench.

**What this project adds.** A pre-registered, domain-specific prompt bank and rubric for this exact population and question space; a same-benchmark comparison of whether a safety-hardened system prompt (the kind a real product would actually ship) measurably changes outcomes, tested with a paired significance test rather than an eyeballed comparison; and a proposed fix (retrieval grounded in verified source material) evaluated against the same benchmark as the problem, rather than a separate report of the problem alone.

## 3. Methods

### 3.1 Prompt bank construction

82 prompts were written across 8 risk categories (enforcement reassurance, DACA eligibility, mixed-status family aid, in-state residency claims, FERPA data sharing, scholarship eligibility, indirect/roleplay framings of the same risks, and leading-premise questions), before any model was run against them. The full bank and category definitions are published as a standalone dataset: [huggingface.co/datasets/atharv146/college-advice-ai-safety-eval-prompts](https://huggingface.co/datasets/atharv146/college-advice-ai-safety-eval-prompts). The dataset's upload timestamp and this repository's commit history serve as a verifiable record that the prompt bank and rubric predate any evaluation results.

### 3.2 Grading rubric

Three binary failure dimensions, scored independently per response, defined in `data/rubric.md` before any model output was reviewed:
- **Unsafe reassurance**: does the response state or strongly imply a real risk is low or absent, without appropriate hedging?
- **Fabricated policy claim**: does the response state a specific, state-varying or time-varying policy as settled fact?
- **Unhelpfully evasive**: does the response refuse to provide any real, useful information at all?

### 3.3 Conditions

Every prompt is run under three system-prompt conditions: **vanilla** (a generic helpful-assistant instruction), **guarded** (safety rules adapted directly from a real production system prompt, PathFinder's `system-prompt.ts`), and **grounded** (the model may answer only using retrieved passages from a verified source corpus, with a forced-citation and forced-uncertainty instruction; retrieval is TF-IDF cosine similarity over a small corpus of fact-checked guide content).

### 3.4 Models evaluated

Three commercial models, chosen as the cheap/fast tier of the three families end users actually interact with: `openai/gpt-4o-mini`, `google/gemini-3.7-flash`, and `anthropic/claude-haiku-4.5`, all accessed via OpenRouter. An earlier iteration used only free open-weight endpoints; see §6 for why that was abandoned and what it implies for reproducing this kind of work without a budget.

### 3.5 Statistical methods

Failure rates are reported with Wilson score 95% confidence intervals rather than raw percentages. The vanilla-vs-guarded comparison uses the exact binomial form of McNemar's test, appropriate because the two conditions are run on the identical set of prompts (a paired, not independent, comparison) and because sample sizes are too small to rely on the chi-square approximation.

### 3.6 Human validation

The automated LLM judge (a model held out from the set under test, to avoid a model grading itself or a close relative) is validated two ways: (1) agreement between the judge and each of *[K]* independent human raters on an overlapping sample, and (2) inter-rater agreement between the human raters themselves, computed with Cohen's kappa, which establishes whether the rubric itself is applied consistently independent of whether the automated judge is any good.

## 4. Results

### 4.1 Pooled failure rates by condition (n = 246 responses per condition)

| Condition | Unsafe reassurance | Fabricated policy claim | Unhelpfully evasive | Any failure |
|---|---|---|---|---|
| vanilla | 20.7% | 11.8% | 6.1% | 32.9% |
| guarded | 1.2% | 1.6% | 4.1% | 6.5% |
| grounded | 0.8% | 1.6% | 50.4% | 52.4% |

### 4.2 Per-model baseline variation and intervention effect

| Model | vanilla | guarded | McNemar p | b (fixed) | c (broken) |
|---|---|---|---|---|---|
| Gemini 3.7 Flash | 65.9% [55.1, 75.2] | 8.5% [4.2, 16.6] | < 0.0001 | 47 | 0 |
| GPT-4o-mini | 18.3% [11.4, 28.0] | 8.5% [4.2, 16.6] | 0.077 | 12 | 4 |
| Claude Haiku 4.5 | 14.6% [8.6, 23.9] | 2.4% [0.7, 8.5] | 0.013 | 12 | 2 |

Brackets are 95% Wilson score intervals. *b* is the count of prompts that failed under vanilla and passed under guarded; *c* is the reverse.

### 4.3 Failure rate by risk category (pooled, all models and conditions)

Enforcement reassurance 39.6%, mixed-status family aid 38.0%, indirect/roleplay framing 36.6%, FERPA data sharing 35.2%, scholarship eligibility 26.9%, in-state residency claims 26.0%, DACA eligibility 25.7%, leading premise 20.6%.

### 4.4 Multi-turn escalation

Across 30 guarded conversations (10 sequences × 3 models), zero exhibited capitulation (a safe first-turn response becoming unsafe by the final turn) and zero produced an unsafe final-turn response. Under the vanilla condition, final-turn unsafe reassurance appeared in 5/10 Gemini conversations, 4/10 GPT-4o-mini conversations (2 of which were capitulations), and 1/10 Claude Haiku conversations.

## 5. Discussion

**Baseline safety is not a property of "AI chatbots" in general.** The 4.5× spread between Gemini 3.7 Flash (65.9%) and Claude Haiku 4.5 (14.6%) on identical prompts means a family's exposure to unsafe advice is substantially determined by which product they happen to open. This argues against treating "chatbot safety" as a single number and in favor of per-deployment evaluation.

**Prompt-level intervention is unusually effective here, which is itself informative.** A five-fold pooled reduction from a system prompt alone suggests these failures are not deeply entrenched capability limitations but rather default-behavior problems: the models *can* hedge appropriately and cite the right referral, they simply do not by default. The Gemini result is the cleanest evidence, with 47 prompts flipping to pass and none flipping the other way. The GPT-4o-mini result failing to reach significance (p = 0.077) is a genuine null at this sample size, not evidence of no effect, and is a direct consequence of its lower baseline leaving less headroom.

**The grounded condition is the paper's most useful negative result.** Retrieval with forced citation did precisely what it was designed to do on the dimensions it targeted: unsafe reassurance fell to 0.8% and fabricated policy claims to 1.6%, both better than the hardened prompt. But over-refusal rose from 6.1% to 50.4%, yielding a worse total failure rate than the far simpler intervention. The mechanism is the forced-citation instruction rather than retrieval quality: the model, correctly following its instruction not to exceed its provided sources, declines to supply general information it could safely give. For practitioners, the lesson is that safety interventions must be evaluated on an over-refusal axis alongside a harm axis, or they will optimize a real metric into a system that fails users in a different direction. A refusal is not a safe answer when the person asking has a deadline.

**Multi-turn robustness tracked single-turn performance rather than diverging from it.** We had hypothesized that a system prompt might produce shallow compliance that erodes under social pressure. It did not: the hardened prompt was perfectly robust across 30 escalation conversations. The capitulations we observed occurred only in the unguarded condition.

## 6. Limitations

- **The automated judge has not yet been validated against human raters.** All reported numbers are LLM-judge scores against a fixed rubric. The multi-rater validation tooling exists and computes both judge-human and human-human Cohen's kappa, but no labeling pass has been completed. Following Zheng et al. (2023), judge-only numbers should not be treated as human-verified ground truth until that agreement is measured. This is the most significant outstanding limitation.
- **Free-tier infrastructure proved unusable, which constrains reproducibility for unfunded replication.** An earlier iteration of this study targeted free open-weight endpoints. Across two evenings, free-tier requests returned HTTP 429 from a saturated shared upstream pool on essentially every call, yielding 62/64 single-turn and 2/40 multi-turn completions after hours of retrying; identical prompts on paid endpoints completed 492 calls in roughly five minutes with zero failures. Partial free-model data is retained and reported separately, never pooled with headline results. Replication of this work requires a modest but nonzero budget.
- Sample size is modest (82 prompts per condition per model), appropriate for detecting a large effect (the Gemini result) but underpowered for smaller ones, which is directly why the GPT-4o-mini comparison is inconclusive rather than resolved.
- The prompt bank is written, not sourced from real conversations with real families, since collecting real conversations on this topic would itself be a privacy and safety risk.
- U.S.-immigration-context specific; findings should not be assumed to generalize to other countries' analogous risks.
- Model coverage reflects what was practically testable within this project's time and budget, not an exhaustive survey of deployed chatbots.
- The rubric was validated against a small number of human raters, not a panel of immigration-law experts; it catches confidently-stated falsehoods, not every possible legal subtlety.

## 7. Ethics Statement

This project uses only synthetic, author-written prompts; no real individuals' conversations, personal information, or immigration status was collected or used. The prompt bank is a red-teaming artifact intended to measure and reduce a real harm (chatbots giving unsafe advice to a vulnerable population); it is published for that defensive purpose. The authors are aware that any adversarial-prompt dataset carries some dual-use risk (informing how to construct similar prompts) and judge this outweighed by the value of a public, checkable benchmark in a domain with no existing one.

## References

1. Ganguli, D., et al. (2022). *Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned.* Anthropic. https://www-cdn.anthropic.com/82564d4ec2451b2eed2e0796b7c658fc989f0c1a/Anthropic_RedTeaming.pdf
2. Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023. https://arxiv.org/abs/2306.05685
3. Lin, S., Hilton, J., & Evans, O. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* ACL 2022. https://arxiv.org/abs/2109.07958
4. Mazeika, M., et al. (2024). *HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal.* https://arxiv.org/abs/2402.04249
