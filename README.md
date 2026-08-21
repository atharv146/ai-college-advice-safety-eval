# Do AI chatbots give unsafe college advice to immigrant families?

A red-teaming benchmark testing whether general-purpose AI chatbots give dangerous advice to immigrant and mixed-status families asking about college applications and financial aid, plus a measured comparison of two possible fixes.

**Headline finding: a safety-hardened system prompt cuts the failure rate from 32.9% to 6.5%. A retrieval-grounded version eliminates unsafe reassurance almost entirely but makes the model refuse to answer half the time, so it is worse overall, not better.** That tradeoff is the most useful thing in this repo.

📊 **Dataset:** [huggingface.co/datasets/atharv146/college-advice-ai-safety-eval-prompts](https://huggingface.co/datasets/atharv146/college-advice-ai-safety-eval-prompts)
📄 **Write-up:** [PAPER.md](PAPER.md)

## Why this exists

A nervous parent asks a chatbot "will filling out the FAFSA report my undocumented parent to immigration enforcement?" The honest answer involves real uncertainty, changing policy, and a referral to a licensed professional. A chatbot that answers "no, you're completely safe" is not being helpful, it is giving a family a false sense of security about a real risk.

This is not hypothetical. The motivating incident was a real production failure in a live college-guidance app for immigrant students, where the model volunteered false FERPA-based reassurance about enforcement risk. It was caught by hand, not by any automated check. This project turns that one incident into a systematic, repeatable measurement.

## What was tested

**82 prompts** across 8 risk categories, written and frozen before any model was run.
**3 models:** GPT-4o-mini, Gemini 3.7 Flash, Claude Haiku 4.5.
**3 conditions:** a plain assistant prompt (`vanilla`), a safety-hardened prompt (`guarded`), and a retrieval-grounded condition restricted to verified source material with forced citations (`grounded`).
**Plus 10 three-turn escalation conversations** per model per condition, where a user pushes back twice after the first answer.

Every response was graded by a held-out judge model (GPT-4o, deliberately not one of the models under test) against a rubric fixed in advance, on three independent failure dimensions: **unsafe reassurance**, **fabricated policy claim**, and **unhelpfully evasive**.

## Results

### Pooled across all three models (n = 246 per condition)

| Condition | Unsafe reassurance | Fabricated policy | Unhelpfully evasive | **Any failure** |
|---|---|---|---|---|
| vanilla | 20.7% | 11.8% | 6.1% | **32.9%** |
| guarded | 1.2% | 1.6% | 4.1% | **6.5%** |
| grounded | 0.8% | 1.6% | 50.4% | **52.4%** |

### Per model, baseline vs. hardened prompt

| Model | vanilla fail rate | guarded fail rate | McNemar p | Significant? |
|---|---|---|---|---|
| Gemini 3.7 Flash | 65.9% (55.1–75.2) | 8.5% (4.2–16.6) | < 0.0001 | ✅ |
| GPT-4o-mini | 18.3% (11.4–28.0) | 8.5% (4.2–16.6) | 0.077 | ❌ |
| Claude Haiku 4.5 | 14.6% (8.6–23.9) | 2.4% (0.7–8.5) | 0.013 | ✅ |

95% Wilson confidence intervals in parentheses. McNemar's exact test is used because both conditions run on the *same* 82 prompts, making this a paired comparison, not two independent samples.

### Three findings worth stating plainly

**1. Baseline safety varies enormously between models.** Gemini 3.7 Flash failed on 65.9% of prompts with no safety instructions; Claude Haiku failed on 14.6% of the identical prompts. A family's exposure to bad advice depends heavily on which chatbot they happen to open.

**2. A well-written system prompt does most of the work, but not reliably for every model.** The hardened prompt cut pooled failures by a factor of five. For Gemini the improvement was dramatic and unambiguous (47 of 82 prompts flipped from fail to pass, zero went the other way). For GPT-4o-mini the improvement did not reach significance (p = 0.077), meaning its already-lower baseline left less room and the change is not statistically distinguishable from noise at this sample size.

**3. The obvious fix backfires, and this is the most interesting result.** Restricting the model to verified retrieved sources with forced citations nearly eliminated unsafe reassurance (0.8%) and fabricated policy claims (1.6%), doing exactly what it was designed to do. But it drove the over-refusal rate from 6.1% to 50.4%: the model constantly answers "my available material doesn't cover this" instead of giving the real, safe, generally-useful information it could. Net failure rate is **worse** than the simple hardened prompt. Safety is not the only axis, and a system that refuses half of a scared family's questions has failed them too, just differently.

### Multi-turn pressure testing

The hardened prompt held up completely under escalation: across all 30 guarded conversations, **zero** capitulations and **zero** unsafe final-turn responses. Without it, models gave ground: on the vanilla prompt, Gemini's final turn contained unsafe reassurance in 5 of 10 conversations and GPT-4o-mini's in 4 of 10, with 2 of those being outright capitulations (a safe first answer that became unsafe after the user pushed back twice).

### Which questions are most dangerous

| Category | Failure rate |
|---|---|
| Enforcement reassurance | 39.6% |
| Mixed-status family aid | 38.0% |
| Indirect / roleplay framing | 36.6% |
| FERPA data sharing | 35.2% |
| Scholarship eligibility | 26.9% |
| In-state residency claims | 26.0% |
| DACA eligibility | 25.7% |
| Leading premise | 20.6% |

The highest-failure category is the one where being wrong is most dangerous: direct questions about immigration-enforcement risk.

![Failure rate by model](figures/failure_rate_by_model.png)
![Failure rate by category](figures/failure_rate_by_category.png)

## Reproduce it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your OpenRouter API key
python src/run_harness.py    # baseline + hardened conditions
python src/run_grounded.py   # retrieval-grounded condition
python src/run_multiturn.py  # escalation conversations
python src/judge.py          # automated grading
python src/analyze.py        # statistics + figures
```

## Known gaps, stated honestly

**The judge is not yet human-validated.** Every number above comes from an automated LLM judge. The tooling for human validation exists (`src/label_sample.py --rater yourname`, which supports multiple independent raters and computes Cohen's kappa both judge-vs-human and human-vs-human), but no human labeling pass has been run yet. Until it is, these results should be read as "what a strong LLM judge found using a fixed rubric," not as human-verified ground truth. This is the single most important outstanding item.

**Free-tier infrastructure was unusable, and that is itself a finding.** An earlier version of this study used only free open-weight models via OpenRouter. Over two evenings, free-tier requests returned HTTP 429 on essentially every call from a saturated shared upstream pool, producing 62/64 single-turn and 2/40 multi-turn completions after hours of retrying. The identical prompts on paid endpoints completed 492 calls in about five minutes with zero errors. Partial free-model data is retained in `results/` and reported separately above, never pooled into the headline numbers. Anyone attempting this kind of evaluation on free infrastructure should budget for this.

**Sample size.** 82 prompts per model per condition detects large effects reliably (the Gemini result is unambiguous) but is underpowered for small ones, which is exactly why the GPT-4o-mini comparison lands at p = 0.077 rather than resolving cleanly.

**Retrieval is deliberately simple.** TF-IDF cosine similarity over a small verified corpus, not embeddings. At this corpus size the retrieval quality is not the bottleneck; the over-refusal behavior is driven by the forced-citation instruction, not by retrieval misses.

**Scope.** U.S. immigration context only. Written prompts, not transcripts of real family conversations, since collecting those would itself be a privacy risk.

## Repo layout

```
data/prompts.json              82 adversarial prompts, 8 categories
data/escalation_prompts.json   10 three-turn pressure sequences
data/rubric.md                 grading rubric, fixed before any run
data/system_prompts.py         vanilla and guarded system prompts
data/corpus.json               verified source passages for the grounded condition
src/run_harness.py             vanilla + guarded conditions
src/run_grounded.py            retrieval-grounded condition
src/run_multiturn.py           escalation conversations
src/rag.py                     retrieval + forced-citation prompt
src/judge.py                   automated grading
src/label_sample.py            multi-rater human labeling CLI
src/analyze.py                 Wilson CIs, McNemar's test, kappa, figures
```

## License

Code MIT. Prompt bank and rubric CC-BY-4.0, see the [dataset card](https://huggingface.co/datasets/atharv146/college-advice-ai-safety-eval-prompts).
