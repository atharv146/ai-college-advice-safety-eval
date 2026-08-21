"""
The models under test, all reached through one OpenRouter API key so this
project needs exactly one credential, not four separate provider signups.

WHY THESE THREE: they are the cheap/fast tiers of the three model families
people actually talk to (OpenAI, Google, Anthropic). An earlier version of
this file tested only free-tier open-weight models (gpt-oss-20b, glm-5.2,
gemma-4-31b). Those were dropped as PRIMARY subjects for two reasons:

  1. Relevance. A benchmark about what nervous families are actually told
     by a chatbot should test the chatbots they actually use.
  2. Feasibility. Verified 2026-08-19/20: OpenRouter's free tier routes
     through a shared upstream pool that was saturated for hours at a
     stretch, returning HTTP 429 on essentially every request. Two full
     evenings of retrying produced 62/64 single-turn calls and 2/40
     multi-turn conversations. The same prompts on the paid tier return in
     1-2 seconds. That is a real, documented methods-section finding about
     doing this kind of research on free infrastructure, not just an
     inconvenience.

The free-model rows collected during that period are kept in
results/raw_responses.jsonl (they are real data) and are reported
separately in the README rather than silently mixed into the headline
numbers.
"""

MODELS_UNDER_TEST = [
    "openai/gpt-4o-mini",
    "google/gemini-3.7-flash",
    "anthropic/claude-haiku-4.5",
]

# Free-tier models from the earlier runs. Kept as a named list so the
# partial data collected from them stays interpretable, and so a future
# run can include them again if the free pool is ever usable.
LEGACY_FREE_MODELS = [
    "openai/gpt-oss-20b:free",
    "z-ai/glm-5.2:free",
]

# The automated judge. Deliberately NOT a member of MODELS_UNDER_TEST:
# grading with a model that is itself under test (or a close sibling of
# one) is a known source of self-enhancement bias in LLM-as-judge setups
# (Zheng et al. 2023). GPT-4o is a step up in capability from every model
# it grades here, which is the standard arrangement for this design.
JUDGE_MODEL = "openai/gpt-4o"
