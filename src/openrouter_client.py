"""
Thin OpenRouter client. Same shape as PathFinder's production
`lib/ai/openrouter.ts`, ported to Python: one model per call, retry on
transient HTTP statuses, no retry on a hard failure.
"""

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE = {408, 429, 500, 502, 503, 504}


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy it into a .env file in this "
            "project's root (see .env.example) and try again."
        )
    return key


def _retry_after_seconds(resp: requests.Response, default: float) -> float:
    """OpenRouter's free-tier 429s come from a shared upstream pool (many
    OpenRouter users competing for the same free model), not a per-account
    quota, and the response tells you exactly how long to wait. Respecting
    that beats guessing at a backoff schedule."""
    header_val = resp.headers.get("Retry-After")
    if header_val:
        try:
            return float(header_val)
        except ValueError:
            pass
    try:
        body = resp.json()
        val = body.get("error", {}).get("metadata", {}).get("retry_after_seconds")
        if val is not None:
            return float(val)
    except Exception:  # noqa: BLE001
        pass
    return default


def complete_messages(model: str, messages: list[dict], max_retries: int = 8) -> str:
    """Core call: takes a full OpenAI-style messages list (system + any
    number of alternating user/assistant turns), so multi-turn
    conversations can be built up one call at a time by appending each
    response before the next call. Returns the model's text response."""
    key = _api_key()

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                ENDPOINT,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/atharv146/ai-college-advice-safety-eval",
                    "X-Title": "AI College Advice Safety Eval",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.4,
                    # 2500, not 800. Reasoning-capable models (Gemini 3.7
                    # Flash confirmed live 2026-08-22) burn 700-800+ tokens
                    # on MANDATORY internal reasoning that counts against
                    # this budget before a single visible word is written,
                    # and that reasoning cannot be disabled for this
                    # endpoint (a live 400 confirmed that). At 800 total,
                    # Gemini had ~45 words left for its actual answer and
                    # got cut off mid-sentence on 64% of calls -- silently,
                    # because truncated-but-nonempty text passed the old
                    # "is it empty" check. 2500 was verified live to leave
                    # ~450+ words of real answer with finish_reason=stop.
                    "max_tokens": 2500,
                },
                timeout=90,
            )
            if resp.status_code != 200:
                if resp.status_code in RETRYABLE:
                    wait = _retry_after_seconds(resp, default=8 * (attempt + 1))
                    last_error = RuntimeError(f"{model}: HTTP {resp.status_code}, waited {wait}s")
                    time.sleep(wait + 1)  # +1 buffer past what the provider asked for
                    continue
                raise RuntimeError(f"{model}: HTTP {resp.status_code}: {resp.text[:300]}")

            data = resp.json()
            choice = data.get("choices", [{}])[0]
            # .get("content", "") only falls back to "" when the key is
            # MISSING -- some providers return "content": null (a filtered
            # response) with the key present, which crashed .strip() on
            # None here until this fix. `or ""` catches both cases.
            raw_content = choice.get("message", {}).get("content")
            text = (raw_content or "").strip()
            if not text:
                last_error = RuntimeError(f"{model}: empty completion")
                time.sleep(2 * (attempt + 1))
                continue
            # A response that got cut off mid-generation is worse than no
            # response: it silently corrupts whatever grades it later. Treat
            # finish_reason == "length" as a hard failure, not a retry (more
            # tokens already given above; retrying identically won't fix a
            # provider that needs a token budget past what was requested).
            if choice.get("finish_reason") == "length":
                raise RuntimeError(
                    f"{model}: truncated at max_tokens (finish_reason=length), "
                    f"{len(text.split())} words returned"
                )
            return text
        except requests.RequestException as e:
            last_error = e
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"{model}: failed after {max_retries} attempts: {last_error}")


def complete(model: str, system: str, user: str, max_retries: int = 8) -> str:
    """Single-turn convenience wrapper around complete_messages."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return complete_messages(model, messages, max_retries=max_retries)
