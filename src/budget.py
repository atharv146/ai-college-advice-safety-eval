"""
Hard spend guard for OpenRouter.

Added 2026-08-22 after the project ran down its credit balance faster than
expected (a token-budget bug forced a full regeneration of one model's
data, and a frontier judge model was pricier per call than planned). Any
script that spends money should call `require_budget()` before starting
and `checkpoint()` periodically, so an overrun aborts cleanly with data on
disk rather than silently draining the balance mid-run.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

CREDITS_ENDPOINT = "https://openrouter.ai/api/v1/credits"


class BudgetExceeded(RuntimeError):
    pass


def remaining() -> float:
    """Remaining USD on the account. Raises on API failure rather than
    guessing, since guessing here spends real money."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    resp = requests.get(CREDITS_ENDPOINT, headers={"Authorization": f"Bearer {key}"}, timeout=30)
    resp.raise_for_status()
    d = resp.json()["data"]
    return float(d["total_credits"]) - float(d["total_usage"])


def require_budget(need: float, reserve: float = 0.10) -> float:
    """Abort before starting if the run can't afford itself.

    `need` is the estimated cost of the run; `reserve` is a floor kept
    untouched so a later cheap-but-essential step (re-judging, a retry)
    is never blocked by this run having consumed everything.
    """
    have = remaining()
    if have < need + reserve:
        raise BudgetExceeded(
            f"Need ~${need:.2f} + ${reserve:.2f} reserve, but only ${have:.2f} remains. "
            "Reduce scope (fewer prompts or models) or add credit."
        )
    print(f"[budget] ${have:.3f} available, run estimated at ~${need:.2f}, reserve ${reserve:.2f}")
    return have


def checkpoint(floor: float = 0.10, label: str = "") -> float:
    """Call periodically mid-run. Raises if the balance has fallen to the
    floor, so the caller stops with partial results already written."""
    have = remaining()
    if have < floor:
        raise BudgetExceeded(
            f"Balance ${have:.3f} hit the ${floor:.2f} floor{' at ' + label if label else ''}. "
            "Stopping; partial results are on disk and runs are resumable."
        )
    return have
