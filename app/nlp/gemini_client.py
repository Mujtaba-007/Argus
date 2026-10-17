# nlp/gemini_client.py
# Low-level wrapper around the Google Gemini API.
# Keeps all HTTP/SDK details here so the rest of the app never imports
# google.generativeai directly.
#
# Key behaviours:
#   - Model name is driven by settings.gemini_model (default: gemini-2.0-flash-lite)
#   - On 429 RESOURCE_EXHAUSTED, reads the retry-after header (if present)
#     and otherwise uses exponential back-off: 2s → 4s → 8s → 16s → 32s → 60s
#   - Up to 6 retry attempts before re-raising (caller decides whether to skip)
#   - Concurrency and per-minute caps are enforced in nlp/sentiment.py

import asyncio
import logging
import time
from typing import Any

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from app.config import settings

logger = logging.getLogger(__name__)

# Configure SDK once at import time (key stays out of logs).
genai.configure(api_key=settings.gemini_api_key)

# Lazily built so that if settings.gemini_model is overridden at runtime
# the module still picks it up on first call.
_model: genai.GenerativeModel | None = None


def _get_model() -> genai.GenerativeModel:
    global _model
    if _model is None:
        _model = genai.GenerativeModel(settings.gemini_model)
        logger.info("Gemini model initialised: %s", settings.gemini_model)
    return _model


# ---------------------------------------------------------------------------
# 429 / ResourceExhausted retry helper
# ---------------------------------------------------------------------------

_MAX_RETRIES = 6
_BASE_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 60.0

# Global state for Gemini quota handling
from datetime import datetime, timedelta
_consecutive_429: int = 0
_cooldown_until: datetime | None = None

def _set_cooldown() -> None:
    """Activate a 30‑minute cooldown after quota exhaustion.

    Sets a global timestamp until which Gemini calls are skipped and logs a warning.
    """
    global _cooldown_until, _consecutive_429
    _cooldown_until = datetime.utcnow() + timedelta(minutes=30)
    _consecutive_429 = 0
    logger.warning("Gemini daily quota exhausted – cooling down until %s", _cooldown_until.isoformat())

def _reset_quota_counters() -> None:
    """Reset the consecutive 429 counter after a successful call."""
    global _consecutive_429
    _consecutive_429 = 0

def _is_cooldown_active() -> bool:
    """Return True if currently in cooldown period."""
    if _cooldown_until is None:
        return False
    return datetime.utcnow() < _cooldown_until



def _parse_retry_delay(exc: Exception) -> float | None:
    """
    Try to read the 'retry_delay' or 'Retry-After' value from a 429 response.
    Returns seconds as a float, or None if not found.
    """
    try:
        # google-api-core surfaces this in the exception details list
        for detail in getattr(exc, "details", []) or []:
            if hasattr(detail, "retry_delay"):
                rd = detail.retry_delay
                # retry_delay may be a protobuf Duration
                return float(getattr(rd, "seconds", 0)) + getattr(rd, "nanos", 0) / 1e9
    except Exception:
        pass
    return None


def generate_text(prompt: str) -> str:
    """
    Send a prompt to Gemini and return the raw text response.

    Handles quota cooldown: if a cooldown is active, skips the call and raises ResourceExhausted.
    Retries up to _MAX_RETRIES times on 429 RESOURCE_EXHAUSTED using
    exponential back-off (2 → 4 → 8 → 16 → 32 → 60 s), honouring the
    server's retry-after hint when present.

    Raises the last exception after all retries are exhausted so the caller
    (sentiment.py) can decide whether to skip or fall back.
    """
    # If we are in a cooldown period, abort early.
    if _is_cooldown_active():
        logger.warning("Gemini call blocked due to active cooldown until %s", _cooldown_until.isoformat() if _cooldown_until else "unknown")
        raise ResourceExhausted("Gemini quota exhausted, cooldown active")

    model = _get_model()
    last_exc: Exception | None = None
    backoff = _BASE_BACKOFF_S

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = model.generate_content(prompt)
            # Successful call – reset quota counters
            _reset_quota_counters()
            return response.text

        except ResourceExhausted as exc:
            last_exc = exc
            # Increment consecutive 429 counter
            global _consecutive_429
            _consecutive_429 += 1
            if _consecutive_429 >= 2:
                _set_cooldown()
                # After setting cooldown we abort further retries
                raise exc
            delay = _parse_retry_delay(exc) or backoff
            delay = min(delay, _MAX_BACKOFF_S)
            logger.warning(
                "Gemini 429 RESOURCE_EXHAUSTED (attempt %d/%d). Retrying in %.1fs...",
                attempt, _MAX_RETRIES, delay,
            )
            time.sleep(delay)
            backoff = min(backoff * 2, _MAX_BACKOFF_S)

        except Exception as exc:
            last_exc = exc
            # Non-rate-limit errors also get a short exponential retry
            logger.warning(
                "Gemini transient error (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt, _MAX_RETRIES, type(exc).__name__, backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF_S)

    logger.error(
        "Gemini call failed after %d attempts. Last error: %s",
        _MAX_RETRIES, type(last_exc).__name__,
    )
    raise last_exc  # type: ignore[misc]


async def generate_text_async(prompt: str) -> str:
    """Async wrapper — runs generate_text in a thread pool to avoid blocking."""
    return await asyncio.to_thread(generate_text, prompt)
