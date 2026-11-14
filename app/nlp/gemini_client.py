# nlp/gemini_client.py
# Low-level wrapper around the Google Gemini API.
# Keeps all HTTP/SDK details here so the rest of the app never imports
# google.generativeai directly.
#
# Key behaviours:
#   - Model name is driven by settings.gemini_model (default: gemini-2.0-flash-lite)
#   - Explicit generation config: JSON output mode + max_output_tokens, so batch
#     responses are always valid JSON and never silently cut by an unknown default.
#   - ONE shared, thread-safe request budget (settings.gemini_max_rpm) for every
#     caller in the process (sentiment, demographics, ...). Low-priority callers
#     (demographics) may only use half of the window, so they can never starve
#     sentiment analysis.
#   - On 429 RESOURCE_EXHAUSTED the quota payload is parsed (quota id / metric /
#     value / retry delay) and logged in full once per distinct error, and the
#     cooldown length is taken from Google's retry hint when one is present.
#   - Up to 6 retry attempts on other transient errors before re-raising.

import asyncio
import logging
import re
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from app.config import settings

logger = logging.getLogger(__name__)

# Configure SDK once at import time (key stays out of logs).
genai.configure(api_key=settings.gemini_api_key)

# Both callers (sentiment + demographics) expect a JSON array back.
# JSON mode makes Gemini escape control characters and close every bracket.
# 4096 output tokens is ~5x what a 20-item sentiment batch needs (~600-800 tokens).
GEMINI_MAX_OUTPUT_TOKENS = 4096
_GENERATION_CONFIG = genai.GenerationConfig(
    max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
    response_mime_type="application/json",
)

# Lazily built so that if settings.gemini_model is overridden at runtime
# the module still picks it up on first call.
_model: genai.GenerativeModel | None = None


def _get_model() -> genai.GenerativeModel:
    global _model
    if _model is None:
        _model = genai.GenerativeModel(settings.gemini_model, generation_config=_GENERATION_CONFIG)
        logger.info(
            "Gemini model initialised: %s (max_output_tokens=%d, JSON mode).",
            settings.gemini_model, GEMINI_MAX_OUTPUT_TOKENS,
        )
    return _model


# ---------------------------------------------------------------------------
# Shared request budget (one per process, used by every caller)
# ---------------------------------------------------------------------------

_LOW_PRIORITY_MAX_SHARE = 0.5  # low-priority callers may use at most half the RPM window

# Time bounds. generate_text runs in a worker thread, and asyncio.wait_for (the
# 600s cycle_hard_timeout_seconds) cannot cancel a thread, so every wait inside
# generate_text must be bounded here.
GEMINI_BUDGET_MAX_WAIT_S = 90.0    # max time one call may wait for a shared-budget slot
GEMINI_CALL_DEADLINE_S = 240.0     # max total time for one generate_text call (waits + retries)
GEMINI_REQUEST_TIMEOUT_S = 60.0    # max time for a single HTTP request to Gemini


class GeminiBudgetTimeout(RuntimeError):
    """Raised when no shared-budget slot (or call deadline) is available in time."""


class _SharedGeminiBudget:
    """Thread-safe sliding 60s window over ALL Gemini requests in this process.

    generate_text() always runs in a worker thread (directly from demographics,
    or via asyncio.to_thread from sentiment), so a threading.Lock + time.sleep
    here never blocks the asyncio event loop.
    """

    def __init__(self, max_rpm: int) -> None:
        self.max_rpm = max(1, max_rpm)
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()

    def _cap(self, priority: str) -> int:
        if priority == "low":
            return max(1, int(self.max_rpm * _LOW_PRIORITY_MAX_SHARE))
        return self.max_rpm

    def acquire(self, priority: str = "high", deadline: float | None = None) -> bool:
        """Block until a request slot is free.

        Returns False if a cooldown starts while waiting.
        Raises GeminiBudgetTimeout if `deadline` (time.monotonic value) passes first.
        """
        cap = self._cap(priority)
        if deadline is None:
            deadline = time.monotonic() + GEMINI_BUDGET_MAX_WAIT_S
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 60.0:
                    self._timestamps.popleft()
                used = len(self._timestamps)
                if used < cap:
                    self._timestamps.append(now)
                    return True
                # Wait until enough old requests leave the window to get under this cap
                oldest_needed = self._timestamps[used - cap]
                wait_s = 60.0 - (now - oldest_needed) + 0.05
            if _is_cooldown_active():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GeminiBudgetTimeout(
                    f"No Gemini budget slot within wait limit (priority={priority}, {used}/{cap} used)"
                )
            logger.debug(
                "Gemini shared budget full (%d/%d used, priority=%s). Waiting %.1fs.",
                used, cap, priority, wait_s,
            )
            time.sleep(max(0.05, min(wait_s, 5.0, remaining)))


_budget = _SharedGeminiBudget(settings.gemini_max_rpm)


# ---------------------------------------------------------------------------
# 429 / quota state (thread-safe)
# ---------------------------------------------------------------------------

_MAX_RETRIES = 6
_BASE_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 60.0
_FALLBACK_COOLDOWN = timedelta(minutes=30)
_MAX_COOLDOWN = timedelta(hours=24)

_state_lock = threading.Lock()
_consecutive_429: int = 0
_cooldown_until: datetime | None = None
_last_quota_log: tuple[str, float] | None = None  # (signature, monotonic time)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _is_cooldown_active() -> bool:
    """Return True if currently in cooldown period."""
    until = _cooldown_until
    return until is not None and _now() < until


def _reset_quota_counters() -> None:
    """Reset the consecutive 429 counter after a successful call."""
    global _consecutive_429
    with _state_lock:
        _consecutive_429 = 0


def _parse_duration_s(value: Any) -> float | None:
    """Parse '34s', '34.5s', a protobuf Duration, or a number into seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.match(r"\s*([\d.]+)\s*s?\s*$", value)
        return float(m.group(1)) if m else None
    secs = getattr(value, "seconds", None)
    if secs is not None:
        return float(secs) + getattr(value, "nanos", 0) / 1e9
    return None


def _describe_quota_error(exc: Exception) -> dict[str, Any]:
    """Extract which quota was hit from a ResourceExhausted error.

    Handles both protobuf details (gRPC transport) and dict details (REST
    transport), plus a regex fallback over the error text.
    """
    info: dict[str, Any] = {
        "quota_ids": [], "quota_metrics": [], "quota_values": [],
        "retry_delay_s": None, "kind": "unknown",
        "raw": str(exc)[:1500],
    }

    for d in getattr(exc, "details", None) or []:
        if isinstance(d, dict):
            dtype = str(d.get("@type", ""))
            if dtype.endswith("QuotaFailure"):
                for v in d.get("violations", []) or []:
                    info["quota_ids"].append(v.get("quotaId"))
                    info["quota_metrics"].append(v.get("quotaMetric"))
                    info["quota_values"].append(v.get("quotaValue"))
            elif dtype.endswith("RetryInfo"):
                info["retry_delay_s"] = _parse_duration_s(d.get("retryDelay"))
        else:
            for v in getattr(d, "violations", []) or []:
                info["quota_ids"].append(getattr(v, "quota_id", None))
                info["quota_metrics"].append(getattr(v, "quota_metric", None))
                info["quota_values"].append(getattr(v, "quota_value", None))
            if hasattr(d, "retry_delay"):
                info["retry_delay_s"] = _parse_duration_s(d.retry_delay)

    text = info["raw"]
    if not info["quota_ids"]:
        info["quota_ids"] = re.findall(r'quota_?[iI]d"?:\s*"?([\w-]+)', text)
    if not info["quota_metrics"]:
        info["quota_metrics"] = re.findall(r'quota_?[mM]etric"?:\s*"?([\w./-]+)', text)
    if not info["quota_values"]:
        info["quota_values"] = re.findall(r'quota_?[vV]alue"?:\s*"?(\d+)', text)
    if info["retry_delay_s"] is None:
        m = (
            re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", text)
            or re.search(r'retryDelay"?:\s*"?([\d.]+)s', text)
            or re.search(r"retry in ([\d.]+)\s*s", text, re.IGNORECASE)
        )
        if m:
            info["retry_delay_s"] = float(m.group(1))

    ids = " ".join(str(x) for x in info["quota_ids"] + info["quota_metrics"] if x).lower()
    if "perday" in ids or "per_day" in ids:
        info["kind"] = "RPD (requests per day)"
    elif "token" in ids:
        info["kind"] = "TPM (tokens per minute)"
    elif "perminute" in ids or "per_minute" in ids or "requests" in ids:
        info["kind"] = "RPM (requests per minute)"
    return info


def _log_quota_error_once(info: dict[str, Any]) -> None:
    """Log the full quota payload once per distinct error (or every 10 min if it repeats)."""
    global _last_quota_log
    signature = f"{info['kind']}|{info['quota_ids']}|{info['quota_values']}"
    now = time.monotonic()
    with _state_lock:
        last = _last_quota_log
        if last and last[0] == signature and now - last[1] < 600:
            repeat = True
        else:
            _last_quota_log = (signature, now)
            repeat = False
    if repeat:
        logger.debug("Gemini 429 repeat (%s).", info["kind"])
        return
    logger.warning(
        "Gemini quota error: kind=%s quota_ids=%s quota_metrics=%s quota_values=%s retry_delay_s=%s raw=%s",
        info["kind"], info["quota_ids"], info["quota_metrics"], info["quota_values"],
        info["retry_delay_s"], info["raw"],
    )


def _set_cooldown(info: dict[str, Any]) -> None:
    """Start a cooldown. Uses Google's retry hint when present, else the 30 min fallback.

    For per-day quotas the cooldown is never shorter than the fallback, because a
    short retry hint on a daily quota would just produce more failing calls.
    """
    global _cooldown_until, _consecutive_429
    hint = info.get("retry_delay_s")
    if hint:
        duration = timedelta(seconds=max(10.0, float(hint)))
        source = "retry hint"
        if info.get("kind", "").startswith("RPD"):
            duration = max(duration, _FALLBACK_COOLDOWN)
            source = "retry hint (raised to fallback for daily quota)"
    else:
        duration = _FALLBACK_COOLDOWN
        source = "fixed fallback"
    duration = min(duration, _MAX_COOLDOWN)
    with _state_lock:
        _cooldown_until = _now() + duration
        _consecutive_429 = 0
        until = _cooldown_until
    logger.warning(
        "Gemini cooldown started: %s, %.0fs (%s), until %s.",
        info.get("kind", "unknown"), duration.total_seconds(), source, until.isoformat(),
    )


def gemini_cooldown_until() -> datetime | None:
    """Return the end of the active cooldown (UTC), or None if no cooldown is active."""
    return _cooldown_until if _is_cooldown_active() else None


def gemini_cooldown_minutes_remaining() -> float:
    """Return minutes remaining in the active Gemini cooldown, or 0.0 if none."""
    until = gemini_cooldown_until()
    if until is None:
        return 0.0
    return max(0.0, (until - _now()).total_seconds() / 60.0)


# ---------------------------------------------------------------------------
# Public call
# ---------------------------------------------------------------------------

def generate_text(prompt: str, priority: str = "high") -> str:
    """
    Send a prompt to Gemini and return the raw text response.

    priority: "high" (sentiment) may use the whole shared RPM budget;
              "low" (demographics) may use at most half of it.

    Raises ResourceExhausted immediately (no API call) while a cooldown is active.
    Raises GeminiBudgetTimeout if the call cannot finish within GEMINI_CALL_DEADLINE_S
    (budget waits + retry sleeps + requests), so a worker thread can never hang.
    """
    global _consecutive_429
    if _is_cooldown_active():
        raise ResourceExhausted("Gemini quota exhausted, cooldown active")

    model = _get_model()
    last_exc: Exception | None = None
    backoff = _BASE_BACKOFF_S
    call_deadline = time.monotonic() + GEMINI_CALL_DEADLINE_S

    def _remaining() -> float:
        return call_deadline - time.monotonic()

    for attempt in range(1, _MAX_RETRIES + 1):
        if _remaining() <= 0:
            raise GeminiBudgetTimeout(
                f"Gemini call deadline ({GEMINI_CALL_DEADLINE_S:.0f}s) reached after {attempt - 1} attempts"
            )

        # Every attempt (including retries) consumes a slot of the shared budget.
        wait_deadline = min(time.monotonic() + GEMINI_BUDGET_MAX_WAIT_S, call_deadline)
        if not _budget.acquire(priority, deadline=wait_deadline):
            raise ResourceExhausted("Gemini quota exhausted, cooldown active")

        try:
            request_timeout = min(GEMINI_REQUEST_TIMEOUT_S, max(5.0, _remaining()))
            response = model.generate_content(prompt, request_options={"timeout": request_timeout})
            _reset_quota_counters()
            try:
                finish = response.candidates[0].finish_reason
                finish_name = getattr(finish, "name", str(finish))
                if finish_name == "MAX_TOKENS":
                    logger.warning(
                        "Gemini response hit max_output_tokens=%d (truncated output).",
                        GEMINI_MAX_OUTPUT_TOKENS,
                    )
            except Exception:
                pass
            return response.text

        except ResourceExhausted as exc:
            last_exc = exc
            info = _describe_quota_error(exc)
            _log_quota_error_once(info)
            with _state_lock:
                _consecutive_429 += 1
                count = _consecutive_429
            # Daily quota: retrying is pointless. Otherwise allow one retry first.
            if count >= 2 or info["kind"].startswith("RPD"):
                _set_cooldown(info)
                raise
            delay = min(info["retry_delay_s"] or backoff, _MAX_BACKOFF_S, max(0.0, _remaining()))
            logger.warning(
                "Gemini 429 RESOURCE_EXHAUSTED (%s, attempt %d/%d). Retrying in %.1fs...",
                info["kind"], attempt, _MAX_RETRIES, delay,
            )
            time.sleep(delay)
            backoff = min(backoff * 2, _MAX_BACKOFF_S)

        except Exception as exc:
            last_exc = exc
            # Non-rate-limit errors also get a short exponential retry
            delay = min(backoff, max(0.0, _remaining()))
            logger.warning(
                "Gemini transient error (attempt %d/%d): %s. Retrying in %.1fs...",
                attempt, _MAX_RETRIES, type(exc).__name__, delay,
            )
            time.sleep(delay)
            backoff = min(backoff * 2, _MAX_BACKOFF_S)

    logger.error(
        "Gemini call failed after %d attempts. Last error: %s",
        _MAX_RETRIES, type(last_exc).__name__,
    )
    raise last_exc  # type: ignore[misc]


async def generate_text_async(prompt: str, priority: str = "high") -> str:
    """Async wrapper — runs generate_text in a thread pool to avoid blocking."""
    return await asyncio.to_thread(generate_text, prompt, priority)
