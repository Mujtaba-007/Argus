# nlp/sentiment.py
# High-level sentiment analysis module.
# Supports single-item analysis and batched analysis (up to GEMINI_BATCH_SIZE posts per Gemini call).
# Enforces bounded concurrency (asyncio.Semaphore) and configurable rate limits
# (GEMINI_MAX_RPM calls per minute) across the process.
#
# Error handling policy:
#   - If a Gemini batch request fails (e.g. after 6 retries on 429), return None.
#     The caller will NOT write sentiment rows for that batch.
#   - If Gemini answered successfully but the JSON for an individual item was unparseable
#     or missing, confidence = 0.0 and emotion = 'unclassified' (polarity = 'neutral').

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Any, TypedDict

from app.config import settings
from app.nlp.gemini_client import generate_text_async

logger = logging.getLogger(__name__)


class SentimentResult(TypedDict):
    """The dict shape returned for classified posts."""
    polarity: str      # "support" | "neutral" | "against"
    emotion: str       # "support" | "anxiety" | "excitement" | "anger" | "sarcasm" | "neutral" | "unclassified"
    confidence: float  # 0.0 – 1.0


# Safe fallback used ONLY when Gemini responded successfully but an individual post item
# could not be parsed or classified.
_UNCLASSIFIED_ITEM: SentimentResult = {
    "polarity": "neutral",
    "emotion": "unclassified",
    "confidence": 0.0,
}

_VALID_POLARITIES = {"support", "neutral", "against"}
_VALID_EMOTIONS = {"support", "anxiety", "excitement", "anger", "sarcasm", "neutral", "unclassified"}


# ---------------------------------------------------------------------------
# Rate Limiter & Concurrency Gate
# ---------------------------------------------------------------------------

class _GeminiRateLimiter:
    """
    Enforces:
      1. Bounded concurrency via asyncio.Semaphore (settings.gemini_concurrency).
      2. Rate limiting via a sliding timestamp window (settings.gemini_max_rpm).
    """

    def __init__(self, max_rpm: int, concurrency: int) -> None:
        self.max_rpm = max(1, max_rpm)
        self.semaphore = asyncio.Semaphore(concurrency)
        self._lock = asyncio.Lock()
        self._timestamps: list[float] = []

    async def acquire(self) -> None:
        await self.semaphore.acquire()
        try:
            async with self._lock:
                now = time.monotonic()
                # Prune timestamps older than 60 seconds
                self._timestamps = [t for t in self._timestamps if now - t < 60.0]

                if len(self._timestamps) >= self.max_rpm:
                    wait_sec = 60.0 - (now - self._timestamps[0]) + 0.1
                    if wait_sec > 0:
                        logger.info(
                            "Gemini rate limit reached (%d RPM). Throttling for %.1fs...",
                            self.max_rpm, wait_sec,
                        )
                        await asyncio.sleep(wait_sec)
                    now = time.monotonic()
                    self._timestamps = [t for t in self._timestamps if now - t < 60.0]

                self._timestamps.append(now)
        except Exception:
            self.semaphore.release()
            raise

    def release(self) -> None:
        self.semaphore.release()

    @asynccontextmanager
    async def throttle(self):
        await self.acquire()
        try:
            yield
        finally:
            self.release()


_limiter = _GeminiRateLimiter(
    max_rpm=settings.gemini_max_rpm,
    concurrency=settings.gemini_concurrency,
)


# ---------------------------------------------------------------------------
# Prompt Templates & JSON Parsing
# ---------------------------------------------------------------------------

_SINGLE_PROMPT_TEMPLATE = """\
You are a sentiment classifier for social-media posts.
Classify the following text and reply with ONLY a valid JSON object — no markdown, no explanation.

JSON schema:
{{
  "polarity":   "support" | "neutral" | "against",
  "emotion":    "support" | "anxiety" | "excitement" | "anger" | "sarcasm" | "neutral",
  "confidence": <float 0.0–1.0>
}}

Text to classify:
\"\"\"
{text}
\"\"\"
"""

_BATCH_PROMPT_TEMPLATE = """\
You are a sentiment classifier for social-media posts.
Classify each of the following {count} posts and reply with ONLY a valid JSON array of {count} objects — no markdown, no explanation.

Each object must follow this schema:
{{
  "index": <integer corresponding to post index>,
  "polarity": "support" | "neutral" | "against",
  "emotion": "support" | "anxiety" | "excitement" | "anger" | "sarcasm" | "neutral",
  "confidence": <float between 0.0 and 1.0>
}}

Posts to classify:
{posts_formatted}
"""


def _extract_json(raw: str) -> Any:
    """Extract and parse JSON from raw model output, stripping markdown code fences."""
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    return json.loads(cleaned)


def _validate_and_sanitize(item: dict[str, Any]) -> SentimentResult:
    """Validate and sanitize polarity, emotion, and confidence for a single result."""
    polarity = str(item.get("polarity", "neutral")).lower().strip()
    emotion = str(item.get("emotion", "neutral")).lower().strip()

    try:
        confidence = float(item.get("confidence", 0.0))
    except (ValueError, TypeError):
        confidence = 0.0

    if polarity not in _VALID_POLARITIES:
        polarity = "neutral"
    if emotion not in _VALID_EMOTIONS:
        emotion = "neutral"
    confidence = max(0.0, min(1.0, confidence))

    return SentimentResult(polarity=polarity, emotion=emotion, confidence=confidence)


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------

async def analyze_batch_async(texts: list[str]) -> list[SentimentResult] | None:
    """
    Classify up to GEMINI_BATCH_SIZE posts in a single Gemini request.
    Respects concurrency limit (Semaphore) and rate limit (GEMINI_MAX_RPM).

    Returns:
        list[SentimentResult] of the exact same length as `texts` if Gemini succeeded.
        None if the Gemini request itself failed after retries.
        When individual items inside a successful Gemini response cannot be parsed,
        they are given emotion='unclassified' and confidence=0.0.
    """
    if not texts:
        return []

    # Format posts with index labels
    formatted_posts = "\n".join(
        f"[{i}]: \"{(t or '').strip()[:1000].replace(chr(10), ' ')}\""
        for i, t in enumerate(texts)
    )

    prompt = _BATCH_PROMPT_TEMPLATE.format(
        count=len(texts),
        posts_formatted=formatted_posts,
    )

    try:
        async with _limiter.throttle():
            raw_response = await generate_text_async(prompt)
    except Exception as exc:
        logger.warning(
            "Batch Gemini call failed for %d posts: %s. Batch will NOT be saved to sentiments.",
            len(texts), exc,
        )
        return None

    # Gemini answered successfully: prepare results defaulting to unclassified
    results: list[SentimentResult] = [_UNCLASSIFIED_ITEM.copy() for _ in texts]

    try:
        parsed = _extract_json(raw_response)

        # Handle models returning {"results": [...]} or similar wrapper dict
        if isinstance(parsed, dict):
            for val in parsed.values():
                if isinstance(val, list):
                    parsed = val
                    break

        if isinstance(parsed, list):
            for i, item in enumerate(parsed):
                if not isinstance(item, dict):
                    continue

                matched_idx: int | None = None
                idx_val = item.get("index")
                if idx_val is not None:
                    try:
                        int_val = int(idx_val)
                        if 0 <= int_val < len(texts):
                            matched_idx = int_val
                    except (ValueError, TypeError):
                        pass

                if matched_idx is None and 0 <= i < len(texts):
                    matched_idx = i

                if matched_idx is not None:
                    results[matched_idx] = _validate_and_sanitize(item)
        else:
            logger.warning("Gemini batch response was not an array: %.80s", raw_response)

    except Exception as exc:
        logger.warning(
            "Failed to parse Gemini batch response: %s | raw: %.100s. Marking batch items as unclassified.",
            exc, raw_response,
        )

    return results


async def analyze_single_async(text: str) -> SentimentResult | None:
    """Async single-text classification. Returns None if call fails."""
    if not text or not text.strip():
        return _UNCLASSIFIED_ITEM.copy()

    prompt = _SINGLE_PROMPT_TEMPLATE.format(text=text[:2000])

    try:
        async with _limiter.throttle():
            raw_response = await generate_text_async(prompt)
        parsed = _extract_json(raw_response)
        if isinstance(parsed, dict):
            return _validate_and_sanitize(parsed)
        return _UNCLASSIFIED_ITEM.copy()
    except Exception as exc:
        logger.warning("Single sentiment analysis failed: %s", exc)
        return None
