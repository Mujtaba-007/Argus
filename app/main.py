# main.py
# FastAPI entry point for the Argus backend service.
#
# On startup: validates config (fails fast if any env var is missing),
# then launches a background asyncio loop that runs a full ingestion cycle
# every 5 minutes.
#
# HTTP endpoints:
#   GET  /health     → {"status": "ok"}   (liveness probe for Fly.io / Docker)
#   POST /run-cycle  → cycle summary dict  (trigger a cycle manually)

import asyncio
import logging
import logging.config
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI

# Import config first — if any env var is missing this raises immediately
# and the app refuses to start with a clear error message.
from app.config import settings
from app.db.repository import (
    get_active_trackers,
    get_posts_needing_sentiment,
    upsert_sentiments_batch,
)
from app.nlp.sentiment import analyze_batch_async
from app.pipeline.run_cycle import run_cycle

# ---------------------------------------------------------------------------
# Logging — structured, goes to stdout so Docker/Fly.io can capture it.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)

# How often (in seconds) the background loop runs a cycle automatically.
CYCLE_INTERVAL_SECONDS = 5 * 60  # 5 minutes


# ---------------------------------------------------------------------------
# Scheduled alert evaluation (runs every cycle, independent of new posts)
# ---------------------------------------------------------------------------

async def _run_scheduled_alerts() -> None:
    """
    Evaluate alert rules for every tracker that has at least one active rule.
    Runs on every background loop, respecting the 60-minute per-alert cooldown
    inside evaluate_alerts_for_tracker.
    """
    from app.alerts.evaluator import evaluate_alerts_for_tracker
    from app.db.supabase_client import supabase

    try:
        # Fetch distinct tracker_ids that have at least one active alert.
        # We select only tracker_id and use a small page size to avoid loading
        # the full alerts table if it grows large.
        page_size = 500
        start = 0
        tracker_ids: list[str] = []
        seen: set[str] = set()
        while True:
            res = (
                supabase.table("alerts")
                .select("tracker_id")
                .eq("active", True)
                .range(start, start + page_size - 1)
                .execute()
            )
            batch = res.data or []
            for row in batch:
                tid = row.get("tracker_id")
                if tid and tid not in seen:
                    seen.add(tid)
                    tracker_ids.append(tid)
            if len(batch) < page_size:
                break
            start += page_size

        if not tracker_ids:
            logger.debug("Scheduled alerts: no trackers with active rules.")
            return

        total_checked = 0
        total_triggered = 0
        for tid in tracker_ids:
            try:
                res = await asyncio.to_thread(evaluate_alerts_for_tracker, tid)
                total_checked += res["checked"]
                total_triggered += res["triggered"]
            except Exception as exc:
                logger.error("Scheduled alert eval failed for tracker %s: %s", tid, exc)

        logger.info(
            "Scheduled alerts done: trackers=%d, checked=%d, triggered=%d.",
            len(tracker_ids), total_checked, total_triggered,
        )
    except Exception as exc:
        logger.exception("Scheduled alert evaluation error: %s", exc)


# ---------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------

async def _background_loop() -> None:
    """
    Runs forever, triggering a full ingestion + alert-evaluation cycle every
    CYCLE_INTERVAL_SECONDS. Exceptions inside run_cycle are caught there; this
    loop only catches unexpected crashes so it never silently dies.
    """
    logger.info(
        "Background ingestion loop started. Cycle interval: %ds.",
        CYCLE_INTERVAL_SECONDS,
    )
    while True:
        try:
            await run_cycle()
        except Exception as exc:
            # Should not reach here (run_cycle handles its own errors), but
            # just in case — log and continue rather than crashing the loop.
            logger.exception("Unexpected error in background loop: %s", exc)

        # Evaluate alert rules for all trackers with active rules, regardless
        # of whether new posts arrived in this cycle.
        await _run_scheduled_alerts()

        logger.info("Background loop sleeping for %ds.", CYCLE_INTERVAL_SECONDS)
        await asyncio.sleep(CYCLE_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# FastAPI lifespan — starts the background loop on boot, cancels it on shutdown.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Argus backend starting up.")
    loop_task = asyncio.create_task(_background_loop())
    try:
        yield  # server is running; handle requests here
    finally:
        logger.info("Argus backend shutting down.")
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Argus Backend",
    description="Telegram ingestion + Gemini sentiment analysis service.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """
    Liveness probe. Returns 200 OK as long as the process is running.
    Used by Fly.io health checks and Docker HEALTHCHECK.
    """
    return {"status": "ok"}


@app.post("/run-cycle", tags=["ops"])
async def trigger_cycle() -> dict[str, Any]:
    """
    Manually trigger one full ingestion + sentiment + analytics cycle.
    Returns a summary of what happened.
    """
    logger.info("Manual /run-cycle triggered via HTTP.")
    result = await run_cycle()
    return result


@app.post("/run-analytics", tags=["ops"])
async def trigger_analytics(tracker_id: str | None = None) -> dict[str, str]:
    """
    Kick off trend and network analytics in the background and return immediately.
    If tracker_id is provided, runs for that tracker only.
    If omitted, runs across all active Telegram trackers.
    Completion (and per-tracker results) are logged server-side.
    """
    logger.info("POST /run-analytics triggered for tracker_id=%s — starting background task.", tracker_id)

    async def _run_all() -> None:
        from app.analytics.trends import analyze_trends
        targets: list[str]
        if tracker_id:
            targets = [tracker_id]
        else:
            try:
                trackers = await asyncio.to_thread(get_active_trackers)
                targets = [t["id"] for t in trackers]
            except Exception as exc:
                logger.error("/run-analytics background: failed to fetch trackers — %s", exc)
                return
        for tid in targets:
            try:
                result = await analyze_trends(tid)
                logger.info("/run-analytics [%s] done: %s", tid, result)
            except Exception as exc:
                logger.error("/run-analytics [%s] failed: %s", tid, exc)

    asyncio.create_task(_run_all())
    return {"status": "started"}


@app.post("/run-alerts", tags=["ops"])
async def trigger_alerts(tracker_id: str | None = None) -> dict[str, Any]:
    """
    Manually evaluate alert rules.
    If tracker_id is provided, runs for that tracker.
    If omitted, runs across all active Telegram trackers.
    Returns checked and triggered counts.
    """
    logger.info("POST /run-alerts triggered for tracker_id=%s.", tracker_id)
    from app.alerts.evaluator import evaluate_alerts_for_tracker

    if tracker_id:
        res = await asyncio.to_thread(evaluate_alerts_for_tracker, tracker_id)
        return {"checked": res["checked"], "triggered": res["triggered"], "trackers": [tracker_id]}

    trackers = await asyncio.to_thread(get_active_trackers)
    total_checked = 0
    total_triggered = 0
    for t in trackers:
        res = await asyncio.to_thread(evaluate_alerts_for_tracker, t["id"])
        total_checked += res["checked"]
        total_triggered += res["triggered"]

    return {
        "checked": total_checked,
        "triggered": total_triggered,
        "trackers_count": len(trackers),
    }


@app.post("/run-demographics", tags=["ops"])
async def trigger_demographics(tracker_id: str) -> dict[str, Any]:
    """
    Profile authors and aggregate demographics for a tracker.
    Results are aggregated and anonymized — individuals are never exposed.
    Buckets below MIN_GROUP_SIZE are merged for privacy.
    """
    logger.info("POST /run-demographics triggered for tracker_id=%s.", tracker_id)
    from app.analytics.demographics import analyze_demographics
    result = await asyncio.to_thread(analyze_demographics, tracker_id)
    return result


@app.post("/reanalyze", tags=["ops"])
async def reanalyze_sentiments(
    tracker_id: str,
    limit: int = 200,
) -> dict[str, Any]:
    """
    Find posts with no sentiment row, or whose sentiment has confidence = 0
    (emotion 'neutral' or 'unclassified'), and re-run sentiment analysis in rate-limited batches.
    Returns counts: processed, succeeded, failed.
    """
    logger.info("POST /reanalyze triggered for tracker_id=%s, limit=%d", tracker_id, limit)
    posts = await asyncio.to_thread(get_posts_needing_sentiment, tracker_id, limit)

    total_posts = len(posts)
    if total_posts == 0:
        return {
            "tracker_id": tracker_id,
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "message": "No posts needing sentiment re-analysis found.",
        }

    batch_size = settings.gemini_batch_size
    total_batches = (total_posts + batch_size - 1) // batch_size
    succeeded_count = 0
    failed_count = 0

    batch_num = 0
    for i in range(0, total_posts, batch_size):
        batch_num += 1
        batch = posts[i : i + batch_size]
        texts = [p.get("body", "") for p in batch]

        sentiments = await analyze_batch_async(texts)

        if sentiments is not None:
            sentiment_rows = [
                {
                    "post_id": post_item["id"],
                    "tracker_id": tracker_id,
                    "polarity": sent["polarity"],
                    "emotion": sent["emotion"],
                    "confidence": sent["confidence"],
                }
                for post_item, sent in zip(batch, sentiments)
            ]
            try:
                await asyncio.to_thread(upsert_sentiments_batch, sentiment_rows)
                succeeded_count += len(sentiment_rows)
                logger.info("Reanalyze tracker %s: batch %d/%d ok.", tracker_id, batch_num, total_batches)
            except Exception as exc:
                logger.error("Reanalyze tracker %s: upsert_sentiments_batch failed — %s", tracker_id, exc)
                failed_count += len(batch)
        else:
            failed_count += len(batch)
            logger.warning(
                "Reanalyze tracker %s: batch %d/%d failed with Gemini. %d posts skipped.",
                tracker_id, batch_num, total_batches, len(batch),
            )

    return {
        "tracker_id": tracker_id,
        "processed": total_posts,
        "succeeded": succeeded_count,
        "failed": failed_count,
    }

