# pipeline/run_cycle.py
# Orchestrates ingestion, batched sentiment, and triggers analytics per tracker.
# Uses asyncio.Lock to prevent overlapping cycles.

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from app.analytics.scheduler import run_analytics
from app.config import settings
from app.db.repository import (
    get_active_trackers,
    get_existing_external_ids,
    get_post_authors_by_external_ids,
    insert_interactions_batch,
    insert_posts_batch,
    insert_sentiments_batch,
    update_tracker_last_sync,
)
from app.ingestion.telegram_worker import fetch_new_messages
from app.ingestion.x_worker import fetch_new_posts
from app.nlp.sentiment import analyze_batch_async

logger = logging.getLogger(__name__)
_cycle_lock = asyncio.Lock()

# Trends run at most once every 30 minutes per tracker (in-memory throttle)
TRENDS_THROTTLE_SECONDS = 30 * 60
_last_trends_run: dict[str, datetime] = {}


async def run_cycle() -> dict[str, Any]:
    """Execute one full cycle across all active trackers without overlapping."""
    if _cycle_lock.locked():
        logger.warning("Another cycle is already running. Skipping.")
        return {"trackers_processed": 0, "posts_added": 0, "sentiments_added": 0, "errors": ["Cycle in progress"]}

    async with _cycle_lock:
        return await _execute_cycle()


async def _execute_cycle() -> dict[str, Any]:
    t_cycle_start = time.monotonic()
    summary: dict[str, Any] = {
        "trackers_processed": 0,
        "posts_added": 0,
        "sentiments_added": 0,
        "analytics_runs": 0,
        "errors": [],
    }

    try:
        trackers = await asyncio.to_thread(get_active_trackers)
    except Exception as exc:
        msg = f"Failed to fetch active trackers: {exc}"
        logger.error(msg)
        summary["errors"].append(msg)
        return summary

    if not trackers:
        logger.info("No active Telegram trackers found.")
        return summary

    if len(trackers) > 15:
        logger.warning(
            "Found %d active trackers, which exceeds the max limit of 15 per cycle. Processing the first 15.",
            len(trackers),
        )
        trackers = trackers[:15]

    for tracker in trackers:
        tracker_id = tracker["id"]
        summary["trackers_processed"] += 1
        logger.info("Processing tracker id=%s query='%s'.", tracker_id, tracker.get("query"))

        platform = tracker.get("platform", "telegram").lower()

        # Handle X (Twitter) specific gating: enabled check, min interval check, and cooldown
        if platform == "x":
            if not getattr(settings, "X_ENABLED", True):
                logger.debug("Tracker %s: X ingestion is disabled via X_ENABLED. Skipping.", tracker_id)
                continue

            # Check min interval between syncs
            last_sync_raw = tracker.get("last_sync_at")
            if last_sync_raw not in (None, "", "null", "None"):
                try:
                    last_sync_dt = datetime.fromisoformat(last_sync_raw)
                    if last_sync_dt.tzinfo is None:
                        last_sync_dt = last_sync_dt.replace(tzinfo=timezone.utc)
                    min_interval_mins = getattr(settings, "X_MIN_INTERVAL_MINUTES", 60)
                    now_utc = datetime.now(timezone.utc)
                    if (now_utc - last_sync_dt).total_seconds() < min_interval_mins * 60:
                        logger.debug(
                            "Tracker %s: X tracker synced recently (< %d min). Skipping.",
                            tracker_id, min_interval_mins
                        )
                        continue
                except Exception as exc:
                    logger.warning("Could not parse last_sync_at for tracker %s: %s", tracker_id, exc)

        messages = []
        try:
            if platform == "x":
                messages = await fetch_new_posts(tracker)
            else:
                messages = await fetch_new_messages(tracker)
        except Exception as exc:
            msg = f"Tracker {tracker_id} ({platform}): fetch failed — {exc}"
            print(f"[RUN_CYCLE] {msg}")
            logger.exception(msg)
            summary["errors"].append(msg)
            # Never let an error set a misleading last_sync_at
            continue

        if not messages:
            _safe_update_sync_time(tracker_id, summary)
            continue

        # 1. Deduplicate in ONE batched 'in' query
        all_ext_ids = [m["external_id"] for m in messages]
        try:
            existing_ids = get_existing_external_ids(tracker_id, all_ext_ids)
            new_messages = [m for m in messages if m["external_id"] not in existing_ids]
        except Exception as exc:
            msg = f"Tracker {tracker_id}: deduplication query failed — {exc}"
            logger.error(msg)
            summary["errors"].append(msg)
            continue

        total_new = len(new_messages)
        if total_new == 0:
            logger.info("Tracker %s: all messages already exist.", tracker_id)
            _safe_update_sync_time(tracker_id, summary)
            continue

        # 2. Batch sentiment analysis & batched DB insertions
        tracker_posts_added = 0
        batch_size = settings.gemini_batch_size

        batch_num = 0
        total_batches = (total_new + batch_size - 1) // batch_size

        for i in range(0, total_new, batch_size):
            batch_num += 1
            batch = new_messages[i : i + batch_size]
            sentiments = await analyze_batch_async([m["body"] for m in batch])

            try:
                inserted_posts = insert_posts_batch(batch)
            except Exception as exc:
                msg = f"Tracker {tracker_id}: insert_posts_batch failed — {exc}"
                logger.error(msg)
                summary["errors"].append(msg)
                continue

            tracker_posts_added += len(inserted_posts)
            summary["posts_added"] += len(inserted_posts)

            # Insert sentiments only if Gemini batch succeeded
            if sentiments is not None:
                sentiment_rows = [
                    {
                        "post_id": post_row["id"],
                        "tracker_id": tracker_id,
                        "polarity": sent["polarity"],
                        "emotion": sent["emotion"],
                        "confidence": sent["confidence"],
                    }
                    for post_row, sent in zip(inserted_posts, sentiments)
                ]
                try:
                    insert_sentiments_batch(sentiment_rows)
                    summary["sentiments_added"] += len(sentiment_rows)
                except Exception as exc:
                    logger.error("Tracker %s: insert_sentiments_batch failed — %s", tracker_id, exc)
                logger.info("Tracker %s: batch %d/%d ok.", tracker_id, batch_num, total_batches)
            else:
                logger.warning(
                    "Tracker %s: batch %d/%d failed with Gemini. Skipping sentiment insertion for %d posts.",
                    tracker_id, batch_num, total_batches, len(inserted_posts),
                )

            # Insert interactions for replies and forwards
            _record_interactions(tracker_id, batch, inserted_posts)

            logger.info("Tracker %s: processed %d/%d posts.", tracker_id, min(i + batch_size, total_new), total_new)

        _safe_update_sync_time(tracker_id, summary)

        # 3. Trigger TRENDS analytics — at most once every 30 min per tracker.
        #    (Sentiment and network/alert analytics keep their own schedule.)
        now = datetime.now(tz=timezone.utc)
        last_run = _last_trends_run.get(tracker_id)
        elapsed = (now - last_run).total_seconds() if last_run else TRENDS_THROTTLE_SECONDS + 1

        if elapsed >= TRENDS_THROTTLE_SECONDS:
            try:
                from app.analytics.trends import analyze_trends
                logger.info("Triggering trends analytics for tracker %s...", tracker_id)
                result = await asyncio.to_thread(analyze_trends, tracker_id)
                _last_trends_run[tracker_id] = now
                summary["analytics_runs"] += 1
                logger.info(
                    "Trends analytics done for tracker %s: %s", tracker_id, result
                )
            except Exception as exc:
                logger.error("Tracker %s: trends analytics failed — %s", tracker_id, exc)
        else:
            remaining = int(TRENDS_THROTTLE_SECONDS - elapsed)
            logger.debug(
                "Tracker %s: trends skipped (next run in %ds).", tracker_id, remaining
            )

        # 4. Network, demographics, and alerts — run every cycle (scheduler handles them)
        if tracker_posts_added > 0:
            try:
                await asyncio.to_thread(run_analytics, tracker_id)
            except Exception as exc:
                logger.error("Tracker %s: network/demographics analytics failed — %s", tracker_id, exc)

    cycle_elapsed = time.monotonic() - t_cycle_start
    summary["elapsed_seconds"] = round(cycle_elapsed, 2)
    logger.info("Cycle finished in %.2fs: %s", cycle_elapsed, summary)
    if cycle_elapsed > 120.0:
        logger.warning(
            "Cycle took %.2fs (longer than 120 seconds limit)! Consider tuning limits or batch sizes.",
            cycle_elapsed,
        )
    return summary


def _record_interactions(tracker_id: str, batch: list[dict], inserted_posts: list[dict]) -> None:
    """Build and write interaction rows for replies and forwards."""
    reply_ext_ids = [p["reply_to_external_id"] for p in batch if p.get("reply_to_external_id")]
    target_authors = get_post_authors_by_external_ids(tracker_id, reply_ext_ids) if reply_ext_ids else {}
    for p in batch:
        ext_id = p.get("external_id")
        a_hash = p.get("author_hash")
        if ext_id and a_hash and ext_id not in target_authors:
            target_authors[ext_id] = a_hash

    interactions = []
    for raw_p, ins_p in zip(batch, inserted_posts):
        src_hash = raw_p.get("author_hash")
        reply_to = raw_p.get("reply_to_external_id")
        fwd_hash = raw_p.get("forwarded_from_hash")

        target_hash = target_authors.get(reply_to) if reply_to else fwd_hash
        itype = "reply" if reply_to and target_hash else ("forward" if fwd_hash else None)

        if target_hash and target_hash != "unknown" and itype:
            interactions.append({
                "tracker_id": tracker_id,
                "source_hash": src_hash,
                "target_hash": target_hash,
                "type": itype,
                "post_id": ins_p["id"],
                "occurred_at": raw_p.get("posted_at"),
            })

    if interactions:
        try:
            insert_interactions_batch(interactions)
        except Exception as exc:
            logger.warning("Tracker %s: insert_interactions_batch failed — %s", tracker_id, exc)


def _safe_update_sync_time(tracker_id: str, summary: dict[str, Any]) -> None:
    try:
        update_tracker_last_sync(tracker_id, datetime.now(tz=timezone.utc))
    except Exception as exc:
        msg = f"Tracker {tracker_id}: failed to update last_sync_at — {exc}"
        logger.error(msg)
        summary["errors"].append(msg)


async def sync_single_x_tracker(tracker: dict[str, Any]) -> dict[str, Any]:
    """Ingest posts for a single X tracker manually, bypassing interval check but respecting cooldown.

    Returns summary of operations.
    """
    from app.ingestion.x_worker import is_x_in_cooldown, get_x_cooldown_until

    tracker_id = tracker["id"]
    if is_x_in_cooldown():
        cooldown_ts = get_x_cooldown_until()
        return {
            "status": "cooldown",
            "message": f"X is currently in cooldown until {cooldown_ts.isoformat() if cooldown_ts else 'unknown'}",
            "posts_added": 0,
            "sentiments_added": 0,
        }

    summary: dict[str, Any] = {
        "status": "ok",
        "tracker_id": tracker_id,
        "posts_added": 0,
        "sentiments_added": 0,
        "errors": [],
    }

    try:
        messages = await fetch_new_posts(tracker)
    except Exception as exc:
        msg = f"Tracker {tracker_id} manual fetch failed: {exc}"
        logger.exception(msg)
        return {"status": "error", "message": msg, "posts_added": 0, "sentiments_added": 0}

    # If no messages returned (could be rate limited or just no new tweets)
    if is_x_in_cooldown():
        cooldown_ts = get_x_cooldown_until()
        return {
            "status": "cooldown",
            "message": f"X hit cooldown during fetch until {cooldown_ts.isoformat() if cooldown_ts else 'unknown'}",
            "posts_added": 0,
            "sentiments_added": 0,
        }

    if not messages:
        _safe_update_sync_time(tracker_id, summary)
        return {"status": "ok", "message": "No new posts found", "posts_added": 0, "sentiments_added": 0}

    # Deduplicate
    all_ext_ids = [m["external_id"] for m in messages]
    try:
        existing_ids = get_existing_external_ids(tracker_id, all_ext_ids)
        new_messages = [m for m in messages if m["external_id"] not in existing_ids]
    except Exception as exc:
        msg = f"Tracker {tracker_id} deduplication query failed: {exc}"
        logger.error(msg)
        return {"status": "error", "message": msg, "posts_added": 0, "sentiments_added": 0}

    total_new = len(new_messages)
    if total_new == 0:
        _safe_update_sync_time(tracker_id, summary)
        return {"status": "ok", "message": "All posts already exist", "posts_added": 0, "sentiments_added": 0}

    # Process batches
    batch_size = settings.gemini_batch_size
    tracker_posts_added = 0

    for i in range(0, total_new, batch_size):
        batch = new_messages[i : i + batch_size]
        sentiments = await analyze_batch_async([m["body"] for m in batch])

        try:
            inserted_posts = insert_posts_batch(batch)
        except Exception as exc:
            msg = f"Tracker {tracker_id}: insert_posts_batch failed — {exc}"
            logger.error(msg)
            summary["errors"].append(msg)
            continue

        tracker_posts_added += len(inserted_posts)
        summary["posts_added"] += len(inserted_posts)

        if sentiments is not None:
            sentiment_rows = [
                {
                    "post_id": post_row["id"],
                    "tracker_id": tracker_id,
                    "polarity": sent["polarity"],
                    "emotion": sent["emotion"],
                    "confidence": sent["confidence"],
                }
                for post_row, sent in zip(inserted_posts, sentiments)
            ]
            try:
                insert_sentiments_batch(sentiment_rows)
                summary["sentiments_added"] += len(sentiment_rows)
            except Exception as exc:
                logger.error("Tracker %s: insert_sentiments_batch failed — %s", tracker_id, exc)

        _record_interactions(tracker_id, batch, inserted_posts)

    _safe_update_sync_time(tracker_id, summary)

    if tracker_posts_added > 0:
        try:
            await asyncio.to_thread(run_analytics, tracker_id)
        except Exception as exc:
            logger.error("Tracker %s: analytics failed — %s", tracker_id, exc)

    return summary
