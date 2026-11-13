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
    get_posts_needing_sentiment,
    insert_interactions_batch,
    insert_posts_batch,
    insert_sentiments_batch,
    update_tracker_last_sync,
    upsert_sentiments_batch,
)
from app.ingestion.telegram_worker import fetch_new_messages
from app.ingestion.x_worker import fetch_new_posts
from app.nlp.sentiment import analyze_batch_async

logger = logging.getLogger(__name__)
_cycle_lock = asyncio.Lock()

# Trends run at most once every 30 minutes per tracker (in-memory throttle)
TRENDS_THROTTLE_SECONDS = 30 * 60
_last_trends_run: dict[str, datetime] = {}
_cycle_counter: int = 0


def _extract_newest_posted_at(messages: list[dict[str, Any]]) -> datetime | None:
    """Find the newest message timestamp among messages."""
    newest: datetime | None = None
    for m in messages:
        raw_ts = m.get("posted_at")
        if raw_ts:
            try:
                dt = datetime.fromisoformat(raw_ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if newest is None or dt > newest:
                    newest = dt
            except Exception:
                pass
    return newest


async def run_cycle() -> dict[str, Any]:
    """Execute one full cycle across all active trackers without overlapping."""
    if _cycle_lock.locked():
        logger.warning("Another cycle is already running. Skipping.")
        return {"trackers_processed": 0, "posts_added": 0, "sentiments_added": 0, "errors": ["Cycle in progress"]}

    async with _cycle_lock:
        return await _execute_cycle()


async def _execute_cycle() -> dict[str, Any]:
    global _cycle_counter
    t_cycle_start = time.monotonic()
    t_sentiment_start: float | None = None
    sentiments_analyzed_this_cycle = 0
    max_sentiment_per_cycle = getattr(settings, "max_sentiment_per_cycle", 300)
    sentiment_timeout_seconds = getattr(settings, "cycle_sentiment_timeout_seconds", 90.0)
    min_sentiment_per_tracker = getattr(settings, "min_sentiment_per_tracker", 20)

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

    # Fairness: rotate tracker order so a single heavy tracker cannot starve others
    num_active = len(trackers)
    if num_active > 0:
        start_idx = _cycle_counter % num_active
        trackers = trackers[start_idx:] + trackers[:start_idx]
        _cycle_counter += 1
        logger.info("Tracker rotation: start_idx=%d/%d.", start_idx, num_active)

    if len(trackers) > 15:
        logger.warning(
            "Found %d active trackers, which exceeds the max limit of 15 per cycle. Processing the first 15.",
            len(trackers),
        )
        trackers = trackers[:15]

    num_trackers = max(len(trackers), 1)
    fair_ingest_budget_per_tracker = max(
        min_sentiment_per_tracker,
        max_sentiment_per_cycle // num_trackers,
    )

    tracker_ingest_counts: dict[str, int] = {}
    tracker_deferred_ingest: dict[str, int] = {}

    for tracker in trackers:
        tracker_id = tracker["id"]
        tracker_ingest_counts[tracker_id] = 0
        tracker_deferred_ingest[tracker_id] = 0
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
        fetch_timeout = getattr(settings, "per_tracker_fetch_timeout_seconds", 120.0)
        try:
            if platform == "x":
                messages = await asyncio.wait_for(fetch_new_posts(tracker), timeout=fetch_timeout)
            else:
                messages = await asyncio.wait_for(fetch_new_messages(tracker), timeout=fetch_timeout)
        except asyncio.TimeoutError:
            msg = f"Tracker {tracker_id} ({platform}): fetch timed out after {fetch_timeout}s"
            print(f"[RUN_CYCLE] {msg}")
            logger.error(msg)
            summary["errors"].append(msg)
            continue
        except Exception as exc:
            msg = f"Tracker {tracker_id} ({platform}): fetch failed — {exc}"
            print(f"[RUN_CYCLE] {msg}")
            logger.exception(msg)
            summary["errors"].append(msg)
            # Never let an error set a misleading last_sync_at
            continue

        if not messages:
            await _safe_update_sync_time(tracker_id, summary)
            logger.info(
                "Tracker %s ('%s') cycle summary: fetched=0, already_existing=0, new=0, inserted=0, duplicates_ignored=0, sentiment_analyzed=0, deferred=0.",
                tracker_id, tracker.get("query", tracker_id),
            )
            continue

        # 1. Deduplicate in batched chunks
        all_ext_ids = [m["external_id"] for m in messages]
        try:
            existing_ids = await asyncio.to_thread(get_existing_external_ids, tracker_id, all_ext_ids)
            new_messages = [m for m in messages if m["external_id"] not in existing_ids]
        except Exception as exc:
            msg = f"Tracker {tracker_id}: deduplication query failed — {exc}"
            logger.error(msg)
            summary["errors"].append(msg)
            continue

        total_new = len(new_messages)
        if total_new == 0:
            logger.info("Tracker %s: all messages already exist.", tracker_id)
            newest_dt = _extract_newest_posted_at(messages)
            await _safe_update_sync_time(tracker_id, summary, newest_dt)
            logger.info(
                "Tracker %s ('%s') cycle summary: fetched=%d, already_existing=%d, new=0, inserted=0, duplicates_ignored=0, sentiment_analyzed=0, deferred=0.",
                tracker_id, tracker.get("query", tracker_id), len(messages), len(existing_ids),
            )
            continue

        # 2. Batch sentiment analysis & batched DB insertions
        tracker_posts_added = 0
        tracker_sentiment_count = 0
        tracker_had_real_failure = False
        tracker_ingest_budget = min(
            fair_ingest_budget_per_tracker,
            max_sentiment_per_cycle - sentiments_analyzed_this_cycle,
        )
        batch_size = settings.gemini_batch_size

        batch_num = 0
        total_batches = (total_new + batch_size - 1) // batch_size

        for i in range(0, total_new, batch_size):
            batch_num += 1
            batch = new_messages[i : i + batch_size]

            # Enforce per-tracker ingest budget, global cycle cap, and hard time guard
            can_analyze = True
            if sentiments_analyzed_this_cycle >= max_sentiment_per_cycle:
                can_analyze = False
                logger.info(
                    "Tracker %s: per-cycle sentiment cap reached (%d/%d posts). Ingesting batch %d/%d without sentiment (deferred to backfill).",
                    tracker_id, sentiments_analyzed_this_cycle, max_sentiment_per_cycle, batch_num, total_batches,
                )
            elif tracker_sentiment_count >= tracker_ingest_budget:
                can_analyze = False
                logger.info(
                    "Tracker %s: per-tracker ingest sentiment budget reached (%d/%d posts). Ingesting batch %d/%d without sentiment (deferred to backfill).",
                    tracker_id, tracker_sentiment_count, tracker_ingest_budget, batch_num, total_batches,
                )
            elif t_sentiment_start is not None and (time.monotonic() - t_sentiment_start) >= sentiment_timeout_seconds:
                can_analyze = False
                logger.warning(
                    "Tracker %s: cycle sentiment time guard triggered (%.1fs elapsed >= %.1fs). Deferring batch %d/%d to backfill.",
                    tracker_id, time.monotonic() - t_sentiment_start, sentiment_timeout_seconds, batch_num, total_batches,
                )

            sentiments = None
            if can_analyze:
                if t_sentiment_start is None:
                    t_sentiment_start = time.monotonic()
                sentiments = await analyze_batch_async([m["body"] for m in batch])

            try:
                inserted_posts = await asyncio.to_thread(insert_posts_batch, batch)
            except Exception as exc:
                msg = f"Tracker {tracker_id}: insert_posts_batch failed — {exc}"
                logger.error(msg)
                summary["errors"].append(msg)
                tracker_had_real_failure = True
                continue

            tracker_posts_added += len(inserted_posts)
            summary["posts_added"] += len(inserted_posts)

            # Insert sentiments only if Gemini batch succeeded (mapped strictly by external_id)
            if sentiments is not None:
                sentiment_by_ext_id = {
                    m["external_id"]: sent
                    for m, sent in zip(batch, sentiments)
                    if "external_id" in m
                }
                sentiment_rows = []
                for post_row in inserted_posts:
                    ext_id = post_row.get("external_id")
                    sent = sentiment_by_ext_id.get(ext_id)
                    if sent is not None:
                        sentiment_rows.append({
                            "post_id": post_row["id"],
                            "tracker_id": tracker_id,
                            "polarity": sent["polarity"],
                            "emotion": sent["emotion"],
                            "confidence": sent["confidence"],
                        })

                if sentiment_rows:
                    try:
                        await asyncio.to_thread(insert_sentiments_batch, sentiment_rows)
                        sentiments_analyzed_this_cycle += len(sentiment_rows)
                        tracker_sentiment_count += len(sentiment_rows)
                        summary["sentiments_added"] += len(sentiment_rows)
                    except Exception as exc:
                        logger.error("Tracker %s: insert_sentiments_batch failed — %s", tracker_id, exc)
                        tracker_had_real_failure = True
                logger.info(
                    "Tracker %s: batch %d/%d ok (inserted %d, sentiments %d).",
                    tracker_id, batch_num, total_batches, len(inserted_posts), len(sentiment_rows)
                )
            else:
                if not can_analyze:
                    logger.info(
                        "Tracker %s: batch %d/%d inserted without sentiment (deferred to backfill).",
                        tracker_id, batch_num, total_batches,
                    )
                else:
                    logger.warning(
                        "Tracker %s: batch %d/%d failed with Gemini. Skipping sentiment insertion for %d posts.",
                        tracker_id, batch_num, total_batches, len(inserted_posts),
                    )

            # Insert interactions for replies and forwards mapped by external_id
            await _record_interactions(tracker_id, batch, inserted_posts)

            logger.info("Tracker %s: processed %d/%d posts.", tracker_id, min(i + batch_size, total_new), total_new)

        duplicates_ignored = max(0, total_new - tracker_posts_added)
        deferred_count = max(0, tracker_posts_added - tracker_sentiment_count)
        tracker_ingest_counts[tracker_id] = tracker_sentiment_count
        tracker_deferred_ingest[tracker_id] = deferred_count

        # Advance last_sync_at only if all batches finished without real errors
        if tracker_had_real_failure:
            logger.warning(
                "Tracker %s: encountered real batch errors. last_sync_at will NOT be advanced.",
                tracker_id,
            )
        else:
            newest_dt = _extract_newest_posted_at(messages)
            await _safe_update_sync_time(tracker_id, summary, newest_dt)

        logger.info(
            "Tracker %s ('%s') cycle summary: fetched=%d, already_existing=%d, new=%d, inserted=%d, duplicates_ignored=%d, sentiment_analyzed=%d, deferred=%d.",
            tracker_id,
            tracker.get("query", tracker_id),
            len(messages),
            len(existing_ids),
            total_new,
            tracker_posts_added,
            duplicates_ignored,
            tracker_sentiment_count,
            deferred_count,
        )

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

    # -----------------------------------------------------------------------
    # Backfill pass: drain deferred posts using remaining budget & time guard
    # -----------------------------------------------------------------------
    tracker_backfill_counts: dict[str, int] = {}
    tracker_still_awaiting: dict[str, int] = {}

    remaining_cycle_budget = max_sentiment_per_cycle - sentiments_analyzed_this_cycle
    time_exhausted = (
        t_sentiment_start is not None
        and (time.monotonic() - t_sentiment_start) >= sentiment_timeout_seconds
    )

    if remaining_cycle_budget > 0 and not time_exhausted:
        logger.info(
            "Starting backfill pass: %d posts remaining in cycle budget (%.1fs elapsed on sentiment).",
            remaining_cycle_budget,
            time.monotonic() - t_sentiment_start if t_sentiment_start else 0.0,
        )

        for idx, tracker in enumerate(trackers):
            tracker_id = tracker["id"]

            if sentiments_analyzed_this_cycle >= max_sentiment_per_cycle:
                logger.info("Backfill pass: reached cycle sentiment cap (%d posts). Stopping.", max_sentiment_per_cycle)
                break

            if t_sentiment_start is not None and (time.monotonic() - t_sentiment_start) >= sentiment_timeout_seconds:
                logger.warning(
                    "Backfill pass: sentiment time guard reached (%.1fs >= %.1fs). Stopping.",
                    time.monotonic() - t_sentiment_start, sentiment_timeout_seconds,
                )
                break

            # Fair share of remaining budget across remaining trackers
            trackers_left = len(trackers) - idx
            curr_remaining = max_sentiment_per_cycle - sentiments_analyzed_this_cycle
            tracker_backfill_budget = max(
                min_sentiment_per_tracker,
                curr_remaining // max(trackers_left, 1),
            )
            tracker_backfill_budget = min(tracker_backfill_budget, curr_remaining)

            try:
                pending_posts = await asyncio.to_thread(
                    get_posts_needing_sentiment,
                    tracker_id,
                    getattr(settings, "max_per_cycle", 1000),
                )
            except Exception as exc:
                logger.error("Tracker %s backfill: get_posts_needing_sentiment failed — %s", tracker_id, exc)
                continue

            total_pending = len(pending_posts)
            if total_pending == 0:
                tracker_still_awaiting[tracker_id] = 0
                tracker_backfill_counts[tracker_id] = 0
                continue

            to_analyze = pending_posts[:tracker_backfill_budget]
            bf_analyzed = 0
            bf_batch_size = settings.gemini_batch_size

            for j in range(0, len(to_analyze), bf_batch_size):
                if t_sentiment_start is not None and (time.monotonic() - t_sentiment_start) >= sentiment_timeout_seconds:
                    logger.warning("Tracker %s backfill: time guard reached mid-batch.", tracker_id)
                    break
                if sentiments_analyzed_this_cycle >= max_sentiment_per_cycle:
                    break

                bf_batch = to_analyze[j : j + bf_batch_size]
                if t_sentiment_start is None:
                    t_sentiment_start = time.monotonic()

                bf_sentiments = await analyze_batch_async([p.get("body", "") for p in bf_batch])
                if bf_sentiments is not None:
                    sentiment_rows = [
                        {
                            "post_id": post_item["id"],
                            "tracker_id": tracker_id,
                            "polarity": sent["polarity"],
                            "emotion": sent["emotion"],
                            "confidence": sent["confidence"],
                        }
                        for post_item, sent in zip(bf_batch, bf_sentiments)
                    ]
                    try:
                        await asyncio.to_thread(upsert_sentiments_batch, sentiment_rows)
                        bf_analyzed += len(sentiment_rows)
                        sentiments_analyzed_this_cycle += len(sentiment_rows)
                        summary["sentiments_added"] += len(sentiment_rows)
                    except Exception as exc:
                        logger.error("Tracker %s backfill: upsert_sentiments_batch failed — %s", tracker_id, exc)
                else:
                    logger.warning("Tracker %s backfill: Gemini batch failed for %d posts.", tracker_id, len(bf_batch))

            tracker_backfill_counts[tracker_id] = bf_analyzed
            tracker_still_awaiting[tracker_id] = max(0, total_pending - bf_analyzed)

            if bf_analyzed > 0:
                try:
                    await asyncio.to_thread(run_analytics, tracker_id)
                    summary["analytics_runs"] += 1
                except Exception as exc:
                    logger.error("Tracker %s backfill: analytics failed — %s", tracker_id, exc)

    # -----------------------------------------------------------------------
    # Log per-tracker cycle summary: analyzed this cycle, still awaiting
    # -----------------------------------------------------------------------
    for tracker in trackers:
        tid = tracker["id"]
        query = tracker.get("query", tid)
        analyzed_count = tracker_ingest_counts.get(tid, 0) + tracker_backfill_counts.get(tid, 0)
        awaiting_count = tracker_still_awaiting.get(
            tid,
            tracker_deferred_ingest.get(tid, 0),
        )
        logger.info(
            "Tracker %s ('%s'): %d analyzed this cycle, %d still awaiting.",
            tid, query, analyzed_count, awaiting_count,
        )

    cycle_elapsed = time.monotonic() - t_cycle_start
    summary["elapsed_seconds"] = round(cycle_elapsed, 2)
    logger.info("Cycle finished in %.2fs: %s", cycle_elapsed, summary)
    if cycle_elapsed > 120.0:
        logger.warning(
            "Cycle took %.2fs (longer than 120 seconds limit)! Consider tuning limits or batch sizes.",
            cycle_elapsed,
        )
    return summary


async def _record_interactions(tracker_id: str, batch: list[dict], inserted_posts: list[dict]) -> None:
    """Build and write interaction rows for replies and forwards mapped by external_id."""
    reply_ext_ids = [p["reply_to_external_id"] for p in batch if p.get("reply_to_external_id")]
    target_authors = (
        await asyncio.to_thread(get_post_authors_by_external_ids, tracker_id, reply_ext_ids)
        if reply_ext_ids
        else {}
    )
    for p in batch:
        ext_id = p.get("external_id")
        a_hash = p.get("author_hash")
        if ext_id and a_hash and ext_id not in target_authors:
            target_authors[ext_id] = a_hash

    # Map inserted posts by external_id to avoid list index misalignment
    inserted_by_ext_id = {p["external_id"]: p for p in inserted_posts if "external_id" in p}

    interactions = []
    for raw_p in batch:
        ext_id = raw_p.get("external_id")
        ins_p = inserted_by_ext_id.get(ext_id)
        if not ins_p:
            continue  # Post was not newly inserted (e.g. duplicate ignored)

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
            await asyncio.to_thread(insert_interactions_batch, interactions)
        except Exception as exc:
            logger.warning("Tracker %s: insert_interactions_batch failed — %s", tracker_id, exc)


async def _safe_update_sync_time(
    tracker_id: str,
    summary: dict[str, Any],
    synced_at: datetime | None = None,
) -> None:
    target_dt = synced_at or datetime.now(tz=timezone.utc)
    try:
        await asyncio.to_thread(update_tracker_last_sync, tracker_id, target_dt)
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
        await _safe_update_sync_time(tracker_id, summary)
        return {"status": "ok", "message": "No new posts found", "posts_added": 0, "sentiments_added": 0}

    # Deduplicate in batched chunks
    all_ext_ids = [m["external_id"] for m in messages]
    try:
        existing_ids = await asyncio.to_thread(get_existing_external_ids, tracker_id, all_ext_ids)
        new_messages = [m for m in messages if m["external_id"] not in existing_ids]
    except Exception as exc:
        msg = f"Tracker {tracker_id} deduplication query failed: {exc}"
        logger.error(msg)
        return {"status": "error", "message": msg, "posts_added": 0, "sentiments_added": 0}

    total_new = len(new_messages)
    if total_new == 0:
        newest_dt = _extract_newest_posted_at(messages)
        await _safe_update_sync_time(tracker_id, summary, newest_dt)
        return {"status": "ok", "message": "All posts already exist", "posts_added": 0, "sentiments_added": 0}

    # Process batches
    batch_size = settings.gemini_batch_size
    tracker_posts_added = 0
    sentiments_analyzed = 0
    tracker_had_real_failure = False
    max_sentiment_per_cycle = getattr(settings, "max_sentiment_per_cycle", 300)
    sentiment_timeout_seconds = getattr(settings, "cycle_sentiment_timeout_seconds", 90.0)
    t_start = time.monotonic()

    for i in range(0, total_new, batch_size):
        batch = new_messages[i : i + batch_size]

        can_analyze = True
        if sentiments_analyzed >= max_sentiment_per_cycle:
            can_analyze = False
        elif (time.monotonic() - t_start) >= sentiment_timeout_seconds:
            can_analyze = False

        sentiments = await analyze_batch_async([m["body"] for m in batch]) if can_analyze else None

        try:
            inserted_posts = await asyncio.to_thread(insert_posts_batch, batch)
        except Exception as exc:
            msg = f"Tracker {tracker_id}: insert_posts_batch failed — {exc}"
            logger.error(msg)
            summary["errors"].append(msg)
            tracker_had_real_failure = True
            continue

        tracker_posts_added += len(inserted_posts)
        summary["posts_added"] += len(inserted_posts)

        if sentiments is not None:
            sentiment_by_ext_id = {
                m["external_id"]: sent
                for m, sent in zip(batch, sentiments)
                if "external_id" in m
            }
            sentiment_rows = []
            for post_row in inserted_posts:
                ext_id = post_row.get("external_id")
                sent = sentiment_by_ext_id.get(ext_id)
                if sent is not None:
                    sentiment_rows.append({
                        "post_id": post_row["id"],
                        "tracker_id": tracker_id,
                        "polarity": sent["polarity"],
                        "emotion": sent["emotion"],
                        "confidence": sent["confidence"],
                    })
            if sentiment_rows:
                try:
                    await asyncio.to_thread(insert_sentiments_batch, sentiment_rows)
                    sentiments_analyzed += len(sentiment_rows)
                    summary["sentiments_added"] += len(sentiment_rows)
                except Exception as exc:
                    logger.error("Tracker %s: insert_sentiments_batch failed — %s", tracker_id, exc)
                    tracker_had_real_failure = True

        await _record_interactions(tracker_id, batch, inserted_posts)

    if not tracker_had_real_failure:
        newest_dt = _extract_newest_posted_at(messages)
        await _safe_update_sync_time(tracker_id, summary, newest_dt)
    else:
        logger.warning("Tracker %s manual sync: encountered batch errors, last_sync_at not advanced.", tracker_id)

    if tracker_posts_added > 0:
        try:
            await asyncio.to_thread(run_analytics, tracker_id)
        except Exception as exc:
            logger.error("Tracker %s: analytics failed — %s", tracker_id, exc)

    return summary
