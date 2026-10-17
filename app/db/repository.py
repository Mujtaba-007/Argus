# db/repository.py
# Supabase data-access for trackers, posts, sentiments, and interactions.
# Includes batch operations to minimize network roundtrips.

import logging
from datetime import datetime, timezone
from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from app.db.supabase_client import supabase

logger = logging.getLogger(__name__)

_db_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


@_db_retry
def get_active_trackers() -> list[dict[str, Any]]:
    """Return all active trackers (telegram, x, etc.)."""
    result = (
        supabase.table("trackers")
        .select("*")
        .eq("status", "active")
        .execute()
    )
    return result.data or []


@_db_retry
def update_tracker_last_sync(tracker_id: str, synced_at: datetime) -> None:
    """Update last_sync_at timestamp for a tracker."""
    supabase.table("trackers").update(
        {"last_sync_at": synced_at.isoformat()}
    ).eq("id", tracker_id).execute()


@_db_retry
def post_exists(tracker_id: str, external_id: str) -> bool:
    """Single-post existence check (kept for backward compatibility)."""
    result = (
        supabase.table("posts")
        .select("id")
        .eq("tracker_id", tracker_id)
        .eq("external_id", external_id)
        .limit(1)
        .execute()
    )
    return bool(result.data)


@_db_retry
def get_existing_external_ids(tracker_id: str, external_ids: list[str]) -> set[str]:
    """Fetch existing external_ids in a single batched 'in' query."""
    if not external_ids:
        return set()
    result = (
        supabase.table("posts")
        .select("external_id")
        .eq("tracker_id", tracker_id)
        .in_("external_id", external_ids)
        .execute()
    )
    return {r["external_id"] for r in (result.data or []) if "external_id" in r}


@_db_retry
def insert_post(post: dict[str, Any]) -> str:
    """Insert a single post row (kept for backward compatibility)."""
    result = supabase.table("posts").insert(post).execute()
    return result.data[0]["id"]


@_db_retry
def insert_posts_batch(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Insert multiple posts in a single roundtrip, returning created rows."""
    if not posts:
        return []
    result = supabase.table("posts").insert(posts).execute()
    return result.data or []


@_db_retry
def insert_sentiment(sentiment: dict[str, Any]) -> None:
    """Insert a single sentiment row (kept for backward compatibility)."""
    supabase.table("sentiments").insert(sentiment).execute()


@_db_retry
def insert_sentiments_batch(sentiments: list[dict[str, Any]]) -> None:
    """Insert multiple sentiment rows in a single batch."""
    if not sentiments:
        return
    supabase.table("sentiments").insert(sentiments).execute()


@_db_retry
def get_post_authors_by_external_ids(tracker_id: str, external_ids: list[str]) -> dict[str, str]:
    """Return mapping of external_id -> author_hash for replied-to post author lookups."""
    if not external_ids:
        return {}
    result = (
        supabase.table("posts")
        .select("external_id, author_hash")
        .eq("tracker_id", tracker_id)
        .in_("external_id", external_ids)
        .execute()
    )
    return {
        r["external_id"]: r["author_hash"]
        for r in (result.data or [])
        if r.get("external_id") and r.get("author_hash")
    }


@_db_retry
def insert_interactions_batch(interactions: list[dict[str, Any]]) -> None:
    """Insert interaction rows (replies and forwards)."""
    if not interactions:
        return
    supabase.table("interactions").insert(interactions).execute()


@_db_retry
def get_posts_needing_sentiment(tracker_id: str, limit: int = 200) -> list[dict[str, Any]]:
    """
    Find posts for a tracker that either:
    1. Have no sentiment row at all.
    2. Have a bad/fallback sentiment row (confidence == 0, emotion in ['neutral', 'unclassified']).

    Fetches all posts and existing sentiments in pages of 1000 using .range() to overcome
    the Supabase default limit. Returns up to `limit` posts, newest first.
    Logs: total posts, posts with valid sentiment, posts needing analysis.
    """
    page_size = 1000

    # 1. Fetch all posts for this tracker in pages of 1000, ordered newest first
    all_posts: list[dict[str, Any]] = []
    start = 0
    while True:
        res = (
            supabase.table("posts")
            .select("id, tracker_id, body, posted_at")
            .eq("tracker_id", tracker_id)
            .order("posted_at", desc=True)
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = res.data or []
        all_posts.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    total_posts = len(all_posts)
    if total_posts == 0:
        logger.info("Tracker %s: total posts=0, posts with valid sentiment=0, posts needing analysis=0.", tracker_id)
        return []

    # 2. Fetch all sentiments for this tracker in pages of 1000
    all_sentiments: list[dict[str, Any]] = []
    start = 0
    while True:
        res = (
            supabase.table("sentiments")
            .select("post_id, emotion, confidence")
            .eq("tracker_id", tracker_id)
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = res.data or []
        all_sentiments.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    sent_map = {s["post_id"]: s for s in all_sentiments}

    # 3. Classify valid vs needing analysis
    valid_count = 0
    candidates: list[dict[str, Any]] = []
    bad_post_ids_to_clean: list[str] = []

    for post in all_posts:
        pid = post["id"]
        sent = sent_map.get(pid)
        if sent is None:
            if len(candidates) < limit:
                candidates.append(post)
        elif sent.get("confidence", 0.0) == 0.0 and str(sent.get("emotion", "")).lower() in {"neutral", "unclassified"}:
            if len(candidates) < limit:
                candidates.append(post)
                bad_post_ids_to_clean.append(pid)
        else:
            valid_count += 1

    needing_analysis_count = total_posts - valid_count

    logger.info(
        "Tracker %s: total posts=%d, posts with valid sentiment=%d, posts needing analysis=%d.",
        tracker_id, total_posts, valid_count, needing_analysis_count,
    )

    # 4. Clean up any bad sentiment rows for the candidates being re-analyzed
    if bad_post_ids_to_clean:
        try:
            # Delete in chunks if many
            for i in range(0, len(bad_post_ids_to_clean), 100):
                chunk = bad_post_ids_to_clean[i : i + 100]
                supabase.table("sentiments").delete().eq("tracker_id", tracker_id).in_("post_id", chunk).execute()
            logger.info("Deleted %d bad sentiment rows for re-analysis on tracker %s.", len(bad_post_ids_to_clean), tracker_id)
        except Exception as exc:
            logger.warning("Failed to delete bad sentiment rows before reanalyze: %s", exc)

    return candidates


@_db_retry
def upsert_sentiments_batch(sentiments: list[dict[str, Any]]) -> None:
    """Upsert sentiment rows in a single batch (on conflict post_id)."""
    if not sentiments:
        return
    supabase.table("sentiments").upsert(sentiments).execute()

