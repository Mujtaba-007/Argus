# db/repository.py
# Supabase data-access for trackers, posts, sentiments, and interactions.
# Includes batch operations to minimize network roundtrips.

import logging
from datetime import datetime, timezone
from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log, retry_if_exception
from app.db.supabase_client import supabase

logger = logging.getLogger(__name__)


def _is_transient_db_error(exc: BaseException) -> bool:
    """Return True if the exception is transient (network drop, timeout, 5xx) and safe to retry.
    Returns False for client errors (4xx), duplicate keys (23505 / 409), invalid queries, etc.
    """
    # Check PostgREST APIError code or Postgres error class
    code = getattr(exc, "code", None)
    if code is not None:
        code_str = str(code).strip()
        # Postgres error classes: 22 (data exception), 23 (integrity constraint violation e.g. 23505),
        # 42 (syntax error / access rule violation)
        if code_str.startswith(("22", "23", "42")):
            return False
        # HTTP 4xx client errors
        if code_str.isdigit() and 400 <= int(code_str) < 500:
            return False

    exc_str = str(exc)
    if "23505" in exc_str or "duplicate key" in exc_str:
        return False

    # Check for httpx HTTPStatusError status_code
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code is not None and 400 <= status_code < 500:
        return False

    return True


_db_retry = retry(
    retry=retry_if_exception(_is_transient_db_error),
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
def get_tracker(tracker_id: str) -> dict[str, Any] | None:
    """Return a single tracker by ID, or None if not found."""
    result = (
        supabase.table("trackers")
        .select("*")
        .eq("id", tracker_id)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


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
    """Fetch existing external_ids in chunks of at most 200 to prevent PostgREST row caps and URL length issues."""
    if not external_ids:
        return set()
    chunk_size = 200
    existing: set[str] = set()
    for i in range(0, len(external_ids), chunk_size):
        chunk = external_ids[i : i + chunk_size]
        result = (
            supabase.table("posts")
            .select("external_id")
            .eq("tracker_id", tracker_id)
            .in_("external_id", chunk)
            .execute()
        )
        for r in (result.data or []):
            if "external_id" in r:
                existing.add(r["external_id"])
    return existing


@_db_retry
def insert_post(post: dict[str, Any]) -> str:
    """Insert a single post row (kept for backward compatibility)."""
    result = supabase.table("posts").insert(post).execute()
    return result.data[0]["id"]


@_db_retry
def insert_posts_batch(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Upsert multiple posts in a single roundtrip, ignoring duplicate (tracker_id, external_id) rows."""
    if not posts:
        return []
    result = (
        supabase.table("posts")
        .upsert(
            posts,
            on_conflict="tracker_id,external_id",
            ignore_duplicates=True,
        )
        .execute()
    )
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
    """Return mapping of external_id -> author_hash for replied-to post author lookups in chunks of 200."""
    if not external_ids:
        return {}
    chunk_size = 200
    authors: dict[str, str] = {}
    for i in range(0, len(external_ids), chunk_size):
        chunk = external_ids[i : i + chunk_size]
        result = (
            supabase.table("posts")
            .select("external_id, author_hash")
            .eq("tracker_id", tracker_id)
            .in_("external_id", chunk)
            .execute()
        )
        for r in (result.data or []):
            if r.get("external_id") and r.get("author_hash"):
                authors[r["external_id"]] = r["author_hash"]
    return authors


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


@_db_retry
def get_posts_needing_sentiment_windowed(
    tracker_id: str,
    limit: int = 200,
    days: int = 7,
) -> list[dict[str, Any]]:
    """Efficiently return up to `limit` posts from the last `days` days that have
    no valid sentiment row, newest first.

    Uses two bounded, date-filtered queries rather than a full table scan, so it
    never causes a Supabase statement timeout even with thousands of stored posts.

    A 'valid' sentiment is any row that is NOT (confidence==0 AND emotion in
    {neutral, unclassified}).  Posts without any sentiment row are always returned.
    """
    from datetime import timedelta
    since_dt = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    # 1. Posts from the last N days, newest first, bounded by 1000 (safe range)
    res_posts = (
        supabase.table("posts")
        .select("id, tracker_id, body, posted_at")
        .eq("tracker_id", tracker_id)
        .gte("posted_at", since_dt)
        .order("posted_at", desc=True)
        .limit(1000)
        .execute()
    )
    recent_posts = res_posts.data or []
    if not recent_posts:
        return []

    post_ids = [p["id"] for p in recent_posts]

    # 2. Sentiments for only those post IDs (chunked to stay under URL length)
    chunk_size = 200
    sent_map: dict[str, dict] = {}
    for i in range(0, len(post_ids), chunk_size):
        chunk = post_ids[i : i + chunk_size]
        res_sent = (
            supabase.table("sentiments")
            .select("post_id, emotion, confidence")
            .in_("post_id", chunk)
            .execute()
        )
        for s in (res_sent.data or []):
            sent_map[s["post_id"]] = s

    # 3. Filter to posts lacking valid sentiment
    candidates: list[dict[str, Any]] = []
    bad_post_ids: list[str] = []
    for post in recent_posts:
        pid = post["id"]
        sent = sent_map.get(pid)
        if sent is None:
            candidates.append(post)
        elif (
            sent.get("confidence", 0.0) == 0.0
            and str(sent.get("emotion", "")).lower() in {"neutral", "unclassified"}
        ):
            candidates.append(post)
            bad_post_ids.append(pid)
        if len(candidates) >= limit:
            break

    # 4. Delete stale bad-sentiment rows so upsert inserts cleanly
    if bad_post_ids:
        try:
            for i in range(0, len(bad_post_ids), 100):
                chunk = bad_post_ids[i : i + 100]
                supabase.table("sentiments").delete().eq("tracker_id", tracker_id).in_("post_id", chunk).execute()
        except Exception as exc:
            logger.warning("Tracker %s: failed to delete stale sentiment rows — %s", tracker_id, exc)

    logger.info(
        "Tracker %s: windowed backfill found %d posts needing sentiment (last %dd, %d recent posts scanned).",
        tracker_id, len(candidates), days, len(recent_posts),
    )
    return candidates

