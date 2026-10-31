# db/analytics_repository.py
# Supabase data-access specifically for Trends and Network analytics.
# Keeps repository.py modular and under 150 lines.

import logging
from datetime import datetime, timedelta, timezone
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
def get_posts_for_trends(tracker_id: str, since_iso: str, limit: int = 2000) -> list[dict[str, Any]]:
    """
    Fetch the newest *limit* posts from since_iso to now in a single bounded
    query (ordered DESC).  Used for the 24H range where recency matters and
    ~2 000 posts is enough for TF-IDF.

    Paginates in pages of 1 000 only when limit > 1 000, so the common
    2 000-post case makes exactly 2 round-trips.
    """
    page_size = 1000
    all_posts: list[dict[str, Any]] = []
    start = 0
    while len(all_posts) < limit:
        fetch_count = min(page_size, limit - len(all_posts))
        res = (
            supabase.table("posts")
            .select("body, posted_at")
            .eq("tracker_id", tracker_id)
            .gte("posted_at", since_iso)
            .order("posted_at", desc=True)
            .range(start, start + fetch_count - 1)
            .execute()
        )
        batch = res.data or []
        all_posts.extend(batch)
        if len(batch) < fetch_count:
            break
        start += fetch_count
    return all_posts


@_db_retry
def _fetch_bucket(
    tracker_id: str,
    bucket_start_iso: str,
    bucket_end_iso: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Single bucket query: up to *limit* posts in [bucket_start, bucket_end)."""
    res = (
        supabase.table("posts")
        .select("body, posted_at")
        .eq("tracker_id", tracker_id)
        .gte("posted_at", bucket_start_iso)
        .lt("posted_at", bucket_end_iso)
        .order("posted_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


@_db_retry
def get_posts_for_trends_bucketed(
    tracker_id: str,
    window_start: datetime,
    window_end: datetime,
    n_buckets: int = 12,
    posts_per_bucket: int = 50,
) -> list[dict[str, Any]]:
    """
    Evenly-sampled fetch across [window_start, window_end).

    Fetches posts within the entire historical window in a single fast indexed query
    (capped at max 800 posts), and stratifies them across n_buckets in Python memory.
    This replaces 12 sequential HTTP database roundtrips and avoids PostgreSQL statement
    timeouts, executing in milliseconds while maintaining chronological distribution.
    Guarded against non-positive bucket counts or zero-division timedelta math.
    """
    safe_buckets = max(1, n_buckets)
    safe_posts_per_bucket = max(1, posts_per_bucket)
    max_posts = min(800, safe_buckets * safe_posts_per_bucket)

    if window_end <= window_start:
        window_start = window_end - timedelta(days=7)

    w_start_iso = window_start.isoformat()
    w_end_iso = window_end.isoformat()

    # Single fast query utilizing the (tracker_id, posted_at DESC) index
    res = (
        supabase.table("posts")
        .select("body, posted_at")
        .eq("tracker_id", tracker_id)
        .gte("posted_at", w_start_iso)
        .lte("posted_at", w_end_iso)
        .order("posted_at", desc=True)
        .limit(max_posts)
        .execute()
    )
    raw_posts = res.data or []
    if not raw_posts:
        return []

    total_seconds = (window_end - window_start).total_seconds()
    if total_seconds <= 0 or len(raw_posts) <= safe_posts_per_bucket:
        return raw_posts[:max_posts]

    # Time-stratify in memory across n_buckets
    bucket_sec = total_seconds / safe_buckets
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(safe_buckets)]

    for p in raw_posts:
        p_at = p.get("posted_at")
        if not p_at:
            continue
        try:
            dt = datetime.fromisoformat(p_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            offset_sec = max(0.0, (dt - window_start).total_seconds())
            b_idx = int(offset_sec // bucket_sec)
            b_idx = max(0, min(safe_buckets - 1, b_idx))
            if len(buckets[b_idx]) < safe_posts_per_bucket:
                buckets[b_idx].append(p)
        except Exception:
            continue

    stratified: list[dict[str, Any]] = []
    for b in buckets:
        stratified.extend(b)

    # If bucket stratification pruned too aggressively due to sparse timestamps, fallback to raw_posts
    if len(stratified) < min(10, len(raw_posts)):
        return raw_posts[:max_posts]

    return stratified[:max_posts]


@_db_retry
def delete_trends_for_period(tracker_id: str, period_start_iso: str) -> None:
    """Delete trend rows for this tracker and period_start before re-inserting."""
    supabase.table("trends").delete().eq("tracker_id", tracker_id).eq("period_start", period_start_iso).execute()


@_db_retry
def delete_all_trends_for_tracker(tracker_id: str) -> None:
    """Delete ALL trend rows for a tracker, regardless of period_start.

    Call this before each insert run so stale rows from previous periods
    do not accumulate in the table.
    """
    supabase.table("trends").delete().eq("tracker_id", tracker_id).execute()


@_db_retry
def delete_trends_for_tracker_and_range(tracker_id: str, range_label: str) -> None:
    """Delete trend rows for a tracker with a specific range_label."""
    supabase.table("trends").delete().eq("tracker_id", tracker_id).eq("range_label", range_label).execute()


@_db_retry
def insert_trends_batch(trends: list[dict[str, Any]]) -> None:
    """Insert newly computed trend rows."""
    if not trends:
        return
    supabase.table("trends").insert(trends).execute()


@_db_retry
def get_interactions_for_network(
    tracker_id: str, since_iso: str, limit: int = 800
) -> list[dict[str, Any]]:
    """Fetch newest interactions (up to limit) within since_iso for network analysis."""
    res = (
        supabase.table("interactions")
        .select("source_hash, target_hash")
        .eq("tracker_id", tracker_id)
        .gte("occurred_at", since_iso)
        .order("occurred_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


@_db_retry
def upsert_network_nodes(nodes: list[dict[str, Any]]) -> None:
    """Upsert top nodes into network_nodes by (tracker_id, author_hash)."""
    if not nodes:
        return
    supabase.table("network_nodes").upsert(
        nodes, on_conflict="tracker_id,author_hash"
    ).execute()


@_db_retry
def upsert_network_edges(edges: list[dict[str, Any]]) -> None:
    """Upsert edges into network_edges by (tracker_id, source_hash, target_hash)."""
    if not edges:
        return
    supabase.table("network_edges").upsert(
        edges, on_conflict="tracker_id,source_hash,target_hash"
    ).execute()


@_db_retry
def get_authors_needing_profile(tracker_id: str, limit: int) -> list[dict[str, Any]]:
    """Return up to `limit` author_hashes for the tracker.
    Priority: authors with no profile row first, then profiles older than 24h.
    Skip authors profiled in the last 24h so each run moves to new authors.
    Author must have at least 3 posts. Paged diff to bypass 1000-row cap.
    Logs: eligible authors, already profiled, selected this run.
    """
    from datetime import datetime, timedelta, timezone
    from collections import Counter as _Counter

    cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    page_size = 1000

    # 1. Fetch all posts for this tracker in pages of 1000 to count posts per author
    all_post_authors: list[str] = []
    start = 0
    while True:
        res = (
            supabase.table("posts")
            .select("author_hash")
            .eq("tracker_id", tracker_id)
            .neq("author_hash", "unknown")
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = res.data or []
        for r in batch:
            h = r.get("author_hash")
            if h and h != "unknown":
                all_post_authors.append(h)
        if len(batch) < page_size:
            break
        start += page_size

    counts = _Counter(all_post_authors)
    eligible = [h for h, c in counts.items() if c >= 3]

    if not eligible:
        logger.info(
            "Tracker %s demographics: eligible authors=0, already profiled=0, selected this run=0.",
            tracker_id,
        )
        return []

    # 2. Fetch existing profiles for this tracker in pages of 1000
    all_profiles: list[dict[str, Any]] = []
    start = 0
    while True:
        res = (
            supabase.table("author_profiles")
            .select("author_hash, updated_at")
            .eq("tracker_id", tracker_id)
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = res.data or []
        all_profiles.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    profile_map = {r["author_hash"]: r.get("updated_at") for r in all_profiles if r.get("author_hash")}

    # 3. Partition eligible authors: unprofiled first, then stale (>24h). Skip fresh (<24h).
    unprofiled = [h for h in eligible if h not in profile_map]
    stale = [
        h for h in eligible
        if h in profile_map and (not profile_map[h] or profile_map[h] < cutoff_24h)
    ]

    selected_hashes = (unprofiled + stale)[:limit]
    already_profiled_count = sum(1 for h in eligible if h in profile_map)

    logger.info(
        "Tracker %s demographics: eligible authors=%d, already profiled=%d, selected this run=%d (unprofiled=%d, stale=%d).",
        tracker_id,
        len(eligible),
        already_profiled_count,
        len(selected_hashes),
        len(unprofiled),
        len(stale),
    )

    return [{"author_hash": h} for h in selected_hashes]


@_db_retry
def get_all_author_profiles(tracker_id: str) -> list[dict[str, Any]]:
    """Fetch all author_profiles for a tracker in pages of 1000."""
    page_size = 1000
    all_profiles: list[dict[str, Any]] = []
    start = 0
    while True:
        res = (
            supabase.table("author_profiles")
            .select("*")
            .eq("tracker_id", tracker_id)
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = res.data or []
        all_profiles.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return all_profiles


@_db_retry
def get_author_sample_posts(tracker_id: str, author_hash: str, n: int = 10) -> list[str]:
    """Return up to n most recent post bodies for an author. Never logs them."""
    res = (
        supabase.table("posts")
        .select("body")
        .eq("tracker_id", tracker_id)
        .eq("author_hash", author_hash)
        .order("posted_at", desc=True)
        .limit(n)
        .execute()
    )
    return [r["body"] for r in (res.data or []) if r.get("body")]


@_db_retry
def upsert_author_profiles_batch(profiles: list[dict[str, Any]]) -> None:
    """Upsert author_profiles rows (hashes only, no raw text)."""
    if not profiles:
        return
    supabase.table("author_profiles").upsert(
        profiles, on_conflict="tracker_id,author_hash"
    ).execute()


@_db_retry
def replace_demographics_agg(tracker_id: str, rows: list[dict[str, Any]]) -> None:
    """Delete old aggregates for tracker and insert fresh rows."""
    supabase.table("demographics_agg").delete().eq("tracker_id", tracker_id).execute()
    if rows:
        supabase.table("demographics_agg").insert(rows).execute()


@_db_retry
def get_demographics_agg(tracker_id: str) -> list[dict[str, Any]]:
    """Fetch all demographics_agg rows for a tracker."""
    res = (
        supabase.table("demographics_agg")
        .select("*")
        .eq("tracker_id", tracker_id)
        .execute()
    )
    return res.data or []
