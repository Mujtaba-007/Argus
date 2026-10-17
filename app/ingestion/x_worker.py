# ingestion/x_worker.py
# Async ingestion for X (formerly Twitter) API.
# Retrieves recent tweets matching a tracker's query and converts them to the Supabase `posts` schema.

import hashlib
import logging
from datetime import datetime
from typing import Any, List

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def _hash_id(val: Any) -> str | None:
    """Hash a raw identifier (e.g., author_id) for privacy.

    Returns a hex SHA‑256 string or ``None`` if ``val`` is falsy.
    """
    if not val:
        return None
    raw = str(val).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


async def fetch_new_posts(tracker: dict[str, Any]) -> List[dict[str, Any]]:
    """Fetch recent X posts for a given tracker.

    Parameters
    ----------
    tracker: dict
        Tracker row from the ``trackers`` table. Expected keys:
        - ``id`` – tracker UUID used for ``tracker_id`` column.
        - ``query`` – the search string passed directly to the X API.

    Returns
    -------
    list[dict]
        List of post dictionaries ready for insertion into ``posts``.
        Empty list on error or when no new tweets are found.
    """
    token = getattr(settings, "X_BEARER_TOKEN", None)
    if not token:
        logger.error("X_BEARER_TOKEN not configured – skipping X ingestion for tracker %s", tracker.get("id"))
        return []

    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "query": tracker.get("query", "").strip(),
        "max_results": 15,
        "tweet.fields": "created_at,public_metrics,lang,author_id",
    }

    url = "https://api.twitter.com/2/tweets/search/recent"

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code == 429:
                msg = f"[X_WORKER] Rate limit (429) hit for tracker {tracker.get('id')} (query: '{params['query']}')"
                print(msg)
                logger.warning(msg)
                return []
            if resp.status_code == 401:
                msg = f"[X_WORKER] Unauthorized (401) – check X_BEARER_TOKEN for tracker {tracker.get('id')}"
                print(msg)
                logger.error(msg)
                return []
            if resp.status_code != 200:
                msg = f"[X_WORKER] Error response {resp.status_code} for tracker {tracker.get('id')}: {resp.text}"
                print(msg)
                logger.error(msg)
                return []
        except httpx.HTTPError as exc:
            msg = f"[X_WORKER] HTTP error for tracker {tracker.get('id')}: {exc}"
            print(msg)
            logger.error(msg)
            return []
        except Exception as exc:
            msg = f"[X_WORKER] Unexpected network error for tracker {tracker.get('id')}: {exc}"
            print(msg)
            logger.exception(msg)
            return []

    try:
        data = resp.json().get("data", [])
    except Exception as exc:
        msg = f"[X_WORKER] Unable to parse X response JSON for tracker {tracker.get('id')}: {exc}"
        print(msg)
        logger.error(msg)
        return []

    posts: List[dict] = []
    for tweet in data:
        try:
            metrics = tweet.get("public_metrics", {})
            post = {
                "tracker_id": tracker.get("id"),
                "platform": "x",
                "external_id": tweet.get("id"),
                "author_hash": _hash_id(tweet.get("author_id")) or "unknown",
                "body": tweet.get("text", ""),
                "lang": tweet.get("lang", "en"),
                "posted_at": tweet.get("created_at"),
                "like_count": metrics.get("like_count", 0),
                "reply_count": metrics.get("reply_count", 0),
                "share_count": metrics.get("retweet_count", 0),
            }
            posts.append(post)
        except Exception as exc:
            logger.exception("Error converting tweet %s for tracker %s", tweet.get("id"), tracker.get("id"))
            continue

    return posts
