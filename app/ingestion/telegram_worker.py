# ingestion/telegram_worker.py
# Connects to Telegram using an existing Telethon session file and fetches
# new messages from a channel. The session file (argus_session.session)
# must already exist in the working directory — no login flow is triggered here.

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from telethon.tl.types import Message
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from app.config import settings
from app.ingestion.client import get_telegram_client
from app.ingestion.comment_worker import fetch_comments_for_posts

logger = logging.getLogger(__name__)


def _hash_id(val: Any) -> str | None:
    """
    One-way hash an ID or username for privacy.
    Returns a hex string, or None if val is None.
    """
    if val is None:
        return None
    raw = str(val).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _detect_lang(text: str) -> str:
    """
    Lightweight language hint: returns 'ar' if most characters are Arabic-range Unicode,
    'en' otherwise. Real language detection can be added later.
    """
    if not text:
        return "en"
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    return "ar" if arabic_chars / max(len(text), 1) > 0.3 else "en"


async def fetch_new_messages(tracker: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Connect to Telegram (using the existing session), fetch messages from the
    channel specified in tracker['query'], and return them as a list of dicts
    ready to be inserted into the `posts` table.

    Only messages newer than tracker['last_sync_at'] are returned.
    On the first sync (last_sync_at is None), the last _FIRST_SYNC_LIMIT messages
    are fetched.

    Args:
        tracker: A row from the `trackers` table (plain dict).

    Returns:
        List of post dicts. Empty list on error or no new messages.
    """
    raw_query: str = str(tracker.get("query", "")).strip()
    if "t.me/" in raw_query:
        channel = raw_query.split("t.me/")[-1].split("/")[0].strip().lstrip("@")
    else:
        channel = raw_query.lstrip("@")

    tracker_id: str = tracker["id"]
    last_sync_raw = tracker.get("last_sync_at")
    if last_sync_raw in (None, "", "null", "None"):
        last_sync_raw = None

    # Parse last_sync_at to an aware datetime (UTC) for comparison.
    last_sync_dt: datetime | None = None
    if last_sync_raw:
        try:
            last_sync_dt = datetime.fromisoformat(last_sync_raw).replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning("Could not parse last_sync_at='%s' for tracker %s.", last_sync_raw, tracker_id)

    client = get_telegram_client()

    posts: list[dict[str, Any]] = []

    try:
        await client.connect()

        if not await client.is_user_authorized():
            logger.error(
                "Telegram session is not authorised for tracker %s. "
                "Re-run the session setup script.",
                tracker_id,
            )
            return []

        max_fetch = settings.max_per_cycle if last_sync_dt else settings.first_run_limit

        logger.info(
            "Fetching messages -> channel='%s' (raw query='%s') | tracker_id=%s | last_sync_raw=%s | last_sync_dt=%s | limit=%d",
            channel, raw_query, tracker_id, last_sync_raw, last_sync_dt, max_fetch,
        )

        raw_msg_count = 0
        skipped_non_message = 0
        skipped_empty_text = 0
        skipped_date_cutoff = 0

        # Pagination loop: fetch up to 5 pages, each up to max_fetch messages.
        # next_offset_id: Telegram message ID to use as the starting anchor for
        # the next page (iter_messages returns messages strictly older than this ID).
        # Initialised to 0 (== no offset) so the first page always starts from newest.
        page = 0
        reached_end = False
        next_offset_id: int = 0        # 0 means "start from the channel head"
        oldest_fetched_date: datetime | None = None
        while page < 5:
            page += 1
            page_count = 0
            # Build fetch kwargs for this page:
            # - page 1 has next_offset_id=0, so offset_id is omitted (starts from newest).
            # - subsequent pages pass offset_id=<id> to return only messages strictly
            #   older than that value, walking backward through history.
            page_fetch_kwargs: dict[str, Any] = {"limit": max_fetch}
            if next_offset_id:
                page_fetch_kwargs["offset_id"] = next_offset_id

            async for message in client.iter_messages(channel, **page_fetch_kwargs):
                raw_msg_count += 1
                if not isinstance(message, Message):
                    skipped_non_message += 1
                    continue
                if not message.text:
                    skipped_empty_text += 1
                    continue  # skip media-only messages

                msg_date = message.date.replace(tzinfo=timezone.utc)

                # Track oldest fetched date for logging
                oldest_fetched_date = msg_date if (oldest_fetched_date is None or msg_date < oldest_fetched_date) else oldest_fetched_date

                # Skip messages we've already processed.
                if last_sync_dt and msg_date <= last_sync_dt:
                    skipped_date_cutoff += 1
                    reached_end = True
                    break  # iter_messages returns newest-first, so we can stop here

                # --- Per-message try/except: one bad message must not abort the batch ---
                try:
                    # Sum ALL reaction counts across every emoji used on this message.
                    # message.reactions.results is a list of ReactionCount objects,
                    # one per distinct emoji — we want the total, not just index [0].
                    like_count = (
                        sum(r.count for r in message.reactions.results)
                        if message.reactions and message.reactions.results
                        else 0
                    )

                    # reply_count: the `.replies` attribute is None on channels that
                    # have comments disabled, so guard against that.
                    reply_count = (
                        message.replies.replies
                        if message.replies and message.replies.replies is not None
                        else 0
                    )

                    # reply_to_external_id: Telegram message ID of replied-to message
                    reply_to_id = getattr(message, "reply_to_msg_id", None)
                    if reply_to_id is None and getattr(message, "reply_to", None):
                        reply_to_id = getattr(message.reply_to, "reply_to_msg_id", None)
                    reply_to_external_id = str(reply_to_id) if reply_to_id else None

                    # forwarded_from_hash: hashed sender/channel ID of forwarded source
                    fwd_from = getattr(message, "fwd_from", None)
                    fwd_source = None
                    if fwd_from:
                        from_id = getattr(fwd_from, "from_id", None)
                        if from_id:
                            fwd_source = (
                                getattr(from_id, "user_id", None)
                                or getattr(from_id, "channel_id", None)
                                or getattr(from_id, "chat_id", None)
                            )
                        if not fwd_source:
                            fwd_source = getattr(fwd_from, "channel_id", None) or getattr(fwd_from, "from_name", None)
                    forwarded_from_hash = _hash_id(fwd_source) if fwd_source else None

                    author_hash = _hash_id(message.sender_id) or "unknown"

                    posts.append({
                        "tracker_id": tracker_id,
                        "platform": "telegram",
                        "external_id": str(message.id),
                        "author_hash": author_hash,
                        "body": message.text,
                        "lang": _detect_lang(message.text),
                        "posted_at": msg_date.isoformat(),
                        "like_count": like_count,
                        "reply_count": reply_count,
                        "share_count": message.forwards or 0,
                        "reply_to_external_id": reply_to_external_id,
                        "forwarded_from_hash": forwarded_from_hash,
                    })
                    page_count += 1

                except:  # noqa: E722 – intentionally catch all exceptions per original logic
                    logger.exception(
                        "Failed processing message id=%s from '%s':",
                        getattr(message, "id", "?"),
                        channel,
                    )

                # If we've hit the per-cycle page limit, prepare for the next page.
                if page_count >= max_fetch:
                    # Remember the oldest message id of this page for offset.
                    next_offset_id = message.id
                    break

            # End of async for loop for this page.
            if reached_end:
                break
            if page_count < max_fetch:
                # Fewer than max_fetch means channel history is fully exhausted.
                reached_end = True
                break
            if not last_sync_dt:
                # On first run (no last_sync_dt), do not paginate past first_run_limit.
                reached_end = True
                break

        logger.info(
            "iter_messages finished for channel='%s': pages=%d, raw_received=%d, non_message=%d, empty_text=%d, cutoff_skipped=%d, posts_collected=%d",
            channel, page, raw_msg_count, skipped_non_message, skipped_empty_text, skipped_date_cutoff, len(posts),
        )

        # Log final pagination outcome.
        if reached_end:
            logger.info(
                "Tracker %s fully caught up this cycle (pages=%d, oldest_fetched_msg_date=%s, last_sync_dt=%s)",
                tracker_id,
                page,
                oldest_fetched_date.isoformat() if oldest_fetched_date else "N/A",
                last_sync_dt.isoformat() if last_sync_dt else "None",
            )
        else:
            logger.warning(
                "Tracker %s still has unfetched backlog past this cycle's limit (reached %d pages, oldest_fetched_msg_date=%s, last_sync_dt=%s)",
                tracker_id,
                page,
                oldest_fetched_date.isoformat() if oldest_fetched_date else "N/A",
                last_sync_dt.isoformat() if last_sync_dt else "None",
            )

        # Fetch comments for recent channel posts if discussions are enabled
        parent_ids = [
            int(p["external_id"])
            for p in posts
            if p.get("external_id") and p["external_id"].isdigit()
        ][:settings.comment_posts_per_cycle]

        if len(parent_ids) < settings.comment_posts_per_cycle:
            try:
                async for m in client.iter_messages(channel, limit=settings.comment_posts_per_cycle):
                    if m.id not in parent_ids:
                        parent_ids.append(m.id)
                    if len(parent_ids) >= settings.comment_posts_per_cycle:
                        break
            except Exception as exc:
                logger.debug("Could not inspect extra posts for comments on '%s': %s", channel, exc)

        if parent_ids:
            comments = await fetch_comments_for_posts(
                client=client,
                channel=channel,
                tracker_id=tracker_id,
                parent_post_ids=parent_ids,
                hash_id_fn=_hash_id,
                detect_lang_fn=_detect_lang,
            )
            posts.extend(comments)

    except Exception:
        logger.exception(
            "Error fetching messages from channel '%s' (tracker %s):",
            channel, tracker_id,
        )
    finally:
        await client.disconnect()

    return posts
