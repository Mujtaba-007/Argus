# ingestion/comment_worker.py
# Fetches comments from discussion groups linked to channel posts.
# Respects Telegram flood limits and converts comments into standard post dicts.

import asyncio
import logging
from datetime import timezone
from typing import Any, Callable

from telethon import TelegramClient
from telethon.errors import FloodWaitError, MsgIdInvalidError, RPCError
from telethon.tl.types import Message

from app.config import settings
from app.ingestion.client import get_telegram_client

logger = logging.getLogger(__name__)


async def fetch_comments_for_posts(
    client: TelegramClient,
    channel: str,
    tracker_id: str,
    parent_post_ids: list[int],
    hash_id_fn: Callable[[Any], str | None],
    detect_lang_fn: Callable[[str], str],
) -> list[dict[str, Any]]:
    """
    Fetch comments for the given post IDs from the channel's linked discussion group.
    Handles flood waits, sleeps 1s between posts, and skips posts where comments are disabled.
    """
    comments: list[dict[str, Any]] = []

    for post_id in parent_post_ids:
        try:
            async for msg in client.iter_messages(
                channel,
                reply_to=post_id,
                limit=settings.comments_per_post,
            ):
                if not isinstance(msg, Message) or not msg.text:
                    continue

                msg_date = msg.date.replace(tzinfo=timezone.utc)

                # Determine reply target: either the parent post or another comment
                replied_id = getattr(msg, "reply_to_msg_id", None)
                if replied_id is None and getattr(msg, "reply_to", None):
                    replied_id = getattr(msg.reply_to, "reply_to_msg_id", None)

                if replied_id is None or replied_id == post_id:
                    reply_to_ext_id = str(post_id)
                else:
                    reply_to_ext_id = f"{post_id}_c{replied_id}"

                # Forward source hash
                fwd_from = getattr(msg, "fwd_from", None)
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

                like_cnt = sum(r.count for r in msg.reactions.results) if msg.reactions and msg.reactions.results else 0
                reply_cnt = msg.replies.replies if msg.replies and msg.replies.replies is not None else 0

                comments.append({
                    "tracker_id": tracker_id,
                    "platform": "telegram",
                    "external_id": f"{post_id}_c{msg.id}",
                    "author_hash": hash_id_fn(msg.sender_id) or "unknown",
                    "body": msg.text,
                    "lang": detect_lang_fn(msg.text),
                    "posted_at": msg_date.isoformat(),
                    "like_count": like_cnt,
                    "reply_count": reply_cnt,
                    "share_count": msg.forwards or 0,
                    "reply_to_external_id": reply_to_ext_id,
                    "forwarded_from_hash": hash_id_fn(fwd_source) if fwd_source else None,
                })

        except FloodWaitError as flood:
            logger.warning("FloodWaitError on '%s' post %d: sleeping %ds.", channel, post_id, flood.seconds)
            await asyncio.sleep(flood.seconds + 1)
        except (MsgIdInvalidError, RPCError) as rpc_err:
            logger.debug("Comments disabled or unavailable for post %d on '%s': %s", post_id, channel, rpc_err)
        except Exception as exc:
            logger.warning("Error fetching comments for post %d on '%s': %s", post_id, channel, exc)

        # Respect Telegram rate limits between posts
        await asyncio.sleep(1)

    logger.info("Tracker %s: fetched %d comments across %d post(s).", tracker_id, len(comments), len(parent_post_ids))
    return comments
