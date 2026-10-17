# ingestion/client.py
# Factory helper to create a Telethon TelegramClient instance.
# Supports both session strings (for cloud/stateless environments) and session files.
# Never log or print the session string or any credentials.

from telethon import TelegramClient
from telethon.sessions import StringSession

from app.config import settings


def get_telegram_client() -> TelegramClient:
    """
    Creates and returns a TelegramClient configured with either:
    - StringSession(settings.telegram_session_string) if TELEGRAM_SESSION_STRING is provided.
    - File-based session 'argus_session' (.session file in working directory) otherwise.
    """
    session = (
        StringSession(settings.telegram_session_string)
        if settings.telegram_session_string
        else "argus_session"
    )
    return TelegramClient(
        session,
        settings.telegram_api_id,
        settings.telegram_api_hash,
    )
