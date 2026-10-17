# config.py
# Loads and validates all required environment variables using pydantic-settings.
# If any required variable is missing, the app will refuse to start (fail fast).
# Never log or print any of these values — they are secrets.

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All environment variables the service needs.
    pydantic-settings reads them from the .env file automatically.
    If a variable is missing, it raises a clear error at startup.
    """

    # Telegram API credentials (get from https://my.telegram.org)
    telegram_api_id: int
    telegram_api_hash: str
    telegram_session_string: str | None = None

    # Google Gemini API key (for sentiment analysis)
    gemini_api_key: str
    # X (Twitter) API bearer token for ingestion
    X_BEARER_TOKEN: str
    # X API credentials
    x_api_key: str = ""
    x_api_secret: str = ""

    # Supabase connection details
    supabase_url: str
    supabase_service_role_key: str  # Server-side only — never expose to clients

    # Ingestion & Gemini rate limits / batching
    first_run_limit: int = 30
    max_per_cycle: int = 50

    # Gemini model & rate limits
    # Model: gemini-2.0-flash-lite — fast, cheap, ideal for high-volume classification
    gemini_model: str = "gemini-2.0-flash-lite"
    gemini_batch_size: int = 20          # Posts per Gemini call (raised from 10)
    gemini_max_rpm: int = 10             # Hard cap: max Gemini requests per minute
    gemini_calls_per_minute: int = 10    # Alias kept for rate limiter (= gemini_max_rpm)
    gemini_concurrency: int = 4          # Max parallel in-flight Gemini calls

    # Reanalyze endpoint: max posts to process per call
    reanalyze_limit: int = 200

    # Trend analytics configuration
    trend_window_hours: int = 3
    trend_lookback_hours: int = 48

    # Comment ingestion configuration
    comment_posts_per_cycle: int = 10
    comments_per_post: int = 100

    # Demographics settings
    demo_authors_per_run: int = 150    # Max authors profiled per tracker per run
    min_group_size: int = 10           # Minimum bucket size before merging into "Other (small groups)"

    model_config = SettingsConfigDict(
        env_file=".env",          # Read from .env in the working directory
        env_file_encoding="utf-8",
        case_sensitive=False,     # TELEGRAM_API_ID and telegram_api_id both work
    )


# One shared instance — import this wherever you need config values.
# Example: from app.config import settings; use settings.supabase_url
settings = Settings()
