# db/supabase_client.py
# Creates ONE shared Supabase client using the service-role key.
# The service-role key bypasses Row Level Security — keep it server-side only.
# Import `supabase` from this module wherever you need to query the database.

import logging
from supabase import create_client, Client
from app.config import settings

logger = logging.getLogger(__name__)

def _create_supabase_client() -> Client:
    """
    Initialise and return a Supabase client.
    Called once at import time; the result is cached in the module-level `supabase` variable.
    """
    try:
        client: Client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
        logger.info("Supabase client initialised successfully.")
        return client
    except Exception as exc:
        # If this fails, nothing else can work — re-raise immediately.
        logger.critical("Failed to create Supabase client: %s", exc)
        raise


# Module-level singleton — imported by repository.py
supabase: Client = _create_supabase_client()
