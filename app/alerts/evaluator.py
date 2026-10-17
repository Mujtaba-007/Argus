# alerts/evaluator.py
# Evaluates active alert rules after analytics runs without server-side push.
# Secrets only via config.py; never logs keys or message text.

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.db.supabase_client import supabase

logger = logging.getLogger(__name__)

# In-memory top-5 pagerank cache per tracker: {tracker_id: list[str]}
_PREV_TOP5_INFLUENCERS: dict[str, list[str]] = {}


def evaluate_alerts_for_tracker(tracker_id: str) -> dict[str, int]:
    """
    Evaluates all active alert rules for a tracker.
    Returns {"checked": checked_count, "triggered": triggered_count}.
    """
    now = datetime.now(timezone.utc)
    cooldown_cutoff = now - timedelta(minutes=60)

    checked = 0
    triggered = 0

    page_size = 500
    start = 0
    alerts: list[dict[str, Any]] = []

    while True:
        res = (
            supabase.table("alerts")
            .select("*")
            .eq("tracker_id", tracker_id)
            .eq("active", True)
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = res.data or []
        alerts.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    if not alerts:
        logger.debug("Tracker %s: no active alerts found.", tracker_id)
        return {"checked": 0, "triggered": 0}

    # Fetch tracker query name for short message formatting
    tracker_name = "Tracker"
    try:
        t_res = (
            supabase.table("trackers")
            .select("query")
            .eq("id", tracker_id)
            .limit(1)
            .execute()
        )
        if t_res.data:
            q = (t_res.data[0].get("query") or "").strip().lstrip("@")
            if q:
                tracker_name = q
    except Exception as exc:
        logger.warning("Failed to lookup tracker name for %s: %s", tracker_id, exc)

    # Pre-fetch evaluation data once per tracker if needed
    types_present = {a.get("type") for a in alerts}

    shift_data = None
    if "sentiment_shift" in types_present:
        shift_data = _calc_sentiment_shift(tracker_id, now)

    trend_rows = None
    if "trend_spike" in types_present:
        trend_rows = _get_rising_trends(tracker_id)

    curr_top5 = None
    if "new_influencer" in types_present:
        curr_top5 = _get_top5_influencers(tracker_id)

    for alert in alerts:
        checked += 1
        last_trig = alert.get("last_triggered_at")
        if last_trig:
            try:
                dt_trig = datetime.fromisoformat(last_trig)
                if dt_trig.tzinfo is None:
                    dt_trig = dt_trig.replace(tzinfo=timezone.utc)
                if dt_trig > cooldown_cutoff:
                    continue  # within 60 min cooldown
            except Exception:
                pass

        alert_type = alert.get("type")
        threshold = float(alert.get("threshold") or 0.0)
        trigger_msg: str | None = None

        if alert_type == "sentiment_shift" and shift_data:
            # compare share of 'against' in last 3h vs previous 3h
            # trigger if increase >= threshold and last-3h sample >= 20 posts
            sample_count = shift_data["last_3h_count"]
            diff_pp = shift_data["diff_pp"]
            if sample_count >= 20 and diff_pp >= threshold:
                trigger_msg = f"Opposition up {round(diff_pp)} points in {tracker_name}"

        elif alert_type == "trend_spike" and trend_rows is not None:
            # trigger if any trend row has predicted_rising=True and volume >= threshold
            for tr in trend_rows:
                vol = int(tr.get("volume") or 0)
                if tr.get("predicted_rising") and vol >= threshold:
                    term = tr.get("term") or "topic"
                    trigger_msg = f"Trend spike: '{term}' volume {vol} in {tracker_name}"
                    break

        elif alert_type == "new_influencer" and curr_top5 is not None:
            # trigger if a node in current top 5 by pagerank was not in previous top 5
            prev_top5 = _PREV_TOP5_INFLUENCERS.get(tracker_id)
            if prev_top5 is not None:
                new_nodes = [node for node in curr_top5 if node not in prev_top5]
                if new_nodes:
                    trigger_msg = f"New top influencer detected in {tracker_name}"
            # Update cache after evaluation
            _PREV_TOP5_INFLUENCERS[tracker_id] = list(curr_top5)

        if trigger_msg:
            try:
                _write_alert_event(
                    alert_id=alert["id"],
                    tracker_id=tracker_id,
                    owner_id=alert.get("owner_id"),
                    alert_type=alert_type,
                    message=trigger_msg,
                    now_iso=now.isoformat(),
                )
                triggered += 1
            except Exception as exc:
                logger.error("Failed writing alert_event for alert %s: %s", alert.get("id"), exc)

    logger.info("Tracker %s alerts evaluated: checked=%d, triggered=%d", tracker_id, checked, triggered)
    return {"checked": checked, "triggered": triggered}


def _calc_sentiment_shift(tracker_id: str, now: datetime) -> dict[str, Any]:
    t0 = now.isoformat()
    t3 = (now - timedelta(hours=3)).isoformat()
    t6 = (now - timedelta(hours=6)).isoformat()

    # Paged query for sentiments + posts from last 6 hours
    page_size = 1000
    start = 0
    rows: list[dict[str, Any]] = []
    while True:
        res = (
            supabase.table("sentiments")
            .select("polarity, confidence, posts!inner(posted_at)")
            .eq("tracker_id", tracker_id)
            .gte("posts.posted_at", t6)
            .lte("posts.posted_at", t0)
            .gt("confidence", 0)
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = res.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    last_3h_total = 0
    last_3h_against = 0
    prev_3h_total = 0
    prev_3h_against = 0

    for r in rows:
        post = r.get("posts") or {}
        p_at = post.get("posted_at")
        pol = str(r.get("polarity") or "").lower()
        if not p_at:
            continue
        if p_at >= t3:
            last_3h_total += 1
            if pol == "against":
                last_3h_against += 1
        else:
            prev_3h_total += 1
            if pol == "against":
                prev_3h_against += 1

    last_share = (last_3h_against / last_3h_total * 100.0) if last_3h_total > 0 else 0.0
    prev_share = (prev_3h_against / prev_3h_total * 100.0) if prev_3h_total > 0 else 0.0
    diff_pp = last_share - prev_share

    return {
        "last_3h_count": last_3h_total,
        "diff_pp": diff_pp,
    }


def _get_rising_trends(tracker_id: str) -> list[dict[str, Any]]:
    try:
        res = (
            supabase.table("trends")
            .select("term, volume, predicted_rising")
            .eq("tracker_id", tracker_id)
            .eq("predicted_rising", True)
            .order("volume", desc=True)
            .limit(50)
            .execute()
        )
        return res.data or []
    except Exception as exc:
        logger.warning("Failed to fetch trends for alert eval %s: %s", tracker_id, exc)
        return []


def _get_top5_influencers(tracker_id: str) -> list[str]:
    try:
        res = (
            supabase.table("network_nodes")
            .select("author_hash")
            .eq("tracker_id", tracker_id)
            .order("pagerank", desc=True)
            .limit(5)
            .execute()
        )
        nodes = [r["author_hash"] for r in (res.data or []) if r.get("author_hash")]
        if tracker_id not in _PREV_TOP5_INFLUENCERS:
            # Seed cache on first run without triggering false alarm
            _PREV_TOP5_INFLUENCERS[tracker_id] = list(nodes)
        return nodes
    except Exception as exc:
        logger.warning("Failed to fetch network nodes for alert eval %s: %s", tracker_id, exc)
        return []


def _write_alert_event(
    alert_id: str,
    tracker_id: str,
    owner_id: str | None,
    alert_type: str,
    message: str,
    now_iso: str,
) -> None:
    event_payload = {
        "alert_id": alert_id,
        "tracker_id": tracker_id,
        "owner_id": owner_id,
        "type": alert_type,
        "message": message,
        "created_at": now_iso,
    }
    supabase.table("alert_events").insert(event_payload).execute()
    supabase.table("alerts").update({"last_triggered_at": now_iso}).eq("id", alert_id).execute()
