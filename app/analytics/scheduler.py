# analytics/scheduler.py
# Coordinates analytics pipelines (trends & network) for a tracker.
# Called after ingestion in run_cycle.py and on-demand via POST /run-analytics.

import logging
from typing import Any

from app.analytics.demographics import analyze_demographics
from app.analytics.network import analyze_network

logger = logging.getLogger(__name__)


def run_analytics(tracker_id: str) -> dict[str, Any]:
    """
    Execute trend detection and network influence analysis for a tracker.
    Returns metrics from trends (posts, windows, terms found/written) and network.
    """
    logger.info("Starting analytics pipeline for tracker %s.", tracker_id)
    summary: dict[str, Any] = {
        "tracker_id": tracker_id,
        "network_nodes": 0,
        "network_edges": 0,
        "demographics_authors": 0,
        "demographics_buckets": 0,
        "errors": [],
    }

    # NOTE: Trends analytics are NOT run here; they are handled in run_cycle.py
    # with a 30-minute per-tracker throttle (and via POST /run-analytics).

    # 2. Network & Influence Analysis
    try:
        net_res = analyze_network(tracker_id)
        summary["network_nodes"] = net_res.get("nodes", 0)
        summary["network_edges"] = net_res.get("edges", 0)
    except Exception as exc:
        err_msg = f"Network analysis failed for {tracker_id}: {exc}"
        logger.exception(err_msg)
        summary["errors"].append(err_msg)

    # 3. Demographic Intelligence (best-effort, does not block)
    try:
        demo_res = analyze_demographics(tracker_id)
        summary["demographics_authors"] = demo_res.get("authors_profiled", 0)
        summary["demographics_buckets"] = demo_res.get("buckets_written", 0)
    except Exception as exc:
        err_msg = f"Demographics analysis failed for {tracker_id}: {exc}"
        logger.exception(err_msg)
        summary["errors"].append(err_msg)

    # 4. Alert Rule Evaluation (best-effort, does not block)
    try:
        from app.alerts.evaluator import evaluate_alerts_for_tracker
        alert_res = evaluate_alerts_for_tracker(tracker_id)
        summary["alerts_checked"] = alert_res.get("checked", 0)
        summary["alerts_triggered"] = alert_res.get("triggered", 0)
    except Exception as exc:
        err_msg = f"Alerts evaluation failed for {tracker_id}: {exc}"
        logger.exception(err_msg)
        summary["errors"].append(err_msg)

    logger.info("Completed analytics pipeline for tracker %s: %s", tracker_id, summary)
    return summary
