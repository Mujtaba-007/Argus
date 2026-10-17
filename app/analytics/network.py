# analytics/network.py
# Network & influence analysis for social intelligence.
# Builds a directed graph from interactions (last 7 days), computes PageRank
# and Louvain communities, and saves the top 200 nodes and their edges.

import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

import networkx as nx
from networkx.algorithms.community import louvain_communities

from app.db.analytics_repository import (
    get_interactions_for_network,
    upsert_network_edges,
    upsert_network_nodes,
)

logger = logging.getLogger(__name__)


def analyze_network(tracker_id: str) -> dict[str, int]:
    """
    Compute and store PageRank and Louvain communities for a tracker.
    Returns {"nodes": count, "edges": count}.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    try:
        interactions = get_interactions_for_network(tracker_id, cutoff)
    except Exception as exc:
        logger.error("Tracker %s: failed to fetch interactions: %s", tracker_id, exc)
        return {"nodes": 0, "edges": 0}

    # Count interaction frequencies as directed edge weights
    edge_counts: Counter[tuple[str, str]] = Counter()
    for item in interactions:
        s = item.get("source_hash")
        t = item.get("target_hash")
        if s and t and s != "unknown" and t != "unknown":
            edge_counts[(s, t)] += 1

    # Skip trackers with insufficient network structure
    if len(edge_counts) < 5:
        logger.info(
            "Tracker %s: fewer than 5 edges (%d found); skipping network analysis.",
            tracker_id,
            len(edge_counts),
        )
        return {"nodes": 0, "edges": 0}

    # Construct directed graph
    g = nx.DiGraph()
    for (s, t), weight in edge_counts.items():
        g.add_edge(s, t, weight=weight)

    # Compute PageRank (alpha=0.85)
    try:
        pr = nx.pagerank(g, alpha=0.85, weight="weight")
    except Exception as exc:
        logger.error("Tracker %s: PageRank computation failed: %s", tracker_id, exc)
        return {"nodes": 0, "edges": 0}

    # Compute Louvain communities on undirected projection with fixed seed
    g_undirected = g.to_undirected()
    try:
        communities = louvain_communities(g_undirected, weight="weight", seed=42)
    except Exception as exc:
        logger.warning("Tracker %s: Louvain community detection failed (%s); using single community.", tracker_id, exc)
        communities = [set(g.nodes())]

    node_community: dict[str, int] = {}
    for comm_id, members in enumerate(communities):
        for node in members:
            node_community[node] = comm_id

    # Retain top 200 nodes by PageRank to keep the mobile graph performant
    sorted_nodes = sorted(g.nodes(), key=lambda n: pr.get(n, 0.0), reverse=True)[:200]
    top_nodes_set = set(sorted_nodes)

    nodes_payload = [
        {
            "tracker_id": tracker_id,
            "author_hash": node,
            "pagerank": round(float(pr.get(node, 0.0)), 6),
            "community_id": int(node_community.get(node, 0)),
        }
        for node in sorted_nodes
    ]

    edges_payload = [
        {
            "tracker_id": tracker_id,
            "source_hash": u,
            "target_hash": v,
            "weight": int(data.get("weight", 1)),
        }
        for u, v, data in g.edges(data=True)
        if u in top_nodes_set and v in top_nodes_set
    ]

    try:
        upsert_network_nodes(nodes_payload)
        upsert_network_edges(edges_payload)
        logger.info(
            "Tracker %s: upserted %d network nodes and %d edges.",
            tracker_id, len(nodes_payload), len(edges_payload),
        )
        return {"nodes": len(nodes_payload), "edges": len(edges_payload)}
    except Exception as exc:
        logger.error("Tracker %s: failed to save network data to DB: %s", tracker_id, exc)
        return {"nodes": 0, "edges": 0}
