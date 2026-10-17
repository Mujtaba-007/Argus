# analytics/demographics.py
# Demographic Intelligence: infer aggregate, anonymized audience profiles.
# All inference uses text signals only (no device data, no personal info).
# Results are aggregated; individuals are NEVER exposed.
# Estimates are clearly labeled — not ground truth.

import json
import logging
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.db.analytics_repository import (
    get_all_author_profiles,
    get_author_sample_posts,
    get_authors_needing_profile,
    replace_demographics_agg,
    upsert_author_profiles_batch,
)
from app.nlp.gemini_client import generate_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Script-based language detection (no external library needed)
# ---------------------------------------------------------------------------

# Unicode block ranges for Indic scripts
_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0900, 0x097F, "Hindi"),       # Devanagari
    (0x0980, 0x09FF, "Bengali"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0A00, 0x0A7F, "Punjabi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Odia"),
]

# Common Hindi/Urdu words written in Latin (Hinglish markers)
_HINGLISH_WORDS: frozenset[str] = frozenset([
    "hai", "hain", "nahi", "nhi", "kya", "kyun", "kaise", "kab", "kaun",
    "aur", "bhi", "toh", "yeh", "woh", "mein", "se", "pe", "par", "ka",
    "ki", "ke", "ko", "ek", "do", "bhai", "yaar", "sab", "log", "ho",
    "hoga", "gaya", "tha", "thi", "abhi", "phir", "acha", "theek", "bilkul",
])
_HINGLISH_THRESHOLD = 0.08  # fraction of words that are Hinglish markers


def _detect_language(texts: list[str]) -> str:
    """Detect dominant language from a list of text samples."""
    script_hits: Counter[str] = Counter()
    latin_words = 0
    hinglish_hits = 0
    total_chars = 0

    for text in texts:
        for ch in text:
            cp = ord(ch)
            total_chars += 1
            for lo, hi, lang in _SCRIPT_RANGES:
                if lo <= cp <= hi:
                    script_hits[lang] += 1
                    break
        words = re.findall(r"[a-zA-Z]+", text.lower())
        latin_words += len(words)
        hinglish_hits += sum(1 for w in words if w in _HINGLISH_WORDS)

    if script_hits:
        dominant = script_hits.most_common(1)[0]
        if dominant[1] / max(total_chars, 1) > 0.1:
            return dominant[0]

    if latin_words > 0 and hinglish_hits / latin_words >= _HINGLISH_THRESHOLD:
        return "Hinglish"
    if latin_words > 0:
        return "English"
    return "Unknown"


# ---------------------------------------------------------------------------
# Region detection: Indian states / major cities / railway zones
# ---------------------------------------------------------------------------

# Map: lowercase name -> state
_REGION_GAZETTEER: dict[str, str] = {
    # Indian States & Union Territories
    "andhra pradesh": "Andhra Pradesh", "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam", "bihar": "Bihar", "chhattisgarh": "Chhattisgarh",
    "goa": "Goa", "gujarat": "Gujarat", "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh", "jharkhand": "Jharkhand",
    "karnataka": "Karnataka", "kerala": "Kerala", "madhya pradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra", "manipur": "Manipur", "meghalaya": "Meghalaya",
    "mizoram": "Mizoram", "nagaland": "Nagaland", "odisha": "Odisha",
    "orissa": "Odisha", "punjab": "Punjab", "rajasthan": "Rajasthan",
    "sikkim": "Sikkim", "tamil nadu": "Tamil Nadu", "telangana": "Telangana",
    "tripura": "Tripura", "uttar pradesh": "Uttar Pradesh", "uttarakhand": "Uttarakhand",
    "west bengal": "West Bengal", "delhi": "Delhi", "new delhi": "Delhi",
    "jammu": "Jammu & Kashmir", "kashmir": "Jammu & Kashmir", "chandigarh": "Chandigarh",
    "puducherry": "Puducherry", "pondicherry": "Puducherry",

    # Railway Zone HQs & Cities explicitly specified
    "prayagraj": "Uttar Pradesh", "allahabad": "Uttar Pradesh", "prayagraj zone": "Uttar Pradesh",
    "gorakhpur": "Uttar Pradesh", "gorakhpur zone": "Uttar Pradesh",
    "kolkata": "West Bengal", "calcutta": "West Bengal", "kolkata zone": "West Bengal",
    "ahmedabad": "Gujarat", "ahmedabad zone": "Gujarat",
    "jaipur": "Rajasthan", "jaipur zone": "Rajasthan",
    "chennai": "Tamil Nadu", "madras": "Tamil Nadu", "chennai zone": "Tamil Nadu",
    "mumbai": "Maharashtra", "bombay": "Maharashtra", "mumbai zone": "Maharashtra",
    "bhopal": "Madhya Pradesh", "bhopal zone": "Madhya Pradesh",
    "patna": "Bihar", "patna zone": "Bihar",
    "lucknow": "Uttar Pradesh", "lucknow zone": "Uttar Pradesh",
    "delhi zone": "Delhi",
    "bengaluru": "Karnataka", "bangalore": "Karnataka",
    "bengaluru zone": "Karnataka", "bangalore zone": "Karnataka",
    "hyderabad": "Telangana", "hyderabad zone": "Telangana",
    "secunderabad": "Telangana", "secunderabad zone": "Telangana",
    "guwahati": "Assam", "gauhati": "Assam", "guwahati zone": "Assam",
    "ranchi": "Jharkhand", "ranchi zone": "Jharkhand",
    "bhubaneswar": "Odisha", "bhubaneshwar": "Odisha", "bhubaneswar zone": "Odisha",

    # Other Major Indian Cities
    "pune": "Maharashtra", "nagpur": "Maharashtra", "mysore": "Karnataka",
    "coimbatore": "Tamil Nadu", "surat": "Gujarat", "vadodara": "Gujarat",
    "jodhpur": "Rajasthan", "udaipur": "Rajasthan", "kota": "Rajasthan",
    "kanpur": "Uttar Pradesh", "agra": "Uttar Pradesh", "varanasi": "Uttar Pradesh",
    "noida": "Uttar Pradesh", "meerut": "Uttar Pradesh", "ghaziabad": "Uttar Pradesh",
    "indore": "Madhya Pradesh", "gwalior": "Madhya Pradesh", "jabalpur": "Madhya Pradesh",
    "raipur": "Chhattisgarh", "bilaspur": "Chhattisgarh", "bilaspur zone": "Chhattisgarh",
    "dhanbad": "Jharkhand", "jamshedpur": "Jharkhand", "gaya": "Bihar",
    "muzaffarpur": "Bihar", "dehradun": "Uttarakhand", "amritsar": "Punjab",
    "ludhiana": "Punjab", "kochi": "Kerala", "thiruvananthapuram": "Kerala",
    "kozhikode": "Kerala", "visakhapatnam": "Andhra Pradesh", "vijayawada": "Andhra Pradesh",
    "gurgaon": "Haryana", "gurugram": "Haryana", "faridabad": "Haryana",
    "srinagar": "Jammu & Kashmir", "siliguri": "West Bengal", "asansol": "West Bengal",
    "howrah": "West Bengal", "malda": "West Bengal", "ajmer": "Rajasthan",
    "hubli": "Karnataka", "hubballi": "Karnataka",

    # Railway Zones by Name
    "northern railway": "Uttar Pradesh", "southern railway": "Tamil Nadu",
    "eastern railway": "West Bengal", "western railway": "Maharashtra",
    "central railway": "Maharashtra", "north eastern railway": "Uttar Pradesh",
    "south eastern railway": "West Bengal", "south central railway": "Telangana",
    "north central railway": "Uttar Pradesh", "east central railway": "Bihar",
    "east coast railway": "Odisha", "north western railway": "Rajasthan",
    "south western railway": "Karnataka", "west central railway": "Madhya Pradesh",
    "northeast frontier railway": "Assam", "south east central railway": "Chhattisgarh",
}


def _detect_region(texts: list[str]) -> str | None:
    """Return the most frequently mentioned Indian state inferred from text."""
    combined = " ".join(texts).lower()
    hits: Counter[str] = Counter()
    for place, state in _REGION_GAZETTEER.items():
        count = len(re.findall(rf"\b{re.escape(place)}\b", combined))
        if count > 0:
            hits[state] += count
    if hits:
        return hits.most_common(1)[0][0]
    return None


# ---------------------------------------------------------------------------
# Age bracket + interest via Gemini (batched ~15 authors per request)
# ---------------------------------------------------------------------------

_GEMINI_DEMO_BATCH = 15
_AGE_BRACKETS = {"18-24", "25-34", "35-44", "45+", "unknown"}
_INTERESTS = {
    "exam prep", "news and politics", "finance", "technology",
    "entertainment", "sports", "education", "other",
}

_DEMO_PROMPT_TEMPLATE = """
You are a conservative, privacy-preserving text analyst.
Analyze these {n} anonymized author text samples and return ONLY a JSON array.
For each author:
- age_bracket: writing-style estimate. One of: "18-24","25-34","35-44","45+","unknown".
  If ANY sign of a minor (under 18) appears, return "unknown". Treat as rough estimate.
- interest: dominant topic. One of: "exam prep","news and politics","finance",
  "technology","entertainment","sports","education","other".
- confidence: float 0.0-1.0 reflecting how certain you are.
Do NOT include any text, names, or raw content from the samples in the output.
Return ONLY a JSON array of {n} objects in order, no markdown, no explanation:
[{{"age_bracket":"...","interest":"...","confidence":0.0}}, ...]

Samples:
{samples}
"""


def _call_gemini_demo_batch(
    author_hashes: list[str], sample_texts: list[list[str]]
) -> list[dict[str, Any]]:
    """Call Gemini for age/interest inference. Returns list aligned with input."""
    n = len(author_hashes)
    samples_block = ""
    for i, texts in enumerate(sample_texts):
        # Truncate each message to 120 chars to keep prompt size manageable
        truncated = [t[:120] for t in texts[:5]]
        samples_block += f"Author {i+1}:\n" + "\n".join(truncated) + "\n\n"

    prompt = _DEMO_PROMPT_TEMPLATE.format(n=n, samples=samples_block)
    try:
        raw = generate_text(prompt)
        # Strip markdown fences if present
        raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or len(parsed) != n:
            logger.warning("Gemini demographics: unexpected response shape")
            return [{} for _ in range(n)]
        results = []
        for item in parsed:
            age = item.get("age_bracket", "unknown")
            interest = item.get("interest", "other")
            conf = float(item.get("confidence", 0.5))
            results.append({
                "age_bracket": age if age in _AGE_BRACKETS else "unknown",
                "interest": interest if interest in _INTERESTS else "other",
                "confidence": round(min(max(conf, 0.0), 1.0), 3),
            })
        return results
    except Exception as exc:
        logger.warning("Gemini demographics batch failed: %s", type(exc).__name__)
        return [{} for _ in range(n)]


# ---------------------------------------------------------------------------
# Aggregation with privacy rules
# ---------------------------------------------------------------------------

def _aggregate(
    tracker_id: str,
    profiles: list[dict[str, Any]],
    period_start: str,
    min_group: int,
) -> list[dict[str, Any]]:
    """Build demographics_agg rows from profiled authors."""
    conf_threshold = 0.5
    dimensions: dict[str, str] = {
        "language": "language",
        "region": "region",
        "age_bracket": "age",
        "interest": "interest",
    }
    rows: list[dict[str, Any]] = []
    merged_count = 0

    for field, dim_name in dimensions.items():
        bucket_counts: Counter[str] = Counter()
        conf_sum: dict[str, float] = {}
        sample_size = 0
        for p in profiles:
            val = p.get(field)
            conf = float(p.get("confidence", 1.0))
            if not val or val.lower() in ("unknown", "none", ""):
                continue
            if conf < conf_threshold:
                continue
            bucket_counts[val] += 1
            conf_sum[val] = conf_sum.get(val, 0.0) + conf
            sample_size += 1

        if not bucket_counts:
            continue

        total = sum(bucket_counts.values())
        merged_into_other = 0
        other_conf = 0.0

        for bucket, count in bucket_counts.items():
            if count < min_group:
                merged_into_other += count
                other_conf += conf_sum.get(bucket, 0.0)
                merged_count += 1
            else:
                rows.append({
                    "tracker_id": tracker_id,
                    "dimension": dim_name,
                    "bucket": bucket,
                    "user_count": count,
                    "share": round(count / total, 4),
                    "sample_size": sample_size,
                    "avg_confidence": round(conf_sum.get(bucket, 0.0) / count, 3),
                    "period_start": period_start,
                })

        if merged_into_other >= min_group:
            rows.append({
                "tracker_id": tracker_id,
                "dimension": dim_name,
                "bucket": "Other (small groups)",
                "user_count": merged_into_other,
                "share": round(merged_into_other / total, 4),
                "sample_size": sample_size,
                "avg_confidence": round(other_conf / merged_into_other, 3) if merged_into_other else 0.0,
                "period_start": period_start,
            })

    logger.info(
        "Demographics agg for %s: %d rows written, %d buckets merged for privacy.",
        tracker_id, len(rows), merged_count,
    )
    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_demographics(tracker_id: str) -> dict[str, Any]:
    """
    Profile up to DEMO_AUTHORS_PER_RUN authors for a tracker and aggregate.
    Returns counts of authors profiled and buckets written.
    """
    limit = settings.demo_authors_per_run
    min_group = settings.min_group_size
    period_start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()

    # Step 1: pick authors needing profile
    candidates = get_authors_needing_profile(tracker_id, limit)
    if not candidates:
        logger.info("Demographics %s: no authors needing profile.", tracker_id)
        all_existing = get_all_author_profiles(tracker_id)
        if all_existing:
            agg_rows = _aggregate(tracker_id, all_existing, period_start, min_group)
            replace_demographics_agg(tracker_id, agg_rows)
            return {"authors_profiled": 0, "buckets_written": len(agg_rows)}
        return {"authors_profiled": 0, "buckets_written": 0}

    # Step 2: build per-author profiles
    hashes = [c["author_hash"] for c in candidates]

    # Fetch sample posts for all authors
    author_texts: dict[str, list[str]] = {}
    for h in hashes:
        texts = get_author_sample_posts(tracker_id, h, n=10)
        if texts:
            author_texts[h] = texts

    # Local inference: language + region (deterministic, no external calls)
    local_profiles: dict[str, dict[str, Any]] = {}
    matched_region_count = 0
    for h, texts in author_texts.items():
        lang = _detect_language(texts)
        region = _detect_region(texts)
        if region:
            matched_region_count += 1
        local_profiles[h] = {
            "tracker_id": tracker_id,
            "author_hash": h,
            "language": lang,
            "region": region,
            "age_bracket": None,
            "interest": None,
            "confidence": 1.0,  # local rules are deterministic
            "updated_at": period_start,
        }

    logger.info(
        "Tracker %s demographics region detection: %d/%d authors matched the gazetteer.",
        tracker_id, matched_region_count, len(author_texts),
    )

    # Gemini inference: age + interest in batches of ~15
    rpm_delay = 60.0 / max(settings.gemini_max_rpm, 1)
    valid_hashes = list(author_texts.keys())

    for i in range(0, len(valid_hashes), _GEMINI_DEMO_BATCH):
        batch_hashes = valid_hashes[i: i + _GEMINI_DEMO_BATCH]
        batch_texts = [author_texts[h] for h in batch_hashes]
        results = _call_gemini_demo_batch(batch_hashes, batch_texts)
        for h, res in zip(batch_hashes, results):
            if res:  # empty dict means Gemini failed for this batch
                local_profiles[h]["age_bracket"] = res.get("age_bracket")
                local_profiles[h]["interest"] = res.get("interest")
                # Blend confidence: local rules 1.0, Gemini result also factored
                local_profiles[h]["confidence"] = round(
                    (1.0 + res.get("confidence", 0.5)) / 2.0, 3
                )
        if i + _GEMINI_DEMO_BATCH < len(valid_hashes):
            time.sleep(rpm_delay)  # respect GEMINI_MAX_RPM

    profiled = list(local_profiles.values())

    # Step 3: upsert author_profiles (hashes only, no raw text)
    try:
        upsert_author_profiles_batch(profiled)
    except Exception as exc:
        logger.error("Demographics %s: upsert_author_profiles failed — %s", tracker_id, exc)
        return {"authors_profiled": 0, "buckets_written": 0, "error": str(exc)}

    # Step 4: aggregate with privacy rules using all profiled authors for this tracker
    try:
        all_tracker_profiles = get_all_author_profiles(tracker_id)
    except Exception as exc:
        logger.warning(
            "Tracker %s: get_all_author_profiles failed (%s), aggregating current run only.",
            tracker_id, exc,
        )
        all_tracker_profiles = profiled

    agg_rows = _aggregate(tracker_id, all_tracker_profiles or profiled, period_start, min_group)

    try:
        replace_demographics_agg(tracker_id, agg_rows)
    except Exception as exc:
        logger.error("Demographics %s: replace_demographics_agg failed — %s", tracker_id, exc)
        return {"authors_profiled": len(profiled), "buckets_written": 0, "error": str(exc)}

    logger.info(
        "Demographics %s: profiled %d authors, wrote %d agg buckets.",
        tracker_id, len(profiled), len(agg_rows),
    )
    return {"authors_profiled": len(profiled), "buckets_written": len(agg_rows)}
