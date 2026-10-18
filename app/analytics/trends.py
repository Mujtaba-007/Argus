# analytics/trends.py
# Trend & topic detection for social intelligence across 24h, 7d, and 30d ranges.
# Anchored to the newest post timestamp in the analysis range.
# Extracts TF-IDF keywords, hashtags, and capitalized named entities.
#
# Performance notes:
#   - All three ranges are computed from ONE 30D DB fetch (filtered in memory).
#   - Each range is capped to the newest 5 000 posts before TF-IDF.
#   - Term volume per window is computed with a single vectorized matrix pass
#     instead of per-term regex loops.

import logging
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from app.db.analytics_repository import (
    delete_trends_for_tracker_and_range,
    get_posts_for_trends,
    insert_trends_batch,
)

logger = logging.getLogger(__name__)

def _normalize_word(word: str) -> str:
    """Trim trailing 's' for simple English plurals (e.g. funds -> fund), avoiding 'ss'."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


CUSTOM_STOP_WORDS = {
    # Web / platform noise
    "amp", "rt", "http", "https", "com", "co", "t", "telegram", "channel", "group", "zone",
    "live", "user", "id", "user_id",
    # Hindi / Hinglish stopwords & requested conversational noise
    "ka", "ki", "ke", "ko", "hai", "h", "hain", "kab", "kya", "kaise", "kese",
    "nahi", "nhi", "ho", "hoga", "gya", "to", "toh", "bhi", "me", "mein", "se",
    "ye", "yeh", "wo", "woh", "aur", "ya", "par", "pe", "na", "ek",
    "bhai", "bro", "sir", "please", "pls", "ok", "okay", "hi", "ha", "haan",
    "ji", "kr", "kar", "jo", "ab", "mera", "meri", "mere", "dekho", "abe", "saale",
    "tk", "tak", "lia", "liya", "liye", "respond", "tell", "got", "tha", "thi",
    "rha", "raha", "rhi", "rahi", "dikha", "dur", "itna", "city", "intimation", "diya",
    "koi", "aaya", "aaye", "aayi", "aya", "aye", "ayi", "kuch", "sab", "apna", "apni", "apne",
    "fir", "phir", "abhi", "aisa", "aise", "aisi",
    "apka", "mujhe", "kiya",
    "isliye", "pta", "kis", "sahi", "krta", "karta", "kisi",
}

# Combined set of all stop words (English + Custom), including normalized forms
ALL_STOP_WORDS = {
    w
    for sw in ENGLISH_STOP_WORDS.union(CUSTOM_STOP_WORDS)
    for w in (sw, _normalize_word(sw))
}
STOP_WORDS = list(ALL_STOP_WORDS)

# Range configurations: label, lookback_hours, window_hours
RANGE_CONFIGS = [
    ("24H", 24, 1),    # 24 hours lookback, 24 windows of 1 hour
    ("7D", 168, 24),   # 7 days lookback, 7 windows of 24 hours
    ("30D", 720, 72),  # 30 days lookback, 10 windows of 72 hours
]

_MAX_POSTS_PER_RANGE = 3_000


def _clean_text(text: str) -> str:
    """
    Lowercase text, strip URLs, numbers, non-alphanumeric characters,
    and normalize simple plurals by trimming trailing 's' (e.g. funds -> fund).
    """
    no_url = re.sub(r"https?://\S+|www\.\S+", " ", (text or "").lower())
    # Strip numbers and non-alphanumeric characters (keep only a-z and whitespace)
    alpha_only = re.sub(r"[^a-z\s]", " ", no_url)
    # Normalize trailing 's' on words > 3 chars
    words = [_normalize_word(w) for w in alpha_only.split()]
    return " ".join(words)


def _analyze_range(
    tracker_id: str,
    range_label: str,
    valid_posts: list[tuple[datetime, str]],
    lookback_hours: int,
    win_hours: int,
) -> list[dict[str, Any]]:
    """
    Compute trend metrics for a specific time range using vectorized counts.
    Caps to the newest 3 000 posts. Returns trend dicts with volume >= 3.
    """
    if not valid_posts:
        return []

    # Cap each range at 3 000 posts before TF-IDF
    if len(valid_posts) > _MAX_POSTS_PER_RANGE:
        valid_posts = valid_posts[:_MAX_POSTS_PER_RANGE]

    newest_post_dt = valid_posts[0][0]  # already sorted desc
    anchor_end = newest_post_dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    period_start_iso = (anchor_end - timedelta(hours=win_hours)).isoformat()

    num_windows = max(4, lookback_hours // win_hours)

    # Assign each post to its window index
    window_texts: list[list[str]] = [[] for _ in range(num_windows)]
    total_range_texts: list[str] = []

    for dt, body in valid_posts:
        diff_hours = (anchor_end - dt).total_seconds() / 3600.0
        cleaned = _clean_text(body)
        total_range_texts.append(cleaned)
        w_idx = max(0, int(diff_hours // win_hours))
        if 0 <= w_idx < num_windows:
            window_texts[w_idx].append(cleaned)

    if len(total_range_texts) < 3:
        return []

    # ── TF-IDF vectorizer fit on full range ─────────────────────────────────
    try:
        vec = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words=STOP_WORDS,
            min_df=3,
        )
        X_range = vec.fit_transform(total_range_texts)
    except ValueError:
        return []

    vocab = vec.vocabulary_          # term → col index
    feature_names = vec.get_feature_names_out()

    # Total count of each term across the full range (raw occurrence proxy via TF matrix)
    # X_range is (docs × terms) TF-IDF; we use binary presence instead of TF weight
    # for raw volume: count docs in which the term appears (already min_df=3 guarantees >=3)
    term_doc_counts = np.asarray((X_range > 0).sum(axis=0)).flatten()  # shape (n_terms,)

    # ── Per-window vectorized counts ─────────────────────────────────────────
    # For each window, build one doc (join all texts) and transform with same vocab
    def _window_counts(texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(len(feature_names), dtype=np.float32)
        joined = " ".join(texts)
        try:
            wvec = TfidfVectorizer(vocabulary=vocab, ngram_range=(1, 2))
            X_w = wvec.fit_transform([joined])
            # Use raw count (X > 0 is trivially 1 doc; use actual TF values as proxy)
            # Better: re-count occurrences using CountVectorizer with same vocab
            from sklearn.feature_extraction.text import CountVectorizer
            cvec = CountVectorizer(vocabulary=vocab, ngram_range=(1, 2))
            C_w = cvec.fit_transform([joined])
            return np.asarray(C_w.sum(axis=0)).flatten()
        except Exception:
            return np.zeros(len(feature_names), dtype=np.float32)

    # Also compute true raw counts for full range for volume
    from sklearn.feature_extraction.text import CountVectorizer
    try:
        cvec_range = CountVectorizer(vocabulary=vocab, ngram_range=(1, 2))
        C_range = cvec_range.fit_transform(total_range_texts)
        range_counts = np.asarray(C_range.sum(axis=0)).flatten()
    except Exception:
        range_counts = term_doc_counts  # fallback to doc-frequency

    curr_window_counts = _window_counts(window_texts[0])
    prev_window_arrays = [_window_counts(window_texts[w]) for w in range(1, min(4, num_windows))]
    prev_sum = sum(prev_window_arrays) if prev_window_arrays else np.zeros(len(feature_names))
    prev_avg = prev_sum / len(prev_window_arrays) if prev_window_arrays else np.zeros(len(feature_names))

    # ── Also collect hashtags ─────────────────────────────────────────────────
    all_tags = [
        "#" + _normalize_word(h.lstrip("#").lower())
        for _, b in valid_posts
        for h in re.findall(r"#\w+", b)
    ]
    tag_counts = Counter(all_tags)
    hashtag_candidates = {t: c for t, c in tag_counts.items() if c >= 3}

    # Compute post counts per window for status rule
    curr_posts = len(window_texts[0])
    prev_posts_total = sum(len(window_texts[w]) for w in range(1, min(4, num_windows)))

    # ── Build term_stats from vectorized arrays ───────────────────────────────
    term_stats: list[dict[str, Any]] = []

    for col_idx, term in enumerate(feature_names):
        total_range_vol = int(range_counts[col_idx])
        if total_range_vol < 3:
            continue
        words = term.split()
        # Lowercase + strip punctuation each token before checking, in case any
        # non-alpha character survived TF-IDF tokenization.
        # Drop if any constituent word is in ALL_STOP_WORDS, shorter than 3 chars, or digit-only
        if any(
            (w_clean := re.sub(r"[^a-z]", "", w.lower())) in ALL_STOP_WORDS
            or len(w_clean) < 3
            or w_clean.isdigit()
            or w_clean == ""
            for w in words
        ):
            continue
        v_curr = float(curr_window_counts[col_idx])
        v_prev_avg = float(prev_avg[col_idx])
        # Enforce post count condition: if insufficient posts, treat velocity as neutral
        if curr_posts < 5 or prev_posts_total < 5:
            velocity = 0.0
        else:
            if v_prev_avg == 0.0:
                velocity = 0.0
            else:
                velocity = (v_curr - v_prev_avg) / v_prev_avg
        # Cap velocity at +999% (9.99), especially for 7D/30D ranges
        if velocity > 9.99:
            velocity = 9.99
        prev_vols = [float(a[col_idx]) for a in prev_window_arrays]
        is_rising = bool(
            (sum(prev_vols) >= 2 or v_prev_avg >= 1.5)
            and velocity > 0.5
            and v_curr >= 3
        )
        score = total_range_vol * (1.0 + max(velocity, 0.0))
        term_stats.append({
            "tracker_id": tracker_id,
            "term": term,
            "volume": total_range_vol,
            "velocity": round(velocity, 4),
            "predicted_rising": is_rising,
            "period_start": period_start_iso,
            "range_label": range_label,
            "_score": score,
        })

    # ── Add hashtag candidates not already in TF‑IDF terms ───────────────────
    tfidf_terms = {r["term"] for r in term_stats}
    for tag, count in hashtag_candidates.items():
        if tag not in tfidf_terms:
            raw_tag = tag.lstrip("#")
            tag_words = raw_tag.split("_")
            if (
                raw_tag.isdigit()
                or len(raw_tag) < 3
                or re.search(r"\b\d+\b", raw_tag)
                or raw_tag in ALL_STOP_WORDS
                or any(w in ALL_STOP_WORDS or len(w) < 3 or w.isdigit() for w in tag_words)
            ):
                continue
            term_stats.append({
                "tracker_id": tracker_id,
                "term": tag,
                "volume": count,
                "velocity": 0.0,
                "predicted_rising": False,
                "period_start": period_start_iso,
                "range_label": range_label,
                "_score": float(count),
            })

    # Sort and take top terms with sub-gram suppression
    term_stats.sort(key=lambda x: (x["_score"], x["volume"]), reverse=True)

    # Sub-gram suppression: If a bigram (e.g., 'long term') is selected in top trends,
    # suppress standalone unigrams ('long', 'term') from occupying separate top-20 slots
    # unless their independent frequency is significantly higher (>= 1.5x).
    selected_terms: list[dict[str, Any]] = []

    for cand in term_stats:
        cand_term = cand["term"]
        cand_words = cand_term.split()
        cand_vol = cand["volume"]

        # If candidate is a unigram, check if it's a sub-word of an already selected bigram
        if len(cand_words) == 1:
            unigram = cand_words[0]
            suppressed = False
            for sel in selected_terms:
                sel_words = sel["term"].split()
                if len(sel_words) > 1 and unigram in sel_words:
                    if cand_vol < 1.5 * sel["volume"]:
                        suppressed = True
                        break
            if suppressed:
                continue

        # If candidate is a bigram, purge any earlier selected unigrams that are its sub-words
        # unless their volume is >= 1.5x of this bigram
        elif len(cand_words) > 1:
            selected_terms = [
                sel
                for sel in selected_terms
                if not (
                    len(sel["term"].split()) == 1
                    and sel["term"] in cand_words
                    and sel["volume"] < 1.5 * cand_vol
                )
            ]

        selected_terms.append(cand)
        if len(selected_terms) >= 20:
            break

    top_terms = selected_terms[:20]

    # Enforce cooling rows limit (max 50% cooling rows)
    cooling_rows = [t for t in top_terms if t["velocity"] < 0]
    if len(cooling_rows) > len(top_terms) / 2:
        excess = len(cooling_rows) - int(len(top_terms) / 2)
        for t in cooling_rows[:excess]:
            t["velocity"] = 0.0
            t["predicted_rising"] = False
    for row in top_terms:
        row.pop("_score", None)

    return top_terms


def analyze_trends(tracker_id: str) -> dict[str, Any]:
    """
    Compute and store trends for a tracker for 24H, 7D, and 30D ranges.
    Fetches 30D data ONCE and filters in memory for each range.
    Returns metrics dict.
    """
    t_start = time.monotonic()
    logger.info("Trends [%s]: starting analysis.", tracker_id)

    # ONE fetch for the full 30D window
    max_lookback = 720
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_lookback)).isoformat()

    try:
        t_fetch = time.monotonic()
        raw_posts = get_posts_for_trends(tracker_id, cutoff)
        fetch_secs = time.monotonic() - t_fetch
    except Exception as exc:
        msg = f"Failed to fetch posts from database: {exc}"
        logger.error("Trends [%s]: %s", tracker_id, msg)
        return {"posts_analyzed": 0, "windows_used": 0, "terms_found": 0, "terms_written": 0, "reason": msg}

    valid_posts: list[tuple[datetime, str]] = []
    for p in raw_posts:
        body = p.get("body", "")
        if not body:
            continue
        try:
            dt = datetime.fromisoformat(p["posted_at"]).replace(tzinfo=timezone.utc)
            valid_posts.append((dt, body))
        except Exception:
            continue

    # Sort descending once — _analyze_range can then cheaply slice the newest N
    valid_posts.sort(key=lambda x: x[0], reverse=True)

    logger.info(
        "Trends [%s]: fetched %d raw rows, %d valid posts in %.2fs.",
        tracker_id, len(raw_posts), len(valid_posts), fetch_secs,
    )

    if not valid_posts:
        reason = f"No posts with valid timestamps found in lookback period ({max_lookback}h)."
        logger.info("Trends [%s]: 0 terms written. %s", tracker_id, reason)
        return {"posts_analyzed": 0, "windows_used": 0, "terms_found": 0, "terms_written": 0, "reason": reason}

    now_utc = datetime.now(timezone.utc)
    all_range_rows: list[dict[str, Any]] = []

    for range_label, lookback_h, win_h in RANGE_CONFIGS:
        t_range = time.monotonic()
        range_cutoff = now_utc - timedelta(hours=lookback_h)
        # Filter in memory (no additional DB call)
        range_posts = [p for p in valid_posts if p[0] >= range_cutoff]

        range_rows = _analyze_range(
            tracker_id=tracker_id,
            range_label=range_label,
            valid_posts=range_posts,
            lookback_hours=lookback_h,
            win_hours=win_h,
        )

        try:
            delete_trends_for_tracker_and_range(tracker_id, range_label)
            if range_rows:
                insert_trends_batch(range_rows)
        except Exception as exc:
            logger.error(
                "Trends [%s]: failed to save %s rows — %s", tracker_id, range_label, exc
            )

        range_secs = time.monotonic() - t_range
        logger.info(
            "Trends [%s]: range %s done in %.2fs — %d posts → %d terms written.",
            tracker_id, range_label, range_secs, len(range_posts), len(range_rows),
        )
        all_range_rows.extend(range_rows)

    total_secs = time.monotonic() - t_start
    logger.info(
        "Trends [%s]: completed in %.2fs — total rows written: %d.",
        tracker_id, total_secs, len(all_range_rows),
    )
    return {
        "posts_analyzed": len(valid_posts),
        "windows_used": len(RANGE_CONFIGS),
        "terms_found": len(all_range_rows),
        "terms_written": len(all_range_rows),
        "reason": None,
    }
