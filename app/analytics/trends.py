# analytics/trends.py
# Trend & topic detection for social intelligence across 24h, 7d, and 30d ranges.
# Anchored to the newest post timestamp in the analysis range.
# Extracts TF-IDF keywords, hashtags, and capitalized named entities.
#
# Performance notes:
#   - All three ranges are computed from ONE 30D DB fetch (filtered in memory).
#   - Each range is capped to the newest 2 000 posts before TF-IDF.
#   - Term volume per window is computed with a single vectorized matrix pass
#     instead of per-term regex loops.
#   - Dictionary is loaded lazily on first use, filtered to ≤60 000 words,
#     and stored as a frozenset to minimise heap size.
#   - gc.collect() is called after each tracker run to reclaim memory promptly.

import gc
import gzip
import logging
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from app.db.analytics_repository import (
    delete_trends_for_tracker_and_range,
    get_posts_for_trends,
    get_posts_for_trends_bucketed,
    insert_trends_batch,
)

logger = logging.getLogger(__name__)

def _normalize_word(word: str) -> str:
    """Trim trailing 's' for simple English plurals (e.g. funds -> fund), avoiding 'ss'."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


CUSTOM_STOP_WORDS = {
    # ── Web / platform noise ─────────────────────────────────────────────────
    "amp", "rt", "http", "https", "com", "co", "t", "telegram",
    "channel", "group", "zone", "live", "user", "id", "user_id",
    "admin", "bot",

    # ── Pronouns & Demonstratives ────────────────────────────────────────────
    "maine", "tumne", "usne", "humne", "unhone", "inhone", "isne",
    "jisne", "mera", "meri", "mere", "tera", "teri", "tere",
    "uska", "uski", "uske", "inka", "inki", "inke", "unka", "unki", "unke",
    "apna", "apni", "apne", "apka", "apki", "apke",
    "mujhe", "tujhe", "unhe", "inhe",
    "kisi", "kisse", "jisse", "jo", "jin", "jinhone",
    "sab", "sabhi", "kuch", "koi",
    "ye", "yeh", "wo", "woh", "ek",

    # ── Verbs & Auxiliaries ──────────────────────────────────────────────────
    "rhe", "rha", "rhi", "raha", "rahi", "rahe",
    "ho", "hai", "hain", "hoga", "honge", "hangi",
    "tha", "thi", "the",
    "karo", "karna", "karke", "karne",
    "kiya", "diya", "liya", "le", "de",
    "aaya", "aaye", "aayi", "aya", "aye", "ayi",
    "gaya", "gaye", "gya", "gye",
    "bolo", "bol", "pucho", "batao",
    "sakta", "sakti", "sakte",
    "karta", "karti", "karte", "krta", "krti", "krte",
    "kar", "kr", "lia", "liye", "gyi",

    # ── Conjunctions & Prepositions ──────────────────────────────────────────
    "isliye", "kyunki", "lekin",
    "par", "aur", "ya", "to", "toh",
    "tak", "tk", "pe", "se", "me", "mein",
    "bhi", "hi",
    "wala", "wali", "wale", "wal",
    "ne", "ko", "ka", "ki", "ke", "ab", "na",

    # ── Chat Fillers & Questions ─────────────────────────────────────────────
    "abe", "saale", "bhai", "sir", "ji", "bro",
    "yr", "yaar",
    "pls", "please", "hmm", "hmmm",
    "ok", "okay", "haan", "ha", "haan",
    "nahi", "nhi", "na", "naa",
    "ur", "gd",
    "pta", "pata", "sahi", "galat",
    "kya", "kab", "kahan", "kaise", "kese", "kyun",
    "kis", "kon", "kaun",
    "abhi", "fir", "phir",
    "aisa", "aise", "aisi",
    "dekho", "dikha", "itna", "dur",
    "tha", "thi", "ek",

    # ── Legacy / previously observed leakage ─────────────────────────────────
    "h", "hain", "ha", "hoga", "intimation", "city",
    "respond", "tell", "got", "zone", "live",
    "tak", "lia", "diya", "rha", "rhi",
    "koi", "aaya", "kuch", "sab",
    "fir", "phir", "aisa",
    "apka", "mujhe", "kiya",
    "isliye", "pta", "kis", "sahi", "krta", "karta", "kisi",
    "maine", "mujhe", "tumhe", "usse", "inse", "unse",
    "dega", "degi", "denge", "din", "dino", "dono",
    "babu", "kha", "mila", "mile", "mili",
}

# Domain-specific terms & acronyms to always allow even if not in standard English dictionary
DOMAIN_WHITELIST: set[str] = {
    "rrb", "ssc", "ntpc", "cbt", "cbt1", "cbt2", "pnr", "alp", "loco", "groupd",
    # Common financial / crypto / business acronyms & terms for other Argus trackers
    "usdt", "inr", "p2p", "fno", "nifty", "banknifty", "ipo", "etf", "sip",
}

# Fast English dictionary loader (bundled gzipped word list with fallback)
DATA_DIR = Path(__file__).resolve().parent / "data"
WORDS_FILE = DATA_DIR / "words.txt.gz"
_ENGLISH_WORDS_CACHE: frozenset[str] | None = None

# Maximum words to retain from the dictionary (frequency-ordered file assumed).
# Keeping the 60 000 most-common words covers virtually all real content terms
# while limiting peak heap allocation to ~5 MB vs ~30 MB for the full 369k list.
_MAX_DICT_WORDS = 60_000


def _get_english_words() -> frozenset[str]:
    """
    Load and cache the English word list as a frozenset.

    Filtering rules applied once at load time:
    - Lowercase alphabetic characters only (no digits/hyphens/apostrophes).
    - Length 3–15 characters.
    - At most 60 000 words (first N from frequency-ordered file = most common).
    The temporary list is deleted immediately after the frozenset is built.
    """
    global _ENGLISH_WORDS_CACHE
    if _ENGLISH_WORDS_CACHE is not None:
        return _ENGLISH_WORDS_CACHE

    if WORDS_FILE.exists():
        try:
            raw: list[str] = []
            with gzip.open(WORDS_FILE, "rt", encoding="utf-8") as f:
                for line in f:
                    w = line.strip().lower()
                    if w.isalpha() and 3 <= len(w) <= 15:
                        raw.append(w)
                    if len(raw) >= _MAX_DICT_WORDS:
                        break
            _ENGLISH_WORDS_CACHE = frozenset(raw)
            del raw  # release temporary list immediately
            logger.info(
                "Loaded %d English words from %s (capped at %d).",
                len(_ENGLISH_WORDS_CACHE), WORDS_FILE, _MAX_DICT_WORDS,
            )
            return _ENGLISH_WORDS_CACHE
        except Exception as exc:
            logger.warning("Failed to load bundled words dictionary from %s: %s", WORDS_FILE, exc)
    else:
        logger.warning(
            "Dictionary file missing at %s; falling back to sklearn stop-word set.",
            WORDS_FILE,
        )

    # Fallback: sklearn's built-in stop words (tiny but always available)
    _ENGLISH_WORDS_CACHE = frozenset(ENGLISH_STOP_WORDS)
    return _ENGLISH_WORDS_CACHE


def _is_valid_word(word: str) -> bool:
    """Check if a word is in DOMAIN_WHITELIST or a valid English dictionary word."""
    w = word.lower()
    if w in DOMAIN_WHITELIST:
        return True
    dict_words = _get_english_words()
    if not dict_words:
        return True
    return w in dict_words or _normalize_word(w) in dict_words


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

_MAX_POSTS_PER_RANGE = 2_000


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
    Caps to the newest 2 000 posts. Returns trend dicts with volume >= 3.
    """
    if not valid_posts:
        return []

    # Cap each range at 2 000 posts before TF-IDF
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
            max_features=2000,
            ngram_range=(1, 2),
            stop_words=STOP_WORDS,
            min_df=3,
            dtype=np.float32,
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
        cleaned_words = [re.sub(r"[^a-z]", "", w.lower()) for w in words]
        if any(
            w in ALL_STOP_WORDS
            or len(w) < 3
            or w.isdigit()
            or w == ""
            for w in cleaned_words
        ):
            continue

        # English Dictionary & Domain-Acronym Validation:
        # Every constituent word must be in DOMAIN_WHITELIST or a valid English dictionary word.
        # Discards Romanized Hindi verb/noun stems not found in standard English dictionaries.
        if any(not _is_valid_word(w) for w in cleaned_words):
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
            cleaned_tag_words = [re.sub(r"[^a-z]", "", w.lower()) for w in tag_words]
            if (
                raw_tag.isdigit()
                or len(raw_tag) < 3
                or re.search(r"\b\d+\b", raw_tag)
                or raw_tag in ALL_STOP_WORDS
                or any(w in ALL_STOP_WORDS or len(w) < 3 or w.isdigit() or w == "" for w in cleaned_tag_words)
                or any(not _is_valid_word(w) for w in cleaned_tag_words)
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


# ── Memory monitoring helper ─────────────────────────────────────────────────

def _get_process_memory_mb() -> float:
    """Return the current process RSS in MB. Uses psutil when available."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1_048_576
    except ImportError:
        pass
    try:
        import resource  # Unix only
        # getrusage returns kilobytes on Linux
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0.0


_MEMORY_WARN_MB = 400  # log WARNING if RSS exceeds this threshold


def _log_memory(label: str, tracker_id: str) -> float:
    """Log process memory and warn if above threshold. Returns RSS in MB."""
    mb = _get_process_memory_mb()
    if mb <= 0:
        return mb
    if mb >= _MEMORY_WARN_MB:
        logger.warning(
            "Trends [%s] %s: process memory %.1f MB (exceeds %d MB threshold).",
            tracker_id, label, mb, _MEMORY_WARN_MB,
        )
    else:
        logger.info("Trends [%s] %s: process memory %.1f MB.", tracker_id, label, mb)
    return mb


def analyze_trends(tracker_id: str) -> dict[str, Any]:
    """
    Compute and store trends for a tracker for 24H, 7D, and 30D ranges.

    Fetch strategy (memory-bounded — never more than ~2 000 posts in RAM
    at once per range):
    - 24H : one bounded DESC query for the newest 2 000 posts in the window.
    - 7D  : 24-bucket evenly-spaced sample, ≤80 posts per bucket (~1 920 total).
    - 30D : 24-bucket evenly-spaced sample, ≤80 posts per bucket (~1 920 total).

    After each range is processed its post list is deleted and gc.collect()
    is called before the next range begins.
    """
    t_start = time.monotonic()
    logger.info("Trends [%s]: starting analysis.", tracker_id)
    _log_memory("start", tracker_id)

    now_utc = datetime.now(timezone.utc)

    # ── Per-range fetch config ────────────────────────────────────────────────
    # (range_label, lookback_hours, win_hours, use_bucketed_fetch)
    FETCH_CONFIGS = [
        ("24H",  24,   1,  False),  # newest 2 000, single bounded query
        ("7D",   168,  24, True),   # 24-bucket evenly-spaced sample
        ("30D",  720,  72, True),   # 24-bucket evenly-spaced sample
    ]

    total_posts_analyzed = 0
    total_rows_written = 0

    for range_label, lookback_h, win_h, use_buckets in FETCH_CONFIGS:
        t_range = time.monotonic()
        window_start = now_utc - timedelta(hours=lookback_h)
        window_start_iso = window_start.isoformat()

        # ── Fetch ─────────────────────────────────────────────────────────────
        try:
            if use_buckets:
                raw_posts = get_posts_for_trends_bucketed(
                    tracker_id=tracker_id,
                    window_start=window_start,
                    window_end=now_utc,
                    n_buckets=24,
                    posts_per_bucket=80,
                )
            else:
                raw_posts = get_posts_for_trends(
                    tracker_id, window_start_iso, limit=2000
                )
        except Exception as exc:
            logger.error("Trends [%s]: [%s] fetch failed — %s", tracker_id, range_label, exc)
            continue

        # ── Parse & validate timestamps ───────────────────────────────────────
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

        # Sort descending (newest first) — _analyze_range expects this
        valid_posts.sort(key=lambda x: x[0], reverse=True)

        earliest_dt = valid_posts[-1][0] if valid_posts else None
        latest_dt   = valid_posts[0][0]  if valid_posts else None
        logger.info(
            "Trends [%s]: [%s] %d posts fetched (from %s to %s).",
            tracker_id, range_label, len(valid_posts), earliest_dt, latest_dt,
        )

        if not valid_posts:
            logger.info("Trends [%s]: [%s] no valid posts — skipping.", tracker_id, range_label)
            continue

        # ── Analyse ───────────────────────────────────────────────────────────
        range_rows = _analyze_range(
            tracker_id=tracker_id,
            range_label=range_label,
            valid_posts=valid_posts,
            lookback_hours=lookback_h,
            win_hours=win_h,
        )

        # ── Log top-3 terms ───────────────────────────────────────────────────
        top3 = [(r["term"], r["volume"]) for r in range_rows[:3]]
        logger.info(
            "Trends [%s]: [%s] top-3 terms: %s",
            tracker_id, range_label,
            ", ".join(f'"{t}" ({v})' for t, v in top3) if top3 else "(none)",
        )

        # ── Persist ───────────────────────────────────────────────────────────
        try:
            delete_trends_for_tracker_and_range(tracker_id, range_label)
            if range_rows:
                insert_trends_batch(range_rows)
        except Exception as exc:
            logger.error(
                "Trends [%s]: failed to save %s rows — %s", tracker_id, range_label, exc
            )

        range_secs = time.monotonic() - t_range
        n_range_posts = len(valid_posts)
        n_range_rows  = len(range_rows)
        logger.info(
            "Trends [%s]: [%s] done in %.2fs — %d posts → %d terms written.",
            tracker_id, range_label, range_secs, n_range_posts, n_range_rows,
        )

        total_posts_analyzed += n_range_posts
        total_rows_written   += n_range_rows

        # ── Free memory before next range ─────────────────────────────────────
        del valid_posts, raw_posts, range_rows
        gc.collect()

    # ── Final summary ─────────────────────────────────────────────────────────
    total_secs = time.monotonic() - t_start
    logger.info(
        "Trends [%s]: completed in %.2fs — total rows written: %d.",
        tracker_id, total_secs, total_rows_written,
    )
    _log_memory("end", tracker_id)
    gc.collect()

    return {
        "posts_analyzed": total_posts_analyzed,
        "windows_used": len(FETCH_CONFIGS),
        "terms_found": total_rows_written,
        "terms_written": total_rows_written,
        "reason": None,
    }
