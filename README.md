# Argus Backend

A Python service that pulls public Telegram channel messages and runs
AI-powered sentiment analysis on them, writing results to Supabase.

---

## What it does

Every 5 minutes (and whenever you hit `POST /run-cycle`), the service:

1. **Reads** all active Telegram trackers from the `trackers` Supabase table.
2. **Fetches** new messages from each channel using Telegram's API.
3. **Deduplicates** — skips messages already in the database in a single batched query.
4. **Inserts** new messages into the `posts` table (with reply and forward metadata).
5. **Records interactions** — writes replies and forwards to the `interactions` table.
6. **Analyses** each message with Google Gemini in batches (polarity + emotion + confidence score).
7. **Inserts** the result into the `sentiments` table.
8. **Runs Analytics** — computes 24h trend velocity and 7d PageRank/Louvain network graph.
9. **Updates** `trackers.last_sync_at` so the next cycle knows where to start.

---

## Folder structure

```
backend/
├── app/
│   ├── main.py            FastAPI app, /health, /run-cycle, /run-analytics endpoints, background loop
│   ├── config.py          Loads & validates all env vars (fails fast if any are missing)
│   ├── db/
│   │   ├── supabase_client.py   One shared Supabase client (service-role key)
│   │   ├── repository.py        DB access for trackers, posts, sentiments, interactions
│   │   └── analytics_repository.py DB access for trends, network_nodes, network_edges
│   ├── ingestion/
│   │   └── telegram_worker.py   Telethon client; captures replies, forwards, sender hashes
│   ├── nlp/
│   │   ├── gemini_client.py     Low-level Gemini API wrapper with retries
│   │   └── sentiment.py         Batched sentiment classification with rate limits & concurrency gate
│   ├── analytics/
│   │   ├── trends.py            Hourly TF-IDF keyword & hashtag velocity; detects predicted rising terms
│   │   ├── network.py           PageRank & Louvain community detection on interactions graph
│   │   └── scheduler.py         Coordinates trends and network analytics runs
│   └── pipeline/
│       └── run_cycle.py         Orchestrates one full cycle across all trackers
├── requirements.txt
├── Dockerfile
├── .env.example           Copy to .env and fill in your values (never commit .env)
└── README.md
```

---

## Prerequisites

- Python 3.12+
- A `.env` file in this directory with the five variables listed in `.env.example`
- A Telethon session file `argus_session.session` in this directory
  (already exists — do not delete it)

---

## Running locally

Open PowerShell in the `backend/` folder and run these commands **one at a time**:

```powershell
# 1. Create a virtual environment (isolated Python sandbox)
python -m venv .venv

# 2. Activate it
.\.venv\Scripts\Activate.ps1

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Start the server (auto-reloads when you edit code)
uvicorn app.main:app --reload --port 8000
```

The server is ready when you see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Check it's working

```powershell
# Should return {"status":"ok"}
Invoke-RestMethod http://localhost:8000/health

# Manually trigger a full cycle right now (ingestion + sentiment + analytics)
Invoke-RestMethod -Method Post http://localhost:8000/run-cycle

# Manually trigger analytics only (trends + network)
Invoke-RestMethod -Method Post "http://localhost:8000/run-analytics"
```

You can also open `http://localhost:8000/docs` in a browser — FastAPI generates
an interactive API explorer automatically.

---

## Analytics pipelines

### 1. Trend & Topic Detection (`trends.py`)
- Analyzes posts from the **last 24 hours** bucketed into **1-hour windows**.
- Extracts hashtags and terms using scikit-learn's `TfidfVectorizer` (unigrams + bigrams, stopwords removed, min 2 occurrences).
- Calculates hourly velocity: `velocity = (current - previous) / max(previous, 1)`.
- Flags terms as `predicted_rising = true` when `velocity > 0.5`, `volume >= 3`, and occurrences across the last 3 hours are non-decreasing.
- Ranks candidate terms by `velocity * log(1 + volume)` and stores the top 20 terms in the `trends` table (deleting any existing entries for the same period_start first to prevent duplicates).

### 2. Network & Influence Analysis (`network.py`)
- Analyzes all `interactions` (replies and forwards) from the **last 7 days**.
- Builds a directed graph with NetworkX (`DiGraph`), where edge weight equals interaction frequency.
- Computes **PageRank** ($\alpha=0.85$) to score author influence and **Louvain communities** (modularity clustering on the undirected graph with fixed seed) to detect interest clusters.
- Retains the **top 200 nodes** by PageRank and all edges between them to keep mobile graph rendering lightweight and snappy.
- Upserts nodes into `network_nodes` and edges into `network_edges`.

---

## One full cycle — plain English

```
Active trackers in Supabase
        │
        ▼
telegram_worker.py fetches new messages (with reply & forward metadata)
        │
        ▼
repository.py checks: which messages already exist? (1 batched 'in' query)
        │
        ▼
repository.py batch-inserts new posts & interaction rows
        │
        ▼
sentiment.py batches up to 10 posts per Gemini call (polarity + emotion)
        │
        ▼
repository.py batch-inserts results into `sentiments`
        │
        ▼
scheduler.py triggers trends.py (24h TF-IDF) & network.py (7d PageRank)
        │
        ▼
tracker.last_sync_at updated → next cycle starts from here
```

---

## Environment variables

| Variable | Where to get it |
|---|---|
| `TELEGRAM_API_ID` | https://my.telegram.org → API development tools |
| `TELEGRAM_API_HASH` | Same page as API ID |
| `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey |
| `SUPABASE_URL` | Supabase dashboard → Project Settings → API |
| `SUPABASE_SERVICE_ROLE_KEY` | Same page — **keep this secret** |
| `FIRST_RUN_LIMIT` | Optional (default: `30`) — max posts on first sync |
| `MAX_PER_CYCLE` | Optional (default: `50`) — max posts on subsequent cycles |
| `GEMINI_BATCH_SIZE` | Optional (default: `10`) — posts classified per Gemini call |
| `GEMINI_CALLS_PER_MINUTE` | Optional (default: `10`) — Gemini requests rate limit |
| `GEMINI_CONCURRENCY` | Optional (default: `4`) — max concurrent Gemini calls |
| `TREND_WINDOW_HOURS` | Optional (default: `3`) — hours per trend window |
| `TREND_LOOKBACK_HOURS` | Optional (default: `48`) — hours of post lookback for trends |
| `COMMENT_POSTS_PER_CYCLE` | Optional (default: `10`) — recent posts to fetch comments for |
| `COMMENTS_PER_POST` | Optional (default: `100`) — max comments fetched per post |

---

## Deploying to Fly.io

```powershell
# Install flyctl (one time)
winget install flyctl

# Log in
flyctl auth login

# Launch (first time — follow the prompts)
flyctl launch

# Set your secrets (never put them in the Dockerfile)
flyctl secrets set TELEGRAM_API_ID=...
flyctl secrets set TELEGRAM_API_HASH=...
flyctl secrets set GEMINI_API_KEY=...
flyctl secrets set SUPABASE_URL=...
flyctl secrets set SUPABASE_SERVICE_ROLE_KEY=...

# Upload the session file as a Fly volume or secret (ask for help with this step)

# Deploy
flyctl deploy
```

> **Note:** The Telethon session file contains your login credentials.
> Do NOT commit it to Git. On Fly.io you will need to upload it as a
> persistent volume or encode it as a base64 secret.
