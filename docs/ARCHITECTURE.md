# System Architecture

## Overview

The betting system is a microservices application built around a continuous data pipeline:

**Scrape → Validate → Transform → Feature Engineering → ML Inference → API → UI**

Every layer is independently deployable, horizontally scalable, and fully observable.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                           │
│   Next.js Dashboard (port 3001)   │   REST API / WebSocket       │
└───────────────────┬───────────────┴──────────────┬──────────────┘
                    │                              │
                    ▼                              ▼
┌───────────────────────────────────────────────────────────────┐
│                     FastAPI (port 8000)                        │
│  /predictions  /recommendations  /matches  /models  /lottery  │
│  JWT auth  │  Rate limiting  │  Prometheus metrics at /metrics │
│  Celery tasks  │  WebSocket (real-time odds)                   │
└────────┬──────────────────────────┬──────────────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐        ┌─────────────────────┐
│   ML Ensemble   │        │   Feature Engine    │
│  XGBoost        │        │  250+ features:     │
│  LightGBM       │        │  - team_performance │
│  Neural Net     │        │  - head_to_head     │
│  Poisson        │        │  - player_metrics   │
│  Dixon-Coles    │        │  - contextual       │
│  ─────────────  │        │  - market_odds      │
│  SHAP explain   │        │  - temporal         │
│  Kelly Criterion│        │  - derived          │
└────────┬────────┘        └──────────┬──────────┘
         │                            │
         └──────────┬─────────────────┘
                    │
         ┌──────────▼──────────┐       ┌──────────────────┐
         │    PostgreSQL 15    │       │   Redis 7        │
         │  30+ tables:        │◄─────►│  Cache (DB 0)    │
         │  matches, odds,     │       │  Celery (DB 1)   │
         │  predictions,       │       │  Results (DB 2)  │
         │  recommendations,   │       └──────────────────┘
         │  model_perf_logs,   │
         │  features_cache,    │
         │  users, bets, ...   │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────────────────────────────┐
         │         Apache Airflow (port 8080)           │
         │  DAGs:                                       │
         │  live_odds_scraping   (every 1 min)          │
         │  daily_stats_scraping (02:00 UTC)            │
         │  historical_backfill  (one-time / on-demand) │
         │  lottery_scraping     (weekly)               │
         └──────────┬──────────────────────────────────┘
                    │
         ┌──────────▼──────────────────────────────────┐
         │              Scrapers                        │
         │  Bet365 · BetMGM · FBref · Understat        │
         │  Transfermarkt · ESPN · NHL API              │
         │  Tennis · Horse Racing · Lottery            │
         │  Weather (for contextual features)          │
         └─────────────────────────────────────────────┘

Observability (runs alongside):
  Prometheus (9090) → Grafana (3000) → Alertmanager → Email/Telegram
  Loki (3100) ← Promtail (DaemonSet)
  MLflow (5000) ← model training jobs
  Pushgateway (9091) ← batch metric pushes
```

---

## Services

### 1. Data Ingestion (`services/data-ingestion/`)

- **Scrapers**: Inherit `BaseScraper`, implement `scrape()` / `parse()` / `validate()`
- **Anti-detection**: Rotating user agents, configurable delays, optional rotating proxies
- **Deduplication**: Redis-based fingerprinting prevents duplicate inserts
- **Persistence**: All raw + transformed records written to PostgreSQL
- **Orchestration**: Airflow DAGs manage scheduling, retries, and backfill
- **Error handling**: Exponential backoff, dead-letter logging to `scraping_logs` table

### 2. Feature Engineering (`services/feature-engineering/`)

- 250+ features across 7 categories (see table below)
- Registry pattern — features are discovered and composed automatically
- Results cached in `features_cache` table with TTL; Redis layer for hot matches
- Feature validation ensures no NaN/inf values reach the models

| Category | Example features |
|---|---|
| Team performance | form_5, home_win_rate, goals_scored_avg, xG_for, xG_against |
| Head-to-head | h2h_home_wins, h2h_avg_goals, last_5_h2h_results |
| Player metrics | top_scorer_availability, injured_player_xG_loss |
| Contextual | is_derby, neutral_venue, referee_home_bias, days_since_last_match |
| Market/odds | opening_odds, odds_movement, bookmaker_margin, implied_probability |
| Temporal | day_of_week, time_of_day, season_progress, fixture_congestion |
| Derived | value_bet_indicator, kelly_fraction_raw, implied_prob_vs_model_prob |

### 3. ML Models (`services/ml-models/`)

| Model | Type | Use case |
|---|---|---|
| XGBoost | Gradient boosted trees | Primary classifier (tabular features) |
| LightGBM | Gradient boosted trees | Fast secondary classifier |
| Neural Network | PyTorch MLP | Captures non-linear interactions |
| Poisson | Statistical | Goal scoring rate estimation |
| Dixon-Coles | Statistical | Corrected Poisson for low-score matches |
| **Ensemble** | Weighted average | Final probability output |

- Probabilities calibrated with isotonic regression
- SHAP values computed per prediction for explainability
- All experiments tracked in MLflow
- Models serialised to ONNX for fast inference
- Stored in GCS bucket + `/app/models/` local mount

### 4. API Service (`services/api/`)

- **FastAPI** with async handlers and Pydantic v2 validation
- **JWT authentication** — login returns access + refresh token pair
- **Rate limiting** — 100 req/min authenticated, 20 req/min anonymous
- **Celery** for background tasks (batch prediction, email, cleanup)
- **WebSocket** at `/ws/odds` — broadcasts live odds updates via Redis pub/sub
- **Prometheus** metrics exported at `/metrics` (request rates, latencies, model metrics)

### 5. Frontend (`services/frontend/`)

- **Next.js 14** App Router with TypeScript
- **Zustand** for auth and predictions state management
- **TanStack Query** for API data fetching with caching
- **Recharts** for ROI and performance charts
- **Tailwind CSS** + shadcn/ui components
- **WebSocket hook** (`use-websocket.ts`) for live odds stream

---

## Data Flow

```
External sources
      │  (HTTP scraping, API calls)
      ▼
Scrapers → validate → transform → PostgreSQL (raw tables)
                                         │
                                         ▼
                              Feature Engine → features_cache
                                         │
                                         ▼
                              ML Models → predictions table
                                         │
                                         ▼
                              Kelly Criterion → recommendations table
                                         │
                                         ▼
                              FastAPI → Redis cache → Frontend
```

A completed match triggers:
1. `actual_outcome` written to `matches` table
2. Performance monitor recomputes accuracy/ROI → `model_performance_logs`
3. Drift detector runs KS-test on recent vs baseline features
4. If thresholds breached → retraining CronJob triggered

---

## Database Schema (key tables)

| Table | Purpose |
|---|---|
| `matches` | Core match data: teams, date, status, actual_outcome |
| `odds` | Bookmaker odds per market per match (time-series) |
| `predictions` | Model output: probabilities, confidence, model_version |
| `recommendations` | Betting recommendations with Kelly stake |
| `features_cache` | Computed feature vectors per match |
| `model_performance_logs` | Accuracy, ROI, log-loss per evaluation period |
| `model_retraining_logs` | Retraining history and deployment status |
| `betting_recommendations` | User-facing bet records with P&L tracking |
| `scraping_logs` | Per-run scraper audit log |
| `users` | User accounts, bankroll, preferences |
| `leagues`, `teams` | Reference data |

Full schema: `services/data-ingestion/db/migrations/001_create_schema.sql`

---

## Infrastructure

### Local (Docker Compose)
- All services run in a single `betting-network` bridge network
- Data persisted in named Docker volumes
- Prometheus scrapes the API container directly

### Production (single Hetzner VM + Docker Compose)
- The same Compose stack as local, plus Caddy (auto Let's Encrypt TLS) and the
  Next.js frontend, via the `docker-compose.prod.yml` overlay
- Images pulled from GHCR (`docker-compose.ghcr.yml`), SHA-tagged per deploy
- Postgres + Redis are containers with named volumes (not managed services)
- Prometheus + Grafana + Alertmanager + node/postgres/redis exporters run in the
  same stack; alerts route to Telegram
- Backups: daily `pg_dump` → local rotation + Backblaze B2 (`OPERATIONS.md`)

---

## Security

| Concern | Approach |
|---|---|
| Secrets management | `.env` on the VM (git-ignored, `chmod 600`); never committed |
| Database access | Postgres bound to the Docker network only, not published to the host |
| API authentication | JWT (HS256), short-lived access tokens + refresh |
| Input validation | Pydantic v2 on all endpoints |
| Rate limiting | SlowAPI middleware (Redis-backed) |
| Edge / TLS | Caddy terminates TLS (auto Let's Encrypt); only 80/443 exposed publicly |
| Container security | Non-root user in all Dockerfiles |

---

## Key Design Decisions

- **No ORM for read-heavy paths** — raw SQL for features and model queries for performance
- **Dual storage** — Redis for hot-path caching; PostgreSQL as source of truth
- **Model versioning** — every prediction stores `model_version`; enables A/B analysis
- **Ensemble over single model** — reduces variance, improves calibration
- **Kelly Criterion** — ensures the system only recommends bets with positive expected value
- **SHAP explanations** — every prediction is interpretable, not a black box
