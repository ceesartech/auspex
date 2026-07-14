# Data Ingestion Service

Airflow DAGs + the ingestion core (config, DatabaseManager, validators) for the
betting recommendation system.

> The original Selenium/Playwright scraper package that lived in
> `src/scrapers/` was removed in the 2026-07 audit (commit `7222b0a`) — live
> ingestion moved to JSON APIs + lightweight static-HTML parsers in the
> repo-root `scripts/` directory, invoked by the DAGs below via
> `docker compose exec api python /app/scripts/...`.

## What lives here

| Path | Contents |
|------|----------|
| `dags/` | Airflow DAGs: `auspex_pipeline` (15-min ingest → features → predict → recs → grade), `fetch_live_odds` (90-min, quota-gated), `monitor_models` (hourly drift + constant-prior canary), `retrain_models` (weekly), `db_backup_daily`, `airflow_db_maintenance` (monthly metadata clean), `weather_*` (paused — chapter closed) |
| `src/core/` | `ScraperConfig` (connection/rate-limit settings) + `DatabaseManager` (pooled Postgres access) |
| `src/validators/` | `DataValidator` — odds + match sanity checks |
| `src/transformers/` | Data-shaping helpers |
| `db/migrations/` | Canonical SQL schema, applied in order (001…) |

## Data sources (live)

| Source | Data | Fetcher (repo-root `scripts/`) |
|--------|------|-------------------------------|
| the-odds-api.com | Multi-book odds + CLV snapshots | `fetch_live_odds.py` |
| ESPN (JSON APIs) | Fixtures + results, all team sports + tennis/MMA | `fetch_upcoming.py` (`--results` mode) |
| NHL API | NHL enrichment | `fetch_upcoming.py` / `compute_features_nhl.py` |
| The Racing API | Racecards + results | `load_racing_api.py` |
| FBref (static HTML) | Venue metadata | `scrape_fbref_venues.py` |
| NFL injury reports (static HTML) | Injury features | `scrape_nfl_injury_reports.py` |

## Running Tests

```bash
cd services/data-ingestion && PYTHONPATH=src pytest tests -v
```
