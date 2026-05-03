# Betting System API Service

FastAPI backend for the personal sports betting and lottery recommendation system.

## Architecture

- **FastAPI** with sync SQLAlchemy sessions
- **PostgreSQL 15** via SQLAlchemy ORM (maps to existing 28 tables from migrations)
- **Redis 7** for caching and Celery broker
- **Celery** for async tasks (predictions, notifications)
- **JWT** authentication with DOB verification (single-user system)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
cd src && PYTHONPATH=. uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run Celery worker
cd src && PYTHONPATH=. celery -A celery_app worker --loglevel=info

# Run tests
cd services/api && PYTHONPATH=src pytest tests/ -v
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/user/login` | DOB-based authentication |
| POST | `/api/v1/predictions/` | Single match prediction |
| POST | `/api/v1/predictions/bulk` | Bulk predictions |
| GET | `/api/v1/predictions/upcoming` | Upcoming match predictions |
| GET | `/api/v1/predictions/live` | Live match predictions |
| GET | `/api/v1/recommendations/` | Active recommendations |
| GET | `/api/v1/recommendations/high-value` | High EV bets |
| POST | `/api/v1/recommendations/accumulator` | Build accumulator |
| GET | `/api/v1/matches/upcoming` | Upcoming matches with odds |
| GET | `/api/v1/matches/{id}` | Match detail |
| GET | `/api/v1/matches/{id}/stats` | Match stats + form + H2H |
| GET | `/api/v1/odds/live` | Live odds |
| GET | `/api/v1/odds/movements/{id}` | Odds movement history |
| GET | `/api/v1/user/preferences` | Get preferences |
| PUT | `/api/v1/user/preferences` | Update preferences |
| GET | `/api/v1/user/betting-history` | Betting history |
| GET | `/api/v1/user/dashboard` | Dashboard stats |
| GET | `/api/v1/models/performance` | Model metrics |
| GET | `/api/v1/models/active` | Active models |
| GET | `/api/v1/models/comparison` | Compare models |
| GET | `/api/v1/lottery/draws` | Lottery draw results |
| GET | `/api/v1/lottery/analysis` | Number frequency analysis |
| POST | `/api/v1/lottery/recommendations` | Number recommendations |
| WS | `/ws/predictions` | Live prediction updates |
| WS | `/ws/odds` | Live odds updates |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | PostgreSQL connection string |
| `REDIS_URL` | - | Redis connection string |
| `JWT_SECRET` | - | JWT signing secret |
| `USER_DOB` | `1994-05-09` | Owner's date of birth for auth |
