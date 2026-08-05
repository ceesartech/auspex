# CLAUDE.md — agent onboarding

Read this first. It's the map; the details live in the docs it points to.

## What this is

A personal sports-betting prediction system on a **single Hetzner VM** (docker-compose,
~12 services). It ingests fixtures + odds, runs per-sport ML ensembles, derives ~15 soccer
markets from a Dixon-Coles scoreline model, and generates value-bet recommendations
(model probability vs best-book odds → EV / quarter-Kelly).

Sports: **soccer** (the deepest — 23k matches, ~15 derived markets), NFL, NBA, NHL,
**tennis** & **MMA** (1v1), **horse_racing** (its own multi-runner schema), lottery (dormant).

Stack: FastAPI + uvicorn (`services/api`), Next 14 frontend (`services/frontend`), Postgres 15,
Redis, Airflow 3.3 (`auspex_pipeline` DAG runs every 15 min), Caddy (TLS/reverse proxy),
Prometheus + Grafana, MLflow. Models in `services/ml-models`, batch scripts in `scripts/`.

## ⭐ Start here for current state

**[`docs/SYSTEM_AUDIT_AND_ROADMAP.md`](docs/SYSTEM_AUDIT_AND_ROADMAP.md)** is the canonical
state-of-the-system doc: active incidents, the **tested-levers registry** (what's been tried
and closed — do not re-litigate these), the remove-list, cost posture, and the prioritized
roadmap with an execution log. Its §7 log tells you what's done vs open.

## Non-negotiables (how this repo works)

1. **Don't re-test closed levers.** Calibration, soccer match_stats features, weather, and the
   horse-racing ranker ceiling were all rigorously tested and closed with evidence (audit §4).
   Each has a revisit condition; respect it.
2. **Every model-side change passes the validation gate** (audit §4.3): clone the nearest
   `scripts/ab_*.py` harness, run `scripts/walk_forward_predictions.py`, ship only at
   **ΔBrier ≤ −0.005** *and* clear of the Brier noise floor (SE ≈ 0.009 at soccer's n≈3.5k
   test). Two past "wins" died in verification because they sat inside the noise — quote the SE
   in every result.
3. **Silent failure is the enemy.** The worst incident this system had (a month of constant-prior
   soccer predictions) was caused by a swallowed exception. Prefer loud failures, canaries, and
   `logger.error` with context over bare `except: pass`.
4. **Commits:** attribute to the user's GitHub account only — **no `Co-Authored-By` trailers**.
   Direct-to-`main` is the workflow (no PRs); push when the work is verified.

## Prod access & deploy

Prod is `ssh auspex` → `/opt/auspex`. **Read-only by default**; destructive actions
(`VACUUM FULL`, voiding recs, WAL surgery, migrations) need explicit operator approval and a
verified backup first.

```bash
# Query postgres (heredoc to a file — inline quoting through `exec` breaks):
ssh auspex "cd /opt/auspex && cat > /tmp/q.sql <<'SQLEOF'
SELECT ...;
SQLEOF
docker compose cp /tmp/q.sql postgres:/tmp/q.sql >/dev/null 2>&1
docker compose exec -T postgres sh -c 'psql -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" -f /tmp/q.sql'"

# Airflow (3.3 — api-server + scheduler + dag-processor; DAG code uses airflow.sdk imports):
docker compose exec -T airflow-scheduler airflow dags list-runs -d <dag> -o plain
```

**Deploy = push to `main`.** CI builds `ghcr.io/ceesartech/auspex/{api,airflow,frontend}` and
runs `scripts/deploy_remote.sh` on the VM. Crucially: **`scripts/` and `services/` are
bind-mounted**, so a `git pull` on the VM hot-updates batch scripts, training code, and DAGs
immediately — but the **api/frontend/airflow images** only change when CI rebuilds them (and a
running uvicorn needs an api restart to reload Python). Never `docker compose up -d <svc>`
without the prod overlay flags (`-f docker-compose.yml -f docker-compose.prod.yml -f
docker-compose.ghcr.yml`) or you'll rebuild from a local image.

## Commands

```bash
make test          # all service suites + repo-root tests/unit (the money-path math)
make lint          # flake8 + black --check + isort --check + mypy
make format        # black + isort
make db-migrate    # apply services/data-ingestion/db/migrations in order
pytest tests/unit -q --rootdir=.     # ~1,050 fast tests, no DB/Redis needed
```

CI gates on flake8 / black / isort / mypy + every test suite. Lint before committing.

## Key files

| Concern | File |
|---|---|
| Upcoming fixtures **and** results ingestion (ESPN) | `scripts/fetch_upcoming.py` (`--results` mode) |
| Live odds + CLV snapshots | `scripts/fetch_live_odds.py` |
| Soccer serve path (features → predictions) | `scripts/precompute_predictions.py` |
| Rec generation (EV/Kelly, eligibility gate) | `scripts/generate_recommendations.py` |
| Grading / settlement | `scripts/grade_completed_matches.py`, `scripts/grading_outcomes.py` |
| Training + 3-way split + calibration gate | `services/ml-models/src/training/train_all_models.py` |
| Ensemble blend (refuses majority-degraded) | `services/ml-models/src/predictors/ensemble.py` |
| Experiment harnesses | `scripts/ab_*.py`, `scripts/walk_forward_predictions.py` |
| Live model monitor (ECE/Brier + Telegram) | `scripts/monitor_models.py` |
| Backups + restore runbook | `scripts/backup_postgres.py`, `OPERATIONS.md` |
| The pipeline DAG | `services/data-ingestion/dags/auspex_pipeline.py` |

## Docs

`docs/SYSTEM_AUDIT_AND_ROADMAP.md` (state + roadmap) · `docs/ARCHITECTURE.md` ·
`docs/DEPLOYMENT.md` · `docs/API.md` · `OPERATIONS.md` (prod runbook: backups, restore, disk).
