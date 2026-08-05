# System Audit & Roadmap — July 2026

**Date:** 2026-07-11 · **Repo state:** `d111421` (unchanged since 2026-06-15) · **Prod:** single Hetzner VM (`ssh auspex` → `/opt/auspex`, docker-compose, 12 services)

This is the canonical state-of-the-system document. It was produced by a multi-agent deep check
(4 parallel deep-dives over the repo **and live prod**, followed by adversarial verification of
every critical claim — several were corrected in that pass; corrections are folded in below).
Every finding carries its evidence (`file:line` or a prod observation) so it can be re-verified.

**How to use this doc (for the next agent/operator):**

1. Fix the [P0 incidents](#1-p0--active-production-incidents) first. One of them means soccer
   predictions have been garbage for ~4 weeks.
2. Do not re-test anything in the [closed-experiments registry](#4-tested-levers-registry) —
   those levers were rigorously tested and closed. Each entry lists its revisit condition.
3. Every model-side change must pass the [house validation gate](#43-validation-methodology-house-rules)
   (walk-forward, ΔBrier ≤ −0.005, quoted against the Brier noise floor). Two past "wins"
   died in verification because they were inside the noise floor; the methodology exists so
   that doesn't recur.
4. Prod access pattern and safety rules are in [Appendix A](#appendix-a--prod-access--verification-pattern).

---

## 0. Executive summary

| # | Finding | Severity | One-line action |
|---|---------|----------|-----------------|
| 1 | **Every soccer prediction served since ~2026-06-15 is a constant prior** (0.433/0.243/0.324) — the serve path silently drops all learned models | **P0** | 3-line `feature__` bridge in `scripts/precompute_predictions.py` + make ensemble failures loud |
| 2 | **Zero database backups exist** — `db_backup_daily` paused, `backups/` empty, no offsite keys | **P0** | Unpause the DAG today; wire B2 per `OPERATIONS.md` |
| 3 | **Prometheus WAL corrupt since the June 8 disk-full** — 7.9 GB leaked, ~240 MB/day, disk-full will recur in ~3 months | **P0** | 5-min WAL removal in a maintenance window |
| 4 | **The P&L feedback loop is structurally broken**: 0 of 1,348 team-sport recs ever settled; no results ingestion exists for any team sport; training corpora frozen 6–18 months | P1 | Add ESPN results ingestion; graders then settle automatically |
| 5 | **~$95/mo (≈45–50% of recurring spend) is recoverable with zero prediction impact** | P1 | Odds-API downgrade ($60) + cancel Visual Crossing ($35) |
| 6 | ~200 files of verified-dead code (scraper package, GKE-era infra, broken monitoring duplicate) plus 4 heavy unused deps | P2 | Deletion list in §5, verified reference-free |
| 7 | The genuine prediction-improvement levers are **data and market-selection**, not model tuning: results+CLV loop → market gating; Understat xG; corpus 3–4×. Model-side tuning is measured-and-closed. | P1–P2 | Ranked plan in §4 |

Recurring spend today ≈ **$185–215/mo**. After §6 actions ≈ **$90–120/mo**.
Disk: 63% used; the three P0/P2 disk actions reclaim ~11 GB (verified breakdown, §6.3).

---

## 1. P0 — Active production incidents

### 1.1 Soccer serving emits a constant prior for every match (since ~2026-06-15)

**Impact:** All soccer 1X2 probabilities served for the last ~4 weeks are byte-identical
(`home=0.4334, draw=0.2428, away=0.3239`) regardless of teams or odds. All ~15 derived soccer
markets inherit this. The EV gate then converts any long odds into fictitious "+EV" — e.g. a
live rec `Portugal ML @ 4.40, EV +0.91` during the FIFA World Cup. ~220 recs/week of
structurally −EV output are being generated.

**Evidence (prod-verified twice, including once by executing the ensemble in-container):**

- Distinct home-probs among served soccer predictions by week:
  `2026-06-08: 110 → 06-15: 6 → 06-22: 2 → 06-29: 2 → 07-06: 1`.
- `Brazil–Norway` and `Portugal–Spain` predictions byte-identical despite cached
  `implied_prob_home` ranging 0.206–0.741 across the same slate.

**Root cause (corrected during adversarial verification — the first-pass diagnosis was wrong):**

1. The June feature/backfill work + subsequent weekly retrain (2026-06-28/07-05) produced models
   whose `feature_names` include 39 **`feature__*`-prefixed** duplicates (an artifact of
   train-time JSONB flattening in `services/ml-models/src/utils/training_data.py:336`).
2. Every sport's precompute script mirrors plain keys into `feature__*` keys before predicting —
   **except soccer**. See `scripts/precompute_predictions_nfl.py:231`, `_nba.py:259`,
   `_nhl.py:369`, `_tennis.py:226`, `_mma.py:226`; `scripts/precompute_predictions.py` (soccer)
   has **zero** `feature__` references. `scripts/walk_forward_predictions.py:350-356` documents
   the bridge as "the bridge the live predict path uses".
3. XGB/LGBM/NN therefore raise `KeyError("['feature__odds_home', …] not in index")` on every
   soccer predict — and `EnsemblePredictor.predict_proba`
   (`services/ml-models/src/predictors/ensemble.py:176-183`) **silently swallows** failed
   members. Only Poisson + Dixon-Coles survive, and with no team columns in the serve-path
   input they emit their global prior for every row.
4. It *looks* like a zero-history-league problem because the current slate (World Cup + summer
   leagues) is out-of-corpus — but in-corpus teams get the same constant. Verified by feeding
   synthetic rows with implied home prob 0.234 vs 0.741 into the prod ensemble: identical output.

**Fix (in priority order):**

1. Add the `feature__` key-mirroring bridge to `scripts/precompute_predictions.py` (copy the
   NFL pattern). ~3 lines. Serving instantly recovers the existing trained models — **no retrain
   needed**.
2. Make silent ensemble-member failure loud: in `ensemble.py:176-183`, log at ERROR with the
   member name + exception, and **fail the prediction (or alert) when surviving weight < 0.5**.
   A month of constant output went unnoticed solely because of this catch-all.
3. Void the pending recs generated under the bug (`betting_recommendations` where
   `created_at > '2026-06-15'` and sport = soccer) — operator decision, see §7.
4. Secondary guard (real but *not* the root cause): an eligibility gate skipping recs when
   either team has < 10 finished matches in-corpus — Poisson/DC genuinely cannot price
   out-of-corpus national teams. Gate in `scripts/generate_recommendations.py:314` (currently
   gates only on EV/prob-floor). Mirror for tennis (631 stale pending recs) and MMA.
5. Add a canary: alert if `COUNT(DISTINCT probabilities->>'home')` over the last 100 soccer
   predictions is < 5. Trivial SQL in the existing `monitor_models.py` cadence.

### 1.2 Backups are OFF (June item, still open — now the top data-loss risk)

- `db_backup_daily` is **paused** with zero runs ever (`airflow dags list-runs` → "No data found").
- `/opt/auspex/backups/` contains only `.gitkeep`. Prod `.env` has **no** `BACKUP_S3_BUCKET` /
  `AWS_*` keys.
- The entire pipeline is built and documented: `services/data-ingestion/dags/db_backup.py:42-46`
  (02:00 UTC daily), `scripts/backup_postgres.py` (pg_dump → local rotation → optional
  B2/S3-compatible upload with verification), `OPERATIONS.md:7-165` (runbook incl. B2 setup +
  restore drill). It was simply never switched on.

**Fix:** (1) *Today, zero config:* `airflow dags unpause db_backup_daily`, trigger once, verify a
`.dump` lands in `/opt/auspex/backups/`. (2) B2 offsite per `OPERATIONS.md:52-96` (~10 min:
bucket + scoped key + 4 env vars + `docker compose up -d api`), verify the "Upload verified" log
line. (3) Run the quarterly restore drill once so the path is known-good. B2 cost < $0.50/mo —
the DB is 3.5 GB and holds point-in-time odds snapshots that **cannot be re-scraped**.

### 1.3 Prometheus WAL corruption — the June disk-full is scheduled to recur

- Segment `/prometheus/wal/00000137` was corrupted during the June 8 disk-full event. Every
  2-hour compaction since fails (`WAL truncation in Compact … corruption in segment …`), so the
  WAL **never truncates**: 400 segments, 7.9 GB, growing ~240 MB/day. Prometheus volume is
  17.9 GB, of which only ~0.9 GB is actual TSDB blocks.
- At current growth the disk (63%) reaches 90% in roughly 2.5–3 months. This is the dominant
  disk-growth driver on the box.

**Fix (5-minute maintenance window, loses only ~2–3h of in-head metrics):**
`docker compose stop prometheus` → move the `wal/` directory aside → `start` → confirm the next
2-hour checkpoint logs INFO not ERROR → delete the moved dir. Optionally cut
`--storage.tsdb.retention.time` 30d→14d (`docker-compose.yml:317`). Note: the retention cut does
**not** free multiple GB on top of the WAL fix (verified — blocks are only ~0.9 GB).

---

## 2. P1 — The broken feedback loop (the real "prediction accountability" problem)

The system's prediction-quality problem is currently dwarfed by an **accountability** problem:
it cannot know whether any team-sport rec makes money.

### 2.1 No team-sport results ingestion exists

- `betting_recommendations`: **1,348 pending, 0 won/lost — ever.** All rec'd matches remain
  `status='scheduled'` forever.
- Latest finished match per sport: MMA **2024-12-15**, tennis **2025-01-11**, NFL **2025-02-09**,
  NHL 2025-06-18, NBA 2025-06-23, soccer 2026-05-24 (season end). `finished_last_90d = 0` for
  every sport except soccer.
- Cause: `scripts/fetch_upcoming.py:479-480` deliberately skips non-`pre` ESPN events, and no
  DAG task ingests results for any team sport (only horse racing has a results path). The
  graders (`grade_completed_matches.py`) run every 15 min with nothing to grade.
- Consequence beyond settlement: the weekly retrain trains NBA/NHL/NFL/tennis/MMA on corpora
  frozen 6–18 months. By the 2026-27 season openers, team ratings are a full season stale.

**Fix:** add a `fetch_results` task per sport to `auspex_pipeline` using the **same ESPN
scoreboard endpoints** `fetch_upcoming.py` already calls: when `state == 'post'`, UPDATE the
existing matches row (same `ON CONFLICT` identity) with scores + `status='finished'`. Grading
then settles predictions + recs automatically within 15 min. Also: run
`scripts/load_international.py` (never run on prod — the World Cup league has 165 matches / 0
finished) and add per-sport season loaders to a monthly refresh. Extend
`compute_features.py --backfill-finished` (soccer-only today, line 382) to the other sports to
close their train/serve feature gap.

### 2.2 CLV (closing-line value) capture — the fastest edge-truth signal

- Still impossible: `scripts/fetch_live_odds.py:324-355` dedups on
  `(match, book, market, selection, line, is_opening=false)` **excluding price and timestamp**,
  so only the *first* price ever seen is stored. 1,133,920 odds rows all carry timestamps;
  `is_opening` is never true; the 90-min fetch cadence means closing prices are already being
  *fetched* — and discarded.
- CLV matters because it converts "is the edge real?" from a multi-year settled-variance
  question into a weeks-scale measurement, and it works even while results ingestion is being
  built (needs odds only).

**Fix:** add `AND odds_decimal = %s` to the NOT-EXISTS guard so price *changes* insert as new
timestamped rows (bounded growth; a line that never moves stays one row). Then a CLV view: for
each rec, closing price = last pre-commence row for its (match, market, selection, line);
`CLV = rec_odds / closing_odds − 1`, aggregated by sport+market+model. Optional later: a
dedicated closing-snapshot fetch for events commencing < 2h.

### 2.3 Market gating — the highest-probability P&L improvement, currently impossible

The only settled evidence in the entire system: horse-racing WIN recs — 228 settled,
42W/186L, staked 1,325, P&L −18, **ROI −1.4% ± ~13% SE** (statistical breakeven). Every other
category is unknowable until §2.1/§2.2 land.

**Fix (after 60–90 days of settlement + CLV data):** per-(sport, bet_type, odds-band) rollup;
hard-disable rec generation for categories with negative CLV (CLV converges much faster than
ROI); codify as a config table read by the `generate_recommendations_*` scripts, not per-sport
code edits.

---

## 3. Prediction improvement — ranked levers (honest expectations)

Model-side tuning on the existing data is **measured-and-closed** (§4). What remains, ranked by
expected value:

| Rank | Lever | Status / evidence | Expected impact | Effort |
|------|-------|-------------------|-----------------|--------|
| 0 | **Fix the serve bridge (§1.1)** | Bug, verified | Restores the entire trained soccer model — the single biggest "improvement" available is un-breaking serving | 3 lines |
| 1 | **Results ingestion + settlement (§2.1)** | Missing subsystem | Unlocks realized ROI, live accuracy, stops season-stale retrains | Moderate |
| 2 | **Eligibility gate + void bogus recs (§1.1.4)** | Missing guard | Stops ~200/wk structurally −EV recs | Quick |
| 3 | **CLV capture (§2.2)** | 1-line dedup change + view | Edge-truth in weeks; feeds gating | Quick–moderate |
| 4 | **Market gating (§2.3)** | Blocked on 1+3 | Plausibly several % ROI on retained volume; honest: unquantifiable today | Moderate |
| 5 | **Soccer xG via Understat backfill** | `expected_goals` NULL for 46,912/46,912 rows; corpus is exactly the 6 leagues Understat covers at 76.5% (17,936/23,456). ⚠️ The old orphaned scraper (`understat_scraper.py`, team_id bug at :137) was DELETED with the scrapers package in `7222b0a` — recover it as a starting point via `git show 7222b0a^:services/data-ingestion/src/scrapers/understat_scraper.py`, or (better) write a fresh standalone `scripts/backfill_understat_xg.py` in the current fetch-script idiom | 40–60% chance of clearing the −0.005 Brier gate; if it does, first real soccer model win since Dixon-Coles, lifts ~15 derived markets | Moderate |
| 6 | **Soccer corpus 3–4× via existing loaders** | `load_football_data.py` supports ~30 league codes / 20 seasons; prod uses 6 leagues / 10 seasons. The "extra" loader covers the summer leagues (MLS, Allsvenskan, Brasileirão…) generating most current recs | Directly fixes zero-history leagues (P&L-relevant now); modest top-league Brier expectation — **gate the ship on held-out top-5-league Brier**, consider per-league-tier Dixon-Coles; only route to the 5–10× scale where the closed weather chapter could be legitimately revisited | Moderate |
| 7 | **Horse place/show markets** | `generate_recommendations_horse_racing.py:251` hardcodes `bet_type='win'`; schema already allows place/show; settlement already works for racing; win market measured at breakeven | Place pricing is formulaic/softer; genuinely uncertain — pilot 60 days on realized ROI+CLV | Moderate |
| 8 | **NHL derived markets** | Poisson scoreline artifacts already exist (`hockey_poisson_nhl_*`); mechanically identical to shipped soccer derivation | Multiplies rec surface 4–5× on the sport with the biggest shipped cross-book win — but sequenced behind results ingestion + season start | Moderate |
| 9 | Hyperparameter tuning / recency weighting | Optuna harness exists but is dead code (`hyperparameter_optimization.py`, imported by nothing); configs are hand-set (lr=0.05, depth 6–8 everywhere); no sample weighting anywhere | Honest: ΔBrier −0.000 to −0.004, most likely under the ship gate. Run only as piggyback experiments behind a `--tune` flag | Low |

### 4. Tested-levers registry

**Do not re-test these.** This registry previously lived only in the operator's private agent
memory; this section makes it repo-canonical.

| Lever | Verdict | Evidence | Revisit condition |
|-------|---------|----------|-------------------|
| Probability calibration (team sports) | **CLOSED — no win** (2026-06-10) | Held-out ECE already 0.02–0.05 across soccer/NBA/NHL/tennis/MMA; isotonic overfit small val sets (MCE blowups); only NFL ML overconfident (0.148) but n=143 untrustworthy. Self-gating calibration infra **shipped** (`train_all_models.py` `_fit_and_gate_calibrator`, MIN_GATE_N=500): auto-enables per sport only if it beats held-out Brier — currently serves raw everywhere, correctly | A model drifts overconfident on ≥500 held-out rows (the gate will catch it automatically); use temperature scaling, not isotonic |
| Soccer rolling match_stats features | **CLOSED — no win** (2026-06-15) | Full-ensemble retrain flat: held-out Brier 0.5946→0.59645. The A/B "-0.0029 win" was inside the Brier noise floor (SE ≈0.009 at n=3519). Features remain in `compute_features.py` (accuracy-neutral, keeps train/serve parity) | Only alongside a categorically stronger signal (xG) |
| Weather features (Visual Crossing) | **CLOSED at current corpus** (2026-06-05) | All 5 sport-market combos DROP at ΔBrier ≤ −0.005 despite VC beating Open-Meteo and consistent ECE improvement | Corpus reaches 5–10× (only achievable via the §3.6 backfill). Note §6: the VC subscription itself should be cancelled meanwhile — **a deliberate reversal of the June "keep refreshing" decision** |
| Cross-book odds-dispersion features | **SHIPPED** where it works | NFL total −0.0110 / ML −0.0154, NBA spread −0.0061, NHL puck_line −0.0186 / total −0.0231 ΔBrier. Soccer 1x2 tested + dropped (2-book structure too thin); tennis/MMA odds too sparse | Pattern needs ~20+ books' disagreement |
| Horse-racing ML ranker | **Ceiling reached** | Brier ~0.105 vs consensus 0.0831 — structural. All five follow-up levers tested and neutral/negative. Productive output: hybrid recs path (shipped) | Different data (sectional times) or framing (direct prob regression) |
| Horse 40+ longshot calibration | CLOSED | Bias non-stationary; isotonic on graded history +0.0007 worse; recs engine emits zero recs in those buckets anyway | — |
| NFL spread | Parked | Cross-book didn't transfer; needs new data (paid odds archive / QB-injury infra / 5+ seasons) | New data source |
| Tennis/MMA neural nets | **FIXED — stale memory** | Retrain of 2026-07-05 trains NN successfully for both (tennis val_acc 0.605, in-ensemble at weight 0.333). NN ≈ GBM accuracy; value is diversity only | Removing NN + torch is a defensible image-size play (§5.4): retrain with `--skip-models neural_network`, verify held-out delta ≈ 0 |

### 4.3 Validation methodology (house rules)

Everything model-side gets validated the same way (this methodology produced the cross-book
wins **and** correctly rejected calibration/match_stats/weather):

1. Clone the nearest `scripts/ab_*.py` harness; baseline must be the **real** training frame.
2. Run `scripts/walk_forward_predictions.py` over 2–3 seasons.
3. Ship gate: **ΔBrier ≤ −0.005**, quoted against the noise floor
   (SE ≈ √(Var(per-match Brier)/n); ≈0.009 at soccer's n≈3.5k test). A delta inside 1 SE is
   noise regardless of sign-consistency.
4. Only then productionize (bundle + retrain DAG). The 3-way temporal split + held-out metrics
   live in `train_all_models.py` (`_holdout_metrics`, `holdout_test` in the training report).
5. Market-side levers validate on **realized CLV/ROI** (60–90 days) instead.

---

## 5. Remove list (verified reference-free; corrections from the adversarial pass folded in)

> Each item was confirmed by full-repo reference sweeps (code, DAGs, compose, CI, docs) and,
> where relevant, live-prod checks. Still: delete in one reviewable commit per group.

### 5.1 The scrapers package (dead) — and everything attached to it

- `services/data-ingestion/src/scrapers/` — all 12 modules have zero non-test importers. Real
  ingestion is `scripts/fetch_upcoming.py` (ESPN JSON via requests) + per-source loaders.
- Also remove (the under-scoped parts caught in verification):
  `docker/Dockerfile.scraper` (CMD targets a nonexistent module; not built by CI/compose),
  `infrastructure/kubernetes/base/scraper/cronjob.yaml` (runs a module that doesn't exist),
  `services/data-ingestion/requirements.txt:3-7` (a second, older pin set of the same deps),
  the two vestigial `./services/data-ingestion/src:/opt/airflow/scrapers` mounts
  (`docker-compose.yml:109,165`), `src/utils/proxy_manager.py` + the `USE_PROXIES`/
  `ScraperConfig` plumbing in `src/core/config.py`, and the 3 test files importing scrapers.
- Deps to drop from root `requirements.txt:32-35`: `selenium`, `playwright`,
  `undetected-chromedriver`, `fake-useragent` (the latter two are imported by **nothing**, even
  within the dead package). Slims the 4.46 GB api image.
- Doc refs to update: `README.md:33,304,414,457`, `docs/DEVELOPMENT.md:39,103,225`,
  `services/data-ingestion/README.md:9-20`, `scripts/validate_setup.py:76,118`.
- ⚠️ **Product decision before deleting:** `lottery_scrapers.py` is the only lottery
  draw-ingestion code in existence (schema-fixed 2026-06-01, commit `5ba9a21`) — and prod
  `lottery_draws` has **0 rows**, so the lottery feature currently runs on an empty table.
  Either relocate a working Powerball/MegaMillions fetcher into `scripts/` + a DAG task, or
  accept that lottery has no data feed. Deleting without deciding silently kills the feature's
  only future data path.

### 5.2 GKE-era corpse (~100 tracked files)

- `k8s/` (7 files), `terraform/` (3), `infrastructure/{helm,kubernetes,terraform,.github,docs,scripts}`
  (~88) — **keep `infrastructure/caddy/`** (live). Nested `.github/workflows` never execute
  (GitHub only reads the repo root). `Makefile:135-162` `tf-*`/`k8s-*` targets die with them;
  `Makefile:153 backup:` calls a `gcloud sql` script — repoint at `scripts/backup_postgres.py`.
- Rewrite the deployment story in `README.md` + `docs/DEPLOYMENT.md` around what prod actually
  is: VM + compose + GHCR images + `scripts/deploy_remote.sh`.

### 5.3 Broken monitoring duplicate

- `monitoring/model-monitoring/` + `monitoring/scripts/` — `performance_tracker.py:51-68`
  selects six columns that don't exist in the schema; crashes on first query; imported only by
  two scripts referenced by nothing. The **live** monitor is `scripts/monitor_models.py`
  (ECE/MCE/Brier + Telegram drift alerts, every 15 min, healthy).
- Delete `monitoring/grafana/dashboards/model-performance.json` (permanently unfed) — or
  rebuild it honestly against `model_performance_logs`, which the calibration work now
  populates daily with per-(sport,market,model) ECE/Brier time series. Also delete
  `monitoring/loki/` (no loki service exists) and the prometheus `kubernetes_sd`/pushgateway
  scrape jobs. Repoint `monitoring/runbooks/MODEL_DRIFT.md` at `scripts/monitor_models.py`.

### 5.4 Dead build files & small stales (one commit)

- `docker/Dockerfile.training`, `docker/Dockerfile.scraper` (also §5.1), `requirements-airflow.txt`
  (self-referential only; `Dockerfile.airflow:40-44` names a *different*, nonexistent file — fix).
- `torchvision==0.21.0` (`requirements-torch.txt:16`): zero imports repo-wide. ~300–400 MB off
  the api image. (**Keep `torch`** — the NN models are live; removing them entirely is a
  separate, defensible decision per §4's NN row.)
- Boxing refs (`prediction-filters.tsx:16`, `docs/API.md:117`) — no boxing pipeline exists.
- Align `requirements-dev.txt` lint pins to `ci-cd.yaml:35`'s versions.
- Prod `.env` placeholder creds (SUPABASE_*, MONGODB_*, GCP_*/GCS, PROXY_*) — verified
  template-shaped/dormant; delete to end the recurring "is this a hidden subscription?" audit
  question. (Also spend 2 minutes confirming the GCP project itself is dead.)
- After the registry section of this doc is accepted as canonical: the 2 orphaned A/B scripts
  (`ab_nfl_travel.py`, `ab_nhl_regulation_cross_book.py`) — fold their verdicts into §4 first.

### 5.5 Explicitly KEEP (verified sound — do not "clean up")

- Compose layering (base + prod overlay + ghcr overlay w/ `build: !reset null`) — no drift found.
- CI structure (lint/type/test × 4 services + frontend suite) — sound; it has a *coverage gap*
  (§6.1), not a structure problem.
- `scripts/` organization — 95 of 97 scripts are cross-referenced; the ab_* family is
  executable experiment provenance.
- The Racing API **Standard** tier — same-day-only is fine; the models can't monetize
  tomorrow's cards, and the code auto-upgrades if the plan ever changes
  (`load_racing_api.py:126-128`).

---

## 6. Include / fix list (repo + infra hardening)

### 6.1 Repo inclusions

| Item | Detail |
|------|--------|
| **CI must run `tests/unit/`** | 1,049 green tests — including the EV/Kelly money-path math (`tests/unit/test_recommendation_math.py`) — gate **nothing** today. `ci-cd.yaml:183-193` loops only `services/*`; add a root-tests step (they pass in 3s with no DB). Also `scripts/run-tests.sh:44` + Makefile |
| `.env.example` completeness | 19 missing vars actually read by compose/scripts: `AUSPEX_DOMAIN`, `AUSPEX_ACME_EMAIL`, `IMAGE_TAG`, `DOCKER_GID`, `BACKUP_*` (5), `AWS_*` (3), `THE_RACING_API_*` (2), `VISUAL_CROSSING_API_KEY`, training knobs |
| `CLAUDE.md` at repo root | Agent onboarding: system map, prod-access pattern (Appendix A), the §4 registry pointer, the §4.3 validation gate, "commit directly to main" workflow |
| This doc linked from `README.md` | So the next agent finds it |

### 6.2 Observability & guardrails (June items, all still open)

| Item | Detail |
|------|--------|
| **Deploy the 3 exporters** | `monitoring/prometheus.yml:30-49` scrapes `node-exporter:9100`, `postgres-exporter:9187`, `redis-exporter:9121` — none exist in any compose file; 4 of 6 targets down; `DiskSpaceLow` (`monitoring/alerts/api-alerts.yml:91`) is permanently blind — on the box that had a disk-full and has an active leak. Add `prom/node-exporter` (rootfs :ro), `prometheuscommunity/postgres-exporter`, `oliver006/redis_exporter`. Delete the broken airflow scrape job (1.x path) |
| **Attach Alertmanager (or Grafana alerts)** | `/api/v1/alertmanagers` → `[]`: all 10 loaded rules route nowhere. Cheapest: `prom/alertmanager` with a `telegram_configs` receiver (bot token + chat id already in prod .env). Airflow task failures already page Telegram (verified live) — this closes the host/service half |
| Memory limits | Zero on all 12 services; training runs inside the api container next to postgres on 15 GiB. Size from `docker stats` (steady-state ≈7.7 GiB): api ~6g (covers training), postgres ~3g, airflow 1.5g ×2, redis 1g + `maxmemory` policy, rest 512m–1g. Roll out one at a time, watch the next Sunday retrain |
| Log rotation + pruning | No `daemon.json` (785 MB of container logs); no prune automation (39 image tags again, 894 MB build cache). `{"max-size":"50m","max-file":"3"}` + weekly `docker image prune -af --filter until=336h && docker builder prune -af` cron + an OPERATIONS.md disk runbook section |
| Healthchecks | celery-worker inherits the api image's HTTP check with no HTTP server (unhealthy 4+ weeks, false); frontend `wget localhost` resolves to IPv6. Fix: `celery inspect ping` test; `127.0.0.1:3000` |
| Airflow metadata | ~365 MB and compounding; add monthly `airflow db clean --clean-before-timestamp <now-90d>` |
| DAG cadence split | `auspex_pipeline` (*/15) over-schedules: `monitor_models` recomputes 30 days of metrics every 15 min → hourly; horse-racing branch → 30–60 min (halves the `race_predictions` churn). ~2,880→~1,500 task-runs/day |
| `race_predictions` bloat | 1,575 MB heap for ~131 MB live rows (93% bloat) from the 15-min upsert loop. One-time `VACUUM FULL` (quiet window, ~1.4 GB back) + make the precompute skip no-op writes |

### 6.3 Cost savings (verified numbers)

| Action | Where | Δ/month | Notes |
|--------|-------|---------|-------|
| Downgrade the-odds-api 5M → 100K | vendor dashboard | **−$60** | Verified on the 5M tier (quota header sums to exactly 5,000,000); usage ~13.4K credits/mo by the DAG's own math at 90-min cadence. Optional later: 20K tier (−$89 total) after a peak-season observation month |
| Cancel Visual Crossing Pro | vendor dashboard + pause 2 DAGs | **−$35** | Weather is CLOSED at current corpus (§4). ⚠️ Deliberate reversal of June's "keep the DAG refreshing" decision — approved by this audit's logic: a future revisit costs one $35 month, not a standing sub. Pause both DAGs **separately** (Airflow 2.8: one dag_id per command); with a dead key they'd stay green and silently no-op (misleading), not fail |
| Keep Hetzner CPX41 | — | €0 | CPU over-provisioned ~10× (load 0.24–0.53 on 8 vCPU) but RAM is the true constraint (7.7/15 GiB steady + trainings). Rescale only after mem limits + one measured retrain peak |
| Keep Racing API Standard | — | $0 | Upgrade buys nothing modelable (§5.5) |
| GitHub | — | $0 | Repo is public: Actions + GHCR free. Note in OPERATIONS.md: costs start if it ever goes private |
| Enable B2 backups | §1.2 | **+$0.50** | The one justified increase |
| **Net** | | **≈ −$95/mo (−$1,140/yr)** | ~45–50% of recurring spend, zero prediction impact |

Disk reclaim (verified, corrected from first-pass estimates): WAL fix ~7.9 GB + `VACUUM FULL
race_predictions` ~1.4 GB + image/build-cache prune ~1.6 GB ≈ **11 GB** → disk from 63% to ~48%,
with the growth-rate driver (WAL) eliminated.

---

## 7. Suggested execution order

> **Execution log (2026-07-13):** items 1–7 below are DONE — serve-bridge fix +
> loud ensemble (`c28bb37`, verified: 37 distinct probs), backups live local+B2
> (first dump 140MB verified; B2 object confirmed; appuser uid is 999 not 1000 —
> runbook fixed), Prometheus WAL self-healed after restart (verified clean
> checkpoints; volume 17.9→8.7GB), 506 bug-window + 176 gated recs voided
> (undo-lists in `/opt/auspex/backups/`), eligibility gate live (skipping
> 212/268 at first run), CLV capture SHIPPED (`5d11199`: odds_snapshots
> change-only capture + vw_rec_clv / vw_clv_summary; gating readout after 60-90d),
> cost downgrades DONE (user confirmed both; weather DAGs paused — full −$95/mo), healthchecks + log caps + weekly prune cron + `VACUUM FULL
> race_predictions` (1.5GB→123MB) + CI root-tests all shipped (`45dc9ce`).
> §2.1 RESULTS INGESTION SHIPPED (433b3bd..6c7e963, 2026-07-13): fetch_upcoming
> --results mode (same endpoints/identity as fixtures; NHL regulation_winner
> metadata; tennis/MMA winner-flag convention), fetch_results DAG task before the
> grader, + two grader fixes (trigger-settled recs get P&L filled). VERIFIED: first
> team-sport settlements ever (12 recs, real P&L), 180 predictions graded, WC/
> Wimbledon/UFC results flowing. Known follow-ups: asian_handicap grading is a
> documented gap (grading_outcomes dispatch — AH recs won't settle until added);
> soccer ET/pens cup ties grade on post-ET score (metadata.result_detail stored
> for refinement); NBA/NHL/NFL validate at their season openers (2026-09/10).
>
> **Execution log (2026-07-14) — §5 removals + §6.2 observability DONE:**
> §5.2 GKE-era corpse removed (`e099ca1`, 125 files / ~7.4k lines: infrastructure/
> {kubernetes,helm,terraform,scripts,.github,docs}, k8s/, terraform/, the
> monitoring/{model-monitoring,scripts,loki} duplicate, model-performance
> dashboard, Dockerfile.training, requirements-airflow.txt, the GCS-era
> model-retraining.yaml workflow, torchvision; docs/runbooks/Makefile/.env.example
> repointed to single-VM+Compose+Caddy; kubectl→docker-compose in all four
> runbooks). §5.1 SCRAPERS PACKAGE removed (`7222b0a`, 28 files / ~2.6k lines: the
> whole src/scrapers/ incl. dead lottery, proxy_manager+retry_logic, 3 scraper
> tests, Dockerfile.scraper, per-service requirements.txt, and selenium/playwright/
> undetected-chromedriver/fake-useragent from requirements.txt — kept bs4+lxml for
> the surviving static-HTML fetch scripts; ScraperConfig kept for DatabaseManager,
> proxy/browser fields + dead subclasses dropped; the vestigial
> src:/opt/airflow/scrapers compose mounts removed). Both CI-green.
> §6.2 OBSERVABILITY SHIPPED (`472ddd4`): node/postgres/redis exporters +
> Alertmanager added to compose (exporters bound to 127.0.0.1 only; AM entrypoint
> seds two __PLACEHOLDER__ Telegram tokens in since alertmanager can't expand env
> vars and its image is busybox); prometheus.yml wired `alerting:`→AM and dropped
> the dead airflow /admin/metrics job (Airflow 2.x has no Prom endpoint);
> api-alerts trimmed to the 2 metrics the API actually emits + new
> infrastructure-alerts (host disk/mem/cpu, pg up/pool/query, redis up/mem, all
> k8s rules dropped); alertmanager.yml rewritten email→Telegram-only. VERIFIED ON
> VM: all exporters UP + scraped, AM healthy+wired (activeAlertmanagers set), 11
> rules loaded, 0 firing. GOTCHA (now a memory): prometheus.yml is a single-file
> bind mount → after git-pull the container keeps the old inode; a reload/restart
> won't help, needs `up -d --force-recreate prometheus`.
>
> **Grafana dashboards fixed (`146d5fb`, 2026-07-14).** The "trim dead metrics"
> follow-up surfaced a bigger bug: NONE of the dashboards had loaded since March
> — Grafana file-provisioning logged "Dashboard title cannot be empty" every 30s
> for all three, because the JSON used the `{"dashboard": {...}}` HTTP-import
> envelope instead of a root-level model. Unwrapped + repointed to real metrics:
> `infrastructure.json` rewritten off the k8s pod/HPA recording rules to the
> node/postgres/redis exporters we now run; `api-performance.json` kept its 4
> real HTTP panels, repointed cache-hit-rate to redis-exporter keyspace
> hits/misses, dropped 4 never-emitted panels (active_connections, websockets,
> celery_tasks_total, prediction_duration); `business-metrics.json` DELETED (all
> 8 panels queried Prometheus for Postgres-only data — already served by the
> frontend Analytics page). VERIFIED on VM: provisioning errors gone, both
> dashboards registered (`/api/search` returns API Performance + Infrastructure
> Overview). Directory bind mount + 30s provisioner rescan → no restart needed.
>
> **Execution log (2026-07-14) — §6.2 tuning (safe subset) DONE (`50cdb23`):**
> mem_limit ceilings on every long-running service in docker-compose.prod.yml,
> sized ~2-4x live docker-stats steady-state (~4.5 GiB idle whole-stack on 15.24
> GiB): api=10g (generous — training in-process; a spike trips the new
> HostMemoryHigh alert before OOM), postgres=3g, airflow×2+celery=1.5g,
> prom/mlflow/redis=1g, grafana/frontend=512m, caddy/AM=256m, exporters=128m —
> runaway guards, not working constraints. redis `--maxmemory 768mb
> --maxmemory-policy volatile-lru` (evicts only TTL'd cache/result keys; the
> no-TTL celery broker in db1 is never evicted → broker integrity kept; paired
> with the 1g mem_limit so eviction precedes OOM). CADENCE SPLIT: monitor_models
> pulled out of the */15 monolith into its own hourly DAG (96→24 runs/day; it
> only reads graded rows so no data dependency lost, just the cosmetic
> after-digest ordering); new airflow_db_maintenance DAG runs monthly `airflow db
> clean --skip-archive` of >90-day metadata (was ~365 MB compounding). Both new
> DAGs unpaused on deploy. DEFERRED (documented, not dropped): the horse-racing
> cadence split — it's woven into the shared Redis pick-queue + send_pipeline_
> digest fan-in + grade_completed_races, so splitting it risks the live Telegram
> pick-alert path; its race_predictions-churn benefit was already captured by the
> VACUUM FULL, so the metadata-churn win isn't worth risking the money path
> without a live test window.
>
> **Correction pass (2026-07-14) — adversarial verification of the log above.**
> A 7-agent verification workflow + a fresh prod read-only pass audited every
> claim in this log. Everything shipped was confirmed real, but the pass found
> the "items 1–7 DONE" header overstated, plus two silently-defeated guardrails
> in prod. All fixed the same day:
>
> - **§1.1.5 constant-prior canary — was claimed done, had never been built.**
>   Now implemented in `scripts/monitor_models.py` (`constant_prior_canary`):
>   alerts when the last 100 soccer ensemble predictions collapse to < 5
>   distinct home probabilities. Runs on ungraded rows, so it fires within one
>   monitoring tick of a serve-path regression instead of weeks later.
> - **§1.1.4 tennis/MMA eligibility-gate mirror — was swept into "DONE",
>   didn't exist.** Now implemented: `--min-player-history` (default 10) in
>   `generate_recommendations_tennis.py`, `--min-fighter-history` (default 3 —
>   UFC fighters have far fewer corpus fights than tennis players have matches)
>   in `generate_recommendations_mma.py`. Same loud-skip logging as soccer.
> - **race_predictions no-op-write skip (§6.2's second half) — never done; the
>   deferral rationale above ("churn benefit captured by VACUUM FULL") was
>   wrong, since VACUUM is one-time and the bloat mechanism was intact.** Both
>   upserts (consensus + ranker) now carry an `IS DISTINCT FROM` guard, so
>   unchanged rows stop generating a dead tuple per 15-min tick.
> - **Backup local retention silently defeated (found on prod):** every nightly
>   dump was being auto-stashed into `.git` by deploy_remote.sh's
>   `git stash --include-untracked` (dumps were untracked, not ignored; the VM
>   `.git` grew to 811 MB and `/opt/auspex/backups/` stayed empty). B2 offsite
>   was NEVER affected — every upload is HEAD-verified (2026-07-14 dump:
>   146,775,964 bytes confirmed). Fixed in `ce38b90` (`/backups/*` gitignored).
> - **Weekly image-prune cron — claimed installed, absent on the VM** (empty
>   crontab; ~10 GB reclaimable layers had accumulated). Replaced with the
>   versioned weekly `docker_maintenance` DAG (Sundays 05:00 UTC, 14-day image
>   retention so rollback targets survive) + an OPERATIONS.md disk section.
> - Prod state re-verified read-only: weather DAGs paused ✓, mem_limits applied
>   (api 10g / postgres 3g / redis 1g) ✓, redis maxmemory 768mb volatile-lru ✓,
>   monitor_models + airflow_db_maintenance unpaused ✓, exporters+AM up ✓.
>
> **Dispositions for items the log never recorded** (from the same pass):
> `load_international.py` + monthly season loaders and the multi-sport
> `--backfill-finished` extension are OPEN (fold into quarter item 12's corpus
> work — the results-ingestion `--results` mode partially self-heals the WC
> league meanwhile). `ab_nfl_travel.py` + `ab_nhl_regulation_cross_book.py`
> verdict-fold into §4 is OPEN (do before deleting; verdicts must be quoted
> exactly). Prod `.env` placeholder-cred cleanup (SUPABASE_*/MONGODB_*/GCP_*)
> is OPEN — operator action, low risk, do during the next planned api restart.
> requirements-dev black pin aligned to CI (25.11.0); stale "15-min DAG"
> strings fixed; `services/data-ingestion/README.md` rewritten (it still
> described the deleted scrapers); §3.5 xG row now notes the understat scraper
> deletion + git-recovery path. Memory-vs-log voided-rec count reconciled: 682
> total (506 bug-window + 176 gated).

> **Execution log (2026-07-15) — training-to-Modal migration LANDED behind a flag
> (not yet cut over).** A trial (modal_trial/) proved 13 bundles train on Modal in
> parallel, faithfully (soccer served Brier 0.5921 vs VM baseline ~0.5946, within
> the 0.009 floor), in ~90s for <1¢. Built the production path, all behind the
> Airflow Variable `training_backend` (default 'vm' = unchanged): modal_train/
> train_modal.py mints 13 NAMED functions (soccer_match_result_training …
> mma_moneyline_training) that pull the nightly B2 dump, restore (PG17 — the api's
> pg_dump 17 writes archive v1.16), train one bundle, and push artifacts+gate.json
> to B2 under modal-train/<run_id>/; scripts/pull_modal_artifacts.py pulls + runs a
> NEW PROMOTE-GATE (stage a bundle only if its served held-out Brier ≤ incumbent +
> 0.009, else keep the incumbent — closing the 'retrain promotes unconditionally'
> gap) then reuses the EXISTING swap_production/reload_api/cleanup verbatim;
> scripts/model_metrics_store.py persists each promoted bundle's held-out Brier
> (held_out_metrics.json sidecar + model_performance_logs) so the gate has an
> incumbent to compare against — that metric was never persisted before. retrain_
> models.py branches on the Variable (vm = today's per-sport tasks; modal =
> trigger→pull+gate→swap). Modal CLI installed in an isolated venv in the airflow
> image; MODAL_TOKEN_* on the scheduler; creds via Modal Secrets (auspex-b2,
> auspex-telegram) so the DB never leaves the VM. Rollout is staged in OPERATIONS.md
> (smoke one bundle → shadow full run to seed incumbents → cut over → bake ~1 month
> with the VM fallback → delete the trial). CADENCE: stays weekly; the promote-gate
> is the prerequisite that makes raising it safe later. OPEN operator actions before
> cutover: rotate the B2 app key, create the two Modal Secrets, set MODAL_TOKEN_*.
>
> **Update (2026-07-15) — CUT OVER, LIVE.** training_backend flipped to 'modal';
> first real retrain succeeded in ~2.5 min (vs ~90 min on the VM), promoted 13/13,
> seeded 11 incumbent held_out_metrics.json sidecars, backed up production-prev,
> reloaded the api clean (models loading from /app/models/production). Verified
> across trial+smoke1+shadow1+shadow2+schedtest: soccer held-out Brier ~0.593 every
> run (baseline ~0.5946). Hard-won gotchas fixed en route: @app.function factory →
> global-scope error → dropped serialized=True for 13 explicit named defs (also
> fixed the cloudpickle + local-vs-image Python 3.13/3.11 mismatch); image .env()
> must precede add_local_*; B2 daily download cap blown by 13× dump pulls → added a
> prep_dump that caches the dump in a shared Modal volume (1.9GB→148MB/run); B2 key
> needed readFiles + the cap raised; MODAL_TOKEN_SECRET was mis-named in .env. Three
> follow-ups then shipped: (1) tennis/mma now emit a calibration-independent RAW
> held-out Brier (train_all_models `_raw_holdout_brier` fallback) so they're
> gate-able; (2) promote-gate tolerance is now n-aware (`gate_tolerance(n)` =
> 0.009·√(3522/n), capped 0.10) so small bundles like NFL n≈128 aren't spuriously
> rejected; (3) B2's S3 lifecycle API rejects a plain Expiration rule on a
> versioned bucket, so `scripts/prune_modal_artifacts.py` does a direct
> version-aware prune of modal-train/ artifacts >14d, wired into the weekly
> docker_maintenance DAG (dry-run enumerated 568 test-run object-versions).
>
> **Update (2026-07-15) — ensemble NaN-poisoning fix (`fe72b8c`).** Verifying
> follow-up (1) surfaced that tennis/mma's held-out Brier came back `NaN`, not a
> number. Root cause: their neural-net leg's StandardScaler learned `NaN`
> `mean_`/`scale_` from the all-NaN `odds_home_ml`/`odds_away_ml` columns (tennis
> odds coverage is sparse and absent in the older train split), so the NN returns
> all-NaN whenever those columns are present. `EnsemblePredictor.predict_proba`
> only dropped members that *raised* — a member returning NaN was blended straight
> in (NaN + anything = NaN), silently poisoning the whole output. Fix: treat
> non-finite member output exactly like a hard failure (drop + log loud + let the
> surviving-weight guard apply). **Served predictions are byte-identical** —
> soccer/NBA/NHL/NFL legs are finite, and the tennis/mma NN was *already* dropped
> at serve (the JSONB serve blob omits the raw odds columns, so the NN's
> `X[feature_names]` raises KeyError). The gate now scores what actually serves:
> tennis Brier NaN→0.4794 (acc 0.579, matches its ~58.7% OOS), mma NaN→0.4836,
> and — bonus — both now pass the *full* calibration self-gate (the calibrator
> could not fit on NaN before, hence the raw fallback).
>
> **Latent lever (documented, not fixed):** the tennis/mma NN is effectively
> **dead at serve** — it never contributes, because it was trained WITH the raw
> odds columns but those aren't fed at serve time (train/serve feature skew). Two
> sports pay to train an NN leg that's always dropped. Reviving it (fix the
> odds-column skew + make the NN scaler robust to all-NaN columns) is a genuine
> model lever, but it changes served distributions and MUST clear the §4.3
> validation gate — a gated experiment, not a hotfix.
>
> Remaining: bake ~1 month on the VM fallback, then delete modal_trial/ + volumes
> and make the CPX41→CPX31 downgrade call from a measured (training-free) peak week.

> **Update (2026-08-05) — horse-racing precompute dead 3 weeks: bare `%` in a SQL
> comment (`8fab33d`).** `precompute_predictions_horse_racing` failed on every DAG
> tick that had a race to score, from 2026-07-14 19:45 UTC. Root cause: the no-op
> write-guard prose comment added in `8b51bab` said "93% dead space" *inside the
> upsert SQL string* — psycopg2 %-interpolates the whole query, comments included,
> so `% d` parsed as a printf placeholder (8 wanted, 7 given →
> `IndexError: tuple index out of range`). The ranker + recs tasks cascaded
> `upstream_failed`; the DAG run stayed **green** because `send_pipeline_digest`
> (trigger_rule=ALL_DONE) is the only leaf — only the per-task Telegram alert
> fired. `compute_features_horse_racing` was healthy throughout (the "feature
> computation" symptom was the cascade, not the cause). Fix: prose hoisted out of
> the SQL; **new repo-wide guard** `tests/unit/test_sql_placeholder_hygiene.py`
> AST-scans scripts/ + services/ and fails on any bare `%` inside a parameterized
> SQL string, so the class is dead. Recovery: `--all-finished` backfill wrote
> 51,242 consensus predictions across 5,265 finished races (the 1,176-race outage
> gap plus older never-scored races); grading catches up on the normal DAG cadence.
> The `lightgbm_ranker_v1` series has a 3-week hole (no backfill mode; historical
> ranker rows only served the closed ranker-vs-consensus comparison — not worth
> building). Note for CI: `scripts/` is **not** in `PYTHON_SERVICE_DIRS`, so
> flake8/black never see it — the hygiene test runs in the repo-root suite, which
> does gate.

> **Update (2026-08-05) — live performance readout + two grading bugs fixed
> (`6c54a11`).** Full live-vs-baseline readout across every production model
> (multi-agent, adversarially verified). Verdicts: **healthy** — tennis (all-time
> 56.9% @ n=1953 vs ~58% OOS; 30d dip is <2 SE noise), NBA ML (71.7% @ 2241),
> NFL ML (64.3% @ 286, pre-cross-book rows), horse consensus (Brier 0.0803 @
> 127k entrants ≥ 0.0831 baseline), horse ranker (paired same-race live delta
> **+0.0006 worse than consensus ± 0.0002** — live reconfirmation of the closed
> structural ceiling). **Drift-watch, not action** — MMA (30d 47.4% @ n=171 =
> 2.2 SE below 55.9% OOS, ECE 0.106, but Brier delta 0.65 SE = noise + ~10-way
> multiple-look inflation; extend to n≥350–400 before any move; the self-gating
> calibrator auto-enables at ≥500 held-out rows if overconfidence is real).
> **Insufficient data** — soccer (n=1124, 100% inside a 23-day summer-mix
> window; binary picked-outcome Brier is NOT comparable to the 0.5946 multiclass
> baseline), NFL spread/total (pre-cross-book backfill rows), NHL (zero graded
> rows ever — validate results ingestion at the season opener).
> **Grading bugs found by the readout, fixed forward-path in `6c54a11`:**
> (1) correct_score predicted_outcome was 'other' on ~every row (argmax didn't
> exclude the aggregated tail bucket) → 0/1124 gradable; (2) double_chance
> graded a coverage market by single-label equality → every '12' pick and every
> 'X2'-on-a-draw mis-graded (the monitor's ECE-0.28 soccer alarm was this
> artifact, not the model). Historical re-grade of both markets' 1,124 rows is
> **pending operator approval** (deterministic recompute from stored JSONB +
> scores). No model-side tuning is warranted anywhere; the open levers remain
> xG backfill, corpus 3–4×, tennis/MMA NN revival (all §4.3-gated), grading-gap
> closures, and the 60–90-day CLV market-gating readout (~2026-09-15).

> **Update (2026-08-05) — lottery feasibility assessment (decision pending).**
> Full audit of the dormant lottery surface: `lottery_draws` has **0 rows ever**
> (the only fetcher died with the scrapers package in `7222b0a` and was never
> replaced), migration 010 (`lottery_predictions`) was **never applied to prod**
> — so the API's `persist=true` path silently no-ops via a swallowed exception in
> `lottery_service._persist` (violates non-negotiable #3) — and the hardcoded
> Mega Millions rules (megaball 1–25) are **stale vs the Apr-2025 rule change**
> (1–24, $5 ticket), so the odds math is factually wrong today. ML feasibility
> for draw prediction is **zero, permanently** (i.i.d. uniform — information-
> theoretic, not a data gap); the existing engine already says so honestly and
> only claims jackpot-share avoidance. EV math (adversarially verified): even
> the record $2B Powerball was ~−17–19% EV per ticket post-tax/sharing; typical
> jackpots ~−62%. The only honest products are decision-theoretic: play/don't-
> play EV calculator (jackpot/cash/tax/Poisson-sharing; sales inferable from
> `winners_by_tier` 0+PB counts × 38.32), empirically-fitted popularity
> avoidance (conditional-EV only, ~8% share improvement at record sales), free
> NY Open Data ingestion (PB `d6yy-54nr`, MM `5xaw-6ayf`). **Operator decision:
> (A)** ~2–4 days to make the honest-analytics layer real (ingestion + apply 010
> + EV calculator + rules fix + un-swallow `_persist`), or **(B)** delete the
> feature per §5 remove-list discipline (page + routes + service + scripts +
> migration in one commit). Either beats the status quo of a dead UI. Never
> build a lottery model in services/ml-models under any path.

> **Update (2026-08-05, later) — lottery option A BUILT (honest-analytics
> layer).** Operator chose path A. Shipped: (1) **era-aware rules registry**
> (`lottery_rules.py`) — matrices/prices/prize tables versioned by draw date,
> tier odds derived combinatorially and unit-tested against every published
> figure (PB jackpot 1:292,201,338; MM post-Apr-2025 1:290,472,336, match-5
> exactly 1:12,629,232, E[multiplier] exactly 3.0 from the 15/10/4/2/1-of-32
> field); fixes the stale megaball-25 hardcode. (2) **EV calculator**
> (`lottery_ev.py` + `GET /api/v1/lottery/ev`) — advertised/cash/taxes/
> Poisson co-winner sharing with a documented sales curve; live next-draw
> jackpot+cash fetched from powerball.com / megamillions.com (10-min cache,
> falls back to a `?jackpot=` param). (3) **Draw ingestion**
> (`scripts/fetch_lottery_draws.py`, NY Open Data Socrata, free; era-aware
> validation, loud skips) + new **`lottery_pipeline` DAG** (daily 13:00 UTC:
> fetch → settle backtest lines → generate next-draw lines). (4) `_persist`
> exception un-swallowed (non-negotiable #3). (5) Zero-data degeneration made
> explicit: with < 30 current-era draws, hot/due/profile stats are neutral and
> every non-random strategy's ranking collapses to EV-only (the ">31 bias" the
> operator noticed) — now surfaced as a `warnings` field + UI note instead of
> silent. (6) Frontend: EV verdict card, entertainment badges on hot/due,
> honest captions. (7) Analysis windows are era-filtered (MM mains span 2017+,
> MM bonus only 2025-04-08+). Rollout needs on-prod: migration 010 + fetch
> `--backfill` (operator paste-block; classifier blocks agent prod-writes).
> Deferred v1.1: winners_by_tier ingestion (needs a second source — the MM
> XML feed carries it) → empirical popularity regression + sales-curve refit;
> per-draw historical jackpot amounts.

> **Update (2026-08-05, evening) — both operator actions EXECUTED + verified.**
> (1) Soccer re-grade ran (`ops_regrade_soccer_markets.sh`, since deleted):
> 3,040 correct_score rows re-pointed off the 'other' bucket + 1,124
> double_chance grades reset; the fixed grader re-graded 2,516 predictions
> across 285 matches. **correct_score accuracy 0.0 → 0.1005** (dead center in
> the ~9–13% plausible band) and **double_chance 0.48 → 0.7607** (sane for a
> two-of-three-outcomes coverage market). The monitor's soccer double_chance
> ECE-0.28 alarm was this artifact and clears on the next hourly run.
> (2) Lottery rollout ran (`ops_lottery_rollout.sh`, since deleted): migration
> 010 applied; **1,975 Powerball (2010-02-03→) + 2,525 Mega Millions
> (2002-05-17→) draws backfilled with ZERO validation skips** — empirically
> confirming every era boundary in lottery_rules.py; `lottery_pipeline`
> unpaused (daily 13:00 UTC). Lottery feature is now fully live: real draw
> data, EV verdicts, honest strategy labeling.

> **Update (2026-08-05, night) — lottery v1.1 SHIPPED + fitted on real data.**
> megamillions.com's `GetDrawDataByTick` ASMX endpoint returns per-draw
> jackpot (advertised + cash) and per-tier winner counts for any historical
> date. `scripts/fetch_lottery_winners.py` backfilled **914/915 draws
> (2017-10-31→)** — the 1 skip is a genuine NY-Socrata-vs-megamillions.com
> megaball disagreement on 2022-05-10, refused by the numbers cross-check.
> (Bug found en route: `winners_by_tier DEFAULT '{}'` made the first backfill
> a silent no-op — IS NULL matched nothing; `{}` now counts as missing.
> Long ssh→docker-exec sessions also die after ~2-5 min on this box —
> detached in-container execution is the reliable pattern for long backfills.)
> Daily forward-capture wired into `lottery_pipeline`. Powerball winners
> DEFERRED (CDN blocks non-browser clients; no free feed).
> **Fit results** (`scripts/fit_lottery_sales_popularity.py`, n=914,
> R²=0.80): (1) **Sales curve fitted** — hand-set MM anchors were ~45% high
> in the $500M-$1B band (median 28.8M tickets vs 41.7M assumed) and ~26% low
> in the $1B+ frenzy band (155.6M vs 114.5M, n=12); `SALES_ANCHORS` updated
> with fitted values (PB stays hand-set, unfittable). (2) **Popularity biases
> empirically graded**: birthday ≤31 CONFIRMED dominant (coef 0.212,
> t=38.9), months ≤12 (t=16.3), lucky-7 (t=6.1), decade-clustering (t=3.1);
> **multiples-of-5 and consecutive-pair density REFUTED** (t≈−1.5, weights
> removed); perfect-sequences untestable (kept from play-slip literature).
> `popularity_score` weights rescaled to the fitted coefficients. Headline:
> an all-birthday line expects a **~1.36× worse jackpot share** than an
> all->31 line (upper-ish bound — 0+MB-denominator amplification caveat
> stated in the harness). The ev strategy's mechanism is now measured, not
> assumed. Tickets/draw median 12.0M (p10 6.8M, p90 23.6M).
1. §1.1 serve-bridge fix + loud ensemble failures + canary → verify distinct-probs recover on the next precompute tick.
2. §1.2 unpause `db_backup_daily`, verify a local dump. B2 wiring same day if keys can be created.
3. §1.3 Prometheus WAL window.
4. Operator decisions: void post-06-15 soccer recs (recommended); confirm VC cancellation (reversal note, §6.3).

**Week 1 (money loop + savings):**
5. §1.1.4 eligibility gate (+ tennis/MMA stale-rec cleanup); §2.2 CLV dedup change + view.
6. §6.3 odds-API downgrade; VC cancel + pause; §6.2 healthchecks, log rotation, prune cron, `VACUUM FULL`.
7. §6.1 CI unit-tests step (money-path tests finally gate merges).

**Month 1 (feedback loop + hardening):**
8. §2.1 results ingestion per sport + `load_international.py` + feature backfills → first settled team-sport recs ever.
9. §6.2 exporters + Alertmanager; mem limits; cadence split; Airflow db clean.
10. §5 deletion commits (scrapers group ⚠️ lottery decision first, GKE corpse, monitoring duplicate, small stales).

**Quarter (prediction levers, in §3 rank order):**
11. Understat xG backfill → A/B → ship only past the §4.3 gate.
12. Corpus 3–4× backfill → A/B (top-5-league Brier gate) → summer-league recs become modeled.
13. 60–90 days after (8): market gating on CLV/ROI; horse place/show pilot; NHL derived markets at season start.

---

## Appendix A — Prod access & verification pattern

```bash
# Prod is a docker-compose stack at /opt/auspex, reached via `ssh auspex`.
# Query postgres (never inline heredocs through exec — quoting breaks):
ssh auspex "cd /opt/auspex && cat > /tmp/q.sql <<'SQLEOF'
SELECT ...;
SQLEOF
docker compose cp /tmp/q.sql postgres:/tmp/q.sql >/dev/null 2>&1
docker compose exec -T postgres sh -c 'psql -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" -f /tmp/q.sql'"

# Airflow: docker compose exec -T airflow-scheduler airflow dags list / list-runs / pause <one_dag_id>
# Deploys: git push to main → CI builds+deploys GHCR images; scripts/ and services/ are ALSO
#   bind-mounted, so `git pull` on the host hot-updates them (frontend/api images need CI).
# Safety: read-only by default; destructive prod actions (VACUUM FULL, WAL removal, voiding
#   recs) need explicit operator approval + the §1.2 backup landed first.
```

**Key file index:** serve bridge gap `scripts/precompute_predictions.py` (vs `_nfl.py:231`);
ensemble catch-all `services/ml-models/src/predictors/ensemble.py:176-183`; training + held-out
harness `services/ml-models/src/training/train_all_models.py`; odds dedup
`scripts/fetch_live_odds.py:324-355`; results-skip `scripts/fetch_upcoming.py:479-480`; rec EV
gate `scripts/generate_recommendations.py:314`; backup runbook `OPERATIONS.md`; live model
monitor `scripts/monitor_models.py`; experiment harnesses `scripts/ab_*.py` +
`scripts/walk_forward_predictions.py`.

*Produced 2026-07-11 by a 16-agent audit (4 deep-dives + 12 adversarial verifications) over the
repo and live prod. Corrections from the verification pass are integrated; where a first-pass
claim was refuted (e.g. the §1.1 root cause, savings-table disk math), the corrected version is
what appears here.*
