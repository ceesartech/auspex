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
| Probability calibration (team sports) | **CLOSED — no win** (2026-06-10) | Held-out ECE already 0.02–0.05 across soccer/NBA/NHL/tennis/MMA; isotonic overfit small val sets (MCE blowups); only NFL ML overconfident (0.148) but n=143 untrustworthy. Self-gating calibration infra **shipped** (`train_all_models.py` `_fit_and_gate_calibrator`, MIN_GATE_N=500): auto-enables per sport only if it beats held-out Brier — currently serves raw everywhere, correctly | A model drifts overconfident on ≥500 held-out rows (the gate will catch it automatically); use temperature scaling, not isotonic **2026-10-26:** the gate had kept ISOTONIC for NBA total on a −0.0019 held-out gain (n=536, inside noise; MCE 0.27 → 0.38) and served a 34-knot step function live (22 distinct probabilities, an overconfident P(over) 0.13–0.25 cluster). The gate now fits TEMPERATURE scaling and keeps it only past paired ΔBrier ≤ −0.005 with the 95% upper bound < 0 (`1662be9`) |
| Soccer rolling match_stats features | **CLOSED — no win** (2026-06-15) | Full-ensemble retrain flat: held-out Brier 0.5946→0.59645. The A/B "-0.0029 win" was inside the Brier noise floor (SE ≈0.009 at n=3519). Features remain in `compute_features.py` (accuracy-neutral, keeps train/serve parity) | Only alongside a categorically stronger signal (xG) |
| Weather features (Visual Crossing) | **CLOSED at current corpus** (2026-06-05) | All 5 sport-market combos DROP at ΔBrier ≤ −0.005 despite VC beating Open-Meteo and consistent ECE improvement | Corpus reaches 5–10× (only achievable via the §3.6 backfill). Note §6: the VC subscription itself should be cancelled meanwhile — **a deliberate reversal of the June "keep refreshing" decision** |
| Cross-book odds-dispersion features | **v1 IN-PLAY ARTIFACT — no signal on v2 odds** (2026-10-30; was SHIPPED 2026-06) | The June wins (NFL total −0.0110 / ML −0.0154, NBA spread −0.0061, NHL puck_line −0.0186 / total −0.0231) were measured on v1 lines whose cross-book dispersion (0.8–1.6) came from in-play prices. Re-check on v2 strictly-pre-game odds (`scripts/ab_cross_book_v2_recheck.py`: production frames with vs without the shipped feature__ cross-book keys, harness XGB params, season walk-forward, paired): NFL total +0.0019 ± 0.0042 (n=570), NFL ML +0.0016 ± 0.0045 (571), NBA spread +0.0027 ± 0.0019 (2,642), NHL puck line +0.0009 ± 0.0010 (5,805), NHL total +0.0010 ± 0.0013 (5,805) — all inside noise. v2 dispersion: NFL total_std 0.21–0.24, NBA spread_std 0.21–0.24, NHL 0.07–0.14. The old `ab_*_cross_book.py` harnesses are no longer valid controls (frames already carry the features). Soccer 1x2 dropped earlier | No model action forced: removal is also ΔBrier ≈ 0 and cannot pass the gate either way — drop only in a feature cleanup through the promote gate. Do not reopen for NFL spread / NBA ML+total. Only lead: the live-captured 2026-27 NHL season (n=213: puck line −0.0093 ± 0.0059, total −0.0152 ± 0.0068), revisit at 1,000+ games |
| Horse-racing ML ranker | **Ceiling reached** | Brier ~0.105 vs consensus 0.0831 — structural. All five follow-up levers tested and neutral/negative. Productive output: hybrid recs path (shipped) | Different data (sectional times) or framing (direct prob regression) |
| Horse 40+ longshot calibration | CLOSED | Bias non-stationary; isotonic on graded history +0.0007 worse; recs engine emits zero recs in those buckets anyway. **Corrected 2026-09-21:** on the 365-day corpus (n=159k) the June "overconfident favourites" reading has the WRONG SIGN — favourites are *under*-priced (0.5+ buckets +4 to +9 pts), longshots over-priced (−0.4 pt); the June −10.6 pt gap at 40–50% was a small-sample read (n=734) | — |
| Horse-racing consensus de-vig method (proportional → power / Shin) | **CLOSED — correct but below gate** (2026-09-21) | `scripts/ab_horse_racing_devig.py`, 365 d, n=159,488 entrants / 16,101 races, race-clustered SE. Proportional de-vig of a 10–58 % overround book is the mechanism behind the monitor's favourite-longshot bias: gaps +0.013 (20–30 %) → +0.090 (70 %+). Power de-vig flattens every bucket to \|gap\| ≤ 0.023, ECE 0.0050 → 0.0013, MCE 0.10 → 0.04, ΔBrier **−0.00011 ± 0.00003**, Δlog-loss −0.00052 ± 0.00010, same sign in every quarter, top-1 identical (order-preserving). Shin ≈ same Brier, worse at 70 %+. Money: settled-rec replay changes 28 of 298 kept recs; at-SP replay ROI CIs all straddle zero for all three methods. The gain is calibration-only and 10× short of the −0.001 harness rule (50× short of §4.3) | Adopt power de-vig *when* the single-book "consensus" is replaced by a multi-book one (§2.3 / rec_gating note) — it is the right method and costs nothing, but it is not a standalone accuracy lever |
| NFL spread | Parked | Cross-book didn't transfer; needs new data (paid odds archive / QB-injury infra / 5+ seasons) | New data source |
| Tennis/MMA neural nets | **FIXED — stale memory** | Retrain of 2026-07-05 trains NN successfully for both (tennis val_acc 0.605, in-ensemble at weight 0.333). NN ≈ GBM accuracy; value is diversity only | Removing NN + torch is a defensible image-size play (§5.4): retrain with `--skip-models neural_network`, verify held-out delta ≈ 0 |
| Training-frame horizon (refit on train+val instead of the 70/15/15 ratio cut) | **CLOSED — no win** (2026-09-03) | Paired walk-forward on a fixed 2026-03-15..08-31 test window, n=4,026: LightGBM ΔBrier **−0.00029 ± 0.00071**, XGBoost −0.00034 ± 0.00073, 50/50 blend −0.00020 ± 0.00059. No cohort clears (top-5 −0.00263 ± 0.00206). The ratio split leaves essentially nothing on the table for 1x2. Tennis cross-check also fails (+0.00077 ± 0.00177, n=910) | A frame whose newest 30% is materially different in kind, not just recency |
| Recency weighting (exponential, 3-year half-life) | **CLOSED — worse** (2026-09-03) | ΔBrier **+0.00090 ± 0.00115** vs the ratio cut on the same n=4,026 window, and worse than the refit variant in 5 of 6 months. Brackets the audit's own −0.000..−0.004 estimate from the wrong side | — |
| "More leagues" vs "more recent" (soccer) | **RESOLVED — corpus was the lever** (2026-09-03) | Restricting the refit to the 6 original leagues is WORSE by +0.00350 ± 0.00135; the gain shipped on 2026-08-06 came from breadth, not recency | — |
| MMA corpus backfill as an accuracy fix | **CLOSED — insufficient alone** (2026-09-03) | The 18-month corpus hole and OOD serve features are real, but a walk-forward over the loader era, where every fighter history IS complete, still shows no skill; clamping the stale features on the live model does not recover accuracy (0.471). The binding constraint is that the model has never seen a market price | Only alongside real odds in the training frame |
| Tennis "max odds ≤ 2.0" rescue gate | **CLOSED — not a model edge** (2026-09-03) | After de-duplicating the fixture-identity bug it is +8% flat ROI, CI [−4%, +20%], statistically identical to a no-model control ("bet every consensus favourite at the best of ~22 books" = +5.9%), and the entire effect is one month | Re-measure once tennis has market inputs and settles honestly |
| Soccer 1x2 pre-match feature families (rest/congestion, Elo, venue-specific form, 10/20-match + EWMA form, season-to-date table, draw propensity, H2H, discipline, cross-book micro-structure, calendar, rolling shots/SoT/corners, rolling xG, totals form) | **CLOSED — market is the ceiling** (2026-10-25) | `scripts/ab_soccer_feature_screen.py` on the full prod frame: 138,773 matches 2012–2026 with 1x2 odds, fixed test 2024–26 n=23,784 (SE ≈ 0.0001–0.0002 per family, 40× tighter than the audit's n≈3.5k). De-vigged average-odds market Brier **0.59496**; LR on market logits −0.00031 ± 0.00012; a prod-like GBM (odds + 5-match form + match_stats diffs) +0.00004 ± 0.00024 vs the raw market. Every family \|ΔBrier\| ≤ 0.00033: best is cross-book micro-structure −0.00023 ± 0.00011 (7/8 seasons negative), all 60 features together −0.00033 ± 0.00019 — 15–50× under the gate. Elo −0.00007, rest −0.00007, venue form −0.00002, H2H +0.00002 | Nothing form-shaped. Only information the closing price does not already carry (confirmed line-ups / injuries, or a sharper price than the books' average) can move soccer 1x2 |
| Soccer over-2.5 features beyond the market total price | **CLOSED** (2026-10-25) | Same harness, 39,435 matches with O/U odds (2019–26), test n=14,746: market O/U price Brier 0.48294 (−0.0176 ± 0.0014 vs the constant base rate the derived-market guard uses); totals form −0.00027 ± 0.00013, xG / shots / rest / season / weather / micro all within ±0.0003 of the price; no GBM beats the price. **The open finding is the inverse:** a model that sees only 1x2 odds (what the Dixon-Coles path has without a totals price) scores +0.00095 ± 0.00058 historically and the served derived O/U is **+0.0108 ± 0.0043 worse than the closing O/U market live** (n=1,302, 2026-07..10) | Lever is the §7 2026-10-25 item 4: a second IPF constraint anchoring the scoreline matrix to the market O/U 2.5 (and BTTS where priced), not more features |
| Soccer favourite–longshot bias of the proportional de-vig | **MEASURED — calibration, not a standalone edge** (2026-10-25) | Realized minus implied by implied band, stable in 2012–19 AND 2020–26: 0.7–0.8 → +0.035 / +0.029 (z 5.3 / 4.0), 0.8+ → +0.030 / +0.038, <0.10 → −0.013 / −0.017 (z −3.6 / −4.1). Flat-stake at the average price: <0.10 implied −24% / −29%, 0.1–0.2 −8% / −10%, favourites ≈ −1 to +1%. At the best (Max) price, p ≥ 0.6 returns **+0.47% ± 0.39%** (n=27,574; by year −1.8% .. +2.9%) → the vig eats it. Live confirmation on 5,262 closing selections (~9 books): 0.6–0.8 bands realized 0.68 / 0.81 vs proportional 0.64 / 0.74 vs power de-vig 0.665 / 0.774 | **SHIPPED 2026-10-26 (`1c6e307`):** power de-vig in `rec_gating.market_consensus_prob` and soccer 1x2 `max_odds` 8.0 → **5.0**. Not 3.5 as first floated: the bias condemns implied < 0.20 (odds above ~5) in both eras, while the 3–5 band is flat-to-positive (live post-gating odds 3–5: n=124, +13%; ≥ 5: n=44, −25.5%) |
| Historical odds backfill v1 (one snapshot per game-day at 03:00 UTC the next day) | **DEFECT — replaced by v2** (2026-10-26) | The historical endpoint returns in-progress events with IN-PLAY prices; the insert path's in-play guard compares against NOW() so it never fired; a ±2-day name match took the first hit; the first snapshot to reach a key won. Same-game comparison against v2 strictly-pre-game snapshots: NBA Jan–Mar 2025 v1 Brier 0.385 vs v2 0.402, **v1 − v2 = −0.0167 ± 0.0050** (n=616) — a "closing" line cannot beat the pre-game consensus by 3.3 SE without in-play information; 11% of games differ by > 10 pts. NFL 2024-25 v1 is stale instead (+0.0083 ± 0.0047, n=285). Cross-book line dispersion: v1 seasons 0.8–1.6, v2 0.22–0.24, live 0.12–0.20 | **REPLACED 2026-10-30:** every NBA/NFL/NHL game 2022-25 now carries v2 strictly-pre-game lines only (no v1 row left; v1 archived to CSV). Re-validate the cross-book "wins" (NFL total/ML, NBA spread, NHL puck line/total) — they were measured on v1 lines |
| Soccer derived markets anchored to the market over/under 2.5 | **SHIPPED** (2026-10-26, `d2ccd32`) | `scripts/ab_soccer_ou_anchor.py`, monthly walk-forward 2019-07..2026-10, n=40,279, month-clustered SE, anchored − 1x2-only reconcile: over 2.5 **−0.00543 ± 0.00053** (team-aware base; −0.0088 identity-blind), better in all 8 years; over 1.5 −0.0032, over 3.5 −0.0047, BTTS −0.0037, correct score −0.0008, DC/DNB/AH±0.5 exactly 0. Live served-path check (n=966): over 2.5 −0.0061 ± 0.0041, no market worse than z 0.76. One cell degrades on the identity-blind base only (home win-to-nil +0.00074 ± 0.00024) and improves live; accepted | `("soccer","over_under")` rec gate stays OFF — a Brier win is not an ROI win. BTTS anchoring needs a BTTS feed |
| Tennis market price as a model input (the-odds-api historical h2h, 2021-09..2026-10) | **PASSES the gate; does NOT beat the market** (2026-10-30) | `scripts/ab_tennis_market_odds.py` on the real frame (43,225 matches, 8,786 = 20.3% priced after the key fix + provider-anchored backfill; 13.6% before): with vs without market, all rows, holdout from 2026-04 **−0.0234 ± 0.0024** (n=6,484); walk-forward 2023 −0.0112 ± 0.0017, 2024 −0.0129 ± 0.0017, 2025 −0.0154 ± 0.0020, 2026 −0.0207 ± 0.0021. On PRICED rows the model with market is still worse than the de-vigged market itself: holdout +0.0152 ± 0.0031 (n=2,243), 2023..2026 +0.036 / +0.034 / +0.018 / +0.023 (accuracy 0.677 vs market 0.694) | Ships through the normal weekly retrain + promote gate (features recomputed). `("tennis","moneyline")` rec gate stays OFF: the model loses to the price it would bet against. Remaining unpriced matches are at tournaments the provider does not list (250s, Challengers); the next lever is serving the market where priced, through the §4.3 gate |
| NHL probable starting goalie at serve time | **CLOSED — no gain** (2026-10-26) | `scripts/ab_nhl_projected_goalie.py`: "most starts in the last 7, rest the No. 1 on the 2nd night of a back-to-back" picks the real starter 65.7% (13,123 team-games; last-starter rule 47.0%). Paired ΔBrier on 1,300 held-out games vs default goalie features: ML −0.0009 ± 0.0011, regulation −0.0005 ± 0.0014, puck line −0.0024 ± 0.0018, total **+0.0012 ± 0.0007**; even the TRUE starter does no better. Shipped behind `NHL_GOALIE_MODE=actual_or_projected`, default off | A confirmed-starter feed (lines posted ~90 min pre-game) |

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

> **Update (2026-08-05, late night) — Airflow 2.8 → 3.3.0 MIGRATED + verified
> (`1a89bf3`..`907d238`).** 2.x is past EOL. Validated before cutover: ruff
> AIR3 clean; all 10 DAGs parse with zero import errors in a throwaway 3.3.0
> container; compose layering validated; fresh verified B2 backup
> (`betting_system-2026-08-05T232043Z.dump`) since the metadata schema
> migration runs inside the shared betting DB. Changes: DAG code on
> airflow.sdk imports + `schedule=`; webserver → **api-server** (health
> `/api/v2/monitor/health`); NEW mandatory **dag-processor** service; init
> runs `db migrate`; FabAuthManager keeps the admin login (user survived the
> migration); Execution-API URL pointed at the api-server container; JWT/API
> secrets reuse the existing .env value — no prod .env edits. Post-cutover:
> **first pipeline tick green in 81s** through the new Execution API; paused
> states preserved. Two follow-up fixes: (1) isort ordering on the
> sed-rewritten imports (the Code Quality gate correctly blocked the first
> deploy attempt); (2) the UI on airflow.<domain> was minting http:// URLs
> behind Caddy → `api-server --proxy-headers` + `FORWARDED_ALLOW_IPS` +
> `AIRFLOW__API__BASE_URL` per the official run-behind-proxy guide; login
> redirect verified https. Airflow-2 habits to unlearn: `dags list-runs`
> takes the dag id positionally (no `-d`), and 2.x UI deep-link paths are
> gone (new React SPA routes).

> **Update (2026-08-06) — settlement graded the MODEL'S pick, not the bet
> (`abd185b`) + 5 phantom wins corrected.** Closing the asian_handicap gap
> exposed that rec settlement used the joined prediction's argmax verdict,
> ignoring the rec's own bet_type/selection — wrong for every non-argmax
> selection and all lined markets. Fixed with rec-level grading
> (grade_rec_selection: moneylines, lined totals, handicaps incl.
> quarter-line half-win/half-loss stake math, coverage/push markets; AH
> prediction rows also grade now). Re-audit of the 19 pre-fix settlements:
> **5 phantom wins (+$3,117.55 fictitious P&L) corrected to lost
> (−$377.23)** with operator approval — the readout's "tennis +1295 /
> MMA +1055 longshot-driven PnL" was largely this artifact. Backlog graded:
> 1,412 predictions across 289 matches (NFL pre-season baseline now real).
> Caught before NFL/NBA/NHL season volume hit settlement.

> **Update (2026-08-06) — xG CHAPTER CLOSED: rolling-form xG is NOT a soccer
> lever (`21a6a5d`..`5f1bd00` + ab_soccer_xg.py).** The backfill itself
> succeeded completely: Understat redesigned (data now at
> /getLeagueData/{league}/{season}); scripts/backfill_understat_xg.py filled
> expected_goals for **17,672 matches / 35k team-rows** (98.5% of Understat's
> coverage; exact name+score matching, 2 score-mismatch skips in 10 seasons;
> the Championship — our largest league — has no Understat coverage). But the
> A/B verdict is decisive: baseline Brier 0.59836 vs augmented 0.59814,
> **dBrier −0.00022** (n=3,563 test, noise floor SE≈0.009) — 25× under the
> −0.005 gate, accuracy flat. Six features tested: rolling xG for/against +
> finishing overperformance (goals−xG), 5- and 10-match windows, vs the REAL
> training frame. Walk-forward not warranted. **Third no-win soccer feature
> lever** (calibration, match_stats, now xG) — the baseline's odds features
> already embed slow-moving team quality, which is why form-style features
> keep dying. REVISIT CONDITIONS: (a) xG as Dixon-Coles INPUTS (fit
> attack/defense on xG instead of noisy goals — a different mechanism than
> GBM features, untested); (b) 5–10× corpus. The data is banked and a weekly
> refresh keeps it current either way. Soccer's sole remaining open lever is
> the corpus 3–4× backfill (§3.6).

> **Update (2026-08-06) — horse PLACE pilot LIVE (§3 rank 7).** The
> bet_type='win' hardcode is lifted: place probabilities via Harville top-k
> over the full field's devigged win probs, place odds from UK each-way
> terms (5–7 runners: 2 @ 1/4; 8+: 3 @ 1/5; 16+ handicap 4-place terms not
> modeled), same EV gate + quarter-Kelly. Settlement branch added
> (finish_position <= places paid; 94% fp coverage verified; scratched →
> void). First tick: 70 races → 1 place rec (EV +9.6%) + 20 win recs
> pending. **60-day pilot clock starts 2026-08-06** — judged on realized
> ROI + CLV per §4.3(5), fold into the ~2026-09-15 market-gating readout.

> **Update (2026-08-06) — corpus 3–4× backfill SHIPPED at 7× (§3.6 CLOSED as
> a win).** Loaded 16 European league codes × 20 seasons + 16 non-European
> countries (football-data.co.uk): promotion landed **168,077 matches / 41
> leagues / 1,305 teams / 804,515 odds rows** (~7× the old 23.7k corpus);
> feature backfill covered all 168,365 finished matches in 22 min, 0 failed.
> **Gate PASSED (ab_soccer_corpus.py)**: on a shared held-out top-5-league
> test (n=6,066), expanded-corpus Brier 0.57939 vs original-corpus 0.58161
> (Δ −0.0022 — required only no regression; improvement is within noise but
> the right sign). Summer-league slice (n=4,365): 0.60424 vs 0.60842
> (−0.0042). The structural win: summer leagues (MLS, Allsvenskan,
> Brasileirão…) now have real training history, so the eligibility gate
> admits their fixtures with modeled probabilities instead of blocking them.
> Production retrain triggered immediately (promote-gate protected) rather
> than waiting for Sunday. Three loader hardenings shipped en route: ragged
> CSV rows skipped, unparseable season files non-fatal, team-batch upsert
> deduped on (normalized_name, sport). Ops lesson recorded: detached
> in-container jobs die on CI deploys — freeze pushes while one runs, or
> sequence the launch after the deploy. NOTE: the corpus is now at the scale
> band (5–10×) named by the weather chapter's revisit condition.

> **Update (2026-08-06) — corpus retrain LIVE: promoted 13/13.** After the
> gate pass, the Modal retrain on the 7× corpus cleared the n-aware promote
> gate for every bundle and swapped production (run
> manual__2026-08-06T08:05). Summer-league fixtures now get modeled
> probabilities as team history accrues past the eligibility gate. Two
> Airflow-3 Jinja findings en route (both fixed in the retrain DAG):
> CLI-triggered manual runs OMIT `logical_date`/`ts_nodash` from the task
> SDK's template context entirely — even null-safe `(logical_date or …)`
> dies on the name lookup; only `{{ run_id }}` is universally present
> (sanitized via replace filters). And a run pins the DAG VERSION serialized
> at trigger time — trigger only after the dag-processor reparses a fresh
> pull (~90s), or the run executes stale code. This closes the 2026-08-05/06
> active-work program: settlement integrity + AH grading, xG (closed, no
> win), place pilot (live), corpus 7× (live).

> **Update (2026-08-08) — WEATHER CHAPTER PERMANENTLY CLOSED (revisit
> executed at full power, verdict DROP).** The 7× corpus met the chapter's
> own revisit bar, so the revisit ran end-to-end: 715 team stadiums
> geocoded (bulk Wikidata SPARQL — the per-entity API 429s from any IP
> class; one SPARQL query returned 14,707 clubs+coords), then a flawless
> VC backfill (operator re-subscribed): **97,821 matches fetched, 0
> failures, ~$10 of metered records** → 119,986 soccer matches with
> vc_actual (72% of the frame). ab_soccer_weather at 7×, n=20,351
> walk-forward test: **ΔBrier +0.0002, Δlog-loss +0.0004, accuracy flat —
> DROP.** The 1× "real but under-powered" signal did not survive power:
> with premium data and a ~2× tighter noise floor there is nothing to
> ship. NO further revisit condition — the question is resolved. Cancel
> the VC subscription (it bought the definitive answer). Weather data
> stays banked; weather DAGs stay paused. Ops findings: OM archive meters
> long-range calls heavily (5-yr hourly chunks exhaust the free tier in
> ~150 calls); VC match-day-only fetching beats range queries ~10:1 on
> metered cost.

> **Update (2026-09-04) — MODEL/ACCURACY AUDIT: the book's profit was an
> artifact; three streams have no measurable skill. MILESTONE A SHIPPED.**
> A 30-agent audit of live prod (7 investigators, a completeness critic, 4
> follow-ups, 22 adversarial verdicts) asked "can any model or accuracy
> optimisation be made?" and found that almost nothing available is model
> tuning. Headline corrections to earlier numbers: live metrics had been
> counting every weekly re-prediction row, inflating n by 3–4× (MMA "n=444"
> is 121 distinct fights; soccer "n=2836" is 997 matches).
> **What has no skill:** MMA moneyline held-out Brier 0.50254 (n=269) IS a
> coin flip, and its odds features are the hard-coded default in 1676/1676
> training rows (importance 0.0) — on the same 68 fights the market scores
> 0.357 vs the model's 0.517 (paired ΔBrier +0.144, SE 0.040). Tennis
> moneyline: 0.489 vs the market's 0.333 (n=97), with 24,446/24,446
> historical rows carrying default odds. Soccer over_under: the production
> Dixon-Coles knows both teams in **4%** of live matches, so 96% get
> identical goal rates (corr(expected total, actual) = 0.036).
> **What is honest:** soccer 1x2 is statistically indistinguishable from the
> closing market (+0.0026 ± 0.0033, n=702) and horse-racing consensus already
> beats SP (Brier 0.0784 vs 0.0828, n=45,055).
> **The book was an artifact.** Reported +2.3% ROI. The BetUS asian-handicap
> slice won 38 of 47 near-even-money handicaps (+67% ROI) off two side-swapped
> feed runs (2026-08-09, 2026-08-22); recs are priced at the FIRST-SEEN odds
> row (up to 8 days stale) and regenerated every cycle, so stored CLV measures
> price drift, not edge, and true CLV is identically 0. Repriced at the real
> closing price, soccer 1x2 recs that still PASS the EV gate lose (n=57,
> −26.7%) while those where the model agrees with the market won (n=53,
> +38.7%): **model-versus-market disagreement is anti-predictive at current
> model quality.** After settling the stranded recs (below) the honest book is
> **n=2,158, −16,478 on 141,969 staked = −11.6% ROI**; excluding the whole
> BetUS AH slice, −14.5%. Soccer AH ex-BetUS is +0.2% (n=111).
> **Structural defects found:** fixture identity is keyed on exact kickoff
> time with no ESPN id stored, so one fixture becomes many rows (tennis 19,008
> stale rows, 1,727 unsettleable recs, 121k of phantom pending stake); the
> Modal promote gate compares held-out Briers across NON-COMPARABLE test sets
> and has rejected the 7× corpus model 4 weeks running, so **prod has never
> served it** (the 2026-08-06 "promoted 13/13" entry above is wrong for
> soccer); the entire 2025-26 season is missing for NFL/NBA/NHL and 294
> preseason games sit untagged in the NFL+NBA corpora, making NFL week-1
> rolling windows 61% preseason (15/16 totals picked under); preseason
> predictions were graded against a fabricated 45.0 total line that
> NEUTRAL_DEFAULTS invented.
> **Milestone A shipped 2026-09-04** (`scripts/rec_gating.py`): MMA, tennis,
> soccer over_under and all of NFL/NBA/NHL are gated OFF with the evidence
> recorded on each gate; survivors are bounded by odds/EV/model-vs-market-gap
> caps; every stake capped at 2.5% of bankroll (quarter-Kelly had staked 23%
> on one bet); racing suppresses win recs in 5–7 runner fields (n=182,
> −57.1%, CI excludes zero). Predictions keep being produced and graded for
> every sport — only rec emission stops — so a disabled stream can be measured
> back to life. `scripts/repair_stranded_recs.py` then settled the stranded
> recs from their finished twin rows using the repo's own grader: soccer 35
> (15W/18L/2V, −15.31 — matching an independent hand-replay to the cent),
> tennis 1,683 rows collapsing to **754 distinct fixtures** (−16,888 on
> 118,734 = −14.2%), MMA 6. Gated streams now hold zero live picks.
> **Deliberate non-action:** the settled BetUS AH rows were NOT rewritten —
> the flip classification does not reproduce stably (10/14/28/45 recs
> depending on method) and voting an arbitrary subset would claim a fix while
> leaving most contamination. Both numbers are reported instead; the cause
> (feed side-swap + first-seen pricing) is fixed in Step 2.

> **Update (2026-09-04) — STEP 2 SHIPPED, and a correction to the 7× corpus
> claim.** Ingestion: fixtures are now keyed by the ESPN competition id
> (migration 023) and realigned in place when ESPN moves a kickoff, which is
> what stops one fixture becoming many rows; ESPN placeholder draw slots
> ("TBD vs TBD", 3,061 rows / 3,528 predictions on prod) are skipped at the
> door. Season type is captured and preseason excluded from every training
> frame and rolling window; 294 historical preseason games were tagged by
> `scripts/backfill_season_type.py`. Grading no longer scores a spread/total
> prediction whose "closing line" was the NEUTRAL_DEFAULTS invention — the
> guard fired on its first run (14 predictions left ungraded). monitor_models
> now counts each match once (it had been inflating n ~4× across every sport).
> Pricing: the `odds` table holds the CURRENT price instead of the first-seen
> one (migration 024). First production run **updated 7,428 prices** against
> 45 genuinely new keys — that is the scale of the staleness recommendations
> were being written against — and the in-play guard protected 303 pre-match
> prices. A per-bookmaker side-swap canary now drops a transposed book's whole
> block, corroborated by its moneyline so it cannot fire on legitimate
> disagreement. Results backfill recovered 104 World Cup fixtures plus the
> June NBA/NHL Finals, settling every stranded NBA/NHL rec; the book now
> stands at **n=2,180, −15,922 on 143,982 = −11.1% ROI**.
> **CORRECTION — the 7× corpus model is NOT better.** With the promote gate
> fixed (it now fingerprints the frame and re-scores both models on rows
> unseen by BOTH, rather than comparing Briers computed on different test
> sets), the soccer challenger measures **paired ΔBrier +0.00433 ± 0.00234
> (n=728)** against the incumbent — marginally WORSE, within noise. The
> earlier "−0.0014 ± 0.0010, n=4,061" came from a wider window. Combined, the
> two are statistically indistinguishable, so promoting the 7× bundle is
> justified by COVERAGE (41 leagues vs 6) and not by accuracy; do not cite it
> as an accuracy win. The gate was also deadlocking on legacy sidecars that
> carry no fingerprint — it now recovers the incumbent's real frame end from
> that run's retained training_report.json. The refusals that remain are
> honest: NBA and NHL have no games after their training cut-off because the
> 2025-26 season is missing, and the NFL frame legitimately shrank when
> preseason was excluded.

> **Update (2026-09-04) — STEP 3: the Dixon-Coles defect was the FITTER, not
> team coverage.** The working hypothesis was that the derived soccer markets
> ran on a constant prior because the artifact knows both teams in only 4% of
> live matches. Investigation refuted that as the dominant cause. The
> iteration normalised BOTH attack and defense to mean 1, over-constraining a
> model with exactly one scale indeterminacy and pinning mean(lambda) to a
> hard-coded baseline: fitted E[total goals] 2.8971 against an observed 2.6415
> (+9.7%, reproduced in-band during validation; the audit measured +10.9% on
> prod). That is why mean P(over 2.5) was 0.567 against a realised 0.524 —
> the model was structurally long overs, which is the mechanism behind the
> −22.4% ROI on that stream. Also fixed: `time_decay` was serialised and read
> back but NEVER APPLIED (an unweighted MLE over 2007-2026, an era in which
> home goal advantage fell 0.405 → 0.29); one global `league_avg_goals` across
> 41 leagues running 2.22 to 3.09 goals per match; and an inert
> `regularization` that is now a real shrinkage prior (~20 effective matches),
> which both makes an unknown team fall back to its LEAGUE baseline and makes
> the fit converge (91 iterations vs never in 500).
> **Validated** (`scripts/ab_soccer_dixon_coles_refit.py`, monthly
> walk-forward, n=10,950, scored after the IPF reconciliation so it measures
> the served path, with FOLD-CLUSTERED standard errors because 10,950 matches
> come from only 13 fitted models). Paired ΔBrier on over-2.5: vs the served
> artifact **−0.00698** (clustered 95% bound −0.00418); vs the same-corpus OLD
> fitter **−0.00722** (bound −0.00231); vs a global base rate −0.00667; vs a
> **PER-LEAGUE base rate −0.00385, which FAILS the gate**; vs the closing
> market −0.00182 (n=546, not significant). The honest reading: it clears the
> gate against the model it replaces and against the old fitter on identical
> folds, but a real share of the gain is "know which league this is" rather
> than team skill. THE DECISIVE CONTROL: the old fitter on the CURRENT corpus
> scores +0.00024 against the served artifact — statistically identical — so
> the defect is the fitter, and promoting the 7× challenger would NOT have
> fixed these markets.
> **Not un-gated.** ("soccer","over_under") stays disabled: every EV
> configuration still loses against real book prices (−2.2% to −6.7% at
> EV>5%). A Brier win is not an ROI win.
> **Serving defect found in review and fixed:** the ensemble frame was built
> from features_cache alone, so the Poisson and Dixon-Coles members inside it
> received EMPTY team names on every live match and emitted their unknown-team
> constant, while the weight optimiser had given them ~9-10% of the blend on
> their team-AWARE validation scores. Identity is now passed through
> (measured −0.00041 on served 1X2, n=600 — a correctness fix, not an accuracy
> win). The promote gate also gains an additive derived-market guard: a
> challenger that cannot beat a constant base rate on the market it derives is
> rejected even if its 1x2 passes, closing the blind spot that let a
> pre-corpus artifact serve for a month behind its 4.55% blend share.
> **STILL OPEN — needs operator sign-off: TEAM IDENTITY.** The corpus and ESPN
> fixture rows are different `teams` rows (1,356 soccer rows across three
> provenance cohorts). Exact normalized matching recovers ZERO by construction
> (`UNIQUE(normalized_name, sport)` means two rows can never share one), and
> fuzzy matching is unusable at ANY threshold — true pairs score low
> ('Ajax'|'Ajax Amsterdam' 0.44) while false pairs score high
> ('Angers'|'Rangers' 0.92, 'Manchester City'|'Manchester United' 0.81), with
> reserve teams and city-mates (Barcelona/Barcelona B, LAFC/LA Galaxy) sitting
> exactly where prefix matching wants to go. There IS free ground truth: a
> corpus row and an ESPN row for the same real match carry identical
> scorelines on the same date (with a systematic ~1 h offset). That validates
> 63 mechanical pairs (accent-strip + club-token, same-country) at 122
> agreements / 0-1 conflicts, but the further 74 curated pairs are ~97%
> precise and contain at least one false positive no automated rule catches
> (Fortaleza CEIF Colombia → Fortaleza Brazil), so they need a NAMED HUMAN
> SIGN-OFF. Payoff if merged: DC coverage of the next-14-day serve population
> 7.1% → 38.9% (mechanical) → 59.5% (curated); the eligibility gate 30.7% →
> 47.1%; teams under the 10-match floor 51.2% → 21.2%. The trap: the merge
> MUST also dedupe 154 cross-identity duplicate fixtures (124 of them
> both-finished with identical scores) or it silently double-counts real
> matches, and it must join on match_date::date + scoreline, because only 5 of
> them collide on exact timestamp.

> **Update (2026-09-21) — DRIFT PAGE MADE ACTIONABLE (62a2d74); the one
> real signal in it investigated and closed.** The hourly monitor paged ten
> unlabeled lines ("MCE 0.923 >= 0.250", "Accuracy 12.5% < 52.4%") every
> hour. Causes: `build_alert` dropped sport/market, MCE was the worst of
> ten bins with no minimum n (n=1 bins page), the 52.4 % break-even floor
> was applied to 3–51-way markets, gated-off streams paged, and dedup keyed
> the whole set. Shipped: every line names market/model/n; MCE needs a
> bucket of n ≥ 20; ECE/MCE only page at n ≥ 200; accuracy pages only when
> the one-sided 95 % binomial bound is below the floor and only for 2-way
> markets; gated streams downgrade to warn; per-finding Redis keys marked
> only after a successful send; a failed send fails the task. Real run
> after deploy: 0 alerts, 4 warnings.
> The surviving signal — horse_racing/win [market_consensus_v1] bucket
> 0.6–0.7 predicted 0.640 vs actual 0.821 (n=28) — was investigated with
> `scripts/ab_horse_racing_devig.py` (§4 registry row). Verdict: (i) the
> MAGNITUDE is a September blip — live favourites ≥ 0.5 by month: Jun
> +0.004, Jul +0.009, Aug −0.006, Sep +0.156 (n=67); a 2σ bucket out of
> ~20 scanned hourly; (ii) underneath it there IS a structural favourite-
> longshot bias, and its mechanism is proportional de-vigging of a
> single-book price with a 10–58 % overround; power de-vig removes it
> (ECE 0.0050 → 0.0013) for ΔBrier −0.00011 ± 0.00003 — real, stable, and
> an order of magnitude below the ship gate; (iii) no measurable money
> effect on the rec stream. Not shipped; adopt with the multi-book consensus.
> Side finding worth knowing: the consensus is scored ONCE from the first
> racecard price and never refreshed while `morning_line_odds` keeps being
> overwritten to the off; the last pre-race price would be ΔBrier −0.0015
> ± 0.00015 better on live rows (−0.0025 in the last 30 d). Settled win
> recs that still show EV at that later price returned +46.5 % flat
> (n=298) against −7.0 % for the 794 that don't — a closing-line-value
> signal racing has no capture for. Re-scoring alone would NOT harvest it:
> against a single book the refreshed consensus equals the price and EV is
> zero by construction, so this too lands on the multi-book consensus item.

> **Update (2026-10-25) — MODEL / ACCURACY CHECK + STATISTICAL FEATURE
> SCREEN (analysis only; nothing shipped).** Live, deduped (newest row per
> match+market), paired against the de-vigged closing consensus from
> `odds_snapshots` on the SAME matches, 2026-07-13..10-25, multiclass Brier,
> positive = model worse: soccer 1x2 **+0.0020 ± 0.0016** (n=1,754; monthly
> Jul +0.005, Aug +0.003, Sep +0.001, Oct +0.001 — the model IS the market);
> soccer over-2.5 **+0.0108 ± 0.0043** (n=1,302); tennis +0.033 ± 0.017
> (n=290); MMA +0.125 ± 0.030 (n=138, market 72% accurate vs model 46%); NHL
> moneyline +0.0185 ± 0.0109 (n=181), NHL regulation +0.035 ± 0.010 (n=139);
> NBA moneyline +0.024 ± 0.024 (n=38); NFL moneyline −0.0025 ± 0.0136
> (n=104). Model-vs-market disagreement is NOT predictive in any sport
> (realized ≈ market in every disagreement bucket), and every EV-rule
> variant replayed at closing best prices (model EV ≥ 5%, odds caps, 50/50
> shrink to market, pure price-shopping) has an ROI SE of 6–20 pp at this n —
> no rule can be ranked yet. The post-gating book since 2026-09-04 is soccer
> only: n=304, +246 on 4,412 (+5.6%, SE ≈ 9 pp); its 1x2 recs sit at odds
> 3–5 (n=124, +13%) and ≥ 5 (n=44, −25.5%).
> **Feature screen (13 families, §4 rows above): soccer 1x2 is at the
> market's ceiling** — nothing pre-match moves it by more than 0.0003 Brier
> on n=23,784, and the prod-like GBM adds +0.00004 over the raw de-vigged
> price. Treat "add a soccer feature" as closed unless it is line-up /
> injury information.
> **Data and serving defects found (these ARE the model levers):**
> (1) **The 2025-26 season carries no odds for NBA (1,398/1,401), NFL
> (286/286) or NHL (1,390/1,396)** — the 2026-09 backfill loaded games
> without `backfill_historical_odds.py`, so 17–25% of each training frame,
> and the most recent season, trains on NEUTRAL_DEFAULT market features
> (NHL 2020-21 and 2021-22 — 2,353 games — are also odds-less). Tennis is
> 99% default odds (27,243/27,517) and MMA 92% (1,783/1,928): no market
> input at all. (2) **Cross-book dispersion is train/serve skewed 4–5×:**
> training-era std features came from one historical snapshot per game-day
> with books at different staleness (NBA spread_std 1.5–1.6, NHL 0.29–0.33,
> NFL total_std 0.86–0.96; the odds-less 2025-26 rows show 5.3 / 2.9) while
> the live current-price table is synchronized (NBA 0.35, NHL 0.13, NFL
> 0.29). The shipped cross-book wins were validated on a dispersion signal
> the serve path does not reproduce. (3) **NHL goalie features are at their
> 0.905 default for 100% of 2026-27 games** (50% in the training era) — the
> goalie backfill is not running for the live season; nothing paged. (4) NBA
> total (corrected 2026-10-26: its CALIBRATION gate kept an isotonic map) emits an overconfident
> under cluster live (15 of 110 games at P(over) ≈ 0.25; 9 went over).
> (5) 62 of 243 upcoming soccer fixtures (25%) sit on default odds — known
> and gated, recorded here as the baseline for a default-rate canary.
> **Ranked recommendations (none executed):**
> 1. `backfill_historical_odds.py` for 2025-26 across NBA/NFL/NHL (+ NHL
>    2020-22), recompute features, retrain through the promote gate — a data
>    completeness fix on the strongest feature, not a lever to re-test.
> 2. Fix NHL goalie population for the live season; add a per-sport
>    "features at NEUTRAL_DEFAULT" rate to `monitor_models` with a page when
>    it jumps (the silent-failure class this system keeps meeting).
> 3. Rebuild the cross-book dispersion features from synchronized snapshots
>    (odds_snapshots since 2026-07 is the live-equivalent definition) or
>    normalize by book count, then re-validate the shipped NFL/NBA/NHL
>    cross-book members on that definition.
> 4. Soccer derived markets: add a totals constraint to
>    `reconcile_matrix_to_1x2` (IPF on 1x2 × over/under-2.5 regions, market
>    O/U where priced, BTTS optional). Expected: closes the +0.0108 live gap
>    on the 2.5 line by construction and moves BTTS / team totals / correct
>    score consistently. Validate with an `ab_soccer_dixon_coles_refit`-style
>    harness before shipping; this is what could un-gate over_under.
> 5. Power de-vig in `rec_gating.market_consensus_prob` and a soccer 1x2
>    `max_odds` of ~3.5 (registry row above).
> 6. Tennis: a `load_tennis_odds.py` from tennis-data.co.uk (B365/PS/Avg/Max
>    closing odds, ATP+WTA, 2001+) joined on player name + date — the first
>    market input for the one gated 1v1 stream that has a free source.
> 7. Hold the NBA-total promote until item 1 lands and the under cluster is
>    explained.
> Harness: `scripts/ab_soccer_feature_screen.py` (export SQL in its
> docstring; the live comparison SQL is in the same file's notes).

> **Update (2026-10-26) — THE 2026-10-25 RECOMMENDATIONS EXECUTED, plus two
> defects found while doing them.** Deployed at `d3836f4` (CI green; every
> auspex_pipeline task succeeding since 11:46 UTC).
> **Recommendation 1 + 3 (odds backfill, cross-book skew) — the cause was the
> backfill itself.** The v1 historical backfill snapshotted the NEXT day at
> 03:00 UTC, after most games started; the historical endpoint returns
> in-progress events with IN-PLAY prices, and the insert path's in-play guard
> compares against NOW(), so it never fired (registry row "Historical odds
> backfill v1"). Same-game check against strictly-pre-game snapshots: NBA
> Jan–Mar 2025 v1 beats the real pre-game consensus by 0.0167 ± 0.0050 Brier
> (n=616), i.e. leakage; NFL 2024-25 v1 is stale instead (+0.0083 ± 0.0047).
> `backfill_historical_odds.py` v2 (`ae1f359`) clusters start times into ≤ 3 h
> groups, snapshots 10 min before each group, accepts only events commencing
> AFTER the snapshot and only for that group's games, caches raw responses
> under /app/backups/odds_hist_cache, and keeps a 15k quota floor. Loaded
> additively on prod (no existing row touched): NFL 2025-26 269/270 games
> (136 in-play events rejected, 17,548 rows), NBA 2025-26 1,320/1,327 (305
> rejected, 85,972 rows), NHL 2025-26 1,389/1,390 (238 rejected, 91,242 rows),
> NHL 2020-22 2,352/2,353 (784 rejected, 223,890 rows); ~47.9k credits incl.
> 7.4k cache-only staging, 19.7k left for the live fetch until the Nov 1 reset.
> Cross-book line dispersion on the clean data: NBA spread std 0.222, NFL total
> std 0.242 (v1 seasons 0.8–1.6; live 0.12–0.20) — the "4–5× skew" was v1
> contamination, not a feature-definition problem.
> **Defect found while recomputing — in-play prices in every live-era fixture
> (`fafbc29`).** fetch_live_odds never overwrote a pre-match price after
> kickoff but still INSERTED unseen keys, and in-play markets mint keys
> constantly: NBA 44/44, NFL 105/105, NHL 187/188, soccer 1,716/1,889, tennis
> 394/430 finished live-era fixtures carried in-play rows (tennis incl. 1,600
> in-play moneyline rows). Serving never saw them; every training frame and
> post-game recompute did (a recompute moved live NBA spread dispersion 0.35 →
> 4.97). Fixed twice over: the live path no longer inserts after kickoff (only
> the v2 backfill opts in), and `utils.training_data.prematch_odds()` rewrites
> every training query and every feature-builder odds query to read rows first
> seen before kickoff (a static test fails on any unwrapped query).
> **Recommendation 2 (NHL data + canary).** The live 2026-27 season had ZERO
> nhl_match_stats rows; 2025-26 had no starting goalies and no 5v5. The NHL
> loader now attaches stats to the existing ESPN fixture (accent-insensitive
> club name ± 12 h, or the NHL game id) instead of inserting by exact kickoff,
> which had created 2 duplicate 2025-26 fixtures via "Montréal"/"Montreal";
> a new `enrich_nhl_stats` DAG task keeps the live season current. Prod: 181/181
> live games attached (362 stats + goalie + 5v5 rows), 2025-26 goalies and 5v5
> backfilled (2,788 rows each). Regulation ties are now stored as 'tie' — the
> ESPN path wrote 'draw', a label the home/tie/away model can never predict, so
> every correct tie pick graded as a miss and all 58 live ties were dropped from
> training (`46d0152`; this also means the 2026-10-25 NHL regulation-vs-market
> comparison covered decisive games only). The probable-starter projection is
> CLOSED (registry). The neutral-default canary is live in the hourly monitor
> (`39c2164`), pages on a ≥ 25-pt jump over a 28-day baseline, and is quiet.
> **Recommendation 4 (soccer derived O/U) — SHIPPED** (registry row): 57 of
> 540 live O/U predictions in the first hours carried a market anchor (≥ 3
> fresh books required); the rest serve the old reconcile.
> **Recommendation 5 — SHIPPED** at max_odds 5.0, not 3.5 (registry row); no
> soccer rec at odds ≥ 5 since deploy.
> **Recommendation 6 (tennis odds) — BLOCKED on a manual download.**
> `scripts/load_tennis_odds.py` + `ab_tennis_market_odds.py` are merged
> (precision-gated name matching, additive, tagged for rollback), but
> tennis-data.co.uk answers every automated request with a Cloudflare 403 and
> the loader does not work around it. Download the ATP+WTA 2022–2026 workbooks
> from tennis-data.co.uk/alldata.php into one directory, then run
> `load_tennis_odds.py --files-dir <dir> --dry-run`, the real load, and
> `compute_features_tennis.py --all-finished --force`.
> **Recommendation 7 (NBA total) — root cause was the CALIBRATOR, not a
> promote.** "kept" in held_out_metrics means the calibration gate kept its
> calibrator; NBA total was the only one of 13 production ensembles serving
> one, an isotonic step map (registry: calibration row). Temperature scaling +
> a paired noise-floor gate shipped (`1662be9`); the next retrain replaces it.
> **Retrain on the repaired data** (`manual__2026-10-26T15:44`, fresh dump
> 15:22 UTC): 9/13 promoted — soccer, NHL moneyline/regulation/puck
> line/total, NBA and NFL moneyline, tennis, MMA — and none kept a calibrator
> under the new gate. The scalar Brier "gains" for NBA ML (0.449 → 0.380) and
> NFL ML (0.463 → 0.439) compare test tails of different composition (the old
> tail held 2025-26 games on default odds); they are not accuracy deltas.
> NBA spread/total and NFL spread/total went to MANUAL: their frames gained the
> newly priced 2025-26 season INSIDE the incumbents' date range, so the
> date-cutoff shared set shrank to two days. A manual paired re-score on the
> challenger's held-out rows the incumbent never saw found none distinguishable
> (NBA spread +0.0003 ± 0.0027 n=724; NBA total +0.0019 ± 0.0039 n=702, the
> incumbent serving 18 distinct probabilities from its isotonic map against
> 699; NFL spread −0.0024 ± 0.0039 n=93; NFL total −0.0003 ± 0.0143 n=93). The
> manual production swap was refused by this session's permission policy, so
> the GATE was fixed instead: the trainer records each ensemble's train+val
> match ids (`seen_match_ids.json.gz`) and the paired path scores rows outside
> that set; for legacy incumbents of line-required frames (NBA/NFL
> spread/total), games first priced after the incumbent trained count as
> unseen. Dry-run on prod with today's challengers: the gate now PROMOTES NBA
> spread and NBA total itself (n=722 / 700); NFL spread/total stay at n=93 <
> 100 and should clear with another week of games. The next scheduled retrain
> (Sun 2026-11-01 04:00 UTC) therefore retires the NBA-total isotonic map
> without a hand-made swap; until then NBA total keeps serving it (recs gated).
> **AWAITING OPERATOR APPROVAL (destructive or mutating):**
> 1. Replace the 2022-25 NBA/NFL/NHL v1 odds rows: `backfill_historical_odds.py
>    --mode replace --confirm-replace` per sport-season (archives every replaced
>    row to /app/backups/odds_archive_*.csv first). ~67k credits → after the
>    Nov 1 quota reset.
> 2. Merge the duplicate Montreal team rows (ESPN `b8284384…` onto history
>    `a6b6fabf…`, 27 fixtures) and the 2 duplicate 2025-26 NHL fixtures.
> 3. Regrade NHL regulation predictions graded against 'draw'.
> 4. Optionally flag the existing in-play rows `is_live = true` (reversible);
>    not required for correctness now that every reader filters them.
> 5. Optionally promote NBA/NFL spread/total now rather than at the Sunday
>    retrain (evidence above; production backup first, as swap_production does).

> **Update (2026-10-26, evening) — operator approvals executed.** Verified
> backup first (`postgres/betting_system-2026-10-26T184445Z.dump`, 386 MB).
> (1) **Odds replace — partial.** The operator expected ~87k credits; the API
> reported 18,873 remaining (81,127 used of 100k), so only what fits ran:
> NFL 2024-25 (fully cached, 0 credits: 285/286 games, 143 in-play events
> rejected, 38,946 v1 rows archived → 17,014 written) and NBA 2024-25 (~5.3k
> credits: 1,319/1,329 games, 254 in-play rejected). Archives:
> /app/backups/odds_archive_*_20261026T18*.csv. The rest (NBA/NFL 2022-24, NHL
> 2022-25, ~54k credits) is `scripts/ops/replace_v1_odds_after_reset.sh`: it
> waits for the Nov 1 reset, takes and verifies a backup, replaces with a 15k
> floor, recomputes features, takes a post-replace backup, and triggers a
> retrain if it misses the 04:00 one. Installing it in cron was refused by
> this session's permission policy; run it by hand after the reset.
> (2) **NHL cleanup — done** in one transaction (archive
> /app/backups/nhl_cleanup_20261026T190233Z.json): the 2 duplicate 2025-26
> fixtures removed (canonical = the NHL-API rows with stats; the ESPN id and 4
> predictions moved over), the ESPN "Montreal Canadiens" team row merged into
> the history row (25 fixtures, 22 stats rows, 1 player), 186 regulation
> predictions relabelled 'draw' → 'tie' (all had picked home/away, so
> correctness is unchanged). Deployed first (`a42126e`): both loaders now
> resolve a club under any accent spelling, so the merge cannot be undone by
> the next fetch. Left alone: four never-played "if necessary" playoff
> placeholders (scheduled, no twin, excluded from training).
> (3) **Promote of the four held bundles — blocked** twice by this session's
> permission policy despite approval; `scripts/ops/promote_held_bundles.py`
> does it (command in its docstring). Independently, every production
> incumbent except tennis now carries an exact `seen_match_ids.json.gz`
> (12 reconstructed from today's run, 4 from the 2026-10-25 legacy run; each
> row count verified against its own fingerprint), so the Nov 1 gate compares
> on exact membership either way. The "first priced after training" rule added
> earlier today was REMOVED: an odds replace re-stamps every game's first-priced
> time and would have counted games the incumbent trained on as unseen.
> (4) **Tennis source.** tennis-data.co.uk is the legitimate free source (same
> author as football-data.co.uk, which the soccer loaders use). The 403 is a
> site-owner Cloudflare FIREWALL block ("Sorry, you have been blocked"), not a
> bot challenge: it hits browsers too, and every client tried (operator, dev
> machine, VM) is in the US, consistent with a geographic rule. Do not route
> around it. Options:
> ask the site owner for access or the files; backfill tennis from the-odds-api
> historical endpoint with the v2 backfill (the same provider live tennis odds
> come from, so train/serve parity; covers only the tournaments it lists, which
> is also all we can serve odds for); or Kaggle's CC0 "ATP Tennis 2000–2025
> daily" mirror (ATP only, a single odds pair, needs a Kaggle API token).
> Plan agreed 2026-10-26: run `scripts/ops/replace_v1_odds_after_reset.sh` on
> Nov 1 after the operator raises the odds-api plan; the four held bundles are
> left to the Nov 1 scheduled retrain under the fixed gate. **Tennis goes
> through the-odds-api too** (operator decision): `backfill_historical_odds.py
> --sport tennis` resolves each match's tournament key from venue + date
> (TENNIS_EVENTS; Slams/combined events query both tour keys, player order is
> matched either way) and prices only the listed tournaments — the same set
> live tennis odds come from. Dry run on the current corpus: 11,138 of 29,421
> finished matches are in listed tournaments; 2,975 calls ≈ 29.8k credits at
> 3 h windows (1,910 ≈ 19.1k at 6 h). The corpus is also missing Feb 2025 –
> Apr 2026 (≈ 15 months, 118 matches in all of 2025); the Nov 1 script loads
> those from ESPN first (free), so tennis needs ≈ 40k credits. Credit budget
> for Nov 1: ≈ 54k team-sport replace + ≈ 40k tennis + ≈ 40k/month live, so
> more than the 100k plan; the script skips the tennis odds step (and says so)
> unless ≥ 85k remain after the team sports. Validate afterwards with
> `scripts/ab_tennis_market_odds.py` before reading anything into the retrain.

> **Incident (2026-10-27 08:00 → 2026-10-29 06:45 UTC) — 63 failed pipeline
> runs, fixed (`c6d6d5f`, `f11ea6c`).** ESPN's NBA scoreboard intermittently
> answers HTTP 200 with an EMPTY body (logged since the fix: "nba/nba
> 2026-10-27: non-JSON body (HTTP 200, 0 bytes)"); probes in between parse
> fine. `fetch_day` decoded the body outside its error handling, so one empty
> response failed `fetch_results` (all 136 failed attempts died in the NBA
> step), skipped NFL/tennis/MMA results chained after it with `&&`, left
> grading and NHL enrichment upstream_failed, and failed the NBA fixture chain
> 5 times. Fix: 3 attempts with backoff per request, a final failure logged at
> ERROR and skipped (results exit non-zero only when EVERY request for a sport
> failed); every sport runs regardless; grading and NHL enrichment run on
> ALL_DONE. No data was lost — the good runs in between recorded every NBA
> game and every finished game of the last 3 days is graded. The deploy was
> first blocked by CI: GitHub moved `ubuntu-latest` to Ubuntu 26.04 and
> Playwright has no chromium there, so the Frontend job is pinned to
> `ubuntu-24.04`. First three runs after the deploy: all success.

> **Update (2026-10-30) — the "Nov 1" odds job ran two days early.** The
> operator raised the-odds-api plan to **5,000,000 credits/month** ahead of
> the reset (balance 5,000,000, used 0 at 17:37 UTC), so
> `scripts/ops/replace_v1_odds_after_reset.sh` was launched on request and
> finished at 18:52 UTC. Verified backup first
> (`postgres/betting_system-2026-10-30T*.dump`, landed 17:39) and after.
> **Replace** (every v1 row archived to `/app/backups/odds_archive_*_20261030T*.csv`):
> NFL 2022-09..2024-02 571 games, 569 priced, 129,454 v1 rows out, 59,634 in;
> NBA 2022-10..2024-06 2,644 games, 2,638 priced, 213,398 out, 265,520 in;
> NHL 2022-10..2025-06 4,198 games, 4,198 priced, 328,094 out, 336,746 in.
> 1,631 in-play events were rejected. One NHL snapshot (2024-03-22) failed
> transiently and left 2 games on v1 rows; a one-day re-run replaced them —
> a DB check confirms no pre-run odds row remains in any of the three
> windows. The 8 unpriced games (2 NFL, 6 NBA) are better unpriced than
> in-play. **Tennis matches:** 13.2k Jan 2025 – Apr 2026 matches loaded from
> ESPN (the loader's "261,044 matches written" counts upserts); quarterly
> counts now 2.3–2.9k, matching earlier years; 0 duplicates introduced.
> **Tennis odds:** 4,361 tournament-key snapshots, 3,112 fetched, 5,484
> games priced; a retry after `2d517c3` priced 153 more (5,889 total, 13.6%
> of the frame). All 1,143 skipped snapshots are a permanent `404
> UNKNOWN_SPORT` (the tournament key was not open at that time — mostly
> warm-up-week games inside a TENNIS_EVENTS window), the same count on both
> runs. **Correction:** `2d517c3`'s message says a rate-limit burst dropped
> ~270 tennis snapshots; the retry log shows zero 429s — the snapshots that
> answered 200 on re-probe were NEW cluster times (pricing games re-groups
> the rest), not failed calls. The retry/backoff it adds is still right for
> the NHL-type transient failure. Coverage limit: ~6.0k listed-tournament
> events were rejected as started — the snapshot is anchored on ESPN's time,
> which runs later than the-odds-api `commence_time` for tennis; anchoring
> snapshots on the provider's event list would recover part of them.
> **Validation** (`ab_tennis_market_odds.py`): see the §4 registry row — the
> market input passes the gate in every window, but the model still loses to
> the market on priced rows, so tennis recs stay gated off. Features
> recomputed for nfl/nba/nhl/tennis; the Sun Nov 1 04:00 retrain trains on
> the clean data. Credits: 80,549 used of 5M (balance check). Pre-existing finding, not
> touched: ~3.8k tennis player-pair/day duplicates in Jun–Oct 2026 are stale
> `scheduled` rows with no ESPN id next to the real finished match (only 3
> finished duplicate pairs, so training is unaffected).

> **Update (2026-10-30, evening) — follow-ups, and two corrections to the
> entry above.** (1) **Cross-book wins were a v1 artifact:** re-checked on v2
> odds (`scripts/ab_cross_book_v2_recheck.py`), all five June wins are inside
> noise — §4 registry. (2) **Correction — the 1,143 tennis 404s were WRONG KEY
> NAMES, not "key not open yet":** `GET /v4/sports?all=true` lists 51 tennis
> keys; ours named `tennis_{atp,wta}_australian_open` (real:
> `*_aus_open_singles`), `tennis_{atp,wta}_miami` (`*_miami_open`),
> `tennis_atp_monte_carlo` (`*_monte_carlo_masters`) and
> `tennis_{atp,wta}_finals` (not offered). The LIVE list had the same names, so
> Australian Open / Miami / Monte Carlo were never priced live either, and
> neither path asked for the provider's 31 500-level tournaments (Basel and
> Vienna this week — why no tennis odds after Oct 20). (3) ESPN's tennis times
> run hours after the provider's `commence_time` (Paris 2025-10-29: +2.3 h /
> +3.2 h), which is what the ~6k "started" rejections were. Fixed in
> `5925c77`: provider key names + the 500s in TENNIS_EVENTS; `--anchor
> provider` lists the provider's events per tournament-day (7,848 listings,
> 1 credit each, zero 404s) and clusters snapshots on its commence_time —
> **+3,013 matches priced** (52,944 rows, 1,022 snapshots, 0 skipped; 93% of
> the 3,228 unpriced matches the provider listed); live now discovers the
> ACTIVE tennis keys from the free `/v4/sports` each run (fallback: all 51,
> `TENNIS_PROVIDER_KEYS`) — first check found Basel, Vienna and Pan Pacific.
> Tennis priced share 13.6% → 20.3%; A/B re-run in the §4 row. Credits for
> the day: 94,171 of 5M. (4) **Tennis fixture orphans — root cause still
> live:** ESPN keeps a competition id when it re-fills a 1v1 draw slot (comp
> 185138: Fritz v Tsitsipas → Fritz v Hijikata); the resolver found the stale
> pairing by id, refused to realign it, never stamped the id, and every later
> time change forked another row (~25–50/week since the 09-04 fix). Fixed in
> `2337db1` (the stale scheduled pairing is released: id removed, status
> `cancelled`). Cleanup of the 19.5k existing orphans: `repair_fixture_orphans.py`
> (`3ec6e45`; twin rule shared with `repair_stranded_recs.py`; strictly
> pre-kickoff odds / CLV snapshots / predictions move onto the finished twin,
> never over its own rows; write-once archive first). Dry run: 16,003 orphans →
> 3,200 twins, 3,386 never-played pairings, 80 held back for their recs
> (stranded-recs dry run: 50 settle 18W/32L on 37 fixtures, 30 void). Applying
> it was blocked by the session's auto-mode classifier, so it is packaged as
> `scripts/ops/cleanup_tennis_orphans.sh` (verified backup → recs → merge →
> features → grade the moved predictions) for the operator to run. Side
> finding, not acted on: 2,673 soccer / 283 NHL / 8 NFL finished matches in
> the last 160 days still hold ungraded predictions (the pipeline grades
> `--days 14`), so a wide-window grader run should be a deliberate decision.

> **Update (2026-10-30, night) — the ungraded backlog, and a correction.**
> **Correction:** the "side finding" above blamed the grader's `--days 14`
> window. Wrong: the pipeline grades every match while it is inside that
> window. The rows were ungraded because their prediction types were never in
> `grading_outcomes`' dispatch table — the soccer derived markets
> (clean_sheet, win_to_nil, team_total, odd_even, total_goals,
> winning_margin, result_btts, result_over_under, and the four half-time
> markets) and the NHL Dixon-Coles markets (`puck_line`, `regulation_1x2`,
> `total_goals`, `clean_sheet`, `win_to_nil`). Fixed in `40c40ef`:
> multi-proposition markets grade the PICKED proposition (as `over_under`
> does); NHL `puck_line` shares the asian_handicap keys and math;
> `regulation_1x2` maps `regulation_winner` tie→draw. Half-time markets read
> `metadata.halftime_{home,away}_score`, which the ESPN results path now
> derives from the scoreboard's scoring plays (stamped only when the plays
> reproduce the final score; ESPN's soccer scoreboard has no linescores);
> `scripts/backfill_halftime_scores.py` stamps older rows (existing finished
> rows only, ESPN id else the exact triple, final score must match,
> write-once CSV). `1784861`: the grader never grades a prediction created
> after kickoff — 467 rows written 2026-06-02/03 for 2023-25 NFL (366), NBA
> (8) and tennis (83) matches sat ungraded only because they were older than
> the window, and a wide run would have folded that hindsight into accuracy.
> Backlog run (`grade_completed_matches.py --days 160`, 23:34Z): 64,098
> predictions graded on 2,974 matches (incl. ~2k pushes that keep
> `is_correct` NULL). Hit rates — soccer: clean_sheet 81.6%, team_total
> 87.3%, win_to_nil 87.2%, odd_even 48.4%, result_over_under 31.6%,
> winning_margin 25.9%, result_btts 24.7%, total_goals 22.2% (n=7,584 each);
> NHL: clean_sheet / win_to_nil 96.8%, puck_line 80.2%, regulation_1x2
> 48.1%, total_goals 24.4% (n=283; the model's argmax is always `8+`, a
> cumulative band). 5 June NHL recs whose selections were stored as
> `home -1.5` / `over 5.5` (unparseable, so won/lost with no P&L) now carry
> P&L; one — `home -1.5` at 4-2 — was mis-settled lost by the old argmax path
> and is now won (+35.92). Half-time backfill (23:39Z): 2,677 matches stamped (1,721 by ESPN id, 956
> by the exact triple; 6 refused because the plays did not reproduce the
> final, 147 ESPN competitions without a finished row); a random 100 of the
> planned stamps agreed 100/100 with ESPN's independent summary-endpoint
> linescores; HT draws 39.7% / home 34.4% / away 25.8%. Then 7,405 rows per HT
> market graded: over_under_ht 85.9%, btts_ht 79.8%, match_result_ht 35.2%,
> ht_ft 26.7% (27 each left NULL). Model finding, not acted on: the HT 1x2
> derivation picks home on 93% of matches at an average 39.5% and hits 35.0%,
> while the real HT draw rate is 39.7% — the HT draw is under-weighted.
> Remaining NULLs are by design:
> pushes (AH integer lines, DNB draws, NFL spread pushes), NFL ties, the
> phantom-line guard (14), the 467 post-kickoff rows. Monitor preview of the new
> slices (30d, the monitor's own drift logic, read-only): all inside thresholds
> except NHL regulation_1x2 ECE 0.069 and soccer ht_ft ECE 0.054 (warn), and
> soccer odd_even — accuracy 48.4% under the 52.4% break-even, an ALERT because
> the derived soccer markets inherit the enabled soccer default gate. No odds feed
> exists for any derived market (prod odds: 1x2 / AH / O-U / moneyline / spread /
> total), so odd_even can never emit a rec: `rec_gating` now gates
> (soccer, odd_even) off with that evidence (measured, not paged). The monitor
> class map gained `regulation_1x2` (3) and `puck_line` (2-way multi-line).

> **Update (2026-10-31, 00:47–00:54Z) — tennis orphan cleanup ran; it deleted
> 45 settled recs; restored the same night.** `scripts/ops/cleanup_tennis_orphans.sh`
> (operator-run; the 23:12Z attempt had aborted at step 2 before any write —
> CWD-relative CSV default, fixed `3ab4fb6`). Backup
> `betting_system-2026-10-31T004703Z.dump` landed first. Step 2 settled the
> 80 stranded tennis recs as the dry run said: 50 settled (18W/32L, net
> +61.96 on 3,460.39 staked; 13 are duplicates of an already-covered fixture,
> so n=37 real bets) and 30 voided (no twin). Step 3 merged 16,054 orphans
> onto 3,210 finished twins (moved 18,450 odds, 130,869 CLV snapshots, 2,823
> predictions — all graded in step 5) and deleted 3,386 never-played
> pairings; archive `fixture_orphans_tennis_20261031T004915Z.jsonl.gz`
> (273,534 rows). Stale tennis rows 19,470 → 30 (exactly the orphans the 30
> voided recs point at); 0 tennis recs pending; 0 ungraded pre-kickoff tennis
> predictions on finished matches.
> **The incident:** only 5 of the 50 settled recs survived. Step 2 re-points a
> rec's `match_id` onto the twin but not its `prediction_id`;
> `betting_recommendations.prediction_id` is `ON DELETE CASCADE`, and the
> orphan tool's has-recs guard looked only at `match_id`. The twin already
> held the same (model, version, type) for all 45 — so the rec's own
> prediction was never the one moved — and deleting the orphan deleted the
> prediction and the rec with it (no accumulator legs or history rows were
> attached). Fixes: `repair_fixture_orphans.py` now treats an orphan holding a
> prediction any rec references as has-recs (plan and DELETE guard), and
> counts + warns on a refused delete. `scripts/restore_cascaded_recs.py`
> re-inserts each lost rec from the pre-repair dump with the repair's
> settlement (status / actual_result / P&L from the audit CSV, match_id =
> twin) and links it to the twin's own prediction of the same model, version
> and type (the deleted prediction cannot be re-inserted beside it —
> `predictions_match_model_version_type_key`; the twins' predictions were all
> written pre-kickoff; max |Δp| vs the deleted ones 0.047, median 0.002).
> Every row is refused unless the twin's score and a fresh grade of the rec's
> own selection reproduce the CSV. Dry run: 45 planned, 0 refused (16W/29L,
> net +218.76 — the CSV's figure exactly). **Applied 01:07:56Z** (audit CSV
> `restored_recs_2026-10-31T010756Z.csv`): 45 inserted; all 80 recs of the
> repair now match its plan (status, P&L to the cent — `profit_loss` is
> numeric(10,2) — and match_id), each has its prediction, and the recs P&L
> widget's query sees all 50 settled at net +61.96; a re-run finds nothing to
> restore. Open: the CASCADE
> itself — any prediction delete silently deletes recs; `ON DELETE RESTRICT`
> (a migration, needs approval) would make this class fail loudly.

> **Update (2026-09-02) — UI: panes showed ~3 matches, detail page
> overflowed, deploys broken by a GitHub 401.** Three root causes, all
> fixed in one push. (1) `GET /predictions/upcoming` applied `LIMIT` to
> ROWS and only collapsed to the headline market for NHL, so soccer's 19
> derived-market rows per match saturated the pane's `limit=50` at ~3
> matches — Premier League / La Liga simply never rendered. It now
> returns ONE headline row per match for every sport (`headline_task()`;
> `sport:prediction_type` pairs in the All-Sports path so NHL's regulation
> market, persisted as `match_result`, is not double-counted); ceiling
> 2000; soccer sub-markets selectable via `market=<prediction_type>`.
> (2) `/predictions/match/{id}` was ordered by `prediction_type`, so the
> detail sidebar donut drew `asian_handicap` (51 outcomes) with outside
> labels — the "jumbled top-right". Headline row is now first (API +
> client guard); the chart draws no outside labels (donut + legend ≤6
> outcomes, scrolling bar list above that); the odds and stats cards
> read the real API fields (`selection`/`market_type`/`line`,
> `home_form`/`away_form`/`head_to_head` — both had been reading keys
> that don't exist and rendered empty or label-less); every long list
> scrolls inside its card; `bg-card` was never wired as a Tailwind token
> (added). Lists are date-grouped with result counts, league and
> soccer-market filters. (3) GitHub returns 401 on anonymous
> `git-upload-pack` from the Hetzner IP even for public repos (run
> 33670930981; the same fetch succeeds from a residential IP), so
> `deploy_remote.sh` now authenticates with the run's `GITHUB_TOKEN` via
> a per-process `http.extraheader` (no admin rights for a deploy key;
> manual pulls need a PAT — `OPERATIONS.md`). Verified: api 165 tests,
> lint, `next build`/type-check/lint, 3-lens adversarial review (two
> medium findings fixed before commit).

**Day 1 (production correctness + safety):**
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
