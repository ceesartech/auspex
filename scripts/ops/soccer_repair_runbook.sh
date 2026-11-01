#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval. Every step below
# changes prod data through its own wrapper, and each wrapper takes its own
# verified B2 backup before its first write (CLAUDE.md: data repairs need a
# verified backup first) and checks its own exact after-state.
#
# THE SOCCER REPAIR RUNBOOK (2026-10-31): team identity, team merge, results
# path / grading / seasons, football-data kickoffs, soccer orphans, trigger
# 027 — the whole sequence, in order, one wrapper per step. Expected counts
# are prod's (read only, 2026-10-31) and the rehearsal's: this runbook and the
# real wrappers and tools, end to end through a docker/Airflow shim, against
# a local Postgres built from the migrations (011's trigger restored) and a
# read-only export of prod taken 11:08Z (compute_features stubbed). Rehearsed
# again on a 13:02Z export (TimeZone UTC, as prod) with every count below
# unchanged, through three runs: step 4's backup never landing (exit 1, the
# retrain DAGs kept paused with an ALERT) then --from 4 (adopted, steps 4-11,
# unpaused at the end); the merge's grading failing after its commit (STATUS
# failed, unsettled_recs_on_adopted 3) then --from 3 (step 3 finishes that
# apply: 52 rows graded, 0 unsettled, COMPLETED); and a full re-run that only
# reads (every step ALREADY DONE, 0 backups, row fingerprint unchanged).
# Start it right after the deploy: until step 1 runs, results for the mapped
# Liga MX fixtures are refused as identity drift, and the results sweep only
# looks back 3 days (ESPN's scoreboards for today back through today-3, UTC;
# ESPN files a fixture under its local date). Four mapped fixtures were not
# finished on prod at 2026-11-01 08:10Z (scoreboard date in brackets):
#   Guadalajara v Atlante  6b2133b1  11-01 01:07Z [10-31]  played: 4-2
#   Santos v Necaxa        5b94347b  11-01 23:00Z [11-01]
#   Atlante v Santos       52c8d673  11-07 03:00Z [11-06]
#   León v Guadalajara     0e7f388c  11-09 02:00Z [11-08]
# Step 1 must run before the sweep stops covering each — ~2026-11-04 00:00Z,
# 11-05 00:00Z, 11-10 00:00Z, 11-12 00:00Z — for the sweep to record it.
# One step 1 reaches too late stays a stale 'scheduled' row, and joins the
# never_resulted class 96 h after kickoff (11-05 01:07Z, 11-05 23:00Z,
# 11-11 03:00Z, 11-13 02:00Z). From then step 12 heals it only if the
# reviewed list holds a 'heal' row for it (only Guadalajara v Atlante does:
# ESPN 401876915 final 4-2); any other it lists "NOT in the reviewed list"
# and leaves (healing it takes its own reviewed row, a commit and a deploy,
# then --only 12 and --only 16). Before its 96 h step 12 lists it as
# PENDING (soccer rows still 'scheduled' 6-96 h after kickoff) and leaves
# it; --only 12 (then --only 16) after the time shown reaches it. If that
# late --only 12 aborts because the orphan cleanup has work again (rows that
# went stale since step 7), run --only 7, then --only 12, then --only 16.
#
#  1 scripts/ops/repoint_team_matches.sh --pass 1
#    Moves 33 Liga MX sides (31 match rows) off Santos FC's 'Santos' and CD
#    Guadalajara's 'Guadalajara' onto Santos Laguna / Guadalajara Chivas with
#    their match_stats, stamps ESPN ids 2674/225/219. Expect plan 33 repoint /
#    0 done / 6 refused; step 2 refuses unless exactly the map's 39 sides sit
#    on santos/guadalajara. After: 0/33/6, 6 sides left.
#  2 scripts/ops/migrate_027_outcome_trigger.sh
#    Function body only (calculate_match_outcome skips post-kickoff rows), no
#    rows. Expect "97384c8327efa61669af1012c06c6232|1|1|1" and
#    "m027 pre_pred=true post_pred=null pre_rec=won post_rec=pending".
#  3 scripts/ops/merge_soccer_teams.sh
#    Drains then pauses auspex_pipeline + fetch_live_odds while it runs.
#    171 pairs merge (0 refused, 0 kickoffs adopted): 2,469 match sides
#    re-pointed, 308 twin fixtures folded (Lecce v Roma and Olympiakos v OFI
#    Crete take their ESPN twins' results), 171 team rows deleted, aliases and
#    ESPN ids recorded (168 stamped), 47 odds-feed spellings held (0 unmatched
#    before the merge; any would fail the step).
#  4 scripts/ops/repair_football_data_kickoffs.sh
#    dedupe merge 68, alias_merge 4, fold 26, refuse 0, fold_unsettled_recs 0;
#    shift <= 63,533 (63,532 on the 11:08Z export). Soccer rows fall by 98,
#    stale rows by 26; --check passes. The trigger grades ~138 predictions on
#    the folded rows and settles no rec.
#  5 scripts/ops/repoint_team_matches.sh --pass 2
#    Moves the 6 sides pass 1 refused (4 rows the fd repair alias-merged + 2
#    stale rows). Expect 6 repoint / 32 done / 1 refused (cb4e21fc 'match row
#    not found': folded by step 3); step 2 refuses unless exactly 6 sides are
#    left. After: 0 Liga MX rows on santos/guadalajara.
#  6 scripts/ops/repair_stranded_recs_soccer.sh
#    12 candidates: 7 recs voided as written after kickoff (profit 0, 'void:
#    written after kickoff') and re-pointed onto their twins, 0 settled, 5 left
#    on the row without a twin (Panathinaikos v Kifisia).
#  7 scripts/ops/cleanup_soccer_orphans.sh
#    Stale soccer rows: merges <= 204 (merge_delete + merge_cancel), World Cup
#    placeholders <= 65, ESPN-verified postponements <= 6, no no-twin delete,
#    0 blocked; recs untouched. Rehearsal: merge_delete 94, merge_cancel 19,
#    placeholders 65, postponed 6 (stale rows 288 -> 104). Then features +
#    grading for the twins (107; 800 predictions graded).
#  8 scripts/ops/backfill_soccer_regulation_scores.sh, then its re-grade
#    15 AET/PENS rows get their 90-minute score (5 change basis). The
#    re-grade (prediction rows only) is applied here ONLY when its dry run
#    plans 132-138 changes (138 on the rehearsal), pinned to that exact count
#    (--expect-changes); a fresh dry run must then plan 0.
#  9 scripts/ops/backfill_season_labels.sh
#    relabel_promote_raw <= 62,430 and relabel_espn <= 2,200, reviewed leagues
#    only (rehearsal: 62,156 and 1,603); after: re-plan 0.
# 10 compute_features.py --backfill-finished soccer
#    Recomputes features_cache for every finished soccer match (replaces the
#    per-step feature commands steps 1, 3 and 7 print). Skipped as already
#    done when its last completion is recorded after every data-changing
#    step's last change (this runbook's state file, below).
# 11 merge_soccer_teams.py --check-after <plan> --odds-since <MERGED_AT> --after-later-steps
#    Read-only. Waits (up to ODDS_WAIT_POLLS x AF_POLL_SECONDS = 240 x 30 s,
#    2 h; fetch_live_odds runs every 90 min) for a fetch_live_odds run that
#    started after the merge and finished; then
#    teams_reminted, resolver_misses, odds_not_yet_observed and
#    odds_fixtures_stopped must be 0 (and the merge's other after-state counts:
#    only repoints_missing is reported, not failed — steps 4 and 7 delete some
#    re-pointed rows on purpose; 17 on the rehearsal). The plan and MERGED_AT
#    come from /app/backups/merge_soccer_teams_last_apply.env.
#
# Steps 12-16 (added 2026-11-01, the follow-ups the runbook used to leave to
# separate approvals) need steps 1-11 done: each one REFUSES unless the
# state file records steps 1-11 complete (each one's last record 'changed',
# 'already_done' or 'checked'). After a full run of the earlier, 11-step
# runbook, run --from 12; with no such record (a lost state file), run the
# runbook from step 1: a finished step re-plans read-only and records
# ALREADY DONE, which verifies its effect, then 12-16 follow. Rehearsed
# 2026-11-01, with real ESPN scoreboards, through the shim, first as fifteen
# steps (no labels repair yet) on read-only prod exports of 02:42Z and
# 05:15Z: the 11-step runbook (d592a4d) from step 1 (every count above
# reproduced), then --from 12 on its state file; a full re-run of that (all
# ALREADY DONE / checked, no backup, row fingerprint unchanged); a run from
# step 1 on a fresh copy (the same counts); --from 12 refused (no pause, no
# backup) on a state file without step 11; and a never_resulted row outside
# the list (Guadalajara v Atlante, its kickoff moved 6 days back on the
# replica) listed and left alone while step 12 went through. Then as the
# sixteen steps below on a 06:48:46Z export (ESPN fetched ~07:07Z): the
# 11-step runbook (d592a4d) from step 1, every count above again; --from 12
# on its state file — step 12 38/38/3,173/3 and the re-run's merge_cancel 1,
# step 13 create 2 / tag 0 / relabel 10, 10, 20 / rename 1, 68 /
# cross_country_after 0, step 14 match_stats 4 / odds 13 / refused_raw 0,
# step 15 26 / 504, step 16 run, the retrain DAGs unpaused; a full re-run of
# the sixteen (all ALREADY DONE / checked: step 13 still done after step
# 14's promote_raw; backups 14 -> 14, row fingerprint unchanged); this
# runbook from step 1 on a fresh copy (the same counts in all sixteen steps;
# recs, predictions and the trigger function identical, the other tables
# equal in counts: new uuids and timestamps);
# --from 12 and --from 13 refused (no pause, no backup) on a state file
# without step 11; and promote_raw's checked dry run before step 13:
# refused_raw 68, exit 3 (why the labels repair comes first):
# 12 scripts/ops/backfill_soccer_never_resulted.sh, then
#    scripts/ops/cleanup_soccer_orphans.sh --after-never-resulted
#    The cleanup's never_resulted rows (42): ESPN's result recorded, through
#    the results path, on the 38 the rehearsal reviewed for healing
#    (scripts/data/soccer_never_resulted_reviewed.csv), and those rows graded
#    (scripts/backfill_never_resulted.py, pinned, one transaction). Expect
#    candidates 38, heal 38, refuse 0, predictions graded 3,173, recs changed
#    0, teams stamped 3 (the wrapper's bounds); started late (above),
#    Guadalajara v Atlante's reviewed 'heal' row adds 1 / 1 / 76 / 0, and
#    only then are the bounds 39, 39, 3,249, 3. The list's other 4, any
#    never_resulted row it does not hold (a fixture whose result never
#    arrived joins the class 96 h after kickoff), and every PENDING row
#    (still 'scheduled' 6-96 h after kickoff) are listed and left as they
#    are; an orphan cleanup with work left (rows that went stale after step
#    7) aborts it before its backup. One of the 4 is a fork of a healed row (Coquimbo Unido v O'Higgins
#    2026-06-15), which the cleanup re-run then merges (merge_cancel 1: it
#    holds void recs; that re-run's ceilings are merges <= 1, no placeholder,
#    no postponement); ESPN does not report the other 3 final. A corpus step.
# 13 scripts/ops/repair_soccer_labels.sh
#    The rows football-data's files filed under the wrong league or decoded
#    wrongly (scripts/repair_soccer_labels.py, the reviewed map
#    scripts/data/soccer_misfiled_division_rows.csv; one transaction, pinned
#    on every PLAN key, write-once audit CSV and reversal SQL): leagues SC1 /
#    SC2 created (or a code-less one of that name tagged: tag_league), the 10
#    Scottish fixtures of 2026-08-01 and their raw rows relabelled from La
#    Liga / La Liga 2, the 20 clubs on them moved to their Scottish league,
#    'PreuÃ\x9fen MÃ¼nster' and its 68 raw rows renamed. Expect
#    create_league 2, tag_league 0, relabel_raw 10, relabel_match 10,
#    relabel_team 20, rename_team 1, rename_raw 68, refuse 0, mojibake_left 0,
#    cross_country_after 0 (the repoint passes moved Guadalajara / Santos).
#    A corpus step: matches.league_id feeds the per-league Dixon-Coles /
#    Poisson baselines and the team name keys their teams (no feature value
#    changes: compute_features reads neither). It must precede step 14: until
#    it has run, promote_raw commits everything else and exits 3 on the 68
#    mis-decoded raw rows (its screen), and step 14 aborts at its step 0.
# 14 scripts/ops/promote_raw_after_repair.sh
#    promote_raw in checked mode (scripts/promote_raw.py --dry-run, then
#    --apply --expect-plan): re-writes the 4 match_stats rows step 4 archived
#    for the alias-merged Liga MX fixtures (it refuses nothing once step 5 has
#    run, and screens out nothing once step 13 has) and adds 13 closing odds;
#    every other PLAN key 0, refused_raw included. A corpus step.
# 15 scripts/ops/grade_folded_soccer_rows.sh
#    The rest of the predictions on the 26 rows step 4 folded
#    (scripts/grade_matches.py, pinned, one transaction): 504 graded, 0 recs
#    changed. Predictions only: not a corpus step.
# 16 compute_features.py --backfill-finished soccer
#    As step 10, after steps 12-14 changed the corpus (the same skip: when no
#    corpus change is recorded after the last feature recompute, step 10's or
#    16's, it reads ALREADY DONE).
#
# Before step 1 (read-only) it REFUSES TO START unless:
#   * no other runbook run is going: it takes ~/.auspex_ops/soccer_repair_runbook.lock
#     (scripts/ops/_airflow_lib.sh af_lock; a lock whose run is gone — killed
#     outright — is taken over, and the ledger says so; a refused run touches
#     nothing). The merge wrapper takes its own lock, so step 3 runs under this
#     one without a wait;
#   * the file being run is byte-identical to this runbook as committed at
#     the VM HEAD (git rev-parse HEAD:<path> equals git hash-object of the
#     file), so HEAD carries the commit that holds it, and scripts/ and
#     services/ carry no local edits: the bind-mounted wrappers and tools are
#     HEAD's (a git merge-base --is-ancestor test of the runbook's own commit
#     would be true by construction, so none is made);
#   * void_post_kickoff_recs.sh has run (>= 22 recs carry its 'void: written
#     after kickoff (was ...)' label) and fix_unplayed_soccer_rows.sh has run
#     (its 4 rows are 'cancelled');
#   * no repair tool's --apply is still running in the api container
#     (repoint_team_matches, merge_soccer_teams, repair_football_data_kickoffs,
#     repair_stranded_recs, repair_fixture_orphans,
#     backfill_soccer_regulation_scores, regrade_predictions,
#     backfill_season_labels, backfill_never_resulted, repair_soccer_labels,
#     promote_raw, grade_matches): `docker compose exec -T` does not forward an
#     interrupt, so an apply cut short on the host runs on in the container,
#     and a re-run would race it. It lists them; `docker compose top api`
#     shows when they have ended;
#   * neither retrain_models (Sun 04:00 UTC) nor retrain_halftime_dc (the 1st,
#     05:30 UTC) has a run in flight. Both fit on the soccer corpus this
#     sequence rewrites, so each one that is unpaused is drained and paused
#     (Airflow 3.3 never advances a paused DAG's runs) — a DAG paused before
#     the run stays paused and is never unpaused here, unless its marker names
#     an earlier runbook run that is gone (then it is adopted; a marker naming
#     a run still going refuses the start).
# They are unpaused when the runbook exits ONLY if the corpus is consistent:
# no corpus step (1, 3, 4, 5, 7, 12, 13, 14) failed or was cut short in this
# run, and the state file records a feature recompute (step 10 or 16) after the
# last corpus change (or no corpus change at all). A corpus change is a corpus
# step's 'changed' record or its 'failed' one: a corpus step that fails or is
# cut short is recorded 'failed' (it may have committed part of its change —
# the repoint passes and the fd repair commit their apply before their
# after-state), so a later run that reads that step ALREADY DONE does not
# clear it; only the next completion of step 10 or 16 does. Each corpus step is also recorded 'started'
# before it runs, so a run killed outright during one (kill -9, a host
# crash: no trap, no 'failed') still leaves the change on record; a
# read-only re-run closes its own 'started'. Otherwise — a failed or interrupted corpus
# step, or a run that leaves step 10 (16 after steps 12-14) still to do — they
# STAY PAUSED and the runbook prints an ALERT: unpause them only after that
# recompute has run (re-run with --from N, or --only 10 / --only 16, which
# unpauses them itself once it completes; the ALERT's command, for after
# that, also removes the marker).
# Unpaused now, either would start its missed run at once and train on a
# half-repaired corpus with stale features (retrain_models' swap_production
# then serves it and restarts the api). A later runbook run (--from N) adopts
# them (their markers in ~/.auspex_ops, scripts/ops/_airflow_lib.sh) and
# unpauses them once the recompute has completed. A failure in steps 2, 6, 8,
# 9, 11 or 15 (none feeds the training corpus or its features) leaves that
# decision to the state file.
# Every pause, unpause and ALERT is also appended to
# ~/.auspex_ops/airflow_pauses.log, which survives a dropped terminal.
#
# It stops at the first step that exits non-zero and prints "fix, then re-run
# with --from N". Every step is safe to re-run after it succeeded: each
# wrapper first re-plans read-only and exits 0 ("ALREADY DONE") when it has
# nothing left to do, without a backup or a write (the merge and the cleanup
# first finish follow-ups an earlier run of theirs left pending: the merge's
# grading, the cleanup's features and grading). A full re-run of a finished
# sequence therefore only reads.
#
# Usage (VM host; deploy the integrated branch to main first). Run it inside
# tmux, so a dropped SSH session interrupts nothing (an interrupt is safe —
# every wrapper's trap restores what it paused — but it stops the sequence):
#   tmux new -s soccer_repair
#   bash /opt/auspex/scripts/ops/soccer_repair_runbook.sh 2>&1 | tee -a ~/soccer_repair_runbook.log
#   ... --from N     run steps N..16
#   ... --only N     run step N alone
# State: RUNBOOK_STATE (default ~/.soccer_repair_runbook.state) records each
# step's completion, in order ("changed", "already_done", or "checked" for
# step 11), each corpus step's start ("started") and each corpus step that
# failed or was cut short ("failed"), one line per record, "step=N
# result=R at=<epoch> head=<commit>" (the 11-step runbook's format, unchanged);
# the feature steps' skip, the retrain unpause and steps 12-16's refusal read
# it (a missing file means step 10 / 16 runs and steps 12-16 refuse; a record
# that cannot be written keeps the retrain DAGs paused at the end).
# Read-only check on prod 2026-10-31 ~12:10Z (VM HEAD a071d6a, not deployed):
# the preflight reads 22 relabelled / 4 cancelled, 39 Liga MX sides sit on
# santos/guadalajara, calculate_match_outcome is 011's.
# (AUSPEX_DIR, TMPDIR, AF_POLL_SECONDS, AF_STATE_DIR and ODDS_WAIT_POLLS exist for tests and rehearsals.)
set -uo pipefail
# A dead terminal (Ctrl-C or an SSH drop with the output through tee) must not
# kill the exit path by SIGPIPE before it has decided about the retrain DAGs.
trap '' PIPE
FROM=1
ONLY=""
usage() { echo "usage: $0 [--from N | --only N]   (N = 1..16; see the header)"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --from) [ $# -ge 2 ] || { usage; exit 2; }; FROM="$2"; shift 2 ;;
    --only) [ $# -ge 2 ] || { usage; exit 2; }; ONLY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
for n in "$FROM" ${ONLY:+"$ONLY"}; do
  if ! [[ "$n" =~ ^([1-9]|1[0-6])$ ]]; then usage; exit 2; fi
done

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
OPS="$(dirname "$SELF")"
REL=scripts/ops/soccer_repair_runbook.sh
AUSPEX_DIR="${AUSPEX_DIR:-/opt/auspex}"
cd "$AUSPEX_DIR" || { echo "ABORT: cannot cd to $AUSPEX_DIR"; exit 1; }
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] runbook: $*"; }
# shellcheck source=scripts/ops/_airflow_lib.sh
. "$OPS/_airflow_lib.sh" || { log "ABORT: cannot load $OPS/_airflow_lib.sh"; exit 1; }
STATE_FILE="${RUNBOOK_STATE:-$HOME/.soccer_repair_runbook.state}"
ODDS_WAIT_POLLS="${ODDS_WAIT_POLLS:-240}"
RETRAIN_DAGS="retrain_models retrain_halftime_dc"
CURRENT_STEP=""
STATE_WRITE_FAILED=""
# The steps that change the training corpus or its features, and the steps that recompute the features after them.
CORPUS_STEPS=" 1 3 4 5 7 12 13 14 "
FEATURE_STEPS=" 10 16 "
LAST_STEP=16
# One runbook run at a time, taken before anything this run could leave behind (a refused run exits here, with no
# trap: it touches nothing of the run that holds the lock). finish releases it last; the merge wrapper takes its own
# lock, so step 3 does not wait.
af_lock soccer_repair_runbook || exit 1
RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/soccer_runbook.XXXXXX") || { log "ABORT: mktemp failed"; af_unlock; exit 1; }
# The state file's last corpus record (records are appended in order): a feature step (10 or 16), a corpus step, or
# nothing. A feature step's completions count, and a corpus step's 'changed' and 'failed'. So does its 'started',
# written before it runs, until the same step records a read-only result (ALREADY DONE / checked) right after it: a
# run killed outright during a corpus step (kill -9, a host crash: finish never runs) leaves 'started' behind,
# unclosed, and that counts as a change for good — a later re-run that reads the step ALREADY DONE still finds the
# corpus changed since the last feature recompute.
last_corpus_record() {
  [ -r "$STATE_FILE" ] || return 0
  awk -v corpus="$CORPUS_STEPS" -v features="$FEATURE_STEPS" '
    $1 !~ /^step=/ || $2 !~ /^result=/ { next }
    { n = substr($1, 6); r = substr($2, 8) }
    index(features, " " n " ") { last = n; open = ""; next }
    !index(corpus, " " n " ") { next }
    r == "changed" || r == "failed" { last = n; open = ""; next }
    r == "started" { before = (open != "" ? open : last); open = n; last = n; next }
    open == n { last = before; open = "" }
    END { print last }
  ' "$STATE_FILE"
}
is_feature_step() { case "$FEATURE_STEPS" in *" $1 "*) return 0 ;; esac; return 1; }
# 0 when a feature step (10 or 16) recomputed the features after the last corpus change.
features_current() { is_feature_step "$(last_corpus_record)"; }
# 0 when a retrain would fit a consistent corpus: no corpus step failed or was cut short in this run, the state
# file was writable, and nothing changed the corpus after the last feature recompute (or nothing ever did).
corpus_consistent() {
  local last
  [ -z "$STATE_WRITE_FAILED" ] || return 1
  if [ -n "$CURRENT_STEP" ]; then case "$CORPUS_STEPS" in *" $CURRENT_STEP "*) return 1 ;; esac; fi
  last=$(last_corpus_record)
  [ -z "$last" ] || is_feature_step "$last"
}
# The feature step that follows a corpus step: 10 after steps 1-9, 16 after 12-14 (both run the same recompute).
feature_step_after() { if [ "${1:-0}" -ge 12 ] 2>/dev/null; then echo 16; else echo 10; fi; }
# Steps 1-11 whose last state-file record is not a completion ('changed', 'already_done', 'checked'), as
# "N(last result)" words; nothing when all eleven are complete. Steps 12-16 run only then.
old_steps_incomplete() {
  [ -r "$STATE_FILE" ] || { echo "all (no state file $STATE_FILE)"; return 0; }
  # seen[] apart from last[]: reading last[i] for a step never recorded creates it, so 'i in last' cannot tell.
  awk '
    $1 ~ /^step=/ && $2 ~ /^result=/ {
      n = substr($1, 6) + 0
      if (n >= 1 && n <= 11) { last[n] = substr($2, 8); seen[n] = 1 }
    }
    END {
      for (i = 1; i <= 11; i++) {
        r = (i in seen) ? last[i] : "none"
        if (r != "changed" && r != "already_done" && r != "checked") printf "%d(%s) ", i, r
      }
    }
  ' "$STATE_FILE"
}
OLD_STEPS_HELP="steps 12-16 need steps 1-11 done: after the 11-step runbook's full run, its state file records them; otherwise run the runbook from step 1 (finished steps only re-plan, read-only, and record ALREADY DONE)"
finish() {
  local rc=$?
  trap '' INT TERM HUP
  # A corpus step that failed or was cut short may have committed part of its change (the repoint passes and the fd
  # repair commit their apply, then check the after-state): recorded 'failed', only a later feature recompute (step
  # 10 or 16) clears it — a re-run that reads that step ALREADY DONE writes no 'changed' record.
  local fs
  fs=$(feature_step_after "${CURRENT_STEP:-${ONLY:-$FROM}}")
  if [ -n "$CURRENT_STEP" ]; then
    case "$CORPUS_STEPS" in
      *" $CURRENT_STEP "*)
        if echo "step=$CURRENT_STEP result=failed at=$(date -u +%s) head=${head:-unknown}" >> "$STATE_FILE" 2>/dev/null; then
          af_say "step $CURRENT_STEP failed or was cut short (exit $rc): recorded in $STATE_FILE — step $fs runs again before the retrain DAGs are unpaused"
        else
          STATE_WRITE_FAILED=1
          af_note "WARNING: could not record step $CURRENT_STEP's failure in $STATE_FILE — after the fix, run steps $CURRENT_STEP-$fs (--from $CURRENT_STEP) before any retrain; do not rely on step $fs's skip"
        fi
        ;;
    esac
  fi
  if [ -n "${AF_PAUSED_BY_US// /}" ]; then
    if corpus_consistent; then
      af_resume
    else
      af_keep_paused "the soccer repair is not complete (exit $rc${CURRENT_STEP:+ in step $CURRENT_STEP}) and no feature recompute (step $fs) has run since the last corpus change: a retrain now would fit a half-repaired corpus" \
        "Unpause it only after step $fs has run: re-run the runbook (--from ${CURRENT_STEP:-N}, or --only $fs), which unpauses it itself once step $fs completes; by hand, only after step $fs"
    fi
  fi
  rm -f "$RUN_DIR"/*.sql
  $DC exec -T postgres rm -f /tmp/soccer_runbook_q.sql </dev/null >/dev/null 2>&1
  af_say "logs of this run: $RUN_DIR"
  af_unlock
}
trap finish EXIT
trap 'exit 130' INT TERM HUP
should_run() { if [ -n "$ONLY" ]; then [ "$1" = "$ONLY" ]; else [ "$1" -ge "$FROM" ]; fi; }

log "start (from ${ONLY:-$FROM}${ONLY:+ only}); AUSPEX_DIR=$AUSPEX_DIR"

# ---- preflight (read-only) -------------------------------------------------
# The file being run must be HEAD's version of this runbook (so HEAD carries the commit that holds it), and the
# bind-mounted scripts/ and services/ must be HEAD's too. (`merge-base --is-ancestor` of the commit that last
# touched the runbook would be true by construction: that commit is found from HEAD.)
head=$(git -C "$AUSPEX_DIR" rev-parse HEAD 2>/dev/null)
head_blob=$(git -C "$AUSPEX_DIR" rev-parse "HEAD:$REL" 2>/dev/null)
self_blob=$(git -C "$AUSPEX_DIR" hash-object "$SELF" 2>/dev/null)
self_commit=$(git -C "$AUSPEX_DIR" log -1 --format=%h HEAD -- "$REL" 2>/dev/null)
if [ -z "$head" ] || [ -z "$head_blob" ] || [ "$head_blob" != "$self_blob" ]; then
  log "REFUSED: this runbook ($SELF) is not the one committed at the VM HEAD (${head:-no git HEAD} in $AUSPEX_DIR) — deploy the integrated branch (push to main, let CI deploy) and run /opt/auspex/$REL"
  exit 1
fi
if ! git -C "$AUSPEX_DIR" diff --quiet HEAD -- scripts services; then
  log "REFUSED: scripts/ or services/ differ from HEAD $head (bind-mounted code that is not the deployed commit):"
  git -C "$AUSPEX_DIR" diff --stat HEAD -- scripts services | tail -5
  exit 1
fi
log "this runbook is HEAD's version (VM HEAD $head, last changed in $self_commit); scripts/ and services/ are clean"
# A run that starts at step 12 or later: steps 1-11 must be recorded complete before anything is paused (run_step
# checks again before each of steps 12-16, for a run that reaches them).
if [ "${ONLY:-$FROM}" -ge 12 ]; then
  incomplete=$(old_steps_incomplete)
  if [ -n "$incomplete" ]; then
    log "REFUSED: steps 1-11 are not all recorded complete in $STATE_FILE (not complete: $incomplete) — $OLD_STEPS_HELP; nothing changed"
    exit 1
  fi
  log "steps 1-11 are recorded complete in $STATE_FILE"
fi

cat > "$RUN_DIR/preflight.sql" <<'SQLEOF'
BEGIN READ ONLY;
SELECT (SELECT count(*) FROM betting_recommendations
         WHERE actual_result LIKE 'void: written after kickoff (was %') || '|' ||
       (SELECT count(*) FROM matches
         WHERE id IN ('d7d8a9b7-39a6-4340-bb53-09982171fa8a', 'ff74d919-d543-4729-a438-5f59e826b7b9',
                      '681794ae-f1e5-4e10-a365-8e71f86ef4a1', '83ce848f-64ad-4147-ab32-4a770a5eed15')
           AND status = 'cancelled');
ROLLBACK;
SQLEOF
$DC cp "$RUN_DIR/preflight.sql" postgres:/tmp/soccer_runbook_q.sql >/dev/null 2>&1
pre=$($DC exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -At -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/soccer_runbook_q.sql' 2>&1 | grep -E '^[0-9]+\|[0-9]+$' | tail -1)
relabelled=${pre%%|*}; unplayed=${pre##*|}
if ! [[ "$pre" =~ ^[0-9]+\|[0-9]+$ ]]; then log "REFUSED: the preflight query failed (got '$pre')"; exit 1; fi
if [ "$relabelled" -lt 22 ]; then
  log "REFUSED: void_post_kickoff_recs.sh has not run ($relabelled recs carry its label, it relabels 22) — run it first"
  exit 1
fi
if [ "$unplayed" != "4" ]; then
  log "REFUSED: fix_unplayed_soccer_rows.sh has not run ($unplayed of its 4 rows are cancelled) — run it first"
  exit 1
fi
log "void_post_kickoff_recs has run ($relabelled relabelled) and fix_unplayed_soccer_rows has run (4/4 cancelled)"

# An apply a run cut short may still be running in the api container: docker exec does not forward an interrupt.
REPAIR_TOOLS="repoint_team_matches merge_soccer_teams repair_football_data_kickoffs repair_stranded_recs repair_fixture_orphans backfill_soccer_regulation_scores regrade_predictions backfill_season_labels backfill_never_resulted repair_soccer_labels promote_raw grade_matches"
# shellcheck disable=SC2086  # $REPAIR_TOOLS is a word list
running=$(af_applies_running $REPAIR_TOOLS)
case $? in
  0)
    log "REFUSED: a repair apply is still running in the api container (an earlier run was cut short during it; docker exec does not forward an interrupt):"
    echo "$running" | sed 's/^/  /'
    log "  wait until it has ended (docker compose top api), then re-run the runbook with the same arguments — nothing changed"
    exit 1
    ;;
  1) log "no repair tool's --apply is running in the api container" ;;
  *)
    log "REFUSED: could not read the api container's processes to rule out an apply still running (is the api container up?) — nothing changed"
    exit 1
    ;;
esac

# shellcheck disable=SC2086  # $RETRAIN_DAGS is a word list
if ! af_drain_and_pause 1 4 $RETRAIN_DAGS; then
  log "REFUSED: $RETRAIN_DAGS: a run in flight, a DAG that cannot be paused, or one another run still holds (above) — start the runbook after it finishes"
  exit 1
fi

# ---- steps -----------------------------------------------------------------
record() {  # record <step> <started|changed|already_done|checked>
  echo "step=$1 result=$2 at=$(date -u +%s) head=$head" >> "$STATE_FILE" 2>/dev/null \
    || { STATE_WRITE_FAILED=1; log "WARNING: could not write $STATE_FILE (the feature steps' skip then re-runs them; the retrain DAGs stay paused at the end)"; }
}
# A step's result is 'already_done' when its output says ALREADY DONE, 'checked' when it says CHECKED (read-only),
# else 'changed' — unless the step wrote $RUN_DIR/step<n>.result itself (a step that runs two wrappers: either one's
# ALREADY DONE alone must not hide the other's change).
run_step() {  # run_step <n> <label> <command...>
  local n="$1" label="$2" rc result incomplete
  shift 2
  should_run "$n" || return 0
  if [ "$n" -ge 12 ]; then
    incomplete=$(old_steps_incomplete)
    if [ -n "$incomplete" ]; then
      log "STEP $n REFUSED: steps 1-11 are not all recorded complete in $STATE_FILE (not complete: $incomplete) — $OLD_STEPS_HELP"
      exit 1
    fi
  fi
  CURRENT_STEP="$n"
  case "$CORPUS_STEPS" in *" $n "*) record "$n" started ;; esac
  log "STEP $n/$LAST_STEP START: $label"
  "$@" 2>&1 | tee "$RUN_DIR/step$n.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    if [ "$n" = "7" ] && [ "$rc" = "2" ]; then
      log "STEP 7 INCOMPLETE (exit 2): the cleanup is verified but its feature/grading follow-ups failed — fix, then re-run with --from 7 (the cleanup then runs only its pending follow-ups)"
    else
      log "STEP $n FAILED (exit $rc): $label — fix, then re-run with --from $n"
    fi
    exit "$rc"
  fi
  result=changed
  if [ -f "$RUN_DIR/step$n.result" ]; then
    result=$(cat "$RUN_DIR/step$n.result")
  else
    grep -q "ALREADY DONE" "$RUN_DIR/step$n.log" && result=already_done
    grep -q "CHECKED (read-only)" "$RUN_DIR/step$n.log" && result=checked
  fi
  record "$n" "$result"
  CURRENT_STEP=""
  log "STEP $n/$LAST_STEP DONE ($result)"
}

step8() {
  local out="$RUN_DIR/step8_wrapper.log" line ids types changes min max rc
  bash "$OPS/backfill_soccer_regulation_scores.sh" 2>&1 | tee "$out"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || return "$rc"
  line=$(grep '^REGRADE ' "$out" | tail -1)
  ids=$(echo "$line" | sed -n 's/.* ids_file=\([^ ]*\).*/\1/p')
  types=$(echo "$line" | sed -n 's/.* types=\([^ ]*\).*/\1/p')
  changes=$(echo "$line" | sed -n 's/.* changes=\([0-9]*\).*/\1/p')
  min=$(echo "$line" | sed -n 's/.* min=\([0-9]*\).*/\1/p')
  max=$(echo "$line" | sed -n 's/.* max=\([0-9]*\).*/\1/p')
  if [ -z "$ids" ] || [ -z "$types" ] || [ -z "$changes" ] || [ -z "$min" ] || [ -z "$max" ]; then
    log "the wrapper printed no readable REGRADE line ('$line')"; return 1
  fi
  if [ "$changes" = "0" ]; then
    log "re-grade: ALREADY DONE (the dry run plans 0 changes)"; return 0
  fi
  if [ "$changes" -lt "$min" ] || [ "$changes" -gt "$max" ]; then
    log "re-grade: $changes changes planned, outside the reviewed $min-$max — not applied"; return 1
  fi
  log "re-grade: applying $changes change(s), pinned (--expect-changes $changes)"
  $DC exec -T api sh -c "python /app/scripts/regrade_predictions.py --match-ids-file $ids --prediction-type $types --apply --expect-changes $changes --out-csv /app/backups/regrade_regulation_apply_\$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/runbook_regrade.log 2>&1; rc=\$?; grep -E 'Scanned|Rewrote|Pinned|ERROR|Done:|Traceback|Error' /tmp/runbook_regrade.log | tail -8; exit \$rc"
  rc=$?
  [ "$rc" -eq 0 ] || { log "re-grade apply exited $rc (a pin mismatch writes nothing)"; return "$rc"; }
  $DC exec -T api sh -c "python /app/scripts/regrade_predictions.py --match-ids-file $ids --prediction-type $types --out-csv /app/backups/regrade_regulation_verify_\$(date -u +%Y%m%dT%H%M%SZ).csv > /tmp/runbook_regrade_verify.log 2>&1; rc=\$?; grep -o '[0-9]* change(s) planned' /tmp/runbook_regrade_verify.log | tail -1; exit \$rc"
  rc=$?
  left=$($DC exec -T api sh -c "grep -o '[0-9]* change(s) planned' /tmp/runbook_regrade_verify.log | tail -1 | cut -d' ' -f1")
  if [ "$rc" -ne 0 ] || [ "$left" != "0" ]; then log "re-grade verify: '${left:-unreadable}' change(s) still planned (want 0)"; return 1; fi
  log "re-grade applied: $changes prediction row(s); a fresh dry run plans 0"
}

# Steps 10 and 16: the same recompute, skipped when a feature step's completion is the state file's last corpus record.
recompute_features() {
  if features_current; then
    log "ALREADY DONE: features were recomputed after the last data-changing step (state file $STATE_FILE)"
    return 0
  fi
  $DC exec -T api sh -c "python /app/scripts/compute_features.py --backfill-finished soccer > /tmp/runbook_features.log 2>&1; rc=\$?; tail -5 /tmp/runbook_features.log; exit \$rc"
}

# Step 12: the never_resulted backfill, then the orphan cleanup again — a row it healed can leave a fork of the same
# fixture with a finished twin now, which the cleanup merges (ALREADY DONE when there is none; --after-never-resulted:
# that re-run's own reviewed ceilings, the one fork and nothing else). Exit 2 from the
# cleanup (its feature/grading follow-ups failed) fails the step; --from 12 then finds the backfill ALREADY DONE and
# the cleanup runs only its pending follow-ups.
step12() {
  local out1="$RUN_DIR/step12_backfill.log" out2="$RUN_DIR/step12_cleanup.log" rc
  bash "$OPS/backfill_soccer_never_resulted.sh" 2>&1 | tee "$out1"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || return "$rc"
  log "step 12: the forks of the rows healed above — scripts/ops/cleanup_soccer_orphans.sh --after-never-resulted"
  bash "$OPS/cleanup_soccer_orphans.sh" --after-never-resulted 2>&1 | tee "$out2"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || return "$rc"
  if grep -q "ALREADY DONE" "$out1" && grep -q "ALREADY DONE" "$out2"; then
    echo already_done > "$RUN_DIR/step12.result"
  else
    echo changed > "$RUN_DIR/step12.result"
  fi
}

step11() {
  local st plan merged_at runs starts s found="" i rc d
  st=$($DC exec -T api sh -c "cat /app/backups/merge_soccer_teams_last_apply.env 2>/dev/null")
  plan=$(echo "$st" | sed -n 's/^PLAN=//p'); merged_at=$(echo "$st" | sed -n 's/^MERGED_AT=//p')
  if [ -z "$plan" ] || [ -z "$merged_at" ]; then
    log "no merge apply recorded in /app/backups/merge_soccer_teams_last_apply.env (got '$st') — step 3 has not applied"; return 1
  fi
  log "merge plan $plan, merged at $merged_at; waiting for a fetch_live_odds run that started after it to finish"
  for d in auspex_pipeline fetch_live_odds; do
    if [ "$(af_dag_paused "$d")" = "True" ]; then log "$d is paused — unpause it: $(af_unpause_cmd "$d") — then re-run with --from 11"; return 1; fi
  done
  for i in $(seq 1 "$ODDS_WAIT_POLLS"); do
    runs=$($DC exec -T airflow-scheduler sh -c "airflow dags list-runs fetch_live_odds --state success -o json 2>/dev/null | grep '^\\[' | tail -1")
    starts=$(echo "$runs" | grep -o '"start_date": "[^"]*"' | cut -d'"' -f4 | cut -c1-19)
    for s in $starts; do
      if [[ "$s" > "${merged_at:0:19}" ]]; then found="$s"; break; fi
    done
    [ -n "$found" ] && break
    sleep "$AF_POLL_SECONDS"
  done
  if [ -z "$found" ]; then log "no fetch_live_odds run has finished since $merged_at after $ODDS_WAIT_POLLS polls — re-run with --from 11 later"; return 1; fi
  log "fetch_live_odds run started $found finished; checking odds and resolution since $merged_at"
  $DC exec -T api sh -c "python /app/scripts/merge_soccer_teams.py --check-after $plan --odds-since $merged_at --after-later-steps > /tmp/runbook_odds_after.log 2>&1; rc=\$?; grep -E 'AFTER|REFUSED|ERROR|stopped|repoints_missing|Traceback' /tmp/runbook_odds_after.log | tail -15; exit \$rc"
  rc=$?
  [ "$rc" -eq 0 ] && log "CHECKED (read-only): no merged team re-minted, every merged spelling resolves, the odds feed still reaches the merged clubs' fixtures"
  return "$rc"
}

run_step 1 "repoint_team_matches --pass 1" bash "$OPS/repoint_team_matches.sh" --pass 1
run_step 2 "migrate_027_outcome_trigger" bash "$OPS/migrate_027_outcome_trigger.sh"
run_step 3 "merge_soccer_teams" bash "$OPS/merge_soccer_teams.sh"
run_step 4 "repair_football_data_kickoffs" bash "$OPS/repair_football_data_kickoffs.sh"
run_step 5 "repoint_team_matches --pass 2" bash "$OPS/repoint_team_matches.sh" --pass 2
run_step 6 "repair_stranded_recs_soccer" bash "$OPS/repair_stranded_recs_soccer.sh"
run_step 7 "cleanup_soccer_orphans" bash "$OPS/cleanup_soccer_orphans.sh"
run_step 8 "backfill_soccer_regulation_scores + re-grade" step8
run_step 9 "backfill_season_labels" bash "$OPS/backfill_season_labels.sh"
run_step 10 "compute_features --backfill-finished soccer" recompute_features
run_step 11 "merge odds after-check" step11
run_step 12 "never_resulted results backfill + grading, then the orphan cleanup again" step12
run_step 13 "repair_soccer_labels (the Scottish fixtures filed under La Liga, the mis-decoded Preussen Muenster)" bash "$OPS/repair_soccer_labels.sh"
run_step 14 "promote_raw, checked (the match_stats step 4 archived)" bash "$OPS/promote_raw_after_repair.sh"
run_step 15 "grade the rows step 4 folded" bash "$OPS/grade_folded_soccer_rows.sh"
run_step 16 "compute_features --backfill-finished soccer (after steps 12-14)" recompute_features
scope="from step $FROM"
[ -z "$ONLY" ] || scope="step $ONLY only"
if corpus_consistent; then
  log "ALL STEPS DONE ($scope)"
else
  log "ALL STEPS DONE ($scope) — but no feature recompute (step 10 or 16) since the last corpus change: the repair is not complete"
fi
