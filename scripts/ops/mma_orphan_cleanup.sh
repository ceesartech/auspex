#!/bin/bash
# NOT YET APPROVED — run only on explicit operator approval (CLAUDE.md: data
# repairs need a verified backup first, which step 1 takes).
# Delete the 8 bout-less MMA fighter rows the user approved on 2026-11-02
# ("Yes" to adding the 8 empty MMA fighter records, including 'TBA' and
# 'Opponent TBA', to the clean-up), with scripts/mma_orphan_cleanup.py, in ONE
# transaction: Rodrigo Vera, Max Gimenis, Ramiro Jimenez, Uran Satybaldiev,
# Henrique da Silva Lopes, Ethyn Ewing, 'TBA' and 'Opponent TBA'. Run B of
# scripts/ops/repair_stranded_recs_mma.sh deleted their 10 bouts on
# 2026-11-02 (ESPN had dropped or re-drawn them); the rows carry no ESPN
# athlete id, no alias, no other metadata, and nothing references them.
# Reviewed list with the evidence: scripts/data/mma_orphan_fighters_20261103.csv.
#
# Reviewed plan — a local replica of the 2026-11-02T23:16:10Z read-only export
# of every UFC team, match, rec, prediction, odds row, snapshot and feature row
# (taken after the identity clean-up of 22:41Z and the held-rows settle of
# 22:44Z), with this commit's code:
#   PLAN listed=8 delete=8 done=0 refuse=0
#   WRITTEN orphans_deleted=8
#   A re-run plans the done plan (DONE_P below) and writes nothing.
# Steps:
#   0. refuse while another scripts/ops wrapper or the soccer runbook runs on
#      this host (a shell running one; this run's own parents, such as sudo,
#      tmux or the `bash -c` of an ssh one-liner, and its subshells never
#      count; an unreadable process list refuses), while an apply of a
#      scripts/ tool still runs in the api container, or while a UFC card is
#      on (a UFC bout kicked off within the last $CARD_BEFORE_H h or starts
#      within $CARD_AFTER_H h, or one is 'live'); refuse unless the list holds
#      exactly $EXPECTED_ORPHANS rows and the identity clean-up is done and
#      verified (scripts/ops/mma_identity_cleanup.sh --verify, read only: the
#      list was reviewed against the state it leaves).
#      Already done? The tool's read-only dry run must plan exactly the
#      reviewed plan (not done) or exactly the done plan; a done plan is checked
#      with --check-after on the plan the last apply recorded ($STATE_FILE)
#      before ALREADY DONE (exit 0, no backup). Anything else — a row that
#      gained a bout or a reference, an ESPN id, aliases or other metadata, a
#      renamed row, a row deleted by hand — aborts before the backup;
#   1. refuse unless the backups filesystem has at least $MIN_FREE_KB KiB free
#      (a dump is ~0.4 GB; the root filesystem also holds Postgres and its WAL),
#      then a verified backup (B2) — abort if it does not land;
#   2. dry run (write-once CSV + plan file) — abort unless it exits 0 with
#      the reviewed plan;
#   3. record the plan (STATUS=applying) in $STATE_FILE, then
#      apply --expect-plan <the dry run's plan file>: it locks the teams table
#      IN EXCLUSIVE MODE (lock_timeout 10s: a sweep waits seconds; no bout can
#      be inserted onto any team and no team inserted, stamped or renamed until
#      it commits), re-plans, refuses (exit 2, nothing written) unless that is
#      the dry run's plan digest, writes the gzip JSONL archive (the reversal
#      record) before its first write, deletes each row guarded on the row as
#      the plan saw it and on no reference under any foreign key to teams, and
#      commits only once every listed row is gone, teams fell by exactly 8 and
#      no table with a foreign key to teams lost a row, inside it;
#   4. after-state: --check-after on the dry run's plan file and a fresh dry
#      run that plans exactly the done plan; STATUS=done.
#   No follow-ups: no bout, prediction, feature or rec changes.
#   --verify (read only; for a later wrapper): the done plan, a recorded apply
#      whose after-state passed (STATUS=done), and --check-after on the
#      recorded plan — exit 0 when done and verified, else 1.
# Exit: 0 done (or already done, or verified); 1 aborted or an after-state failed.
# The other MMA wrappers' done states hold afterwards (rehearsed): the rows are
# in none of their lists (the tool refuses a list that names a row of the
# identity clean-up's lists or the fighter merge map), so
# mma_identity_cleanup.sh still reads ALREADY DONE / VERIFIED,
# settle_mma_held_rows.sh --verify, merge_mma_fighters.sh --verify and
# repair_stranded_recs_mma.sh --verify-b still pass.
# Reversal: the apply's archive /app/backups/mma_orphan_cleanup_apply_<stamp>.jsonl.gz
# holds the full pre-image of every deleted row: re-insert them with their own
# ids. Restoring any of run B's deleted bouts of these fighters (its archive
# fixture_orphans_mma_20261102T043520Z.jsonl.gz) needs these rows back first.
# Needs this commit on the VM (scripts/ is bind-mounted: a git pull is enough).
# WHEN: any time no UFC card is on (step 0; next cards 2026-11-07 22:00Z,
# 2026-11-14 22:00Z, 2026-11-21 15:00Z), after scripts/ops/mma_identity_cleanup.sh
# (done on prod 2026-11-02 22:41Z).
# Run on the VM host (an interactive shell is the usual way; sudo or an ssh
# one-liner work too):
#   bash /opt/auspex/scripts/ops/mma_orphan_cleanup.sh 2>&1 | tee -a ~/mma_orphan_cleanup.log
# (AUSPEX_DIR, TMPDIR, AF_STATE_DIR, MMA_ORPHAN_PS and MMA_ORPHAN_DF exist for tests and rehearsals.)
set -uo pipefail
MODE=apply
case "${1:-}" in
  "") ;;
  --verify) MODE=verify ;;
  *) echo "usage: $0 [--verify]   (see the header)"; exit 2 ;;
esac
OPS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || { echo "ABORT: cannot find scripts/ops — nothing changed"; exit 1; }
cd "${AUSPEX_DIR:-/opt/auspex}" || { echo "ABORT: cannot cd to ${AUSPEX_DIR:-/opt/auspex} — nothing changed"; exit 1; }
DC="docker compose"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
# shellcheck source=_airflow_lib.sh
. "$OPS/_airflow_lib.sh" || { log "ABORT: cannot load $OPS/_airflow_lib.sh — nothing changed"; exit 1; }
W=$(mktemp -d "${TMPDIR:-/tmp}/mma_orphan.XXXXXX") || { log "ABORT: mktemp failed — nothing changed"; exit 1; }
cleanup() {
  rm -rf "$W"
  $DC exec -T postgres rm -f /tmp/mma_orphan_card.sql >/dev/null 2>&1
  af_unlock
}
trap cleanup EXIT
latest_dump() {
  $DC exec -T api python -c "import sys; sys.path.insert(0,'/app/scripts'); import b2_io; s=b2_io.s3_client(); o=max(b2_io._list(s,'postgres/'), key=lambda o:o['LastModified']); print(int(o['LastModified'].timestamp()), o['Key'])" 2>/dev/null | tail -1
}
plan_num() {  # plan_num <PLAN line> <key>: the count, or "missing"
  local v
  v=$(echo "$1" | grep -o " $2=[0-9]*" | cut -d= -f2)
  if [[ "$v" =~ ^[0-9]+$ ]]; then echo "$v"; else echo missing; fi
}
same_plan() {  # same_plan <PLAN line> <reviewed 'k=v ...'>: every reviewed key has exactly its count
  local k
  for k in $2; do
    [ "$(plan_num "$1" "${k%%=*}")" = "${k#*=}" ] || return 1
  done
}

LIST=scripts/data/mma_orphan_fighters_20261103.csv
EXPECTED_ORPHANS=8
TOOL="python /app/scripts/mma_orphan_cleanup.py"
PLAN_R="listed=8 delete=8 done=0 refuse=0"
DONE_P="listed=8 delete=0 done=8 refuse=0"
CARD_BEFORE_H=8
CARD_AFTER_H=3
MIN_FREE_KB=2097152  # 2 GiB
STATE_FILE=/app/backups/mma_orphan_cleanup_last_apply.env
STAMP=$(date -u +%Y%m%dT%H%M%SZ)_$$
PLAN_FILE=/app/backups/mma_orphan_cleanup_dryrun_$STAMP.plan.json
PS_CMD="${MMA_ORPHAN_PS:-ps -eo pid=,ppid=,args=}"
DF_CMD="${MMA_ORPHAN_DF:-df -Pk ${AUSPEX_DIR:-/opt/auspex}/backups}"

# run_tool <log name> <command>: the command in the api container; prints its summary lines and exit
run_tool() {
  $DC exec -T api sh -c "$2 > /tmp/$1.log 2>&1; rc=\$?; grep -E 'PLAN |PLAN_FILE|PLAN_DIGEST|OTHER_BOUTLESS|AFTER|APPLIED|Archived|WRITTEN|ROLLED BACK|REFUSED|ERROR|Traceback|Error' /tmp/$1.log | tail -30; echo \"tool exit \$rc\"; exit \$rc" </dev/null
}
plan_of() { $DC exec -T api sh -c "grep -oE 'PLAN [a-z_]+=[0-9]+( [a-z_]+=[0-9]+)*' /tmp/$1.log | tail -1" </dev/null; }
read_state_file() { REC=$($DC exec -T api sh -c "cat $STATE_FILE 2>/dev/null" </dev/null); }
rec_val() { echo "$REC" | sed -n "s/^$1=//p" | tail -1; }
write_state_file() {  # write_state_file <plan> <status>
  printf 'PLAN=%s\nSTATUS=%s\n' "$1" "$2" | $DC exec -T api sh -c "cat > $STATE_FILE.tmp && mv $STATE_FILE.tmp $STATE_FILE"
}
check_recorded() { run_tool mma_orphan_after "$TOOL --check-after $1"; }

# others_running: prints every OTHER run of a scripts/ops wrapper or the soccer runbook on this host; 0 when there is
# one, 1 when there is none, 2 when the process list ($PS_CMD: "pid ppid args" lines) cannot be read.
# A run is a SHELL (sh, bash, dash, zsh, any path, options allowed) whose script word is a scripts/ops/*.sh path or
# names one of the wrappers in this directory (a run from inside scripts/ops) or the soccer runbook. Not a run: an
# editor or pager open on a wrapper, `tail -f` on a log, tee, tmux/screen/sudo themselves (their shell child is the
# run). This run's own process tree is never 'other': its ancestors (sudo, the `bash -c "... | tee"` of an ssh
# one-liner, a tmux server, a wrapper that called this one) and its descendants (its subshells, the
# mma_identity_cleanup.sh --verify it runs). A second run of this script is another run (af_lock holds it off too).
# (The same check as scripts/ops/mma_identity_cleanup.sh.)
others_running() {
  local names out
  names="soccer_repair_runbook.sh $(cd "$OPS" && ls -- *.sh 2>/dev/null | tr '\n' ' ')"
  out=$($PS_CMD 2>/dev/null) || return 2
  [ -n "$out" ] || return 2
  out=$(echo "$out" | awk -v self="$$" -v parent="$PPID" -v names="$names" '
    function script_of(a,   r) {
      if (!match(a, /^([^ ]*\/)?(ba|da|z)?sh( +-o +[A-Za-z]+| +-[A-Za-z]+)* +/)) return ""
      r = substr(a, RSTART + RLENGTH); sub(/ .*/, "", r); return r
    }
    BEGIN { n = split(names, nm, " "); for (i = 1; i <= n; i++) known[nm[i]] = 1 }
    /^[ \t]*[0-9]+[ \t]+[0-9]+/ {
      a = $0; sub(/^[ \t]*[0-9]+[ \t]+[0-9]+[ \t]*/, "", a)
      par[$1] = $2; cmd[$1] = a; order[++cnt] = $1
    }
    END {
      mine[self] = 1
      p = parent; hops = 0
      while (p != "" && p != "0" && !(p in mine) && hops++ < 500) { mine[p] = 1; if (!(p in par)) break; p = par[p] }
      for (i = 1; i <= cnt; i++) {
        q = order[i]; c = q; hops = 0
        while ((c in par) && hops++ < 500) { if (par[c] == self) { mine[q] = 1; break }; c = par[c] }
      }
      for (i = 1; i <= cnt; i++) {
        q = order[i]
        if (q in mine) continue
        s = script_of(cmd[q]); b = s; sub(/.*\//, "", b)
        if (s != "" && (s ~ /(^|\/)scripts\/ops\/[^\/]+\.sh$/ || (b in known))) print "  pid " q ": " cmd[q]
      }
    }')
  [ -n "$out" ] || return 1
  echo "$out"
  return 0
}
# ufc_card_on: prints the UFC bouts that make a card 'on' now; 0 when there is one, 1 when none, 2 unreadable.
ufc_card_on() {
  local out
  cat > "$W/card.sql" <<SQL
SELECT m.id, m.match_date, m.status FROM matches m JOIN leagues l ON l.id = m.league_id
 WHERE l.sport = 'mma' AND m.status NOT IN ('cancelled', 'postponed')
   AND (m.status = 'live' OR m.match_date BETWEEN now() - interval '$CARD_BEFORE_H hours' AND now() + interval '$CARD_AFTER_H hours')
 ORDER BY m.match_date;
SELECT 'CARD_CHECK_OK';
SQL
  $DC cp "$W/card.sql" postgres:/tmp/mma_orphan_card.sql >/dev/null 2>&1 || return 2
  out=$($DC exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -At -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /tmp/mma_orphan_card.sql' </dev/null) || return 2
  case "$out" in *CARD_CHECK_OK*) ;; *) return 2 ;; esac
  out=$(echo "$out" | grep -v '^CARD_CHECK_OK$' | grep .)
  [ -n "$out" ] || return 1
  echo "$out" | sed 's/^/  /'
  return 0
}
# free_kb: the KiB available on the backups filesystem ($DF_CMD, POSIX output), or empty when unreadable.
free_kb() {
  local v
  v=$($DF_CMD 2>/dev/null | awk 'NR == 2 { print $4 }')
  if [[ "$v" =~ ^[0-9]+$ ]]; then echo "$v"; fi
}

if [ "$MODE" = apply ]; then
  log "0/4 nothing else running? (scripts/ops wrappers, the soccer runbook, applies in the api container, a UFC card)"
  others_running
  case $? in
    0) log "ABORT at step 0: the process(es) above run a scripts/ops wrapper or the soccer runbook — wait for them to end; nothing changed (no backup)"; exit 1 ;;
    1) ;;
    *) log "ABORT at step 0: could not list this host's processes ($PS_CMD) — nothing changed (no backup)"; exit 1 ;;
  esac
  af_lock mma_orphan_cleanup || { log "ABORT at step 0: another run of this script holds its lock — nothing changed"; exit 1; }
  af_applies_running mma_orphan_cleanup mma_identity_cleanup settle_mma_held_rows merge_mma_fighters repair_fixture_orphans backfill_results '*'
  case $? in
    0) log "ABORT at step 0: the apply above still runs in the api container — wait for it to end; nothing changed (no backup)"; exit 1 ;;
    1) ;;
    *) log "ABORT at step 0: could not scan the api container for a running apply — nothing changed (no backup)"; exit 1 ;;
  esac
  ufc_card_on
  case $? in
    0) log "ABORT at step 0: a UFC card is on (the bouts above kicked off within ${CARD_BEFORE_H} h, start within ${CARD_AFTER_H} h, or are live) — run it after the card; nothing changed (no backup)"; exit 1 ;;
    1) ;;
    *) log "ABORT at step 0: could not read whether a UFC card is on — nothing changed (no backup)"; exit 1 ;;
  esac
fi

[ -r "$LIST" ] || { log "ABORT: $LIST is missing — pull this commit first; nothing changed"; exit 1; }
n_orphans=$(grep -v '^#' "$LIST" | tail -n +2 | grep -c .)
if [ "$n_orphans" != "$EXPECTED_ORPHANS" ]; then
  log "ABORT: $LIST holds $n_orphans row(s); this script expects $EXPECTED_ORPHANS — nothing changed"
  exit 1
fi

if [ "$MODE" = apply ]; then
  log "0/4 the identity clean-up done and verified? (mma_identity_cleanup.sh --verify, read only)"
  if ! bash "$OPS/mma_identity_cleanup.sh" --verify; then
    log "ABORT at step 0: the MMA identity clean-up is not done and verified — this list is reviewed against the state it leaves; nothing changed (no backup)"
    exit 1
  fi
fi

log "0/4 already done? (read-only dry run)"
run_tool mma_orphan_pre "$TOOL --out-csv /app/backups/mma_orphan_cleanup_precheck_$STAMP.csv --plan-out /app/backups/mma_orphan_cleanup_precheck_$STAMP.plan.json"
pre=$(plan_of mma_orphan_pre)
if same_plan "$pre" "$DONE_P"; then
  read_state_file
  rec_plan=$(rec_val PLAN); rec_status=$(rec_val STATUS)
  if [ -z "$rec_plan" ] || [ -z "$rec_status" ]; then
    log "FAILED: the rows read gone ('$pre') but no apply is recorded in $STATE_FILE (got '${REC:-nothing}') — deleted outside this script? Inspect /app/backups/mma_orphan_cleanup_*; nothing changed"
    exit 1
  fi
  if [ "$MODE" = verify ] && [ "$rec_status" != done ]; then
    log "NOT VERIFIED: the rows read gone ('$pre') but the recorded apply is STATUS=$rec_status (its after-state never passed) — re-run scripts/ops/mma_orphan_cleanup.sh to check it"
    exit 1
  fi
  if ! check_recorded "$rec_plan"; then
    log "FAILED: the rows read gone but the after-state of the recorded plan ($rec_plan, STATUS=$rec_status) fails (above) — inspect; nothing changed"
    exit 1
  fi
  if [ "$MODE" = verify ]; then
    log "ORPHAN CLEANUP VERIFIED: '$pre'; the after-state of $rec_plan holds (STATUS=$rec_status)"
    exit 0
  fi
  if [ "$rec_status" = done ]; then
    log "ALREADY DONE: '$pre' and the after-state of $rec_plan holds — nothing to write (no backup)"
    exit 0
  fi
  write_state_file "$rec_plan" done || { log "FAILED: could not record STATUS=done in $STATE_FILE"; exit 1; }
  log "COMPLETED: the delete was committed and its after-state holds; STATUS=done recorded (it was $rec_status)"
  exit 0
fi
if [ "$MODE" = verify ]; then
  log "NOT VERIFIED: the orphan clean-up is not done (pre-plan '${pre:-unreadable}', done plan $DONE_P) — run scripts/ops/mma_orphan_cleanup.sh first"
  exit 1
fi
if ! same_plan "$pre" "$PLAN_R"; then
  log "ABORT at step 0: the pre-check plan ('${pre:-unreadable}') is neither the reviewed plan ($PLAN_R) nor the done plan ($DONE_P) — a listed row gained a bout or a reference, carries an ESPN id, aliases or other metadata, was renamed or deleted outside this script (see the refusals above); nothing changed (no backup)"
  exit 1
fi
log "not done yet (pre-plan '$pre') — continuing"

log "1/4 disk, then backup"
free=$(free_kb)
if [ -z "$free" ] || [ "$free" -lt "$MIN_FREE_KB" ]; then
  log "ABORT at step 1: the backups filesystem has ${free:-an unreadable amount of} KiB free ($DF_CMD); a backup needs at least $MIN_FREE_KB KiB (2 GiB) there, so a dump never fills the disk Postgres writes to — free space first (OPERATIONS.md, disk); nothing changed (no backup)"
  exit 1
fi
log "backups filesystem: $free KiB free (at least $MIN_FREE_KB)"
before=$(date -u +%s)
$DC exec -T airflow-scheduler airflow dags trigger db_backup_daily >/dev/null 2>&1
landed=""
for _ in $(seq 1 40); do
  sleep 30
  l=$(latest_dump); ts=${l%% *}
  if [ -n "$ts" ] && [ "$ts" -gt "$before" ] 2>/dev/null; then landed="$l"; break; fi
done
if [ -z "$landed" ]; then log "ABORT: backup did not land in B2 within 20 min — nothing changed"; exit 1; fi
log "backup landed: $landed"

log "2/4 dry run (plan file $PLAN_FILE)"
run_tool mma_orphan_dry "$TOOL --out-csv /app/backups/mma_orphan_cleanup_dryrun_$STAMP.csv --plan-out $PLAN_FILE"
rc=$?
plan=$(plan_of mma_orphan_dry)
if [ "$rc" -ne 0 ] || ! same_plan "$plan" "$PLAN_R"; then
  log "ABORT at step 2: the dry run failed (exit $rc: a refusal, see above) or its plan ('$plan') is not the reviewed one ($PLAN_R) — review the dry-run CSV; nothing changed"
  exit 1
fi
log "plan: $plan"

write_state_file "$PLAN_FILE" applying \
  || { log "ABORT at step 3: could not record the apply (STATUS=applying) in $STATE_FILE before it — nothing changed"; exit 1; }
log "3/4 apply (--expect-plan $PLAN_FILE; recorded STATUS=applying)"
run_tool mma_orphan_apply "$TOOL --apply --expect-plan $PLAN_FILE --out-csv /app/backups/mma_orphan_cleanup_apply_$STAMP.csv --archive /app/backups/mma_orphan_cleanup_apply_$STAMP.jsonl.gz"
rc=$?
if [ "$rc" -ne 0 ]; then
  why="it rolled back"
  [ "$rc" = "2" ] && why="it refused before writing (its plan differs from the dry run's, the teams lock timed out, or a refusal above)"
  log "ABORT at step 3 (exit $rc; full log /tmp/mma_orphan_apply.log in the api container): $why — one transaction, nothing committed; a re-run starts from step 0"
  exit 1
fi

log "4/4 after-state (from the dry run's plan file)"
check_recorded "$PLAN_FILE"
after_rc=$?
run_tool mma_orphan_verify "$TOOL --out-csv /app/backups/mma_orphan_cleanup_verify_$STAMP.csv --plan-out /app/backups/mma_orphan_cleanup_verify_$STAMP.plan.json"
replan=$(plan_of mma_orphan_verify)
log "after: --check-after exit $after_rc (expected 0), re-plan '$replan' (expected $DONE_P)"
if [ "$after_rc" -ne 0 ] || ! same_plan "$replan" "$DONE_P"; then
  log "FAILED after-state at step 4 — inspect (the apply's archive /app/backups/mma_orphan_cleanup_apply_$STAMP.jsonl.gz and its CSV are the reversal record); STATUS stays 'applying', so a re-run checks the recorded plan again before anything else"
  exit 1
fi
write_state_file "$PLAN_FILE" done || { log "FAILED: could not record STATUS=done in $STATE_FILE (the delete is committed and verified; a re-run records it)"; exit 1; }
log "DONE: the 8 bout-less MMA fighter rows are deleted (archive /app/backups/mma_orphan_cleanup_apply_$STAMP.jsonl.gz)"
