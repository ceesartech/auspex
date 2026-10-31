#!/bin/bash
# Shared helpers for scripts/ops wrappers (sourced, never run on its own; not a
# step of any run order): Airflow 3.3 pausing, a run lock, and the check for
# an apply still running in the api container. The caller defines DC (the
# compose command) and log().
#
# Airflow 3.3 never advances a PAUSED DAG's runs: the scheduler examines only
# runs of unpaused DAGs (dagrun.py get_running_dag_runs_to_examine and the
# queued-run query filter DagModel.is_paused == false, and task instances are
# queued only where ~is_paused). A run in flight when its DAG is paused stays
# 'running' for good. So a wrapper that wants "no run of these DAGs in flight"
# DRAINS FIRST (waits for 0 runs while the DAGs still run), then pauses, then
# re-checks 0; a run that slipped in between makes it unpause and try again.
#
#   af_drain_and_pause <attempts> <polls> <dag>...   0 when drained and paused
#   af_resume                                        unpause what we paused
#   af_resume_on_exit                                the same, AF_EXIT_NOTE, then af_unlock: an EXIT trap
#   af_keep_paused <why> [<when to unpause>]         leave ours paused, loudly
#   af_dag_paused <dag>                              True / False / '' (unreadable)
#   af_runs_in_flight <dag>...                       queued+running runs, or exit 1
#   af_lock <name>, af_unlock                        one run at a time per name
#   af_applies_running <tool>...                     an /app/scripts/<tool>.py --apply still running?
# AF_PAUSED_BY_US lists the DAGs this run paused or adopted. A DAG that was
# already paused is left as it is and never unpaused, unless a marker says a
# wrapper paused it and never unpaused it (below). AF_POLL_SECONDS (default
# 30) is the wait between polls. AF_EXIT_NOTE, when set, is written to the
# ledger and the log by af_resume_on_exit (what an exit at that point leaves
# behind, and what to run next).
#
# Pausing keeps its books BEFORE it acts: the DAG joins AF_PAUSED_BY_US and
# gets its marker and a 'pausing' ledger line, then `dags pause` runs, then a
# second read must say True. A read that fails, or an interrupt landing
# anywhere in between (bash runs the trap as soon as the child returns),
# leaves the DAG listed, so the exit path unpauses it — or ALERTs, marker
# kept, when it still reads anything but False.
#
# Surviving a dead terminal (Ctrl-C or an SSH drop while the output goes
# through `| tee`): the terminal signals the whole process group, tee dies,
# and every later write to the pipe would kill the shell by SIGPIPE — in the
# middle of its unpausing. So SIGPIPE is ignored, af_resume unpauses every
# DAG before it verifies or logs anything, the exit path ignores a second
# INT/TERM/HUP while it unpauses, and its lines are written from a subshell
# (af_say): a builtin echo that fails on a dead pipe leaves its text in the
# shell's stdout buffer (bash before 4.3 never purges it), and that text then
# leaks into the next redirection or $(...) — a DAG's 'False' would read as
# something else and the ledger would fill with stale lines. Every pause,
# unpause and ALERT is also appended to $AF_STATE_DIR/airflow_pauses.log
# (default ~/.auspex_ops), and each DAG we pause gets a marker
# $AF_STATE_DIR/paused_<dag> naming this run (script and pid), removed only
# once it reads unpaused again: a DAG still paused with its marker (the
# wrapper was killed outright, or the unpause failed) is adopted by the next
# wrapper or runbook run that drains it, and unpaused when that run ends —
# but only when the run the marker names is gone. A marker whose run is still
# going (`ps -p <pid> -o args=` names its script) is left to that run, loudly,
# and this run does not start. Every ALERT's unpause command also removes the
# marker, so a marker does not outlive a hand unpause and later adopt (and
# undo) the operator's own pause.
#
# af_lock <name>: the directory $AF_STATE_DIR/<name>.lock holds "<pid>
# <script>" (mkdir is atomic; flock is not on every host). A live holder
# refuses this run; a holder that is gone (killed outright) is taken over.
# Each script takes its own name, so the runbook (which holds its lock) runs
# the merge wrapper (which takes the merge lock) as a child without a wait.
#
# af_applies_running: `docker compose exec -T` does not forward an interrupt,
# so an apply cut short on the host keeps running in the api container. The
# api image has no ps or pgrep: the scan reads /proc/<pid>/cmdline (skipping
# its own shell; the '[.]' keeps its own text from matching).

trap '' PIPE
AF_PAUSED_BY_US=""
AF_LOCK=""
AF_EXIT_NOTE=""
AF_POLL_SECONDS="${AF_POLL_SECONDS:-30}"
AF_STATE_DIR="${AF_STATE_DIR:-${HOME:-/tmp}/.auspex_ops}"
AF_LEDGER="$AF_STATE_DIR/airflow_pauses.log"

# log() from a subshell: a write that fails on a dead terminal never dirties this shell's stdout buffer.
af_say() { ( log "$*" ) 2>/dev/null; }

# A line for the ledger (written first: it survives a dead terminal) and the log.
af_note() {
  { mkdir -p "$AF_STATE_DIR" && echo "[$(date -u +%FT%TZ)] $(basename "$0"): $*" >> "$AF_LEDGER"; } 2>/dev/null
  af_say "$*"
}

# The command an ALERT prints: unpause, and drop the marker with it.
af_unpause_cmd() {
  echo "docker compose exec -T airflow-scheduler airflow dags unpause -y $1 && rm -f $AF_STATE_DIR/paused_$1"
}

# af_live_other <pid> <script>: 0 when <pid> is a live process, not this shell, whose command line names <script>.
# When ps cannot read even this shell, the holder counts as live: never take over what may still be running.
af_live_other() {
  case "${1:-}" in '' | *[!0-9]*) return 1 ;; esac
  [ -n "${2:-}" ] && [ "$1" != "$$" ] || return 1
  [ -n "$(ps -p $$ -o args= 2>/dev/null)" ] || return 0
  case "$(ps -p "$1" -o args= 2>/dev/null)" in *"$2"*) return 0 ;; esac
  return 1
}

# "<pid> <script>" of the run a marker names, or nothing.
af_marker_holder() {
  sed -n 's/^paused by \([^ ][^ ]*\) pid \([0-9][0-9]*\) .*/\2 \1/p' "$1" 2>/dev/null | head -1
}

# Airflow 3.3 CLI: dag_id positional, -o json; structlog lines are filtered
# out by keeping the last line that starts with '['.
af_dag_paused() {
  $DC exec -T airflow-scheduler sh -c "airflow dags details $1 -o json 2>/dev/null | grep '^\\[' | tail -1" </dev/null \
    | grep -o '"is_paused": "[A-Za-z]*"' | head -1 | sed 's/.*: "\(.*\)"/\1/'
}

af_runs_in_flight() {
  local n=0 c d s
  for d in "$@"; do
    for s in queued running; do
      c=$($DC exec -T airflow-scheduler sh -c "airflow dags list-runs $d --state $s -o json 2>/dev/null | grep '^\\[' | tail -1" </dev/null)
      [ -n "$c" ] || return 1
      n=$((n + $(echo "$c" | grep -o '"run_id"' | wc -l)))
    done
  done
  echo "$n"
}

af_resume() {
  local d
  # Unpause all of them first: nothing below may stand between a DAG and its unpause.
  for d in $AF_PAUSED_BY_US; do
    $DC exec -T airflow-scheduler airflow dags unpause -y "$d" </dev/null >/dev/null 2>&1
  done
  for d in $AF_PAUSED_BY_US; do
    if [ "$(af_dag_paused "$d")" = "False" ]; then
      rm -f "$AF_STATE_DIR/paused_$d" 2>/dev/null
      af_note "unpaused $d"
    else
      af_note "ALERT: $d is STILL PAUSED (or unreadable) — unpause it now: $(af_unpause_cmd "$d")"
    fi
  done
  AF_PAUSED_BY_US=""
}

# For an EXIT trap: a second Ctrl-C (or the HUP of a dropped session) must not cut the unpausing short. The run lock
# goes last, so no other run starts while this one still unpauses.
af_resume_on_exit() {
  trap '' INT TERM HUP
  af_resume
  [ -z "$AF_EXIT_NOTE" ] || af_note "$AF_EXIT_NOTE"
  af_unlock
}

# Leave the DAGs this run paused PAUSED (their markers stay, so the next run adopts them), loudly:
# af_keep_paused <why> [<when to unpause, ending where the command follows>].
af_keep_paused() {
  local d
  for d in $AF_PAUSED_BY_US; do
    af_note "ALERT: $d stays PAUSED — $1. ${2:-Unpause it by hand}: $(af_unpause_cmd "$d")"
  done
  AF_PAUSED_BY_US=""
}

af_drain_and_pause() {
  local attempts="$1" polls="$2" attempt i n st d already marker holder
  shift 2
  for attempt in $(seq 1 "$attempts"); do
    n=""
    for i in $(seq 1 "$polls"); do
      n=$(af_runs_in_flight "$@") || n=""
      [ "$n" = "0" ] && break
      sleep "$AF_POLL_SECONDS"
    done
    if [ "$n" != "0" ]; then
      already=""
      for d in "$@"; do [ "$(af_dag_paused "$d")" = "True" ] && already="$already $d"; done
      log "runs of $* still in flight after $polls polls (${n:-unreadable})${already:+; already paused before this run (its runs never advance — unpause it or clear the run):$already}"
      return 1
    fi
    for d in "$@"; do
      st=$(af_dag_paused "$d")
      marker="$AF_STATE_DIR/paused_$d"
      if [ "$st" != "True" ] && [ "$st" != "False" ]; then
        log "cannot read whether $d is paused"
        return 1
      fi
      case " $AF_PAUSED_BY_US " in *" $d "*) continue ;; esac
      holder=""
      [ -f "$marker" ] && holder=$(af_marker_holder "$marker")
      if [ -n "$holder" ] && af_live_other "${holder%% *}" "${holder#* }"; then
        af_note "$d is held by a run that is still going (${holder#* } pid ${holder%% *}; its marker $marker) — left to that run, which unpauses it when it ends: this run does not start while another run holds $d. Wait for that run to end, then re-run"
        return 1
      fi
      if [ "$st" = "True" ]; then
        if [ -f "$marker" ] && [ "${holder%% *}" = "$$" ]; then
          AF_PAUSED_BY_US="$AF_PAUSED_BY_US $d"
          af_note "$d is still paused by this run (an unpause above did not take) — back on its list: it is unpaused when this run ends"
        elif [ -f "$marker" ]; then
          AF_PAUSED_BY_US="$AF_PAUSED_BY_US $d"
          af_note "$d is paused by an earlier wrapper run that never unpaused it ($(cat "$marker" 2>/dev/null); that run is gone) — adopted: this run unpauses it when it ends"
        else
          log "$d is already paused — left as it is"
        fi
        continue
      fi
      # Books first (a stale marker is overwritten): from here on, an interrupt or a failed read leaves $d listed.
      AF_PAUSED_BY_US="$AF_PAUSED_BY_US $d"
      { mkdir -p "$AF_STATE_DIR" && echo "paused by $(basename "$0") pid $$ at $(date -u +%FT%TZ)" > "$marker"; } 2>/dev/null \
        || af_note "WARNING: could not write $marker (a run killed outright would leave $d paused unannounced)"
      af_note "pausing $d"
      $DC exec -T airflow-scheduler airflow dags pause -y "$d" </dev/null >/dev/null 2>&1
      st=$(af_dag_paused "$d")
      if [ "$st" != "True" ]; then
        af_note "could not confirm that $d is paused (it reads '${st:-nothing}') — it stays on this run's list: the exit unpauses it, or ALERTs"
        return 1
      fi
      af_note "paused $d"
    done
    n=$(af_runs_in_flight "$@") || n=""
    if [ "$n" = "0" ]; then
      return 0
    fi
    log "attempt $attempt/$attempts: ${n:-an unreadable number of} run(s) of $* slipped in between the drain and the pause — unpausing and trying again"
    af_resume
  done
  log "could not pause $* with no run in flight in $attempts attempts"
  return 1
}

# A lock directory: its holder file, then the (then empty) directory — nothing else is ever removed.
af_drop_lockdir() {
  unlink "$1/holder" 2>/dev/null
  rmdir "$1" 2>/dev/null
}

# af_lock <name>: 0 when this run holds $AF_STATE_DIR/<name>.lock (af_unlock releases it); 1, loudly, when another
# run that is still going holds it, or it cannot be taken.
af_lock() {
  local dir="$AF_STATE_DIR/$1.lock" held try
  mkdir -p "$AF_STATE_DIR" 2>/dev/null
  for try in 1 2; do
    if mkdir "$dir" 2>/dev/null; then
      if echo "$$ $(basename "$0")" > "$dir/holder" 2>/dev/null; then
        AF_LOCK="$dir"
        return 0
      fi
      af_drop_lockdir "$dir"
      log "REFUSED: could not write $dir/holder — is $AF_STATE_DIR writable? Nothing changed"
      return 1
    fi
    held=$(cat "$dir/holder" 2>/dev/null)
    if af_live_other "${held%% *}" "${held#* }"; then
      log "REFUSED: another run holds $dir: pid ${held%% *} ($(ps -p "${held%% *}" -o args= 2>/dev/null)) — wait for it to end; nothing changed"
      return 1
    fi
    [ "$try" = 1 ] || break
    # Its holder is gone (killed outright): move it aside, which only one of two runs taking it over at once can do,
    # and put it back if it turns out to be a fresh lock that a live run took meanwhile.
    if mv "$dir" "$dir.stale.$$" 2>/dev/null; then
      if [ "$(cat "$dir.stale.$$/holder" 2>/dev/null)" = "$held" ]; then
        af_drop_lockdir "$dir.stale.$$"
        af_note "lock $dir: its holder (${held:-none recorded}) is gone — taken over"
      else
        mv "$dir.stale.$$" "$dir" 2>/dev/null
      fi
    fi
  done
  log "REFUSED: could not take $dir (another run took it at the same moment?) — re-run; nothing changed"
  return 1
}

af_unlock() {
  if [ -n "$AF_LOCK" ] && [ "$(cat "$AF_LOCK/holder" 2>/dev/null)" = "$$ $(basename "$0")" ]; then
    af_drop_lockdir "$AF_LOCK"
  fi
  AF_LOCK=""
}

# af_applies_running <tool>...: prints "pid <n>: <command line>" for every /app/scripts/<tool>.py process with
# --apply running in the api container and returns 0; 1 when there is none; 2 when the scan did not run.
af_applies_running() {
  local pat="" t out
  for t in "$@"; do pat="$pat${pat:+|}*/app/scripts/$t[.]py*--apply*"; done
  out=$($DC exec -T api sh -c "for f in /proc/[0-9]*/cmdline; do p=\${f#/proc/}; p=\${p%/cmdline}; [ \"\$p\" = \"\$\$\" ] && continue; c=\$(tr '\\000' ' ' < \"\$f\" 2>/dev/null); case \"\$c\" in $pat) echo \"APPLY_RUNNING pid \$p: \$c\" ;; esac; done; echo PROC_SCAN_OK" </dev/null 2>/dev/null)
  case "$out" in *PROC_SCAN_OK*) ;; *) return 2 ;; esac
  echo "$out" | sed -n 's/^APPLY_RUNNING //p' | grep . || return 1
}
