#!/bin/bash
# Shared Airflow 3.3 helpers for scripts/ops wrappers (sourced, never run on
# its own; not a step of any run order). The caller defines DC (the compose
# command) and log().
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
#   af_resume_on_exit                                the same, for an EXIT trap
#   af_keep_paused <why>                             leave ours paused, loudly
#   af_dag_paused <dag>                              True / False / '' (unreadable)
#   af_runs_in_flight <dag>...                       queued+running runs, or exit 1
# AF_PAUSED_BY_US lists the DAGs this run paused or adopted. A DAG that was
# already paused is left as it is and never unpaused, unless a marker says a
# wrapper paused it and never unpaused it (below). AF_POLL_SECONDS (default
# 30) is the wait between polls.
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
# $AF_STATE_DIR/paused_<dag>, removed only once it reads unpaused again: a
# DAG still paused with its marker (the wrapper was killed outright, or the
# unpause failed) is adopted by the next wrapper or runbook run that drains
# it, and unpaused when that run ends.

trap '' PIPE
AF_PAUSED_BY_US=""
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
      af_note "ALERT: $d is STILL PAUSED (or unreadable) — unpause it now: docker compose exec -T airflow-scheduler airflow dags unpause -y $d"
    fi
  done
  AF_PAUSED_BY_US=""
}

# For an EXIT trap: a second Ctrl-C (or the HUP of a dropped session) must not cut the unpausing short.
af_resume_on_exit() {
  trap '' INT TERM HUP
  af_resume
}

# Leave the DAGs this run paused PAUSED (their markers stay, so the next run adopts them), loudly.
af_keep_paused() {
  local d
  for d in $AF_PAUSED_BY_US; do
    af_note "ALERT: $d stays PAUSED — $1. Unpause by hand: docker compose exec -T airflow-scheduler airflow dags unpause -y $d"
  done
  AF_PAUSED_BY_US=""
}

af_drain_and_pause() {
  local attempts="$1" polls="$2" attempt i n st d already marker
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
      if [ "$st" = "True" ]; then
        if [ -f "$marker" ]; then
          AF_PAUSED_BY_US="$AF_PAUSED_BY_US $d"
          af_note "$d is paused by an earlier wrapper run that never unpaused it ($(cat "$marker" 2>/dev/null)) — adopted: this run unpauses it when it ends"
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
