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
#   af_dag_paused <dag>                              True / False / '' (unreadable)
#   af_runs_in_flight <dag>...                       queued+running runs, or exit 1
# AF_PAUSED_BY_US lists the DAGs af_drain_and_pause paused (a DAG that was
# already paused is left as it is and never unpaused). AF_POLL_SECONDS
# (default 30) is the wait between polls.

AF_PAUSED_BY_US=""
AF_POLL_SECONDS="${AF_POLL_SECONDS:-30}"

# Airflow 3.3 CLI: dag_id positional, -o json; structlog lines are filtered
# out by keeping the last line that starts with '['.
af_dag_paused() {
  $DC exec -T airflow-scheduler sh -c "airflow dags details $1 -o json 2>/dev/null | grep '^\\[' | tail -1" \
    | grep -o '"is_paused": "[A-Za-z]*"' | head -1 | sed 's/.*: "\(.*\)"/\1/'
}

af_runs_in_flight() {
  local n=0 c d s
  for d in "$@"; do
    for s in queued running; do
      c=$($DC exec -T airflow-scheduler sh -c "airflow dags list-runs $d --state $s -o json 2>/dev/null | grep '^\\[' | tail -1")
      [ -n "$c" ] || return 1
      n=$((n + $(echo "$c" | grep -o '"run_id"' | wc -l)))
    done
  done
  echo "$n"
}

af_resume() {
  local d
  for d in $AF_PAUSED_BY_US; do
    $DC exec -T airflow-scheduler airflow dags unpause -y "$d" >/dev/null 2>&1
    if [ "$(af_dag_paused "$d")" = "False" ]; then
      log "unpaused $d"
    else
      log "ALERT: $d is STILL PAUSED — unpause it now: docker compose exec -T airflow-scheduler airflow dags unpause -y $d"
    fi
  done
  AF_PAUSED_BY_US=""
}

af_drain_and_pause() {
  local attempts="$1" polls="$2" attempt i n st d already
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
      if [ "$st" = "True" ]; then
        case " $AF_PAUSED_BY_US " in *" $d "*) ;; *) log "$d is already paused — left as it is" ;; esac
      elif [ "$st" = "False" ]; then
        $DC exec -T airflow-scheduler airflow dags pause -y "$d" >/dev/null 2>&1
        if [ "$(af_dag_paused "$d")" != "True" ]; then log "could not pause $d"; return 1; fi
        AF_PAUSED_BY_US="$AF_PAUSED_BY_US $d"
        log "paused $d"
      else
        log "cannot read whether $d is paused"
        return 1
      fi
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
