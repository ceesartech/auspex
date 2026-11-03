"""Daily disk guards for the VM root filesystem (OPERATIONS.md "Disk maintenance").

The root disk (75 GB) filled to 95% on 2026-11-02. Two of the growers are
handled here, daily, because a weekly pass let a week of growth land at once:

* prune_airflow_task_logs — Airflow task-log FILES are unbounded (`airflow db
  clean` prunes metadata rows only). Measured 2026-10-01..11-02: 120-290 MB a
  day, mean ~190 MB, ~3,300 files a day, 96% of it auspex_pipeline (every 15
  min). Keeping 14 days holds ~2.7 GB (1.7-4.0 GB at the measured extremes);
  the old 30-day weekly prune let it reach 7.4 GB. 14 days still covers every
  run anyone debugs from the UI (older task logs show "log not found"; the run
  rows stay for 90 days). It also deletes rotated files (`*.log.1`, `*.log.2`):
  the Airflow 2 dag_processor_manager.log.1/.2 (100 MB each, frozen since July)
  never matched the old `*.log` pattern. Empty directories go once they have
  been empty for an hour, so a task directory Airflow has just created is never
  removed under it.
* prune_docker_images — scripts/prune_docker_images.py (its docstring has the
  policy and why `docker image prune --filter until=` was dropped): keeps every
  image a container uses plus the newest 3 deploys' images as the local
  rollback target, removes the rest once 24 h old.

The two tasks are independent: one failing does not skip the other. Both fail
loudly (non-zero exit → red task → Telegram via notify_failure); neither
swallows an error.

The weekly docker_maintenance DAG keeps the build-cache and B2 modal-artifact
prunes.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG
from alerting import notify_failure  # shared Telegram failure alerting

AIRFLOW_LOG_DIR = "/opt/airflow/logs"
TASK_LOG_RETENTION_DAYS = 14
KEEP_DEPLOYS = 3
IMAGE_GRACE_HOURS = 24


def task_log_prune_command(log_dir: str = AIRFLOW_LOG_DIR, days: int = TASK_LOG_RETENTION_DAYS) -> str:
    """Delete Airflow log files (`*.log`, rotated `*.log.N`) older than
    `days`, then directories empty for over an hour; print what it freed.
    Fails the task on any find/du error (set -e + pipefail) AND when a file
    the delete pass selected is still there afterwards: that check catches a
    refused delete even where find exits 0 on one (BSD find does).

    The check reads back the exact list the delete pass printed (NUL-
    separated, in a temp file) instead of running a second `-mtime` find:
    each find measures age from its own start, so a second pass would also
    see the ~2 files a minute that cross the 14-day line in between and fail
    the task on a delete that was never refused (review, 2026-11-02)."""
    match = f"-type f \\( -name '*.log' -o -name '*.log.[0-9]*' \\) -mtime +{days}"
    return (
        "set -euo pipefail\n"
        f"d='{log_dir}'\n"
        'test -d "$d" || { echo "log dir $d is missing" >&2; exit 1; }\n'
        'list=$(mktemp "${TMPDIR:-/tmp}/prune_task_logs.XXXXXX")\n'
        "trap 'rm -f \"$list\"' EXIT\n"
        'before=$(du -sk "$d" | cut -f1)\n'
        f'find "$d" {match} -print0 -delete > "$list"\n'
        "files=0; left=0; eg=''\n"
        "while IFS= read -r -d '' f; do\n"
        "  files=$((files + 1))\n"
        '  if [ -e "$f" ] || [ -L "$f" ]; then\n'
        "    left=$((left + 1))\n"
        '    if [ "$left" -le 5 ]; then eg="$eg $f"; fi\n'
        "  fi\n"
        'done < "$list"\n'
        'if [ "$left" -gt 0 ]; then echo "could not delete $left old log file(s), e.g.:$eg" >&2; exit 1; fi\n'
        'dirs=$(find "$d" -mindepth 1 -type d -empty -mmin +60 -print -delete | wc -l)\n'
        'after=$(du -sk "$d" | cut -f1)\n'
        f'echo "pruned $((files)) log file(s) older than {days} days and $((dirs)) empty dir(s); '
        '$d: $((before / 1024)) MB -> $((after / 1024)) MB"\n'
    )


default_args = {
    "owner": "auspex",
    "on_failure_callback": notify_failure,
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="disk_maintenance",
    description=(
        f"Daily: Airflow task logs > {TASK_LOG_RETENTION_DAYS} d; docker images except in-use + newest "
        f"{KEEP_DEPLOYS} deploys"
    ),
    default_args=default_args,
    start_date=datetime(2026, 11, 1),
    schedule="15 5 * * *",  # 05:15 UTC daily: after the 02:00 backup, outside the 15-min pipeline's heavy tasks
    catchup=False,
    max_active_runs=1,
    # New DAGs start paused on this deployment (core.dags_are_paused_at_creation
    # =True). A paused disk guard is a silent failure, and the image task took
    # over from docker_maintenance in the same deploy, so this one starts live.
    is_paused_upon_creation=False,
    tags=["auspex", "maintenance", "disk"],
) as dag:

    prune_airflow_task_logs = BashOperator(
        task_id="prune_airflow_task_logs",
        bash_command=task_log_prune_command(),
        execution_timeout=timedelta(minutes=30),
    )

    prune_docker_images = BashOperator(
        task_id="prune_docker_images",
        bash_command=(
            "python /opt/auspex/scripts/prune_docker_images.py --apply "
            f"--keep-deploys {KEEP_DEPLOYS} --grace-hours {IMAGE_GRACE_HOURS}"
        ),
        execution_timeout=timedelta(minutes=30),
    )
