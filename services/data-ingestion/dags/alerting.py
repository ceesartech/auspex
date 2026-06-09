"""Shared Airflow failure alerting → Telegram.

Wired as `on_failure_callback` in every DAG's default_args so a broken
task (a failed scrape, a failed retrain, a failed DB backup) pages you
instead of failing silently in the Airflow UI.

Runs inside the scheduler/worker process, which carries
TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID + ENABLE_TELEGRAM_NOTIFICATIONS
in its environment (see docker-compose airflow-scheduler). Sends PLAIN
TEXT (no parse_mode) so a stack-trace snippet or a '<' / '&' in the
error can never trigger Telegram's 400 HTML-parse rejection — the
callback must never raise, or it would mask the original task failure.
"""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_MAX_ERROR_CHARS = 400


def notify_failure(context: dict) -> None:
    """Airflow on_failure_callback. Posts a compact failure summary to
    Telegram. Best-effort: any error here is swallowed (logged) so it
    can't break Airflow's own failure handling."""
    try:
        if os.environ.get("ENABLE_TELEGRAM_NOTIFICATIONS", "false").lower() != "true":
            return
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            logger.warning("Telegram not configured — Airflow failure alert skipped")
            return

        ti = context.get("task_instance")
        dag = context.get("dag")
        dag_id = getattr(dag, "dag_id", None) or context.get("dag_id", "?")
        task_id = getattr(ti, "task_id", None) or context.get("task", "?")
        run_id = context.get("run_id", "?")
        when = context.get("logical_date") or context.get("execution_date")
        exc = context.get("exception")
        try_number = getattr(ti, "try_number", None)
        log_url = getattr(ti, "log_url", "")

        lines = [
            "🔴 Airflow task FAILED",
            f"DAG: {dag_id}",
            f"Task: {task_id}",
            f"Run: {run_id}",
        ]
        if try_number is not None:
            lines.append(f"Try: {try_number}")
        if when is not None:
            lines.append(f"When: {when}")
        if exc is not None:
            lines.append(f"Error: {str(exc)[:_MAX_ERROR_CHARS]}")
        if log_url:
            lines.append(f"Logs: {log_url}")
        text = "\n".join(lines)

        resp = requests.post(
            _TELEGRAM_API.format(token=token),
            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("Airflow failure alert sent for %s.%s", dag_id, task_id)
    except Exception as exc:  # noqa: BLE001 — callback must never raise
        logger.warning("Airflow failure alert failed to send: %s", exc)
