"""Telegram digest helper used by the precompute pipeline.

Both the soccer and NHL precompute scripts push their high-confidence
Alerts into a shared Redis queue with `enqueue_alerts(alerts)`. After
both branches finish, the DAG's final `send_pipeline_digest` task
calls `drain_and_send_digest()` once — that drains the queue, renders
one combined HTML digest containing every sport's picks, and sends
it as a single Telegram message (or chunks across messages if the
digest exceeds 4096 chars).

Why a shared queue + final drain (not per-script send)? The soccer
and NHL precompute tasks run as separate Airflow BashOperators in
parallel branches. Each lives in its own Python process, so they
can't share an in-memory list. The Redis queue stitches them together
without coupling the two scripts.

`send_telegram_digest()` is kept as a low-level primitive: render +
chunk + POST. It's used internally by `drain_and_send_digest()` and
is also testable in isolation.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Redis list key the pipeline pushes pending picks into. The drain
# task pops everything and sends one combined digest.
DEFAULT_QUEUE_KEY = "auspex:pending_alerts"

# Telegram's documented sendMessage hard limit. We chunk just under it
# to leave room for the HTML envelope.
TELEGRAM_MESSAGE_LIMIT = 4096
_SAFE_CHUNK_SIZE = 3900


@dataclass
class Alert:
    """One bundled prediction line.

    `sport` drives the sport emoji (⚽ / 🏒). `market_label` is the
    user-facing market name ("Moneyline", "Puck Line", "1X2"), NOT the
    raw snake_case prediction_type — callers translate before
    constructing the Alert so the digest never leaks internal vocab.
    """

    sport: str
    league_name: str
    home_team: str
    away_team: str
    match_date: datetime
    market_label: str
    predicted_outcome: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)


_SPORT_EMOJI: Dict[str, str] = {
    "soccer": "⚽",
    "nhl": "🏒",
}


def _format_alert_line(alert: Alert) -> str:
    """One HTML-formatted line per pick. Compact enough that ~30 picks
    fit under the 4096-char Telegram limit, but readable on a phone."""
    emoji = _SPORT_EMOJI.get(alert.sport, "•")
    when = alert.match_date.strftime("%a %m/%d %H:%M")
    probs = ", ".join(f"{k} {v:.0%}" for k, v in alert.probabilities.items())
    return (
        f"{emoji} <b>{alert.home_team} vs {alert.away_team}</b> · {alert.league_name}\n"
        f"   {alert.market_label}: <b>{alert.predicted_outcome}</b> "
        f"({alert.confidence:.0%}) · {when}\n"
        f"   <i>{probs}</i>"
    )


def render_digest(alerts: List[Alert], header: Optional[str] = None) -> str:
    """Build the HTML digest body. Pulled out so tests can assert on
    the formatting without needing to mock the HTTP layer."""
    if not alerts:
        return ""
    title = header or f"Auspex picks · {len(alerts)} high-confidence"
    body = "\n\n".join(_format_alert_line(a) for a in alerts)
    return f"<b>{title}</b>\n\n{body}"


def _chunk_text(text: str, limit: int = _SAFE_CHUNK_SIZE) -> List[str]:
    """Split `text` so each chunk stays under `limit`. Splits on the
    blank line between alerts so we never cut a pick in half. If a
    single alert is somehow larger than the limit (shouldn't happen with
    the compact format above) we fall back to a hard substring split."""
    if len(text) <= limit:
        return [text]
    chunks: List[str] = []
    parts = text.split("\n\n")
    cur = ""
    for part in parts:
        candidate = part if not cur else f"{cur}\n\n{part}"
        if len(candidate) > limit:
            if cur:
                chunks.append(cur)
                cur = part
            else:
                # Single part exceeds limit — hard-split it.
                for i in range(0, len(part), limit):
                    chunks.append(part[i : i + limit])
                cur = ""
        else:
            cur = candidate
    if cur:
        chunks.append(cur)
    return chunks


def send_telegram_digest(alerts: List[Alert], *, header: Optional[str] = None) -> int:
    """Send one (or more, if chunked) HTML message containing every
    alert in the batch. Returns the count of messages actually sent.

    Gates (in order):
      1. Empty list → 0
      2. ENABLE_TELEGRAM_NOTIFICATIONS != 'true' → 0
      3. Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID → 0, log warning
      4. Per-chunk POST. A failed chunk logs + stops further sends
         (better to surface a partial digest than to spam retries).
    """
    if not alerts:
        return 0
    if os.environ.get("ENABLE_TELEGRAM_NOTIFICATIONS", "false").lower() != "true":
        return 0
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("Telegram not configured — skipping digest of %d alerts", len(alerts))
        return 0

    digest = render_digest(alerts, header=header)
    chunks = _chunk_text(digest)
    sent = 0
    for chunk in chunks:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": chunk, "parse_mode": "HTML"},
                timeout=10,
            )
            r.raise_for_status()
            sent += 1
        except requests.RequestException as e:
            logger.warning("Telegram digest send failed (%d/%d chunks sent): %s", sent, len(chunks), e)
            break
    return sent


# ── Shared-queue digest path (the soccer+NHL combined message) ──────


def _alert_to_dict(alert: Alert) -> dict:
    """Alert → JSON-safe dict. datetime serializes to ISO so it round-
    trips cleanly through Redis."""
    d = asdict(alert)
    d["match_date"] = alert.match_date.isoformat()
    return d


def _alert_from_dict(d: dict) -> Alert:
    """Inverse of _alert_to_dict. Tolerant of older queue entries that
    were written before we added new fields — defaults applied here so
    a mid-deploy drain of a stale queue still works."""
    return Alert(
        sport=d["sport"],
        league_name=d["league_name"],
        home_team=d["home_team"],
        away_team=d["away_team"],
        match_date=datetime.fromisoformat(d["match_date"]),
        market_label=d["market_label"],
        predicted_outcome=d["predicted_outcome"],
        confidence=float(d["confidence"]),
        probabilities={k: float(v) for k, v in (d.get("probabilities") or {}).items()},
    )


def _redis_client(redis_url: Optional[str] = None):
    """Lazy import + connect. Falls back to REDIS_URL env. Returns None
    if redis isn't reachable so callers can degrade to per-script send."""
    url = redis_url or os.environ.get("REDIS_URL")
    if not url:
        logger.warning("REDIS_URL not set — combined digest queue unavailable")
        return None
    try:
        from redis import Redis  # type: ignore

        return Redis.from_url(url, decode_responses=True)
    except Exception as e:  # pragma: no cover - import / connect guard
        logger.warning("Redis client unavailable (%s) — combined digest queue offline", e)
        return None


def enqueue_alerts(
    alerts: List[Alert],
    *,
    redis_url: Optional[str] = None,
    queue_key: str = DEFAULT_QUEUE_KEY,
    ttl_seconds: int = 6 * 3600,
) -> int:
    """Push every Alert onto the Redis list at `queue_key`. Returns the
    list length AFTER the push so callers can log progress.

    TTL is set on the key so a queue that's never drained (DAG broken /
    misconfigured) auto-expires instead of growing forever. 6 hours is
    well past the 15-min pipeline cadence — if no digest task drains in
    that window something else is wrong and surfacing stale picks would
    be worse than dropping them.

    Returns 0 if the list is empty or Redis is unavailable. The caller
    can fall back to send_telegram_digest in the latter case if it
    wants per-script behavior."""
    if not alerts:
        return 0
    client = _redis_client(redis_url)
    if client is None:
        return 0
    payloads = [json.dumps(_alert_to_dict(a)) for a in alerts]
    pipe = client.pipeline()
    pipe.rpush(queue_key, *payloads)
    pipe.expire(queue_key, ttl_seconds)
    results = pipe.execute()
    return int(results[0]) if results else 0


def drain_and_send_digest(
    *,
    redis_url: Optional[str] = None,
    queue_key: str = DEFAULT_QUEUE_KEY,
    header: Optional[str] = None,
) -> dict:
    """Pop every queued Alert atomically, send one combined digest,
    return a count summary.

    The pop+del is done in a MULTI/EXEC so a concurrent enqueue can't
    race in and have its alerts silently discarded between LRANGE and
    DEL. Returns {drained, sent} so the DAG task log shows how many
    picks were aggregated vs how many Telegram messages were sent.
    """
    client = _redis_client(redis_url)
    if client is None:
        return {"drained": 0, "sent": 0}

    pipe = client.pipeline()
    pipe.lrange(queue_key, 0, -1)
    pipe.delete(queue_key)
    results = pipe.execute()
    raw_payloads = results[0] if results else []

    alerts: List[Alert] = []
    for payload in raw_payloads:
        try:
            alerts.append(_alert_from_dict(json.loads(payload)))
        except Exception as e:  # pragma: no cover - poison-pill guard
            logger.warning("Skipping malformed queued alert: %s", e)

    # Sort: soccer first then NHL (matches the order the user already
    # sees in the dashboard), then by match_date so picks within a
    # sport are ordered by kickoff/puck-drop.
    sport_order = {"soccer": 0, "nhl": 1}
    alerts.sort(key=lambda a: (sport_order.get(a.sport, 99), a.match_date))

    sent = send_telegram_digest(alerts, header=header)
    return {"drained": len(alerts), "sent": sent}
