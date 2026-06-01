"""Telegram digest helper used by both precompute_predictions scripts.

Both the soccer and NHL precompute jobs accumulate one Alert per
high-confidence prediction and call `send_telegram_digest(alerts)` once
at the end of their run. This collapses what used to be one Telegram
message per pick into a single bundled digest — at most one outbound
message per script run instead of dozens.

If the rendered digest exceeds Telegram's 4096-char hard limit it is
split into multiple messages, each capped to stay under the limit
without breaking a line mid-row. Returns the number of messages
actually sent (0 if disabled / unconfigured / no alerts / network
failure).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

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
