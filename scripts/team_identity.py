"""Accent-insensitive team identity shared by the fixture loaders.

Both scripts/fetch_upcoming.py (ESPN) and scripts/load_nhl_historical.py
(NHL API) resolve a team by EXACT teams.normalized_name, and a row can hold
only one spelling. ESPN writes "montreal canadiens", the NHL API
"montréal canadiens", so the same club became two rows: every live-season
Montreal fixture sat on a team with no history (rolling form empty), and a
data-only merge would be undone by the next fetch re-inserting the missing
spelling (audit §7, 2026-10-26). Both loaders now call
find_team_accent_insensitive() before inserting: it reuses an existing row
whose accent-folded name equals the incoming one, and only when exactly one
row qualifies — an ambiguous fold never merges anything.
"""

from __future__ import annotations

import logging
import unicodedata

logger = logging.getLogger(__name__)


def strip_accents(text: str) -> str:
    """'Montréal' -> 'Montreal': NFKD-decompose and drop combining marks."""
    return "".join(ch for ch in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(ch))


def find_team_accent_insensitive(cur, sport: str, normalized_name: str) -> str | None:
    """id of the ONE team of `sport` whose accent-folded normalized_name equals
    the folded `normalized_name`, else None. Candidates are the unaccented
    spelling plus every non-ASCII name of the sport (few rows)."""
    folded = strip_accents(normalized_name)
    cur.execute(
        "SELECT id, normalized_name FROM teams "
        "WHERE sport = %s AND (normalized_name = %s OR normalized_name !~ '^[ -~]*$')",
        (sport, folded),
    )
    hits = {str(r["id"]) for r in cur.fetchall() if strip_accents(r["normalized_name"]) == folded}
    if len(hits) > 1:
        logger.warning(
            "Accent-folded team name %r is ambiguous in %s (%d rows) — not merging", folded, sport, len(hits)
        )
        return None
    return next(iter(hits)) if hits else None
