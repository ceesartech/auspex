"""Team identity shared by the fixture loaders.

Accents. Both scripts/fetch_upcoming.py (ESPN) and scripts/load_nhl_historical.py
(NHL API) resolve a team by EXACT teams.normalized_name, and a row can hold
only one spelling. ESPN writes "montreal canadiens", the NHL API
"montréal canadiens", so the same club became two rows: every live-season
Montreal fixture sat on a team with no history (rolling form empty), and a
data-only merge would be undone by the next fetch re-inserting the missing
spelling (audit §7, 2026-10-26). Both loaders now call
find_team_accent_insensitive() before inserting: it reuses an existing row
whose accent-folded name equals the incoming one, and only when exactly one
row qualifies — an ambiguous fold never merges anything.

Soccer names. ESPN's display names ('Manchester City', 'Tottenham Hotspur',
'Borussia Dortmund') are not football-data's ('Man City', 'Tottenham',
'Dortmund'). The old fuzzy step compared whole strings within the team's
first league at >= 0.85, so most of them missed, ensure_team inserted a
twin, and every later sweep matched the twin exactly: 320 soccer teams
created from 2026-05-27 split from their history ('Man City' 722 finished
matches, 'Manchester City' 11). Lowering the threshold is no fix — it would
merge Kalamata with Kalamaria (0.82) and Man City with Man United (0.81).
Three tools here close it, consulted by fetch_upcoming in this order:

  1. The ESPN team id (teams.external_ids.espn), looked up first and stamped
     onto whatever row a name resolves to, so a name is decided once and every
     later sweep follows the id. A row that already carries a different ESPN
     id is a different ESPN team: it is never reused or overwritten.
  2. Aliases (teams.metadata.aliases, a jsonb array of names), consulted after
     the exact name: the curated answer for names no rule can pair ('Wolves' /
     'Wolverhampton Wanderers'). promote_raw.upsert_teams uses the same
     resolver.
  3. match_club_name, a structural fuzzy match among the established teams of
     ONE country (never 'International', where Barcelona SC and Barcelona
     meet): equal club words, the existing name's words inside the incoming
     one's (for a club that played in the last three years), or abbreviations
     ('Man' for 'Manchester', 'Utd', 'QPR'). It accepts only a unique best
     with a clear margin over the runner-up; anything else is a new team,
     which the caller logs at ERROR as a canary.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

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


# ── 1. ESPN team id ───────────────────────────────────────────────────

# Guarded on the observed state: the row has no ESPN id yet and no other row
# of the sport holds this one, so a stamp never overwrites an id and never
# puts one ESPN team on two rows.
STAMP_ESPN_SQL = """
    UPDATE teams t
       SET external_ids = COALESCE(t.external_ids, '{}'::jsonb) || jsonb_build_object('espn', %(espn)s::text),
           updated_at = NOW()
     WHERE t.id = %(id)s
       AND NOT (COALESCE(t.external_ids, '{}'::jsonb) ? 'espn')
       AND NOT EXISTS (
           SELECT 1 FROM teams o
            WHERE o.sport = t.sport AND o.id <> t.id
              AND o.external_ids @> jsonb_build_object('espn', %(espn)s::text)
       )
    RETURNING t.id
"""


def find_team_by_espn_id(cur, sport: str, espn_id: str | None) -> str | None:
    """id of the team of `sport` stamped with this ESPN team id, else None.

    Two rows holding one id is corruption: logged at ERROR and neither is
    returned, so the caller falls back to the name (and stamp_espn_id will
    not add a third holder)."""
    if not espn_id:
        return None
    cur.execute(
        "SELECT id FROM teams WHERE sport = %s AND external_ids @> jsonb_build_object('espn', %s::text)",
        (sport, str(espn_id)),
    )
    ids = sorted({str(r["id"]) for r in cur.fetchall()})
    if len(ids) > 1:
        logger.error(
            "ESPN team id %s is on %d %s teams (%s): one ESPN team must be one row — resolving by name instead",
            espn_id,
            len(ids),
            sport,
            ", ".join(ids),
        )
        return None
    return ids[0] if ids else None


def espn_id_of(cur, team_id) -> str | None:
    """The ESPN team id stamped on `team_id`, or None."""
    cur.execute("SELECT external_ids->>'espn' AS espn FROM teams WHERE id = %s", (team_id,))
    row = cur.fetchone()
    return row["espn"] if row else None


def stamp_espn_id(cur, team_id, espn_id: str, label: str = "") -> bool:
    """Record ESPN team id `espn_id` on `team_id`. True when the row carries
    that id afterwards; False (logged at ERROR) when the row already holds a
    different ESPN id or another row holds this one — neither is touched."""
    espn_id = str(espn_id)
    cur.execute(STAMP_ESPN_SQL, {"id": team_id, "espn": espn_id})
    if cur.fetchone():
        return True
    cur.execute(
        "SELECT t.external_ids->>'espn' AS espn, "
        "(SELECT o.id FROM teams o WHERE o.sport = t.sport AND o.id <> t.id "
        "AND o.external_ids @> jsonb_build_object('espn', %(espn)s::text) LIMIT 1) AS holder "
        "FROM teams t WHERE t.id = %(id)s",
        {"id": team_id, "espn": espn_id},
    )
    row = cur.fetchone() or {}
    if row.get("espn") == espn_id:
        return True
    logger.error(
        "Not stamping ESPN team id %s onto team %s (%s): the row holds ESPN id %s and team %s holds this one — "
        "one of the two name resolutions is wrong; check both rows",
        espn_id,
        team_id,
        label or "?",
        row.get("espn"),
        row.get("holder"),
    )
    return False


# ── 2. Aliases ────────────────────────────────────────────────────────


def alias_form(name: str) -> str:
    """The form names and aliases are compared in: lower-case, whitespace
    collapsed, accents folded ('Málaga ' and 'malaga' are one alias)."""
    return strip_accents(" ".join((name or "").strip().lower().split()))


def load_alias_index(cur, sport: str) -> dict[str, set[str]]:
    """{alias_form(alias): {team ids}} from teams.metadata.aliases of `sport`.
    A value that is not an array of strings is ignored and logged at ERROR."""
    cur.execute(
        "SELECT id, metadata->'aliases' AS aliases FROM teams WHERE sport = %s AND metadata ? 'aliases'",
        (sport,),
    )
    index: dict[str, set[str]] = {}
    for row in cur.fetchall():
        aliases = row["aliases"]
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            logger.error(
                "teams.metadata.aliases of %s team %s is %r, not an array of names — ignored", sport, row["id"], aliases
            )
            continue
        for alias in aliases:
            if alias_form(alias):
                index.setdefault(alias_form(alias), set()).add(str(row["id"]))
    return index


def resolve_alias(index: dict[str, set[str]], sport: str, name: str) -> str | None:
    """id of the ONE team whose aliases include `name`, else None. An alias two
    teams claim resolves to neither (logged at ERROR)."""
    ids = index.get(alias_form(name))
    if not ids:
        return None
    if len(ids) > 1:
        logger.error(
            "Alias %r is claimed by %d %s teams (%s) — not resolving it; fix teams.metadata.aliases",
            alias_form(name),
            len(ids),
            sport,
            ", ".join(sorted(ids)),
        )
        return None
    return next(iter(ids))


def find_team_by_alias(cur, sport: str, name: str) -> str | None:
    """resolve_alias over the sport's current aliases (one small query)."""
    return resolve_alias(load_alias_index(cur, sport), sport, name)


# ── 3. Structural fuzzy match within a country ────────────────────────

# Words that say what kind of club a name is, not which club: legal forms
# ('FC', 'BK', 'IFK', 'KAA'), and the particles of 'Union de Santa Fe' or
# 'Heart of Midlothian'. Dropped anywhere in both names before comparing.
# 'Club', 'City', 'United', 'Real' are NOT here: they tell Club Brugge from
# Cercle Brugge and Man City from Man United.
GENERIC_CLUB_TOKENS = frozenset(
    {
        "fc", "afc", "cf", "sc", "ac", "as", "aj", "bk", "sk", "fk", "ff", "if", "ik", "ifk", "aif", "is",
        "kv", "kvc", "kaa", "ksv", "krc", "sv", "cd", "sd", "ud", "rcd", "se", "kc", "bc", "ks", "rfc", "cfc",
        "tf", "de", "del", "da", "do", "dos", "das", "the", "of", "and",
    }
)  # fmt: skip

# A country whose leagues gather clubs of many countries. Names collide there
# across borders (Barcelona SC / Barcelona, Independiente del Valle /
# Independiente), so the fuzzy step never runs for them: those teams resolve
# by ESPN id (stamped by their domestic league's sweep) or by alias.
CROSS_COUNTRY = frozenset({"International", "World"})

# Words that make another team of the same club: reserve, youth and women's
# sides. A name carrying one never matches a name without it (Barcelona B is
# not Barcelona).
SIDE_MARKERS = frozenset(
    {
        "b", "c", "ii", "iii", "reserve", "reserves", "res", "youth", "academy", "jong",
        "u17", "u18", "u19", "u20", "u21", "u23", "women", "womens", "ladies", "w", "femenino", "feminino", "frauen",
    }
)  # fmt: skip

# The tiers, a clear MIN_MARGIN apart. Each was checked on prod's 1,049
# established soccer teams by offering every one of them as a NEW name to the
# rest of its country: whole-string ratios (Atletico GO / Atletico-MG 0.89,
# San Martin S.J. / San Martin T. 0.88, Reggina / Reggiana 0.93) and one-word
# abbreviations (Legia / Legnica, Yamaga / Yamagata, Gaziantep / Gaziantepspor)
# merged different clubs, so neither counts at all.
SCORE_EQUAL = 1.0  # the same club words: 'BK Häcken' / 'Hacken'
SCORE_EXPANDED = 0.95  # the same words, some abbreviated: 'Man City' / 'Manchester City'
SCORE_CONTAINED = 0.90  # one name's words all inside the other's: 'Tottenham' / 'Tottenham Hotspur'
SCORE_PARTIAL = 0.85  # inside, with abbreviations: 'West Brom' / 'West Bromwich Albion'
MIN_SCORE = SCORE_PARTIAL
MIN_MARGIN = 0.05

# Containment leaves words unexplained, so it only reaches a club that has
# played within this window. A new club named after a defunct one ('Wuhan
# Three Towns' / Wuhan FC, last match 2022; 'Liaoning Tieren' / Liaoning, 2017)
# is a different club, and the defunct club's own ESPN name never arrives to
# expose the mistake as an ESPN id conflict. Exact club words still match.
CONTAINMENT_ACTIVE_WITHIN = timedelta(days=3 * 365)


def club_key(name: str) -> tuple[str, ...]:
    """The words that identify a club: accent-folded, lower-case, split on
    punctuation, apostrophes dropped ("Nott'm" -> 'nottm'), runs of single
    letters joined ('D.C.' -> 'dc', 'L.P.' -> 'lp'), generic words removed.
    A name made only of generic words keeps them all."""
    text = strip_accents(name or "").lower().replace("'", "").replace("’", "")
    tokens: list[str] = []
    run = ""
    for tok in re.split(r"[^0-9a-z]+", text):
        if len(tok) == 1 and tok.isalpha():
            run += tok
            continue
        if run:
            tokens.append(run)
            run = ""
        if tok:
            tokens.append(tok)
    if run:
        tokens.append(run)
    key = [t for t in tokens if t not in GENERIC_CLUB_TOKENS]
    return tuple(key or tokens)


def _shortens(x: str, y: str) -> bool:
    """x shortens the longer word y: a prefix of 3+ letters ('man', 'ein',
    'brom') or a 3+ letter subsequence keeping y's first and last letters
    ('utd', 'nottm')."""
    if len(x) < 3 or len(x) >= len(y) or x[0] != y[0]:
        return False
    if y.startswith(x):
        return True
    rest = iter(y)
    return x[-1] == y[-1] and all(ch in rest for ch in x)


def _expands(x: tuple[str, ...], y: tuple[str, ...], extra: bool) -> bool:
    """Every word of x maps, in order, onto its own word of y — equal, or
    shortened (_shortens), or the initials of a run of y's words ('oh' ->
    'oud heverlee', 'sg' -> 'saint germain'). With extra=False every word of y
    is used; with extra=True y may carry more ('albion').

    A one-word x may only be the initials of all of y ('qpr' -> 'queens park
    rangers'); a longer x needs at least one word equal. Abbreviations are
    never trusted on their own: 'legia' is a subsequence of 'legnica'."""
    if len(x) == 1:
        word = x[0]
        return 3 <= len(word) <= 4 and word.isalpha() and len(y) == len(word) and word == "".join(w[0] for w in y)

    def walk(i: int, j: int, exact: bool) -> bool:
        if i == len(x):
            return exact and (extra or j == len(y))
        word = x[i]
        for k in range(j, len(y) if extra else min(j + 1, len(y))):
            if word == y[k] and walk(i + 1, k + 1, True):
                return True
            if _shortens(word, y[k]) and walk(i + 1, k + 1, exact):
                return True
            span = len(word)
            if (
                2 <= span <= 4
                and word.isalpha()
                and k + span <= len(y)
                and word == "".join(w[0] for w in y[k : k + span])
                and walk(i + 1, k + span, exact)
            ):
                return True
        return False

    return walk(0, 0, False)


def _inside(x: tuple[str, ...], y: tuple[str, ...]) -> float:
    """The tier at which x sits inside y (x no longer than y), else 0."""
    if len(x) > len(y):
        return 0.0
    if _expands(x, y, extra=False):
        return SCORE_EXPANDED
    if set(x) < set(y):
        return SCORE_CONTAINED
    return SCORE_PARTIAL if _expands(x, y, extra=True) else 0.0


def club_match_tier(incoming: tuple[str, ...], existing: tuple[str, ...]) -> float:
    """The tier at which an existing club's key denotes an incoming name's key
    (see the SCORE_* constants), else 0.

    Containment runs one way only: the existing name inside the incoming one
    ('Tottenham' in 'Tottenham Hotspur'). The incoming name is the fuller,
    official one, so its extra words name the same club; extra words on the
    existing row tell clubs apart ('Wuhan FC' inside 'Wuhan Three Towns',
    'Dundee' inside 'Dundee United', 'Los Angeles FC' inside 'Los Angeles
    Galaxy'). An abbreviation covering both names wholly counts either way."""
    a, b = incoming, existing
    if not a or not b or (set(a) & SIDE_MARKERS) != (set(b) & SIDE_MARKERS):
        return 0.0
    if set(a) == set(b):
        return SCORE_EQUAL
    reverse = SCORE_EXPANDED if _inside(a, b) == SCORE_EXPANDED else 0.0
    return max(_inside(b, a), reverse)


@dataclass(frozen=True)
class ClubMatch:
    """The fuzzy step's verdict: team_id is set only when outcome == 'matched'."""

    outcome: str  # 'matched' | 'ambiguous' | 'no-match'
    team_id: str | None = None
    name: str | None = None
    score: float = 0.0
    runner_up: str | None = None
    runner_up_score: float = 0.0


def match_club_name(name: str, candidates: list[dict], as_of: datetime | None = None) -> ClubMatch:
    """Pick the existing club `name` denotes among `candidates` (dicts with
    id, name, normalized_name and latest_date — the tz-aware time of the
    team's latest match, or None: the established teams of ONE country).

    A candidate whose whole key sits inside another candidate's key ('Paris'
    of Paris FC inside 'Paris SG') cannot win from inside the incoming name:
    a name holding those words ('Paris Saint-Germain') may as well be the
    other club. Containment only reaches a candidate active within
    CONTAINMENT_ACTIVE_WITHIN of `as_of` (default now). The best must reach
    MIN_SCORE and beat the runner-up by MIN_MARGIN, i.e. by a whole tier."""
    as_of = as_of or datetime.now(timezone.utc)
    key = club_key(name)
    keyed = [(c, club_key(c.get("normalized_name") or c.get("name") or "")) for c in candidates]
    scored: list[tuple[float, dict]] = []
    for cand, ckey in keyed:
        score = club_match_tier(key, ckey)
        if 0 < score < SCORE_EQUAL and score == _inside(ckey, key):
            if any(other is not cand and _inside(ckey, okey) for other, okey in keyed):
                score = 0.0
        if score in (SCORE_CONTAINED, SCORE_PARTIAL):
            latest = cand.get("latest_date")
            if latest is None or as_of - latest > CONTAINMENT_ACTIVE_WITHIN:
                score = 0.0
        scored.append((score, cand))
    if not scored:
        return ClubMatch("no-match")
    scored.sort(key=lambda sc: -sc[0])
    (best_score, best), runner = scored[0], (scored[1] if len(scored) > 1 else (0.0, {}))
    if best_score < MIN_SCORE:
        return ClubMatch("no-match")
    outcome = "matched" if best_score - runner[0] >= MIN_MARGIN - 1e-9 else "ambiguous"
    return ClubMatch(
        outcome,
        team_id=str(best["id"]) if outcome == "matched" else None,
        name=best.get("name"),
        score=best_score,
        runner_up=runner[1].get("name") if runner[0] else None,
        runner_up_score=runner[0],
    )


# A team belongs to a country through its first league (teams.league_id) or
# the league of its latest match: promoted, relegated and renamed clubs keep
# their history either way. Index-backed per team (idx_matches_*_team_date).
COUNTRY_CANDIDATES_SQL = """
    SELECT t.id, t.name, t.normalized_name, t.external_ids->>'espn' AS espn, latest.match_date AS latest_date
      FROM teams t
      LEFT JOIN leagues fl ON fl.id = t.league_id
      LEFT JOIN LATERAL (
          SELECT x.league_id, x.match_date FROM (
              (SELECT league_id, match_date FROM matches WHERE home_team_id = t.id
                ORDER BY match_date DESC LIMIT 1)
              UNION ALL
              (SELECT league_id, match_date FROM matches WHERE away_team_id = t.id
                ORDER BY match_date DESC LIMIT 1)
          ) x ORDER BY x.match_date DESC LIMIT 1
      ) latest ON TRUE
      LEFT JOIN leagues ll ON ll.id = latest.league_id
     WHERE t.sport = %(sport)s
       AND (fl.country = %(country)s OR ll.country = %(country)s)
"""


def league_label(cur, league_id) -> tuple[str, str] | None:
    """(name, country) of a league, or None."""
    if not league_id:
        return None
    cur.execute("SELECT name, country FROM leagues WHERE id = %s", (league_id,))
    row = cur.fetchone()
    return (row["name"], row["country"]) if row else None


def country_candidates(cur, sport: str, country: str) -> list[dict]:
    """Every team of `sport` whose first or latest league is in `country`."""
    cur.execute(COUNTRY_CANDIDATES_SQL, {"sport": sport, "country": country})
    return list(cur.fetchall())


def find_team_fuzzy(cur, sport: str, name: str, league_id, espn_id: str | None = None) -> str | None:
    """match_club_name over the teams of the swept league's country, or None.

    With an incoming ESPN id, a candidate already stamped with one is another
    ESPN team (its own id would have matched first) and is left out. Never
    runs for a cross-country league (see CROSS_COUNTRY). A match is logged at
    WARNING: it happens once per team, before the ESPN id takes over, and it
    is the decision to review if two clubs were merged."""
    league = league_label(cur, league_id)
    if league is None or league[1] in CROSS_COUNTRY:
        return None
    league_name, country = league
    candidates = [c for c in country_candidates(cur, sport, country) if not (espn_id and c.get("espn"))]
    verdict = match_club_name(name, candidates)
    if verdict.outcome == "matched":
        logger.warning(
            "Resolved %s team %r (%s, %s) onto existing %r by club-name match %.2f (runner-up %r %.2f) — "
            "if they are different clubs, split them and add aliases",
            sport,
            name,
            league_name,
            country,
            verdict.name,
            verdict.score,
            verdict.runner_up,
            verdict.runner_up_score,
        )
        return verdict.team_id
    if verdict.outcome == "ambiguous":
        logger.warning(
            "Club-name match for %s team %r (%s, %s) is ambiguous: %r %.2f vs %r %.2f — not merging",
            sport,
            name,
            league_name,
            country,
            verdict.name,
            verdict.score,
            verdict.runner_up,
            verdict.runner_up_score,
        )
    return None


def established_team_count(cur, league_id) -> int:
    """Distinct teams with a finished match in `league_id`."""
    cur.execute(
        "SELECT count(*) AS n FROM ("
        "SELECT home_team_id FROM matches WHERE league_id = %(league)s AND status = 'finished' "
        "UNION SELECT away_team_id FROM matches WHERE league_id = %(league)s AND status = 'finished') t",
        {"league": league_id},
    )
    row = cur.fetchone()
    return int(row["n"]) if row else 0


def log_new_soccer_team(cur, name: str, league_id, source: str) -> None:
    """The canary for a soccer team row about to be minted. In a league that
    already has established teams a new name is usually a split of one of
    them, so it is logged at ERROR with the established count; in a league
    with none it is expected and logged at WARNING."""
    league = league_label(cur, league_id)
    label = f"{league[0]} ({league[1]})" if league else f"league {league_id}"
    established = established_team_count(cur, league_id) if league_id else 0
    level = logging.ERROR if established else logging.WARNING
    logger.log(
        level,
        "new soccer team %s in %s that already has %d established teams (%s) — if it is one of them, add its name "
        "to that team's metadata.aliases",
        name,
        label,
        established,
        source,
    )
