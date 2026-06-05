"""Build the team -> home-stadium mapping for soccer weather lookup.

Soccer's `matches.venue` is NULL for 99.6% of historical rows because
the football-data.co.uk CSVs (our main source) ship no venue field,
and PR #17's FBRef scraper hit IP-blocking from hosting-provider
networks. This script generates a per-team fallback so the Visual
Crossing weather fetcher has coords for every soccer match, with
or without `matches.venue`.

Data source: Wikidata SPARQL endpoint. Wikidata indexes football
clubs (Q476028) with their home venue (P115) and stadium coords
(P625) — completely free, no API key, no rate-limit concerns at
our query scale. The SPARQL query below fans across every football
club Wikidata knows about (~3,000 clubs globally) with a home
venue tagged. Results are cross-referenced against our DB's
`teams.name` via case-insensitive matching plus a curated alias
list for known mismatches (e.g. "Man United" vs "Manchester
United F.C.").

Output: `data/soccer_team_stadiums.json` keyed by team UUID with
{stadium, latitude, longitude, timezone, is_indoor}. The VC
weather fetcher (scripts/fetch_weather_visual_crossing.py) reads
this file as a fallback when matches.venue is NULL.

Run from a residential connection (Wikidata is fine from any IP,
but our prod hosts shouldn't proxy this kind of one-off generator):

    python scripts/build_soccer_stadium_map.py \\
        --database-url postgresql://USER:PASS@HOST:PORT/DB \\
        --output data/soccer_team_stadiums.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import psycopg2
import requests
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("build_soccer_stadium_map")


WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

# Pull every football club Wikidata knows about that has BOTH a home
# venue and coordinates for that venue. ?clubLabel is the English
# label; ?coord is "Point(lon lat)" WKT we parse below. The query
# completes in ~5s and returns ~3000 rows.
SPARQL_QUERY = """
SELECT DISTINCT ?club ?clubLabel ?stadium ?stadiumLabel ?coord ?countryLabel
       (GROUP_CONCAT(DISTINCT ?altLabel; separator="||") AS ?altLabels)
WHERE {
  ?club wdt:P31/wdt:P279* wd:Q476028 .
  ?club wdt:P115 ?stadium .
  ?stadium wdt:P625 ?coord .
  OPTIONAL {
    ?club skos:altLabel ?altLabel .
    FILTER(LANG(?altLabel) = "en")
  }
  OPTIONAL { ?club wdt:P17 ?country . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
GROUP BY ?club ?clubLabel ?stadium ?stadiumLabel ?coord ?countryLabel
"""


def fetch_wikidata(max_attempts: int = 4) -> list[dict]:
    """Run the SPARQL query, parse the result. Wikidata's response is
    JSON-LD-ish with each row's columns nested under ``value``.

    Wikidata's public endpoint truncates responses occasionally
    (observed at ~3.6MB on the alt-labels query, gave
    ``JSONDecodeError: Unterminated string``). Retry with exponential
    backoff on JSON or HTTP errors so transient issues don't kill the
    whole regen."""
    import time as _time

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(
                "Querying Wikidata SPARQL (attempt %d/%d)...",
                attempt,
                max_attempts,
            )
            r = requests.get(
                WIKIDATA_ENDPOINT,
                params={"query": SPARQL_QUERY, "format": "json"},
                headers={
                    "User-Agent": "auspex-soccer-stadium-builder/1.0 (chijiokekechi@gmail.com)",
                    "Accept": "application/sparql-results+json",
                },
                timeout=120,  # alt-labels query takes longer
            )
            r.raise_for_status()
            rows = r.json()["results"]["bindings"]
            logger.info("Wikidata returned %d club rows.", len(rows))
            break
        except (
            requests.exceptions.JSONDecodeError,
            requests.exceptions.HTTPError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as e:
            last_exc = e
            if attempt == max_attempts:
                raise
            sleep_for = 2**attempt  # 2s, 4s, 8s
            logger.warning(
                "Wikidata SPARQL failed (%s); retrying in %ds...",
                type(e).__name__,
                sleep_for,
            )
            _time.sleep(sleep_for)
    else:  # pragma: no cover — for-else after a clean break is fine
        raise RuntimeError("Wikidata SPARQL exhausted retries") from last_exc

    out = []
    for r_ in rows:
        club_label = r_.get("clubLabel", {}).get("value")
        stadium_label = r_.get("stadiumLabel", {}).get("value")
        coord = r_.get("coord", {}).get("value")  # "Point(lon lat)"
        country = r_.get("countryLabel", {}).get("value")
        alt_labels_str = (r_.get("altLabels", {}) or {}).get("value") or ""
        # Alt-labels come in "Bayern Munich||FC Bayern München||..." form
        # because we GROUP_CONCAT with "||" separator above. Wikidata's
        # alt-labels are how clubs surface their common short / English
        # name (e.g. "Bayern Munich" is an alt-label of "FC Bayern
        # München"), so indexing on them too dramatically widens the
        # match rate vs the primary clubLabel alone.
        alt_labels = [a.strip() for a in alt_labels_str.split("||") if a.strip()]
        if not (club_label and stadium_label and coord):
            continue
        m = re.match(r"Point\(([-0-9.]+) ([-0-9.]+)\)", coord)
        if not m:
            continue
        lon, lat = float(m.group(1)), float(m.group(2))
        out.append(
            {
                "club_name": club_label,
                "alt_labels": alt_labels,
                "stadium": stadium_label,
                "latitude": lat,
                "longitude": lon,
                "country": country,
            }
        )
    return out


def _normalize(name: str) -> str:
    """Lowercase, strip iteratively-applied prefix + suffix boilerplate,
    drop punctuation, normalise diacritics. Produces a canonical key
    that makes "Manchester United F.C." == "Man United" and
    "AC Milan" == "Milan" == "A.C. Milan" all index to the same slot.

    Diacritic strip is for cross-source matching (e.g. our DB stores
    "Atletico Madrid" without the accent; Wikidata stores
    "Atlético Madrid" with). Lossy on a few non-English names but
    safe at the scale we're operating on."""
    import unicodedata

    s = name.lower().strip()
    # Strip diacritics so "Atlético" → "atletico", "Köln" → "koln",
    # "Mönchengladbach" → "monchengladbach". DB and Wikidata diverge
    # on accents constantly; without this we'd need an alias per
    # accent-variant.
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

    # Iteratively strip suffixes so "Bologna F.C. 1909" peels both
    # " 1909" and " f.c." in turn. Loop terminates when nothing
    # matched in a full pass.
    suffixes = (
        # English club-type abbreviations.
        " f.c.",
        " fc",
        " a.f.c.",
        " afc",
        " s.c.",
        " sc",
        " c.f.",
        " cf",
        # Italian.
        " b.c.",
        " bc",
        " a.c.",
        " ac",
        " s.s.",
        " ss",
        " s.s.c.",
        " ssc",
        " a.s.",
        " as",
        " u.s.",
        " us",
        " calcio",
        " 1913",
        # Swedish ("fotboll" = "football" — same role as Italian
        # "calcio" or English "FC"): AIK Fotboll → "aik".
        " fotboll",
        # Spanish / Portuguese.
        " u.d.",
        " ud",
        " r.c.",
        " rc",
        " c.d.",
        " cd",
        " c.a.",
        " ca",
        " s.a.d.",
        " sad",
        " 05",
        " sociedad anonima deportiva",
        " alsace",
        # French.
        " sco",
        " osc",
        " 29",
        # German.
        " e.v.",
        " ev",
        " sv",
        " 1846",
        " 1899",
        # Year-of-founding tail.
        " 1907",
        " 1908",
        " 1909",
        " 1910",
    )
    # Prefixes — same pattern but at the START. Lots of clubs prepend
    # their club-type (FC Bayern, AC Milan, Real Madrid, etc.). Strip
    # these too so the short DB name matches.
    prefixes = (
        # English / generic.
        "fc ",
        "afc ",
        "a.f.c. ",
        "f.c. ",
        # Italian.
        "ac ",
        "a.c. ",
        "as ",
        "a.s. ",
        "ss ",
        "s.s. ",
        "ssc ",
        "s.s.c. ",
        "us ",
        "u.s. ",
        "acf ",
        # Spanish — "Real" is heavily used (Real Madrid, Real Betis,
        # Real Sociedad). RCD / CD / CA / UD likewise.
        "real ",
        "rcd ",
        "r.c.d. ",
        "cd ",
        "c.d. ",
        "ca ",
        "c.a. ",
        "ud ",
        "u.d. ",
        "athletic ",
        "deportivo ",
        "club atletico ",
        "rc ",
        "r.c. ",
        # German.
        "1 fc ",
        "1. fc ",
        "1 fsv ",
        "1. fsv ",
        "fc ",
        "vfl ",
        "vfb ",
        "sv ",
        "tsv ",
        "tsg 1899 ",
        "bayer 04 ",
        "borussia ",
        "hamburger ",
        "hellas ",
        # French.
        "olympique ",
        "stade ",
        "aj ",
        "rc ",
        # Italian / Portuguese / generic clubs.
        "ssc ",
        "ac ",
    )

    def _peel(s: str) -> str:
        changed = True
        while changed:
            changed = False
            for suffix in suffixes:
                if s.endswith(suffix):
                    s = s[: -len(suffix)].rstrip()
                    changed = True
                    break
            for prefix in prefixes:
                if s.startswith(prefix):
                    s = s[len(prefix) :].lstrip()
                    changed = True
                    break
        return s

    s = _peel(s)
    # Drop punctuation (including apostrophes — "Nott'm" → "nott m"),
    # collapse whitespace.
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Curated overrides for teams whose DB name doesn't match Wikidata's
# canonical name even after normalisation. Add new entries when the
# match script logs "no match for ...". The KEY is the normalised
# DB team name; the VALUE is the normalised Wikidata club label.
TEAM_ALIASES: dict[str, str] = {
    # English Premier shorthand.
    "man united": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "brighton": "brighton hove albion",
    "leicester": "leicester city",
    "leeds": "leeds united",
    "newcastle": "newcastle united",
    "west ham": "west ham united",
    # Premier — football-data.co.uk's idiosyncratic apostrophe gets
    # mangled to a space by the normaliser, so handle the after-
    # normalisation form too.
    "nott m forest": "nottingham forest",
    "nottm forest": "nottingham forest",
    # English Championship — football-data.co.uk drops the suffix
    # ("Derby" instead of "Derby County", "Norwich" instead of
    # "Norwich City"). Each maps to its Wikidata canonical name.
    "derby": "derby county",
    "ipswich": "ipswich town",
    "sheffield weds": "sheffield wednesday",
    "blackburn": "blackburn rovers",
    "coventry": "coventry city",
    "charlton": "charlton athletic",
    "oxford": "oxford united",
    "qpr": "queens park rangers",
    "swansea": "swansea city",
    "norwich": "norwich city",
    "west brom": "west bromwich albion",
    "birmingham": "birmingham city",
    "stoke": "stoke city",
    "preston": "preston north end",
    "hull": "hull city",
    # German.
    "bayern": "bayern munich",
    # French.
    "psg": "paris saint germain",
    # Spanish / La Liga — covers both the abbreviated and the
    # accented-vs-unaccented forms our DB uses vs Wikidata.
    "atletico": "atlético madrid",
    "atletico madrid": "atlético madrid",
    "ath madrid": "atlético madrid",
    "mallorca": "mallorca",  # post-normalise drops "RCD"
    "levante": "levante",  # post-normalise drops "UD"
    "espanol": "espanyol",  # football-data.co.uk drops the ñ
    "betis": "real betis",
    # Italian Serie A — most match directly after the iterative
    # suffix-strip handles " B.C." / " 1909" / " 1907" / " Calcio".
    # Aliases below cover the few that don't.
    "ac milan": "milan",
    "inter milan": "inter milan",
    "internazionale": "inter milan",
    "napoli": "napoli",
    "cagliari": "cagliari",
    # Trailing long-tail matches discovered in the post-regen audit.
    # FIFA World Cup national teams (Germany, Spain, Canada, Norway,
    # etc.) are intentionally NOT in this list — our SPARQL filters to
    # football clubs (Q476028), and national teams have no fixed home
    # stadium anyway. They stay as vc_unknown by design.
    "aik": "aik fotboll",
    "orgryte is": "orgryte",
    "sp gijon": "sporting de gijon",
    "st etienne": "as saint etienne",
}


def match_team_to_stadium(
    db_team_name: str,
    wiki_index: dict[str, dict],
) -> Optional[dict]:
    """Look up a DB team's stadium info via three-tier match:

      1. Direct normalised lookup — covers ~90% of cases.
      2. TEAM_ALIASES dict override — covers known mismatches that
         survive normalisation (e.g. "M'gladbach" → "monchengladbach").
      3. Containment fallback — if the DB norm is at least 4 chars,
         find any wiki_index key that CONTAINS it (or vice versa).
         Catches club-name variants we haven't aliased, e.g. DB
         "Inter" matches Wikidata "inter milan" by containment.

    Returns None if all three tiers fail."""
    norm = _normalize(db_team_name)
    if norm in wiki_index:
        return wiki_index[norm]
    # Alias values may be the human-readable Wikidata label (e.g.
    # "AS Saint-Étienne" or "saint etienne") rather than the
    # post-normalised form. Run them through _normalize too so the
    # alias dict can be maintained in either style and still resolve
    # to the index key.
    aliased = TEAM_ALIASES.get(norm)
    if aliased:
        aliased_norm = _normalize(aliased)
        if aliased_norm in wiki_index:
            return wiki_index[aliased_norm]
    # Containment fallback — DB-name (after normalisation) as a
    # substring of a wiki key. Min 4 chars to avoid spurious matches
    # on short keys ("ac", "fc", etc. would over-match wildly).
    if len(norm) >= 4:
        # Prefer LONGEST match (most specific) to avoid e.g. "athletic"
        # picking the first club whose name starts with "athletic".
        candidates = [k for k in wiki_index if norm in k or k in norm]
        if candidates:
            return wiki_index[max(candidates, key=len)]
    return None


def lookup_timezone(lat: float, lon: float) -> str:
    """Look up an IANA timezone for (lat, lon). Tries
    ``timezonefinder`` if installed (offline + fast); falls back to
    UTC with a warning otherwise (script still ships JSON; user can
    fill timezones later if needed)."""
    try:
        from timezonefinder import TimezoneFinder
    except ImportError:
        return "UTC"
    tf = TimezoneFinder()
    tz = tf.timezone_at(lat=lat, lng=lon)
    return tz or "UTC"


def list_soccer_teams(cur) -> list[dict]:
    cur.execute(
        """
        SELECT t.id::text AS id, t.name AS name, l.name AS league
        FROM teams t
        JOIN leagues l ON l.id = t.league_id
        WHERE l.sport = 'soccer'
        ORDER BY l.name, t.name
        """,
    )
    return cur.fetchall()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--output",
        default="data/soccer_team_stadiums.json",
        help="Where to write the mapping (relative paths from repo root).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    if not args.database_url:
        logger.error("DATABASE_URL not set.")
        return 1

    wiki_rows = fetch_wikidata()
    # Index by normalised name; if a name appears multiple times,
    # keep the first (Wikidata returns multiple language variants for
    # popular clubs, but they all point at the same stadium).
    wiki_index: dict[str, dict] = {}
    for row in wiki_rows:
        # Index by primary label.
        norm = _normalize(row["club_name"])
        if norm and norm not in wiki_index:
            wiki_index[norm] = row
        # ALSO index by every alt-label so DB "Bayern Munich" matches
        # Wikidata "FC Bayern München" via its English alt-label.
        # First-wins on collisions (rare; only happens when two clubs
        # legitimately share a colloquial alt-label).
        for alt in row.get("alt_labels", []) or []:
            alt_norm = _normalize(alt)
            if alt_norm and alt_norm not in wiki_index:
                wiki_index[alt_norm] = row

    with psycopg2.connect(args.database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            db_teams = list_soccer_teams(cur)
    logger.info("Loaded %d soccer teams from DB.", len(db_teams))

    mapping: dict[str, dict] = {}
    matched = 0
    misses: list[str] = []
    for team in db_teams:
        hit = match_team_to_stadium(team["name"], wiki_index)
        if not hit:
            misses.append(f"{team['name']} ({team['league']})")
            continue
        tz = lookup_timezone(hit["latitude"], hit["longitude"])
        mapping[team["id"]] = {
            "team_name": team["name"],
            "league": team["league"],
            "stadium": hit["stadium"],
            "latitude": hit["latitude"],
            "longitude": hit["longitude"],
            "timezone": tz,
            # Wikidata has a "stadium is covered" property but coverage
            # is sparse. Default to False — caller can override per
            # known-indoor venue (Tottenham, Atlanta MLS).
            "is_indoor": False,
            "country": hit.get("country"),
        }
        matched += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False, sort_keys=True)
    logger.info(
        "Wrote %d team mappings to %s (%d misses).",
        matched,
        out_path,
        len(misses),
    )
    if misses:
        logger.info("Sample misses (first 30):")
        for m in misses[:30]:
            logger.info("  - %s", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
