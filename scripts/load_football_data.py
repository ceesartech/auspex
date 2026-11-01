"""Load historical match results + closing odds from football-data.co.uk.

football-data.co.uk publishes free per-season CSVs for European leagues going
back to 1993. Each row has full-time result, half-time result, and closing
odds across multiple books — the closing odds are the single most predictive
feature you can add for football outcome models.

Usage:
    # Load last 5 seasons of Premier League into the configured database
    python scripts/load_football_data.py --leagues E0 --seasons 5

    # Load all top-5 leagues, last 10 seasons
    python scripts/load_football_data.py --leagues E0,D1,I1,SP1,F1 --seasons 10

    # Dump to CSV without touching the database
    python scripts/load_football_data.py --leagues E0 --seasons 3 \\
        --output data/raw/football-data/

League codes (from football-data.co.uk):
    E0  English Premier League
    E1  English Championship
    E2  English League One
    D1  German Bundesliga
    D2  German Bundesliga 2
    I1  Italian Serie A
    I2  Italian Serie B
    SP1 Spanish La Liga
    SP2 Spanish La Liga 2
    F1  French Ligue 1
    F2  French Ligue 2
    N1  Dutch Eredivisie
    B1  Belgian First Division A
    P1  Portuguese Primeira Liga
    SC0 Scottish Premiership
    SC1 Scottish Championship
    SC2 Scottish League One
    SC3 Scottish League Two
    T1  Turkish Super Lig
    G1  Greek Super League

A code promote_raw.LEAGUE_MAP has no league for is refused before anything
is fetched (exit 2): promote_raw could not file its rows.

A file is refused — logged at ERROR, nothing of it written, and the run exits
3 once the other files are written — when (2026-11-01):
  * it is neither UTF-8 nor Windows-1252 (text_encoding.decode_csv_bytes; it
    was read as latin-1, which never fails: 'Preußen Münster' became
    'PreuÃ\\x9fen MÃ¼nster', and the UTF-8 byte-order mark hid the Div column
    of every file from 2024-25 on);
  * it has no Div column, or a row's Div is not the code it was fetched as:
    on 2026-08-06 football-data served the Scottish Championship and League
    One at the 2026-27 SP1 and SP2 URLs, and ten Scottish fixtures were filed
    under La Liga / La Liga 2 (scripts/ops/repair_soccer_labels.sh repairs
    them);
  * a team name looks mojibake (text_encoding.looks_mojibake): text decoded
    wrongly before football-data published it, which would mint a twin club.
A season football-data does not have (HTTP 404) is skipped with a warning, as
before.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_encoding  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("load_football_data")

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Subset of columns we care about. football-data CSVs have ~100 cols; most are
# odds across many bookmakers we don't need.
CORE_COLUMNS = [
    "Div",
    "Date",
    "Time",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HTHG",
    "HTAG",
    "HTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HC",
    "AC",
    "HF",
    "AF",
    "HY",
    "AY",
    "HR",
    "AR",
]
# Closing odds — Bet365 ("B365*"), Pinnacle ("PSC*"), market avg ("Avg*"),
# market max ("Max*"). Pinnacle and market avg are the most useful.
ODDS_COLUMNS = [
    "B365H",
    "B365D",
    "B365A",
    "PSH",
    "PSD",
    "PSA",
    "PSCH",
    "PSCD",
    "PSCA",
    "AvgH",
    "AvgD",
    "AvgA",
    "MaxH",
    "MaxD",
    "MaxA",
    "B365>2.5",
    "B365<2.5",
    "Avg>2.5",
    "Avg<2.5",
]


@dataclass
class LoadResult:
    league: str
    season: str
    rows: int
    path: Path | None


def season_code(year_end: int) -> str:
    """Convert end-year to football-data's season code, e.g. 2025 -> '2425'."""
    start = (year_end - 1) % 100
    end = year_end % 100
    return f"{start:02d}{end:02d}"


class RefusedFile(RuntimeError):
    """A football-data file that is not loaded at all (see the module docstring)."""


def parse_csv(raw: bytes, league: str, season: str, source: str) -> pd.DataFrame:
    """The rows of one season file, checked: decoded strictly, every row's Div
    equal to `league`, no team name that looks mojibake. Raises RefusedFile."""
    try:
        text, encoding = text_encoding.decode_csv_bytes(raw, source)
    except text_encoding.UndecodableText as exc:
        raise RefusedFile(str(exc)) from None
    # Older season files carry ragged rows (stray commas -> more fields
    # than the header); skip those lines rather than dying — one bad row
    # in one of 300+ files must not kill a whole backfill sweep.
    df = pd.read_csv(StringIO(text), on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    if "Div" not in df.columns:
        raise RefusedFile(
            f"{source} ({encoding}) has no Div column (header starts {list(df.columns[:3])}) — nothing says which "
            f"division it holds, so it is not loaded as {league}"
        )
    keep = [c for c in CORE_COLUMNS + ODDS_COLUMNS if c in df.columns]
    df = df[keep].dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    divs = df["Div"].fillna("").astype(str).str.strip()
    wrong = sorted(set(divs[divs != league]))
    if wrong:
        raise RefusedFile(
            f"{source} was fetched as {league} {season} but {int((divs != league).sum())} of its {len(df)} row(s) "
            f"name division(s) {wrong} — football-data served another division's file at that URL; not loaded"
        )
    names = pd.concat([df["HomeTeam"], df["AwayTeam"]]).astype(str)
    garbled = sorted({n for n in names if text_encoding.looks_mojibake(n)})
    if garbled:
        raise RefusedFile(
            f"{source} ({encoding}) has team name(s) that look mis-decoded: {garbled[:5]} — not loaded (a name like "
            "that would mint a twin of the club)"
        )
    df["league"] = league
    df["season"] = season
    return df


def fetch_csv(league: str, season: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{season}/{league}.csv"
    logger.info("Fetching %s", url)
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        raise FileNotFoundError(f"No data for {league} {season} (HTTP 404)")
    resp.raise_for_status()
    return parse_csv(resp.content, league, season, url)


def load_seasons(
    leagues: Iterable[str],
    n_seasons: int,
    end_year: int | None = None,
    refused: list[str] | None = None,
) -> list[pd.DataFrame]:
    """The season files of `leagues`; each file refused (RefusedFile) is
    logged at ERROR and appended to `refused` as 'league season'."""
    end_year = end_year or datetime.utcnow().year + (1 if datetime.utcnow().month >= 8 else 0)
    out: list[pd.DataFrame] = []
    for league in leagues:
        for year in range(end_year - n_seasons + 1, end_year + 1):
            season = season_code(year)
            try:
                df = fetch_csv(league, season)
            except FileNotFoundError:
                logger.warning("Skipping missing %s %s", league, season)
                continue
            except RefusedFile as e:
                logger.error("REFUSED %s %s: %s", league, season, e)
                if refused is not None:
                    refused.append(f"{league} {season}")
                continue
            except Exception as e:
                # Malformed file (parser errors etc.) — skip loudly, keep
                # the sweep alive. A missing season is recoverable later;
                # a dead 300-file backfill is not.
                logger.error("Skipping unparseable %s %s: %s", league, season, e)
                continue
            out.append(df)
            logger.info("Loaded %s %s: %d rows", league, season, len(df))
    return out


def write_csv(frames: list[pd.DataFrame], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for df in frames:
        league = df["league"].iloc[0]
        season = df["season"].iloc[0]
        path = output_dir / f"{league}_{season}.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    return paths


def write_database(frames: list[pd.DataFrame], database_url: str) -> int:
    """Insert into a staging table `raw_football_data`. The transformer then
    promotes rows into leagues/teams/matches/odds. Promotion is left to a
    follow-up step so this loader stays idempotent and side-effect-free.
    """
    import psycopg2
    from psycopg2.extras import execute_values

    if not frames:
        return 0
    all_df = pd.concat(frames, ignore_index=True)

    cols = list(all_df.columns)
    quoted_cols = ['"' + c + '"' for c in cols]
    schema_cols = ", ".join(q + " TEXT" for q in quoted_cols)
    create_sql = f"""
        CREATE TABLE IF NOT EXISTS raw_football_data (
            {schema_cols},
            loaded_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE ("Date", "HomeTeam", "AwayTeam")
        );
    """
    insert_sql = "INSERT INTO raw_football_data (" + ", ".join(quoted_cols) + ") " "VALUES %s ON CONFLICT DO NOTHING"
    rows = [tuple(None if pd.isna(v) else str(v) for v in row) for row in all_df.itertuples(index=False)]

    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(create_sql)
            execute_values(cur, insert_sql, rows)
            conn.commit()
            return cur.rowcount


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--leagues", default="E0", help="Comma-separated league codes (default: E0).")
    p.add_argument("--seasons", type=int, default=5, help="Number of seasons back to load (default: 5).")
    p.add_argument("--end-year", type=int, default=None, help="End-year of the most recent season (default: current).")
    p.add_argument("--output", type=Path, default=None, help="Optional output dir for raw CSVs.")
    p.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="Postgres URL. Falls back to $DATABASE_URL. Pass empty to skip DB load.",
    )
    return p.parse_args(argv)


# main()'s exit code when a file was refused (everything else written), as
# promote_raw exits on a refused raw row.
REFUSED_EXIT_CODE = 3


def unknown_league_codes(leagues: Iterable[str]) -> list[str]:
    """The codes promote_raw.LEAGUE_MAP has no league for (their rows could not be promoted)."""
    import promote_raw  # it imports psycopg2: the CSV-only path needs that installed too

    return sorted(set(leagues) - set(promote_raw.LEAGUE_MAP))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    leagues = [s.strip() for s in args.leagues.split(",") if s.strip()]
    unknown = unknown_league_codes(leagues)
    if unknown:
        logger.error(
            "Unknown league code(s) %s: promote_raw.LEAGUE_MAP has no league for them — add them there first; "
            "nothing fetched",
            unknown,
        )
        return 2
    refused: list[str] = []
    frames = load_seasons(leagues, args.seasons, args.end_year, refused)
    if not frames:
        logger.error("No data loaded for leagues=%s seasons=%d (refused: %s)", leagues, args.seasons, refused or "none")
        return REFUSED_EXIT_CODE if refused else 1
    total = sum(len(df) for df in frames)
    logger.info("Loaded %d total rows across %d season-files", total, len(frames))

    if args.output:
        paths = write_csv(frames, args.output)
        logger.info("Wrote %d CSV files to %s", len(paths), args.output)

    if args.database_url:
        inserted = write_database(frames, args.database_url)
        logger.info("Inserted %d new rows into raw_football_data", inserted)
    else:
        logger.info("--database-url not set; skipped DB load")

    if refused:
        logger.error(
            "REFUSED %d file(s) (ERROR lines above): %s — everything else is loaded; exiting %d",
            len(refused),
            ", ".join(refused),
            REFUSED_EXIT_CODE,
        )
        return REFUSED_EXIT_CODE
    return 0


if __name__ == "__main__":
    sys.exit(main())
