"""Load historical international football match results.

Pulls the maintained `martj42/international_results` Kaggle/GitHub
dataset — all international matches since 1872 in a single CSV. Covers:

    - FIFA World Cup (group stage, knockouts, final)
    - WC qualifiers (UEFA, CONMEBOL, CAF, AFC, CONCACAF, OFC)
    - UEFA European Championship
    - Copa América
    - CONCACAF Gold Cup
    - Africa Cup of Nations
    - AFC Asian Cup
    - OFC Nations Cup

Source: https://github.com/martj42/international_results (CC0).

What this dataset has:
    date, home_team, away_team, home_score, away_score, tournament,
    city, country, neutral

What it doesn't:
    No odds data. International matches in our DB will have NULL for
    odds-derived feature columns and the model trains on team identity
    + score history alone.

Usage:
    # Default: pull WC + WC qualifiers (the dominant summer events)
    python scripts/load_international.py

    # Specific tournaments (comma-separated, exact tournament names)
    python scripts/load_international.py --tournaments "FIFA World Cup,UEFA Euro"

Every row this writes must be one promote_raw can promote (2026-11-01):
promote_raw refuses a raw row whose league code is not in its LEAGUE_MAP or
whose team name looks mis-decoded, and exits 3 on every run while such a row
sits in raw_football_data (nothing deletes it). So a tournament with no code
in TOURNAMENT_CODES, or whose code promote_raw.LEAGUE_MAP has no league for,
is refused before anything is fetched (exit 2) — it was written as league
'UNK' — and a dataset with a row promote_raw would refuse
(promote_raw.raw_row_problem) is not written at all (exit 3). Friendlies are
not loaded: --include-friendlies wrote league 'FRIEND', which no league
stands for (the option is gone; a real friendlies league would be a
LEAGUE_MAP entry and a decision about training on friendlies first).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from io import StringIO

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("load_international")

DATASET_URL = "https://raw.githubusercontent.com/martj42/international_results/" "master/results.csv"

# Map dataset tournament names → our internal (league_code, league_name).
# Codes are non-football-data so they don't collide with the club-league
# codes; each must be one of promote_raw.LEAGUE_MAP's International leagues
# (pinned by tests/unit/test_load_international.py), which name the league.
TOURNAMENT_CODES: dict[str, tuple[str, str]] = {
    "FIFA World Cup": ("WC", "FIFA World Cup"),
    "FIFA World Cup qualification": ("WCQ", "FIFA World Cup Qualifiers"),
    "UEFA Euro": ("EURO", "UEFA European Championship"),
    "UEFA Euro qualification": ("EUROQ", "UEFA Euro Qualifiers"),
    "UEFA Nations League": ("UNL", "UEFA Nations League"),
    "Copa América": ("COPA", "Copa América"),
    "CONCACAF Gold Cup": ("GOLD", "CONCACAF Gold Cup"),
    "African Cup of Nations": ("AFCON", "Africa Cup of Nations"),
    "AFC Asian Cup": ("AFCASIAN", "AFC Asian Cup"),
    "Confederations Cup": ("CONFED", "FIFA Confederations Cup"),
}

DEFAULT_TOURNAMENTS = [
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "UEFA Nations League",
    "Copa América",
    "CONCACAF Gold Cup",
    "African Cup of Nations",
    "AFC Asian Cup",
]


def _season_from_date(date_str: str) -> str:
    """International tournaments don't have a `season` per se; use the year."""
    try:
        return date_str[:4]
    except Exception:
        return ""


def _result_from_scores(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def fetch_dataset() -> pd.DataFrame:
    logger.info("Fetching %s", DATASET_URL)
    r = requests.get(DATASET_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.content.decode("utf-8")))
    logger.info("Loaded %d rows total from dataset", len(df))
    return df


def normalize(df: pd.DataFrame, tournaments: list[str]) -> pd.DataFrame:
    """Filter to requested tournaments + reshape to raw_football_data schema."""
    df = df[df["tournament"].isin(tournaments)].copy()
    if df.empty:
        return df

    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score", "date"])
    df["FTHG"] = df["home_score"].astype(int)
    df["FTAG"] = df["away_score"].astype(int)
    df["FTR"] = df.apply(lambda r: _result_from_scores(r["FTHG"], r["FTAG"]), axis=1)
    df["HomeTeam"] = df["home_team"]
    df["AwayTeam"] = df["away_team"]
    # Date as DD/MM/YYYY to match the football-data.co.uk format that
    # promote_raw.parse_match_date already understands.
    df["Date"] = pd.to_datetime(df["date"]).dt.strftime("%d/%m/%Y")

    unknown = sorted(set(df["tournament"]) - set(TOURNAMENT_CODES))
    if unknown:  # main() refuses these before fetching; never a made-up code
        raise ValueError(f"tournament(s) {unknown} have no code in TOURNAMENT_CODES")
    df["league"] = df["tournament"].map(lambda t: TOURNAMENT_CODES[t][0])
    df["season"] = df["date"].map(_season_from_date)

    keep = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "league", "season"]
    return df[keep]


def unknown_tournaments(tournaments: list[str]) -> list[str]:
    """The tournaments promote_raw could not file: no code in TOURNAMENT_CODES, or a code promote_raw.LEAGUE_MAP has
    no league for."""
    import promote_raw  # it imports psycopg2, as write_database does

    return sorted(
        t for t in tournaments if t not in TOURNAMENT_CODES or TOURNAMENT_CODES[t][0] not in promote_raw.LEAGUE_MAP
    )


def unpromotable_rows(df: pd.DataFrame) -> list[tuple[str, dict]]:
    """[(problem, row)] for each row promote_raw would refuse (promote_raw.raw_row_problem)."""
    import promote_raw

    out = []
    for row in df.to_dict("records"):
        problem = promote_raw.raw_row_problem(row)
        if problem is not None:
            out.append((problem, row))
    return out


# main()'s exit code when the dataset holds a row promote_raw would refuse (nothing written).
REFUSED_EXIT_CODE = 3


def write_database(df: pd.DataFrame, database_url: str) -> int:
    """Insert into the same raw_football_data table the EU + extra loaders use.
    Only columns present in both ends are written."""
    import psycopg2
    from psycopg2.extras import execute_values

    if df.empty:
        return 0

    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT column_name FROM information_schema.columns " "WHERE table_name = 'raw_football_data'")
            existing = {r[0] for r in cur.fetchall()}
            if not existing:
                raise RuntimeError("raw_football_data doesn't exist. Run load_football_data.py first.")

            cols = [c for c in df.columns if c in existing]
            quoted_cols = ['"' + c + '"' for c in cols]
            insert_sql = f"INSERT INTO raw_football_data ({', '.join(quoted_cols)}) " "VALUES %s ON CONFLICT DO NOTHING"
            rows = [tuple(None if pd.isna(v) else str(v) for v in row) for row in df[cols].itertuples(index=False)]
            execute_values(cur, insert_sql, rows)
            conn.commit()
            return cur.rowcount


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--tournaments",
        default=",".join(DEFAULT_TOURNAMENTS),
        help="Comma-separated tournament names (use --list to discover).",
    )
    p.add_argument("--list", action="store_true", help="List distinct tournaments in the dataset and exit.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tournaments = [t.strip() for t in args.tournaments.split(",") if t.strip()]
    if not args.list:
        unknown = unknown_tournaments(tournaments)
        if unknown:
            logger.error(
                "Tournament(s) %s have no league promote_raw could file them under (TOURNAMENT_CODES -> "
                "promote_raw.LEAGUE_MAP): add both first — nothing fetched",
                unknown,
            )
            return 2
    df = fetch_dataset()

    if args.list:
        for t in sorted(df["tournament"].unique()):
            n = (df["tournament"] == t).sum()
            print(f"{t:<60} {n:>6} matches")
        return 0

    df = normalize(df, tournaments)
    if df.empty:
        logger.warning("No matches matched tournaments=%s", tournaments)
        return 1

    logger.info("Filtered to %d matches across %d tournaments", len(df), df["league"].nunique())
    refused = unpromotable_rows(df)
    if refused:
        for problem, row in refused[:10]:
            logger.error(
                "REFUSED %s %s: %s v %s — %s", row["league"], row["Date"], row["HomeTeam"], row["AwayTeam"], problem
            )
        logger.error(
            "%d row(s) promote_raw would refuse (and then exit 3 on every run while they sit in raw_football_data) "
            "— nothing written; exiting %d",
            len(refused),
            REFUSED_EXIT_CODE,
        )
        return REFUSED_EXIT_CODE

    if args.database_url:
        inserted = write_database(df, args.database_url)
        logger.info("Inserted %d new rows into raw_football_data", inserted)
    else:
        logger.info("--database-url not set; printed-only mode")
        print(df.head())
    return 0


if __name__ == "__main__":
    sys.exit(main())
