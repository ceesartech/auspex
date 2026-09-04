"""Load historical match results + closing odds for non-European leagues.

Companion to load_football_data.py. football-data.co.uk publishes
non-European league CSVs under /new/<COUNTRY>.csv — one big file per
country covering many seasons. The column shape differs slightly from
the EU per-season files (Home/Away vs HomeTeam/AwayTeam, HG vs FTHG,
PH/PD/PA Pinnacle vs PSCH/PSCD/PSCA, no Bet365 closing).

This script:
  1. Pulls each requested country CSV.
  2. Renames columns to match the EU raw_football_data shape so a
     single staging table holds both sources.
  3. Inserts into raw_football_data with the appropriate league code
     and season label so promote_raw.py picks them up unchanged.

Country codes (football-data.co.uk file name → league code stored
in raw_football_data.league, matched against promote_raw.LEAGUE_MAP):

    ARG -> AR1   Argentina Primera División
    AUT -> AT1   Austria Bundesliga
    BRA -> BR1   Brasileirão Série A
    CHN -> CN1   Chinese Super League
    DNK -> DK1   Danish Superliga
    FIN -> FI1   Finnish Veikkausliiga
    IRL -> IE1   Irish Premier Division
    JPN -> JP1   J1 League
    MEX -> MX1   Liga MX
    NOR -> NO1   Eliteserien
    POL -> PL1   Polish Ekstraklasa
    ROU -> RO1   Liga I
    RUS -> RU1   Russian Premier League
    SWE -> SE1   Allsvenskan
    SWZ -> CH1   Swiss Super League
    USA -> MLS   Major League Soccer

Usage:
    # Default: pull all known countries
    python scripts/load_football_data_extra.py

    # Specific countries
    python scripts/load_football_data_extra.py --countries USA,BRA,ARG

    # CSV-only dump (no DB write)
    python scripts/load_football_data_extra.py --output data/raw/extra/

Run after load_football_data.py + before promote_raw.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("load_football_data_extra")

BASE_URL = "https://www.football-data.co.uk/new"

# country code -> (league code, league name, country name)
COUNTRY_MAP: dict[str, tuple[str, str, str]] = {
    "ARG": ("AR1", "Primera División", "Argentina"),
    "AUT": ("AT1", "Bundesliga", "Austria"),
    "BRA": ("BR1", "Brasileirão Série A", "Brazil"),
    "CHN": ("CN1", "Chinese Super League", "China"),
    "DNK": ("DK1", "Superliga", "Denmark"),
    "FIN": ("FI1", "Veikkausliiga", "Finland"),
    "IRL": ("IE1", "Premier Division", "Ireland"),
    "JPN": ("JP1", "J1 League", "Japan"),
    "MEX": ("MX1", "Liga MX", "Mexico"),
    "NOR": ("NO1", "Eliteserien", "Norway"),
    "POL": ("PL1", "Ekstraklasa", "Poland"),
    "ROU": ("RO1", "Liga I", "Romania"),
    "RUS": ("RU1", "Premier League", "Russia"),
    "SWE": ("SE1", "Allsvenskan", "Sweden"),
    "SWZ": ("CH1", "Super League", "Switzerland"),
    "USA": ("MLS", "MLS", "USA"),
}

# Translate the extra-leagues CSV columns to the EU schema so a single
# staging table holds both sources. Values not present in extra files
# simply remain unmapped → columns missing from the resulting frame.
COLUMN_MAP: dict[str, str] = {
    "Home": "HomeTeam",
    "Away": "AwayTeam",
    "HG": "FTHG",
    "AG": "FTAG",
    "Res": "FTR",
    # Pinnacle in extra files is "PH/PD/PA"; treat as closing odds to
    # match the EU "PSCH/PSCD/PSCA" columns.
    "PH": "PSCH",
    "PD": "PSCD",
    "PA": "PSCA",
    # Market average and max are common to both formats.
    "AvgH": "AvgH",
    "AvgD": "AvgD",
    "AvgA": "AvgA",
    "MaxH": "MaxH",
    "MaxD": "MaxD",
    "MaxA": "MaxA",
    # Over/under 2.5 — keep if present.
    "AvgCAHH": "Avg>2.5",
    "AvgCAHA": "Avg<2.5",
}

# Keep only columns we care about post-rename (mirrors load_football_data.py).
KEEP_COLUMNS = [
    "Date",
    "Time",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "PSCH",
    "PSCD",
    "PSCA",
    "AvgH",
    "AvgD",
    "AvgA",
    "MaxH",
    "MaxD",
    "MaxA",
    "Avg>2.5",
    "Avg<2.5",
    "league",
    "season",
]


def fetch_country(code: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{code}.csv"
    logger.info("Fetching %s", url)
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        raise FileNotFoundError(f"No file for {code} (HTTP 404)")
    r.raise_for_status()
    text = r.content.decode("latin-1")
    df = pd.read_csv(StringIO(text))
    df.columns = [c.strip() for c in df.columns]
    return df


def normalize(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """Rename extra-league columns to match the EU raw_football_data shape."""
    league_code, _, _ = COUNTRY_MAP[country]

    rename = {src: dst for src, dst in COLUMN_MAP.items() if src in df.columns}
    df = df.rename(columns=rename)

    # `league` and `season` are tagged per-row so promote_raw can group
    # rows back into the right leagues table.
    df["league"] = league_code
    if "Season" in df.columns:
        df["season"] = df["Season"]
    elif "season" not in df.columns:
        df["season"] = ""

    # Drop rows missing essentials.
    if "HomeTeam" in df.columns and "AwayTeam" in df.columns and "Date" in df.columns:
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"])
    else:
        logger.warning("%s file missing core columns; skipping", country)
        return pd.DataFrame()

    keep = [c for c in KEEP_COLUMNS if c in df.columns]
    return df[keep]


def load_all(countries: list[str]) -> list[pd.DataFrame]:
    out: list[pd.DataFrame] = []
    for code in countries:
        if code not in COUNTRY_MAP:
            logger.warning("Unknown country code: %s — skipping", code)
            continue
        try:
            df = fetch_country(code)
        except FileNotFoundError:
            logger.warning("No data for %s", code)
            continue
        df = normalize(df, code)
        if df.empty:
            continue
        out.append(df)
        logger.info("Loaded %s: %d rows", code, len(df))
    return out


def write_csv(frames: list[pd.DataFrame], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    for df in frames:
        code = df["league"].iloc[0]
        df.to_csv(output_dir / f"{code}.csv", index=False)
    return len(frames)


def write_database(frames: list[pd.DataFrame], database_url: str) -> int:
    """Insert into raw_football_data with whatever schema is already there.
    The first run of load_football_data.py creates the table with the EU
    column set; missing columns in extra rows become NULL.
    """
    import psycopg2
    from psycopg2.extras import execute_values

    if not frames:
        return 0
    all_df = pd.concat(frames, ignore_index=True)

    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            # Discover the actual columns in raw_football_data so we
            # only insert what the table already has.
            cur.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'raw_football_data'
            """
            )
            existing = {r[0] for r in cur.fetchall()}
            if not existing:
                raise RuntimeError(
                    "raw_football_data table doesn't exist. Run " "scripts/load_football_data.py first to create it."
                )

            cols = [c for c in all_df.columns if c in existing]
            quoted_cols = ['"' + c + '"' for c in cols]
            insert_sql = f"INSERT INTO raw_football_data ({', '.join(quoted_cols)}) " "VALUES %s ON CONFLICT DO NOTHING"
            rows = [tuple(None if pd.isna(v) else str(v) for v in row) for row in all_df[cols].itertuples(index=False)]
            execute_values(cur, insert_sql, rows)
            conn.commit()
            return cur.rowcount


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--countries", default=",".join(COUNTRY_MAP.keys()), help="Comma-separated country codes (default: all known)."
    )
    p.add_argument("--output", type=Path, default=None, help="Optional output dir for raw normalized CSVs.")
    p.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL"), help="Postgres URL. Pass empty to skip DB load."
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    codes = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    frames = load_all(codes)
    if not frames:
        logger.error("No data loaded for countries=%s", codes)
        return 1

    total = sum(len(df) for df in frames)
    logger.info("Loaded %d total rows across %d countries", total, len(frames))

    if args.output:
        n = write_csv(frames, args.output)
        logger.info("Wrote %d files to %s", n, args.output)

    if args.database_url:
        inserted = write_database(frames, args.database_url)
        logger.info("Inserted %d new rows into raw_football_data", inserted)
    else:
        logger.info("--database-url not set; skipped DB write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
