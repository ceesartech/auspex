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

A file is refused — logged at ERROR, nothing of it written, and the run exits
3 once the other files are written — when it is neither UTF-8 nor
Windows-1252 (every /new/ file is UTF-8 with a byte-order mark; it was read
as latin-1 until 2026-11-01), when its Country column is missing or a row
names another country than the one requested (the per-season files' Div
check, load_football_data.parse_csv: on 2026-08-06 football-data served
Scottish divisions at the 2026-27 SP1/SP2 URLs), or when a team name looks
mojibake (text_encoding.looks_mojibake). A file that does not parse, or has
no Home/Away/Date columns, is logged at ERROR and counted with the refused
ones (exit 3). An unknown country code is refused before anything is fetched
(exit 2).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import text_encoding  # noqa: E402

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
    # No over/under source: AvgCAHH/AvgCAHA are the closing Asian-handicap
    # average (home/away), not over/under 2.5, and were mapped onto
    # Avg>2.5/Avg<2.5 until 2026-10-31 (no /new/ file carried them, so no row
    # was written that way). Pinned by tests/unit/test_load_football_data_extra.py.
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


class RefusedFile(RuntimeError):
    """A /new/ country file that is not loaded at all (see the module docstring)."""


# main()'s exit code when a file was refused (everything else written).
REFUSED_EXIT_CODE = 3


def parse_country_csv(raw: bytes, code: str, source: str) -> pd.DataFrame:
    """The rows of one /new/ country file, checked: decoded strictly, every
    row's Country equal to COUNTRY_MAP's country for `code`, no team name that
    looks mojibake. Raises RefusedFile."""
    try:
        text, encoding = text_encoding.decode_csv_bytes(raw, source)
    except text_encoding.UndecodableText as exc:
        raise RefusedFile(str(exc)) from None
    df = pd.read_csv(StringIO(text))
    df.columns = [c.strip() for c in df.columns]
    want = COUNTRY_MAP[code][2]
    if "Country" not in df.columns:
        raise RefusedFile(
            f"{source} ({encoding}) has no Country column (header starts {list(df.columns[:3])}) — nothing says "
            f"which country it holds, so it is not loaded as {want}"
        )
    teams = [c for c in ("Home", "Away") if c in df.columns]
    if teams:
        df = df.dropna(subset=teams, how="all")
    countries = df["Country"].fillna("").astype(str).str.strip()
    wrong = sorted(set(countries[countries != want]))
    if wrong:
        raise RefusedFile(
            f"{source} was fetched as {code} ({want}) but {int((countries != want).sum())} of its {len(df)} row(s) "
            f"name {wrong} — football-data served another country's file at that URL; not loaded"
        )
    names = pd.concat([df[c] for c in teams]).dropna().astype(str) if teams else pd.Series(dtype=str)
    garbled = sorted({n for n in names if text_encoding.looks_mojibake(n)})
    if garbled:
        raise RefusedFile(
            f"{source} ({encoding}) has team name(s) that look mis-decoded: {garbled[:5]} — not loaded (a name like "
            "that would mint a twin of the club)"
        )
    return df


def fetch_country(code: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{code}.csv"
    logger.info("Fetching %s", url)
    r = requests.get(url, timeout=60)
    if r.status_code == 404:
        raise FileNotFoundError(f"No file for {code} (HTTP 404)")
    r.raise_for_status()
    return parse_country_csv(r.content, code, url)


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

    # Drop rows missing essentials; a file without those columns is refused (it was a warning and a silent skip).
    missing = [c for c in ("HomeTeam", "AwayTeam", "Date") if c not in df.columns]
    if missing:
        raise RefusedFile(f"{country} file has no {missing} column(s) after the rename — not loaded")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"])

    keep = [c for c in KEEP_COLUMNS if c in df.columns]
    return df[keep]


def load_all(countries: list[str], refused: list[str] | None = None) -> list[pd.DataFrame]:
    """The country files of `countries` (all of them in COUNTRY_MAP: main
    refuses an unknown code first); each file refused (RefusedFile) or that
    does not parse is logged at ERROR and appended to `refused`."""
    out: list[pd.DataFrame] = []
    for code in countries:
        if code not in COUNTRY_MAP:
            raise ValueError(f"unknown country code {code!r} (COUNTRY_MAP has {sorted(COUNTRY_MAP)})")
        try:
            df = normalize(fetch_country(code), code)
        except FileNotFoundError:
            logger.warning("No data for %s", code)
            continue
        except RefusedFile as e:
            logger.error("REFUSED %s: %s", code, e)
            if refused is not None:
                refused.append(code)
            continue
        except Exception as e:
            # A malformed file (parser error etc.): logged and counted (exit 3), the other countries still load.
            logger.error("Skipping unparseable %s: %s", code, e)
            if refused is not None:
                refused.append(f"{code} (unparseable)")
            continue
        if df.empty:
            logger.warning("%s: no rows with teams and a date", code)
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
    unknown = sorted(set(codes) - set(COUNTRY_MAP))
    if unknown:
        logger.error("Unknown country code(s) %s (COUNTRY_MAP has %s) — nothing fetched", unknown, sorted(COUNTRY_MAP))
        return 2
    refused: list[str] = []
    frames = load_all(codes, refused)
    if not frames:
        logger.error("No data loaded for countries=%s (refused: %s)", codes, refused or "none")
        return REFUSED_EXIT_CODE if refused else 1

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
