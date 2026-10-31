"""How matches.season spells a soccer season: one scheme, keyed by league.

Every loader that writes a soccer row labels its season from here, so one
league's season never carries two spellings (until 2026-10-31 fetch_upcoming
labelled every league Aug->May while promote_raw labelled football-data's
calendar-year files by the year, splitting one Brasileirao or MLS season
across '2025-2026' and '2026-2027' at July):

  * a league whose season is one calendar year: the year, '2026';
  * a league whose season straddles two years: 'YYYY-YYYY', cut in July
    (Jul-Dec -> YYYY-YYYY+1, Jan-Jun -> YYYY-1-YYYY). This is also the
    default for a league not listed here (the European leagues, the
    A-League, the UEFA club competitions).

The leagues, from ESPN's scoreboard season blocks (2025 and 2026, fetched
2026-10-31) and football-data's season labels (raw_football_data on prod):

  calendar year  MLS, Brasileirao (BR1), Argentina (AR1, from 2021),
                 Chinese Super League (CN1), Eliteserien (NO1), Allsvenskan
                 (SE1), Chile (CL1, from 2018), Colombia (CO1), Finland
                 (FI1), Ireland (IE1), K League (KR1); ESPN files each as
                 'YYYY ...' running 01-01..12-31, football-data as '2026'.
                 The international competitions are labelled by the year of
                 the kickoff (load_international's convention; the World
                 Cup, Gold Cup and Libertadores are within a year anyway).
  cross-year     Liga MX (MX1): Apertura Jul-Dec + Clausura Jan-May; ESPN
                 '2025-26 Liga BBVA MX' 2025-06-01..2026-06-01, football-data
                 '2025/2026'.
  switched       J1 (JP1): calendar years through the 2026 special season
                 (ESPN '2026-j1-100-year-vision-league', 2026-02-06 to the
                 placement play-offs of 2026-06-06), then Aug-May from
                 2026-27 (ESPN's 'regular-season' from 2026-08-08, calendar
                 to 2027-06-06). football-data's JPN file ends with '2025'.
  irregular      Argentina before 2021 (football-data '2016/2017' ..
                 '2019/2020', then '2020' for the Copa Diego Maradona of
                 Oct 2020 - Jan 2021), Chile before 2018 (Apertura /
                 Clausura across two years, then the 2017 Transicion), and
                 two Covid seasons that ran into the next year (football-data
                 BR1 '2020' to 2021-02-26, CN1 '2021' to 2022-01-12): no date
                 rule reproduces those labels, so season_for_kickoff returns
                 None there and only a source that names the season
                 (football-data's raw season) may label such a row.

FOOTBALL_DATA_YEAR_CODED is a different question, about football-data's own
season codes: which leagues' files name a season by its years ('2016',
'2016/2017') instead of the per-season European files' 'YYZZ' code ('1617').
promote_raw reads a 4-digit code from one of those leagues as a year. It
holds Austria and Liga MX too, whose seasons are cross-year: their files
spell them '2016/2017', never as four digits.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

CALENDAR = "calendar"
CROSS_YEAR = "cross_year"

_UTC = timezone.utc

# The international competitions (promote_raw.LEAGUE_MAP country
# 'International'; pinned by tests/unit/test_soccer_seasons.py).
INTERNATIONAL_CODES: frozenset[str] = frozenset(
    {"WC", "LIB", "WCQ", "EURO", "EUROQ", "UNL", "COPA", "GOLD", "AFCON", "AFCASIAN", "CONFED"}
)

# League code -> its eras, oldest first: (first kickoff of the era, scheme).
# A scheme of None is an irregular era (see the module docstring). A league
# not listed is CROSS_YEAR throughout.
SEASON_ERAS: dict[str, tuple[tuple[Optional[datetime], Optional[str]], ...]] = {
    "MLS": ((None, CALENDAR),),
    # Covid seasons that ran into the next January/February.
    "BR1": ((None, CALENDAR), (datetime(2021, 1, 1, tzinfo=_UTC), None), (datetime(2021, 3, 1, tzinfo=_UTC), CALENDAR)),
    "AR1": ((None, None), (datetime(2021, 2, 1, tzinfo=_UTC), CALENDAR)),
    "CN1": ((None, CALENDAR), (datetime(2022, 1, 1, tzinfo=_UTC), None), (datetime(2022, 2, 1, tzinfo=_UTC), CALENDAR)),
    "NO1": ((None, CALENDAR),),
    "SE1": ((None, CALENDAR),),
    "CL1": ((None, None), (datetime(2018, 1, 1, tzinfo=_UTC), CALENDAR)),
    "CO1": ((None, CALENDAR),),
    "FI1": ((None, CALENDAR),),
    "IE1": ((None, CALENDAR),),
    "KR1": ((None, CALENDAR),),
    "JP1": ((None, CALENDAR), (datetime(2026, 7, 1, tzinfo=_UTC), CROSS_YEAR)),
    "MX1": ((None, CROSS_YEAR),),
    **{code: ((None, CALENDAR),) for code in INTERNATIONAL_CODES},
}

# Leagues whose football-data files name seasons by their years: the /new/
# country files (load_football_data_extra.COUNTRY_MAP) and the internationals.
FOOTBALL_DATA_YEAR_CODED: frozenset[str] = frozenset(
    {"AR1", "AT1", "BR1", "CH1", "CN1", "DK1", "FI1", "IE1", "JP1", "MLS", "MX1", "NO1", "PL1", "RO1", "RU1", "SE1"}
    | INTERNATIONAL_CODES
)


def _utc(kickoff: datetime) -> datetime:
    """A kickoff as aware UTC (a naive one is read as UTC)."""
    return kickoff.astimezone(_UTC) if kickoff.tzinfo is not None else kickoff.replace(tzinfo=_UTC)


def cross_year_label(kickoff: datetime) -> str:
    """'YYYY-YYYY' with a July cutoff, on the UTC date. This is also what
    fetch_upcoming wrote for EVERY soccer league until 2026-10-31."""
    k = _utc(kickoff)
    return f"{k.year}-{k.year + 1}" if k.month >= 7 else f"{k.year - 1}-{k.year}"


def season_scheme(league: Optional[str], kickoff: datetime) -> Optional[str]:
    """CALENDAR, CROSS_YEAR, or None for an irregular era of that league."""
    k = _utc(kickoff)
    scheme: Optional[str] = CROSS_YEAR
    for start, era_scheme in SEASON_ERAS.get(league or "", ((None, CROSS_YEAR),)):
        if start is None or k >= start:
            scheme = era_scheme
    return scheme


def season_for_kickoff(league: Optional[str], kickoff: datetime) -> Optional[str]:
    """The season label of a `league` (football-data code) fixture kicking
    off at `kickoff`, or None when that league's season had no regular
    shape then (the label must come from a source that names it)."""
    scheme = season_scheme(league, kickoff)
    if scheme is None:
        return None
    return str(_utc(kickoff).year) if scheme == CALENDAR else cross_year_label(kickoff)
