"""Backfill NHL historical games and team stats from the NHL Stats API.

The NHL Stats API (https://api-web.nhle.com/v1, no key required) is the
canonical source for finished NHL games. We use two endpoints:

  * /club-schedule-season/{tri-code}/{season-id}
        Enumerate every game in a team-season. Each game appears under
        both teams' schedules, so we dedupe by game ID.
  * /gamecenter/{game-id}/right-rail
        Team-level stats (SOG, faceoffs, PP, hits, blocks, giveaways,
        takeaways, PIM) and the period-by-period linescore (used to
        derive regulation scores and OT/SO tiebreak flags).

Idempotent: matches.{home_team_id, away_team_id, match_date} is the
upsert key for matches, and (match_id, team_id) is unique on
nhl_match_stats. Re-running picks up missing games and overwrites
stat rows for games already loaded.

Usage:
    # Single season (good for sanity checks)
    python scripts/load_nhl_historical.py --seasons 2024-2025

    # Full roadmap backfill (10 seasons, ~13k games, ~1h with throttle)
    python scripts/load_nhl_historical.py \\
        --seasons 2015-2016,2016-2017,2017-2018,2018-2019,2019-2020,\\
2020-2021,2021-2022,2022-2023,2023-2024,2024-2025

    # Skip games already in nhl_match_stats (default), or refetch all
    python scripts/load_nhl_historical.py --seasons 2024-2025 --refetch
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import psycopg2
import requests
from psycopg2.extras import Json, RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("load_nhl_historical")

NHL_API = "https://api-web.nhle.com/v1"

# NHL team tri-codes used by the API. Includes every franchise active
# at any point in the 10-season window (2015-16 → 2024-25):
#   * The 30 teams pre-Vegas
#   * VGK (2017-18 onwards)
#   * SEA (2021-22 onwards)
#   * UTA, the Utah franchise that replaced Arizona in 2024-25
# Inactive (or not-yet-active) teams in a given season's schedule call
# return an empty `games` array, which we treat as a no-op.
NHL_TEAM_CODES = [
    "ANA", "ARI", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI",
    "COL", "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL",
    "NJD", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA",
    "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG",
    "WSH",
]

# gameType values from the schedule feed.
GAME_TYPE_PRESEASON = 1
GAME_TYPE_REGULAR = 2
GAME_TYPE_PLAYOFF = 3

# gameState values that mean the game is final and stats are settled.
FINAL_STATES = {"FINAL", "OFF", "OFFICIAL"}

# Be polite — NHL doesn't publish rate limits but we still don't want
# to hammer them. ~3 req/s is the working number used by the nhl-api
# Python clients in the wild.
REQUEST_DELAY_SEC = 0.3


@dataclass(frozen=True)
class TeamGameStats:
    """Subset of /right-rail teamGameStats + linescore we persist.

    `power_play_goals` / `power_play_opps` come from the API's
    "powerPlay" formatted string "G/O". `penalty_kill_*` are derived
    from the opponent's power-play numbers (opp PP opps = our PK opps,
    opp PP goals = goals against us short-handed).
    """

    shots_on_goal: int
    power_play_goals: int
    power_play_opps: int
    power_play_pct: float | None
    pim: int
    hits: int
    blocked_shots: int
    giveaways: int
    takeaways: int
    faceoffs_won: int
    faceoffs_taken: int
    faceoff_win_pct: float | None
    period_scores: dict[str, int]


def season_id(season_str: str) -> str:
    """'2024-2025' -> '20242025' (the NHL API's season-id format)."""
    parts = season_str.split("-")
    if (
        len(parts) != 2
        or len(parts[0]) != 4
        or len(parts[1]) != 4
        or not parts[0].isdigit()
        or not parts[1].isdigit()
    ):
        raise ValueError(f"Bad --seasons entry: {season_str!r} (expected YYYY-YYYY)")
    return parts[0] + parts[1]


def get_json(url: str) -> dict | None:
    """Polite GET with throttle. Returns None on 404 (skip silently)."""
    time.sleep(REQUEST_DELAY_SEC)
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException as e:
        logger.warning("GET %s failed: %s", url, e)
        return None
    if r.status_code == 404:
        return None
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        logger.warning("GET %s -> HTTP %d: %s", url, r.status_code, e)
        return None
    try:
        return r.json()
    except ValueError:
        logger.warning("GET %s -> non-JSON body", url)
        return None


def parse_wins_taken(value: str | None) -> tuple[int, int]:
    """Parse the API's "wins/taken" formatted strings like "34/59" or
    "2/4". Returns (0, 0) on malformed input — we still record the
    game even if one stat string is missing."""
    if not value or "/" not in value:
        return 0, 0
    try:
        w, t = value.split("/", 1)
        return int(w), int(t)
    except ValueError:
        return 0, 0


def parse_team_game_stats(rr_payload: dict, side: str) -> TeamGameStats:
    """Extract one team's TeamGameStats from a /right-rail response.

    `side` is 'awayValue' or 'homeValue' — the field name the right-rail
    feed uses for each team's value in the teamGameStats list.
    """
    stats = {row["category"]: row.get(side) for row in rr_payload.get("teamGameStats") or []}

    pp_goals, pp_opps = parse_wins_taken(stats.get("powerPlay"))
    fo_wins, fo_taken = parse_wins_taken(stats.get("faceoffWins"))

    # Period scores: keyed by period number ("1", "2", "3", "OT") so the
    # NHL feature pipeline can derive regulation totals without re-reading
    # the right-rail payload. SO goals aren't included in the linescore —
    # they only affect the final score and we capture that via
    # matches.{home,away}_score + matches.metadata.period_type_final.
    period_scores: dict[str, int] = {}
    linescore = rr_payload.get("linescore") or {}
    score_key = "home" if side == "homeValue" else "away"
    for period in linescore.get("byPeriod") or []:
        desc = period.get("periodDescriptor") or {}
        ptype = desc.get("periodType")
        pnum = desc.get("number")
        # Regular periods: "1", "2", "3". Overtime: "OT".
        if ptype == "OT":
            key = "OT"
        elif pnum is not None:
            key = str(pnum)
        else:
            continue
        period_scores[key] = int(period.get(score_key) or 0)

    return TeamGameStats(
        shots_on_goal=int(stats.get("sog") or 0),
        power_play_goals=pp_goals,
        power_play_opps=pp_opps,
        power_play_pct=float(stats["powerPlayPctg"]) if stats.get("powerPlayPctg") is not None else None,
        pim=int(stats.get("pim") or 0),
        hits=int(stats.get("hits") or 0),
        blocked_shots=int(stats.get("blockedShots") or 0),
        giveaways=int(stats.get("giveaways") or 0),
        takeaways=int(stats.get("takeaways") or 0),
        faceoffs_won=fo_wins,
        faceoffs_taken=fo_taken,
        faceoff_win_pct=float(stats["faceoffWinningPctg"]) if stats.get("faceoffWinningPctg") is not None else None,
        period_scores=period_scores,
    )


def regulation_outcome(home_periods: dict[str, int], away_periods: dict[str, int]) -> tuple[int, int, str]:
    """Sum periods 1-3 only (exclude OT/SO) to get the regulation score,
    then label the regulation winner. Used downstream for both the
    moneyline target (incl. OT/SO winner via matches.{home,away}_score)
    and the 60-minute regulation 3-way target."""
    reg_home = sum(home_periods.get(str(p), 0) for p in (1, 2, 3))
    reg_away = sum(away_periods.get(str(p), 0) for p in (1, 2, 3))
    if reg_home > reg_away:
        winner = "home"
    elif reg_away > reg_home:
        winner = "away"
    else:
        winner = "tie"
    return reg_home, reg_away, winner


def ensure_nhl_league(cur) -> str:
    cur.execute(
        """
        INSERT INTO leagues (name, country, sport, external_ids)
        VALUES ('NHL', 'USA', 'nhl', %s)
        ON CONFLICT (name, country, sport) DO UPDATE
            SET external_ids = leagues.external_ids || EXCLUDED.external_ids
        RETURNING id
        """,
        (Json({"nhl_api": "NHL"}),),
    )
    return cur.fetchone()["id"]


def ensure_nhl_team(cur, team_payload: dict, league_id: str) -> str | None:
    """Resolve an NHL team payload (from schedule's awayTeam/homeTeam) to
    a teams.id. NHL names are canonical, so we skip the soccer fuzzy
    path and rely on (normalized_name, sport='nhl') uniqueness."""
    place = (team_payload.get("placeName") or {}).get("default")
    common = (team_payload.get("commonName") or {}).get("default")
    abbrev = team_payload.get("abbrev")
    nhl_id = team_payload.get("id")
    if not (place and common):
        return None
    full = f"{place} {common}"
    norm = " ".join(full.strip().lower().split())
    cur.execute(
        """
        INSERT INTO teams (name, normalized_name, league_id, sport, external_ids)
        VALUES (%s, %s, %s, 'nhl', %s)
        ON CONFLICT (normalized_name, sport) DO UPDATE
            SET league_id = COALESCE(teams.league_id, EXCLUDED.league_id),
                external_ids = teams.external_ids || EXCLUDED.external_ids
        RETURNING id
        """,
        (full, norm, league_id, Json({"nhl_api_id": nhl_id, "nhl_api_abbrev": abbrev})),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def upsert_match(cur, game: dict, league_id: str, home_id: str, away_id: str,
                 reg_home: int, reg_away: int, reg_winner: str, season: str) -> str | None:
    """Upsert one game's row in `matches`. Returns the match id.

    Final score = NHL `awayTeam.score` / `homeTeam.score` (includes OT
    and SO goals). Regulation scores + final-period type land in
    matches.metadata for the feature pipeline.
    """
    try:
        match_dt = datetime.fromisoformat((game["startTimeUTC"]).replace("Z", "+00:00"))
    except (KeyError, ValueError):
        return None

    home_score = (game.get("homeTeam") or {}).get("score")
    away_score = (game.get("awayTeam") or {}).get("score")
    period_type = (game.get("gameOutcome") or {}).get("lastPeriodType") or "REG"
    venue = (game.get("venue") or {}).get("default")
    game_type = "playoff" if game.get("gameType") == GAME_TYPE_PLAYOFF else "regular"

    metadata = {
        "regulation_home_score": reg_home,
        "regulation_away_score": reg_away,
        "regulation_winner": reg_winner,
        "period_type_final": period_type,
        "game_type": game_type,
        "nhl_game_id": game.get("id"),
    }

    cur.execute(
        """
        INSERT INTO matches (
            league_id, home_team_id, away_team_id, match_date, status,
            home_score, away_score, season, venue, external_ids, metadata
        )
        VALUES (%s, %s, %s, %s, 'finished', %s, %s, %s, %s, %s, %s)
        ON CONFLICT (home_team_id, away_team_id, match_date) DO UPDATE
            SET status = 'finished',
                home_score = EXCLUDED.home_score,
                away_score = EXCLUDED.away_score,
                venue = COALESCE(matches.venue, EXCLUDED.venue),
                external_ids = matches.external_ids || EXCLUDED.external_ids,
                metadata = matches.metadata || EXCLUDED.metadata,
                updated_at = NOW()
        RETURNING id
        """,
        (
            league_id, home_id, away_id, match_dt,
            home_score, away_score, season, venue,
            Json({"nhl": str(game.get("id"))}),
            Json(metadata),
        ),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def upsert_team_stats(cur, match_id: str, team_id: str, stats: TeamGameStats,
                      opp_stats: TeamGameStats) -> None:
    """Persist one team's NHL stats. Goalie-side numbers (saves, save_pct,
    goals_against) are derived from the opponent's shots and goals — the
    right-rail feed doesn't expose them as team-level stats directly."""
    goals = sum(stats.period_scores.values())
    goals_against = sum(opp_stats.period_scores.values())
    shots_faced = opp_stats.shots_on_goal
    saves = max(shots_faced - goals_against, 0)
    save_pct = (saves / shots_faced) if shots_faced > 0 else None
    pk_pct = (
        1.0 - (opp_stats.power_play_goals / opp_stats.power_play_opps)
        if opp_stats.power_play_opps > 0
        else None
    )
    cur.execute(
        """
        INSERT INTO nhl_match_stats (
            match_id, team_id,
            shots_on_goal, goals,
            saves, save_pct, goals_against,
            power_play_goals, power_play_opportunities, power_play_pct,
            penalty_kill_goals_against, penalty_kill_opportunities, penalty_kill_pct,
            hits, blocked_shots,
            faceoffs_won, faceoffs_taken, faceoff_win_pct,
            giveaways, takeaways, pim,
            period_scores
        )
        VALUES (%s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s)
        ON CONFLICT (match_id, team_id) DO UPDATE SET
            shots_on_goal = EXCLUDED.shots_on_goal,
            goals = EXCLUDED.goals,
            saves = EXCLUDED.saves,
            save_pct = EXCLUDED.save_pct,
            goals_against = EXCLUDED.goals_against,
            power_play_goals = EXCLUDED.power_play_goals,
            power_play_opportunities = EXCLUDED.power_play_opportunities,
            power_play_pct = EXCLUDED.power_play_pct,
            penalty_kill_goals_against = EXCLUDED.penalty_kill_goals_against,
            penalty_kill_opportunities = EXCLUDED.penalty_kill_opportunities,
            penalty_kill_pct = EXCLUDED.penalty_kill_pct,
            hits = EXCLUDED.hits,
            blocked_shots = EXCLUDED.blocked_shots,
            faceoffs_won = EXCLUDED.faceoffs_won,
            faceoffs_taken = EXCLUDED.faceoffs_taken,
            faceoff_win_pct = EXCLUDED.faceoff_win_pct,
            giveaways = EXCLUDED.giveaways,
            takeaways = EXCLUDED.takeaways,
            pim = EXCLUDED.pim,
            period_scores = EXCLUDED.period_scores
        """,
        (
            match_id, team_id,
            stats.shots_on_goal, goals,
            saves, save_pct, goals_against,
            stats.power_play_goals, stats.power_play_opps, stats.power_play_pct,
            opp_stats.power_play_goals, opp_stats.power_play_opps, pk_pct,
            stats.hits, stats.blocked_shots,
            stats.faceoffs_won, stats.faceoffs_taken, stats.faceoff_win_pct,
            stats.giveaways, stats.takeaways, stats.pim,
            Json(stats.period_scores),
        ),
    )


def already_loaded_game_ids(cur, season: str) -> set[int]:
    """Return NHL game IDs already in nhl_match_stats for this season.
    Used to skip re-fetching boxscores on resumed runs."""
    cur.execute(
        """
        SELECT DISTINCT (m.external_ids->>'nhl')::bigint AS nhl_id
        FROM matches m
        JOIN nhl_match_stats s ON s.match_id = m.id
        JOIN leagues l ON l.id = m.league_id
        WHERE l.sport = 'nhl' AND m.season = %s
          AND m.external_ids ? 'nhl'
        """,
        (season,),
    )
    return {row["nhl_id"] for row in cur.fetchall() if row.get("nhl_id") is not None}


def discover_games(season_str: str) -> dict[int, dict]:
    """Walk every team's season schedule and return {game_id: game} for
    regular-season + playoff games in FINAL state. Inactive teams in
    that season silently no-op."""
    games: dict[int, dict] = {}
    sid = season_id(season_str)
    for tri in NHL_TEAM_CODES:
        payload = get_json(f"{NHL_API}/club-schedule-season/{tri}/{sid}")
        if not payload:
            continue
        for g in payload.get("games") or []:
            if g.get("gameType") not in (GAME_TYPE_REGULAR, GAME_TYPE_PLAYOFF):
                continue
            if (g.get("gameState") or "").upper() not in FINAL_STATES:
                continue
            gid = g.get("id")
            if gid is None:
                continue
            games.setdefault(int(gid), g)
        logger.info("season %s team %s: %d cumulative finished games", season_str, tri, len(games))
    return games


# How often to log progress inside the per-game loop. The discovery
# step is chatty enough on its own; here we just need a heartbeat every
# ~30s of work (0.3s/req × 100 games ≈ 30s) so a multi-season run
# doesn't look hung between season-boundary banners.
GAME_LOG_INTERVAL = 100


def load_season(database_url: str, season_str: str, refetch: bool) -> dict[str, int]:
    counts = {"games_seen": 0, "games_loaded": 0, "games_skipped": 0, "games_failed": 0}
    logger.info("=== season %s: discovering games ===", season_str)
    games = discover_games(season_str)
    counts["games_seen"] = len(games)
    if not games:
        logger.info("=== season %s: no games discovered, skipping ===", season_str)
        return counts

    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            league_id = ensure_nhl_league(cur)
            already = set() if refetch else already_loaded_game_ids(cur, season_str)
            total = len(games)
            logger.info(
                "=== season %s: %d games to consider (%d already loaded) ===",
                season_str, total, len(already),
            )

            for processed, (gid, game) in enumerate(games.items(), start=1):
                if gid in already:
                    counts["games_skipped"] += 1
                else:
                    rr = get_json(f"{NHL_API}/gamecenter/{gid}/right-rail")
                    if rr is None:
                        counts["games_failed"] += 1
                    else:
                        try:
                            home_stats = parse_team_game_stats(rr, "homeValue")
                            away_stats = parse_team_game_stats(rr, "awayValue")
                            reg_home, reg_away, reg_winner = regulation_outcome(
                                home_stats.period_scores, away_stats.period_scores
                            )
                            home_team_id = ensure_nhl_team(cur, game["homeTeam"], league_id)
                            away_team_id = ensure_nhl_team(cur, game["awayTeam"], league_id)
                            if not (home_team_id and away_team_id):
                                counts["games_failed"] += 1
                            else:
                                match_id = upsert_match(
                                    cur, game, league_id, home_team_id, away_team_id,
                                    reg_home, reg_away, reg_winner, season_str,
                                )
                                if not match_id:
                                    counts["games_failed"] += 1
                                else:
                                    upsert_team_stats(cur, match_id, home_team_id, home_stats, away_stats)
                                    upsert_team_stats(cur, match_id, away_team_id, away_stats, home_stats)
                                    counts["games_loaded"] += 1
                                    # Commit per game so a crash mid-season leaves a
                                    # usable partial backfill instead of an aborted
                                    # transaction.
                                    conn.commit()
                        except Exception as e:  # noqa: BLE001 — one bad game shouldn't kill the season
                            logger.warning("game %s failed: %s", gid, e)
                            counts["games_failed"] += 1

                # Heartbeat so a long silent loop looks like progress
                # instead of a hang. Fires on the interval and on the
                # final game so the season-end count is always visible.
                if processed % GAME_LOG_INTERVAL == 0 or processed == total:
                    logger.info(
                        "season %s: processed %d/%d (loaded=%d skipped=%d failed=%d)",
                        season_str, processed, total,
                        counts["games_loaded"], counts["games_skipped"], counts["games_failed"],
                    )

    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--seasons",
        required=True,
        help="Comma-separated season strings (e.g. 2024-2025,2023-2024).",
    )
    p.add_argument(
        "--refetch",
        action="store_true",
        help="Re-fetch right-rail stats for games already in nhl_match_stats.",
    )
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.database_url:
        logger.error("DATABASE_URL not set and --database-url not provided")
        return 2
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    for season in seasons:
        counts = load_season(args.database_url, season, args.refetch)
        logger.info("season %s: %s", season, counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
