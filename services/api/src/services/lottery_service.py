"""Lottery analysis + combination-generation service.

Wraps the pure engine in services/lottery_analysis.py with DB access: loads
historical draws, generates scored combinations, and (optionally) persists them
to lottery_predictions for honest backtesting. Read the framing in
lottery_analysis — this service never claims to forecast a draw; `score` ranks
lines by statistical-profile fit and expected value (jackpot-share avoidance),
not by any (impossible) win probability.
"""

import json
import logging
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import httpx
from services.lottery_analysis import DISCLAIMER, MIN_DRAWS_FOR_STATS
from services.lottery_analysis import analyze as engine_analyze
from services.lottery_analysis import generate_combinations, get_game_config
from services.lottery_ev import ev_report
from services.lottery_live import (
    JACKPOT_SOURCES,
    REQUEST_HEADERS,
    parse_megamillions_payload,
    parse_powerball_api,
    parse_powerball_homepage,
)
from services.lottery_rules import DRAW_WEEKDAYS, analysis_window_starts
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Live next-draw jackpot estimates are best-effort: the EV endpoint falls
# back to an explicit `jackpot` query param when unreachable. Successes are
# cached in-process for 10 minutes; failures for only 60s — powerball.com's
# CDN serves an occasional undecodable/odd variant (some edge nodes return
# brotli regardless of Accept-Encoding), and a flake shouldn't blank the EV
# card for 10 minutes. Parsers + source quirks live in lottery_live.
_JACKPOT_CACHE_TTL_S = 600
_JACKPOT_NEGATIVE_TTL_S = 60
_jackpot_cache: Dict[str, Tuple[float, Dict]] = {}


def _fetch_jackpot_once(game: str) -> Optional[Dict[str, float]]:
    resp = httpx.get(JACKPOT_SOURCES[game], headers=REQUEST_HEADERS, timeout=8.0, follow_redirects=True)
    resp.raise_for_status()
    if game != "powerball":
        return parse_megamillions_payload(resp.text)
    # Powerball two-step: the JSON API serves the SPA homepage HTML to
    # non-browser clients (CDN bot-gating); when the JSON parse comes up
    # empty, scrape the jackpot from the homepage markup we DO receive.
    result = parse_powerball_api(resp.text)
    if result is None:
        page = httpx.get(
            JACKPOT_SOURCES["powerball_fallback_page"],
            headers=REQUEST_HEADERS,
            timeout=8.0,
            follow_redirects=True,
        )
        page.raise_for_status()
        result = parse_powerball_homepage(page.text)
    return result


def fetch_live_jackpot(game: str) -> Optional[Dict[str, float]]:
    """Current next-draw estimated jackpot {advertised, cash_value} or None.
    Never raises — a missing live estimate degrades to requiring the caller
    to pass the jackpot explicitly. One retry on a failed attempt: the CDN
    variance is per-request, so a fresh request usually lands a good node."""
    cached = _jackpot_cache.get(game)
    if cached:
        age = time.monotonic() - cached[0]
        ttl = _JACKPOT_CACHE_TTL_S if cached[1] else _JACKPOT_NEGATIVE_TTL_S
        if age < ttl:
            return cached[1] or None
    result: Optional[Dict[str, float]] = None
    for attempt in (1, 2):
        try:
            result = _fetch_jackpot_once(game)
        except Exception as e:
            logger.warning("Live jackpot fetch failed for %s (attempt %d): %s", game, attempt, e)
        if result is not None:
            break
    if result is None:
        logger.warning("Live jackpot unavailable for %s after retries", game)
    _jackpot_cache[game] = (time.monotonic(), result or {})
    return result


class LotteryService:
    """Service for lottery analysis and combination generation."""

    def __init__(self, db: Session):
        self.db = db

    def _load_draws(self, game: str, limit: int) -> Tuple[List[List[int]], List[int]]:
        """Most-recent-first main-number lists and bonus numbers for a game.

        Era-aware: mains only from draws whose MAIN matrix matches current
        rules, bonus only from the current bonus-pool era. Mixing matrices
        poisons frequency stats — e.g. Mega Millions megaball 25 exists in
        pre-Apr-2025 draws but is not in today's 1-24 pool."""
        main_start, bonus_start = analysis_window_starts(game)
        rows = self.db.execute(
            text(
                """
                SELECT numbers, bonus_number, draw_date
                FROM lottery_draws
                WHERE game = :game AND draw_date >= :main_start
                ORDER BY draw_date DESC
                LIMIT :limit
                """
            ),
            {"game": game, "main_start": main_start, "limit": limit},
        ).fetchall()
        main = [list(r.numbers) for r in rows]
        bonus = [r.bonus_number for r in rows if r.draw_date >= bonus_start]
        return main, bonus

    def analyze(self, game: str, num_draws: int = 100) -> Optional[Dict]:
        """Hot/cold/overdue + historical profile. None if no draws."""
        cfg = get_game_config(game)
        main, bonus = self._load_draws(game, num_draws)
        if not main:
            return None
        result = engine_analyze(main, bonus, cfg)
        result["game"] = game
        return result

    def next_draw_date(self, game: str, today: Optional[date] = None) -> Optional[date]:
        """Next scheduled draw date after `today`, for tagging persisted lines."""
        today = today or date.today()
        days = DRAW_WEEKDAYS.get(game, set())
        if not days:
            return None
        for i in range(1, 8):
            d = today + timedelta(days=i)
            if d.weekday() in days:
                return d
        return None

    def recommend(
        self,
        game: str,
        strategy: str = "blend",
        num_sets: int = 5,
        *,
        persist: bool = False,
        user_id: Optional[str] = None,
        seed: Optional[int] = None,
        window: int = 200,
    ) -> Dict:
        """Generate `num_sets` scored combinations; optionally persist them."""
        cfg = get_game_config(game)
        main, bonus = self._load_draws(game, window)
        combos = generate_combinations(main, bonus, cfg, strategy=strategy, n=num_sets, seed=seed)

        if persist and combos:
            self._persist(game, strategy, combos, user_id)

        warnings: List[str] = []
        if len(main) < MIN_DRAWS_FOR_STATS and strategy not in ("ev", "random"):
            warnings.append(
                f"Only {len(main)} current-era draws on record (< {MIN_DRAWS_FOR_STATS}): "
                "hot/due/profile statistics are neutral, so this strategy's ranking is "
                "effectively EV-only (unpopular high numbers). Ingest draw history or use "
                "the 'ev'/'random' strategies explicitly."
            )

        return {
            "game": game,
            "strategy": strategy,
            "total_draws_analyzed": len(main),
            "generated_at": datetime.utcnow(),
            "combinations": [c.to_dict() for c in combos],
            "warnings": warnings,
            "disclaimer": DISCLAIMER,
        }

    def ev(
        self,
        game: str,
        *,
        jackpot: Optional[float] = None,
        state_tax: float = 0.0,
    ) -> Optional[Dict]:
        """Per-ticket EV verdict for the next draw. Uses the live advertised
        jackpot + cash value when `jackpot` isn't supplied; returns None if
        neither is available (route turns that into a 422 asking for the
        param)."""
        if jackpot is not None:
            advertised = jackpot
            # Live cash value only applies to the live jackpot, not an override.
            cash_value = None
        else:
            live = fetch_live_jackpot(game)
            if not live:
                return None
            advertised = live["advertised"]
            cash_value = live.get("cash_value") or None
        report = ev_report(game, advertised, cash_value=cash_value, state_tax=state_tax)
        report["jackpot_source"] = "live" if jackpot is None else "user"
        report["next_draw_date"] = self.next_draw_date(game)
        report["disclaimer"] = DISCLAIMER
        return report

    def _persist(self, game: str, strategy: str, combos: Sequence, user_id: Optional[str]) -> None:
        target = self.next_draw_date(game)
        query = text(
            """
            INSERT INTO lottery_predictions
            (game, strategy, numbers, bonus_number, score, features, rationale,
             target_draw_date, user_id)
            VALUES (:game, :strategy, :numbers, :bonus, :score,
                    CAST(:features AS jsonb), :rationale, :target,
                    CAST(:user_id AS uuid))
            """
        )
        try:
            for c in combos:
                self.db.execute(
                    query,
                    {
                        "game": game,
                        "strategy": strategy,
                        "numbers": c.numbers,
                        "bonus": c.bonus_number,
                        "score": c.score,
                        "features": json.dumps(c.features),
                        "rationale": c.rationale,
                        "target": target,
                        "user_id": user_id,
                    },
                )
            self.db.commit()
        except Exception:
            # Loud failure by charter (audit non-negotiable #3). The old
            # swallow-and-rollback here hid a missing lottery_predictions
            # table on prod for over a year — persist=true "succeeded" while
            # writing nothing.
            logger.exception("Failed to persist lottery predictions (game=%s strategy=%s)", game, strategy)
            self.db.rollback()
            raise
