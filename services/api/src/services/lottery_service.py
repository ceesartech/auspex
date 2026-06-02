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
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from services.lottery_analysis import DISCLAIMER
from services.lottery_analysis import analyze as engine_analyze
from services.lottery_analysis import generate_combinations, get_game_config
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Scheduled draw weekdays (Mon=0 … Sun=6): Powerball Mon/Wed/Sat, MM Tue/Fri.
DRAW_WEEKDAYS: Dict[str, set] = {
    "powerball": {0, 2, 5},
    "mega_millions": {1, 4},
}


class LotteryService:
    """Service for lottery analysis and combination generation."""

    def __init__(self, db: Session):
        self.db = db

    def _load_draws(self, game: str, limit: int) -> Tuple[List[List[int]], List[int]]:
        """Most-recent-first main-number lists and bonus numbers for a game."""
        rows = self.db.execute(
            text(
                """
                SELECT numbers, bonus_number
                FROM lottery_draws
                WHERE game = :game
                ORDER BY draw_date DESC
                LIMIT :limit
                """
            ),
            {"game": game, "limit": limit},
        ).fetchall()
        main = [list(r.numbers) for r in rows]
        bonus = [r.bonus_number for r in rows]
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

        return {
            "game": game,
            "strategy": strategy,
            "total_draws_analyzed": len(main),
            "generated_at": datetime.utcnow(),
            "combinations": [c.to_dict() for c in combos],
            "disclaimer": DISCLAIMER,
        }

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
        except Exception as e:  # pragma: no cover - persistence is best-effort
            logger.error("Failed to persist lottery predictions: %s", e)
            self.db.rollback()
