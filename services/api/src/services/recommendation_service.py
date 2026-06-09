"""Recommendation service - value bet detection, Kelly criterion, accumulators"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional

from models.responses import AccumulatorResponse, MatchInfo, RecommendationResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for generating betting recommendations"""

    def __init__(self, db: Session):
        self.db = db

    def get_active_recommendations(
        self,
        confidence_level: Optional[str] = None,
        market_type: Optional[str] = None,
        min_odds: Optional[float] = None,
        max_odds: Optional[float] = None,
        limit: int = 20,
        user_id: Optional[str] = None,
    ) -> List[RecommendationResponse]:
        """Get active betting recommendations from the vw_active_recommendations view"""

        query = text(
            """
            SELECT recommendation_id, match_date, league_name,
                   home_team, away_team, bet_type, selection,
                   odds_at_recommendation, bookmaker, confidence_rating,
                   expected_value, recommended_stake, reasoning,
                   status, model_name, model_confidence
            FROM vw_active_recommendations
            WHERE (:confidence_level IS NULL OR UPPER(confidence_rating) = :confidence_level)
            AND (:market_type IS NULL OR bet_type = :market_type)
            AND (:min_odds IS NULL OR odds_at_recommendation >= :min_odds)
            AND (:max_odds IS NULL OR odds_at_recommendation <= :max_odds)
            LIMIT :limit
        """
        )

        results = self.db.execute(
            query,
            {
                # SQL above does UPPER(confidence_rating), so the bind value
                # must also be uppercased — otherwise no row matches.
                "confidence_level": confidence_level.upper() if confidence_level else None,
                "market_type": market_type,
                "min_odds": min_odds,
                "max_odds": max_odds,
                "limit": limit,
            },
        ).fetchall()

        recommendations = []
        for row in results:
            recommendations.append(
                RecommendationResponse(
                    recommendation_id=str(row.recommendation_id),
                    match_info=MatchInfo(
                        match_id=str(row.recommendation_id),
                        league_name=row.league_name,
                        home_team=row.home_team,
                        away_team=row.away_team,
                        match_date=row.match_date,
                        venue=None,
                    ),
                    market_type=row.bet_type,
                    outcome=row.selection,
                    recommended_odds=float(row.odds_at_recommendation),
                    recommended_stake=float(row.recommended_stake or 0),
                    expected_value=float(row.expected_value or 0),
                    confidence_level=row.confidence_rating or "medium",
                    reasoning=row.reasoning or "",
                    expires_at=row.match_date,
                    sport=None,
                    model_probability=(float(row.model_confidence) if row.model_confidence is not None else None),
                )
            )

        # Merge in horse-racing recs (separate schema: race_recommendations).
        # They share the RecommendationResponse shape via _horse_racing_recs.
        recommendations.extend(
            self._horse_racing_recs(
                confidence_level=confidence_level,
                market_type=market_type,
                min_odds=min_odds,
                max_odds=max_odds,
                limit=limit,
            )
        )
        recommendations.sort(key=lambda r: r.expected_value, reverse=True)
        return recommendations[:limit]

    def _horse_racing_recs(
        self,
        *,
        confidence_level: Optional[str] = None,
        market_type: Optional[str] = None,
        min_odds: Optional[float] = None,
        max_odds: Optional[float] = None,
        min_ev: Optional[float] = None,
        limit: int = 20,
    ) -> List[RecommendationResponse]:
        """Active horse-racing win recommendations, mapped into the
        unified RecommendationResponse shape so they appear in the
        general /recommendations feed alongside team sports.

        Horse racing lives in its own schema (race_recommendations +
        race_predictions + races + race_entrants + horses). The 'win'
        market is the only one generated today; a non-'win' market_type
        filter therefore yields nothing.
        """
        # Horse racing only has the 'win' market. If the caller filters
        # to a different market, there are no horse-racing matches.
        if market_type is not None and market_type != "win":
            return []

        query = text(
            """
            SELECT rec.id            AS recommendation_id,
                   r.race_date,
                   r.track_name,
                   r.race_number,
                   h.name            AS horse_name,
                   rec.odds_at_recommendation,
                   rec.confidence_rating,
                   rec.expected_value,
                   rec.recommended_stake,
                   rec.reasoning,
                   rp.confidence     AS consensus_prob
            FROM race_recommendations rec
            JOIN races r          ON r.id = rec.race_id
            JOIN race_entrants e  ON e.id = rec.entrant_id
            JOIN horses h         ON h.id = e.horse_id
            LEFT JOIN race_predictions rp ON rp.id = rec.race_prediction_id
            WHERE rec.status IN ('pending', 'placed', 'won')
              AND rec.bet_type = 'win'
              AND r.race_date > NOW()
              AND (:confidence_level IS NULL OR UPPER(rec.confidence_rating) = :confidence_level)
              AND (:min_odds IS NULL OR rec.odds_at_recommendation >= :min_odds)
              AND (:max_odds IS NULL OR rec.odds_at_recommendation <= :max_odds)
              AND (:min_ev IS NULL OR rec.expected_value >= :min_ev)
            ORDER BY rec.expected_value DESC
            LIMIT :limit
            """
        )
        rows = self.db.execute(
            query,
            {
                "confidence_level": confidence_level.upper() if confidence_level else None,
                "min_odds": min_odds,
                "max_odds": max_odds,
                "min_ev": min_ev,
                "limit": limit,
            },
        ).fetchall()

        out: List[RecommendationResponse] = []
        for row in rows:
            league = row.track_name + (f" R{row.race_number}" if row.race_number else "")
            out.append(
                RecommendationResponse(
                    recommendation_id=str(row.recommendation_id),
                    match_info=MatchInfo(
                        match_id=str(row.recommendation_id),
                        league_name=league,
                        home_team=row.horse_name,
                        away_team="Field",
                        match_date=row.race_date,
                        venue=None,
                    ),
                    market_type="win",
                    outcome=row.horse_name,
                    recommended_odds=float(row.odds_at_recommendation),
                    recommended_stake=float(row.recommended_stake or 0),
                    expected_value=float(row.expected_value or 0),
                    confidence_level=row.confidence_rating or "medium",
                    reasoning=row.reasoning or "",
                    expires_at=row.race_date,
                    sport="horse_racing",
                    model_probability=(float(row.consensus_prob) if row.consensus_prob is not None else None),
                )
            )
        return out

    def get_high_value_recommendations(
        self,
        min_ev: float = 0.05,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> List[RecommendationResponse]:
        """Get highest expected value recommendations"""

        query = text(
            """
            SELECT br.id as recommendation_id, m.match_date,
                   l.name as league_name, ht.name as home_team,
                   at.name as away_team, br.bet_type, br.selection,
                   br.odds_at_recommendation, br.bookmaker,
                   br.confidence_rating, br.expected_value,
                   br.recommended_stake, br.reasoning,
                   p.confidence as model_confidence
            FROM betting_recommendations br
            JOIN predictions p ON br.prediction_id = p.id
            JOIN matches m ON br.match_id = m.id
            JOIN leagues l ON m.league_id = l.id
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE br.status IN ('pending', 'placed')
            AND m.match_date > NOW()
            AND br.expected_value >= :min_ev
            ORDER BY br.expected_value DESC
            LIMIT :limit
        """
        )

        results = self.db.execute(query, {"min_ev": min_ev, "limit": limit}).fetchall()

        recommendations = []
        for row in results:
            recommendations.append(
                RecommendationResponse(
                    recommendation_id=str(row.recommendation_id),
                    match_info=MatchInfo(
                        match_id=str(row.recommendation_id),
                        league_name=row.league_name,
                        home_team=row.home_team,
                        away_team=row.away_team,
                        match_date=row.match_date,
                        venue=None,
                    ),
                    market_type=row.bet_type,
                    outcome=row.selection,
                    recommended_odds=float(row.odds_at_recommendation),
                    recommended_stake=float(row.recommended_stake or 0),
                    expected_value=float(row.expected_value or 0),
                    confidence_level=row.confidence_rating or "medium",
                    reasoning=row.reasoning or "",
                    expires_at=row.match_date,
                    sport=None,
                    model_probability=(float(row.model_confidence) if row.model_confidence is not None else None),
                )
            )

        # Merge in high-EV horse-racing recs so the unified "High Value
        # Bets" list spans every sport.
        recommendations.extend(self._horse_racing_recs(min_ev=min_ev, limit=limit))
        recommendations.sort(key=lambda r: r.expected_value, reverse=True)
        return recommendations[:limit]

    def build_accumulator(
        self,
        min_legs: int = 2,
        max_legs: int = 5,
        min_total_odds: float = 2.0,
        max_total_odds: float = 10.0,
        confidence_threshold: float = 0.65,
        user_id: Optional[str] = None,
    ) -> AccumulatorResponse:
        """Build optimized accumulator bet using available recommendations"""

        # Fetch high-confidence recommendations for accumulator legs
        query = text(
            """
            SELECT br.id as recommendation_id, m.id as match_id,
                   m.match_date, l.name as league_name,
                   ht.name as home_team, at.name as away_team,
                   br.bet_type, br.selection, br.odds_at_recommendation,
                   br.bookmaker, br.confidence_rating,
                   br.expected_value, br.recommended_stake, br.reasoning,
                   p.confidence as model_confidence
            FROM betting_recommendations br
            JOIN predictions p ON br.prediction_id = p.id
            JOIN matches m ON br.match_id = m.id
            JOIN leagues l ON m.league_id = l.id
            JOIN teams ht ON m.home_team_id = ht.id
            JOIN teams at ON m.away_team_id = at.id
            WHERE br.status = 'pending'
            AND m.match_date > NOW()
            AND p.confidence >= :confidence_threshold
            ORDER BY p.confidence DESC, br.expected_value DESC
            LIMIT :max_candidates
        """
        )

        results = self.db.execute(
            query,
            {"confidence_threshold": confidence_threshold, "max_candidates": max_legs * 3},
        ).fetchall()

        if len(results) < min_legs:
            raise ValueError(f"Not enough qualifying recommendations. Found {len(results)}, need {min_legs}.")

        # Select legs optimizing for total odds within range
        selected_legs: List = []
        total_odds = 1.0
        combined_prob = 1.0

        for row in results:
            leg_odds = float(row.odds_at_recommendation)
            new_total = total_odds * leg_odds

            if new_total > max_total_odds and len(selected_legs) >= min_legs:
                break

            selected_legs.append(row)
            total_odds = new_total
            combined_prob *= float(row.model_confidence)

            if len(selected_legs) >= max_legs:
                break

        if len(selected_legs) < min_legs:
            raise ValueError(f"Could not build accumulator within odds range. " f"Best total odds: {total_odds:.2f}")

        # Calculate Kelly criterion for accumulator
        ev = combined_prob * total_odds - 1
        kelly_stake = max(0, ev / (total_odds - 1)) if total_odds > 1 else 0

        # Get user bankroll preference for stake calculation
        bankroll = self._get_user_bankroll()
        recommended_stake = round(bankroll * kelly_stake * 0.25, 2)  # Quarter Kelly

        # Create accumulator in database
        acc_id = self._create_accumulator(selected_legs, total_odds, recommended_stake, combined_prob, ev)

        # Build leg responses
        leg_responses = []
        for row in selected_legs:
            leg_responses.append(
                RecommendationResponse(
                    recommendation_id=str(row.recommendation_id),
                    match_info=MatchInfo(
                        match_id=str(row.match_id),
                        league_name=row.league_name,
                        home_team=row.home_team,
                        away_team=row.away_team,
                        match_date=row.match_date,
                        venue=None,
                    ),
                    market_type=row.bet_type,
                    outcome=row.selection,
                    recommended_odds=float(row.odds_at_recommendation),
                    recommended_stake=float(row.recommended_stake or 0),
                    expected_value=float(row.expected_value or 0),
                    confidence_level=row.confidence_rating or "medium",
                    reasoning=row.reasoning or "",
                    expires_at=row.match_date,
                )
            )

        return AccumulatorResponse(
            accumulator_id=str(acc_id),
            legs=leg_responses,
            total_odds=round(total_odds, 4),
            recommended_stake=recommended_stake,
            potential_return=round(recommended_stake * total_odds, 2),
            combined_probability=round(combined_prob, 6),
            expected_value=round(ev, 4),
            confidence_level=self._get_accumulator_confidence(combined_prob),
        )

    def _get_user_bankroll(self) -> float:
        """Get user bankroll from preferences"""
        query = text(
            """
            SELECT preference_value FROM user_preferences
            WHERE preference_key = 'bankroll'
        """
        )
        result = self.db.execute(query).fetchone()
        if result and result.preference_value:
            val = result.preference_value
            if isinstance(val, dict):
                return float(val.get("value", 1000))
            return float(val)
        return 1000.0

    def _create_accumulator(self, legs, total_odds: float, stake: float, combined_prob: float, ev: float) -> uuid.UUID:
        """Create accumulator and legs in database"""
        acc_id = uuid.uuid4()

        query = text(
            """
            INSERT INTO accumulators (id, name, total_odds, stake, potential_return,
                                      status, confidence_rating, reasoning)
            VALUES (:id, :name, :total_odds, :stake, :potential_return,
                    'pending', :confidence_rating, :reasoning)
        """
        )

        self.db.execute(
            query,
            {
                "id": str(acc_id),
                "name": f"Auto-Accumulator {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                "total_odds": total_odds,
                "stake": stake,
                "potential_return": round(stake * total_odds, 2),
                "confidence_rating": self._get_accumulator_confidence(combined_prob),
                "reasoning": f"{len(legs)}-fold accumulator. Combined probability: {combined_prob:.4f}, EV: {ev:.4f}",
            },
        )

        for i, leg in enumerate(legs):
            leg_query = text(
                """
                INSERT INTO accumulator_legs (accumulator_id, recommendation_id, leg_order)
                VALUES (:acc_id, :rec_id, :leg_order)
            """
            )
            self.db.execute(
                leg_query,
                {
                    "acc_id": str(acc_id),
                    "rec_id": str(leg.recommendation_id),
                    "leg_order": i + 1,
                },
            )

        self.db.commit()
        return acc_id

    @staticmethod
    def _get_accumulator_confidence(combined_prob: float) -> str:
        """Map combined probability to confidence rating"""
        if combined_prob >= 0.5:
            return "high"
        elif combined_prob >= 0.3:
            return "medium"
        else:
            return "low"
