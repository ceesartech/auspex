"""Response schemas"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class MatchInfo(BaseModel):
    """Match information"""

    match_id: str
    league_name: str
    home_team: str
    away_team: str
    match_date: datetime
    venue: Optional[str] = None


class PredictionResponse(BaseModel):
    """Prediction response"""

    # `model_version` collides with pydantic's protected `model_` namespace
    # in v2. Suppress the warning — the field name is part of our API
    # contract and we don't want to rename it.
    model_config = ConfigDict(protected_namespaces=())

    match_info: MatchInfo
    predicted_outcome: str
    probabilities: Dict[str, float]
    confidence: float
    model_version: str
    timestamp: datetime
    explanation: Optional[Dict[str, Any]] = None
    alternate_models: Optional[List[Dict[str, Any]]] = None
    # Which market this prediction is for — populated by the
    # per-match endpoint so the frontend can render NHL's 4 markets in
    # parallel without re-deriving market from prediction_type strings.
    # Optional + None default to stay backwards-compatible with cached
    # responses that pre-date this field.
    market: Optional[str] = None


class RecommendationResponse(BaseModel):
    """Betting recommendation"""

    recommendation_id: str
    match_info: MatchInfo
    market_type: str
    outcome: str
    recommended_odds: float
    recommended_stake: float
    expected_value: float
    confidence_level: str
    reasoning: str
    expires_at: datetime
    # Sport key so the UI can frame non-team sports (e.g. horse racing,
    # where match_info.home_team is the horse and away_team is "Field").
    # None/"" → a team or 1v1 match (the default rendering).
    sport: Optional[str] = None
    # The model's predicted probability for the recommended selection
    # (devigged consensus win prob for horse racing; ensemble confidence
    # for team sports). Lets the card show the prediction behind the bet.
    model_probability: Optional[float] = None


class AccumulatorResponse(BaseModel):
    """Accumulator bet response"""

    accumulator_id: str
    legs: List[RecommendationResponse]
    total_odds: float
    recommended_stake: float
    potential_return: float
    combined_probability: float
    expected_value: float
    confidence_level: str


class ModelPerformanceResponse(BaseModel):
    """Model performance metrics"""

    # See PredictionResponse for context on protected_namespaces.
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    evaluation_date: datetime
    total_predictions: int
    accuracy: float
    log_loss: float
    brier_score: float
    roi: float
    sharpe_ratio: Optional[float] = None
    by_sport: Optional[Dict[str, Dict[str, float]]] = None


class OddsResponse(BaseModel):
    """Odds data"""

    odds_id: str
    match_id: str
    bookmaker: str
    market_type: str
    selection: str
    odds_decimal: float
    implied_probability: Optional[float] = None
    is_live: bool = False
    timestamp: datetime


class LotteryDrawResponse(BaseModel):
    """Lottery draw result"""

    draw_id: str
    game: str
    draw_date: datetime
    numbers: List[int]
    bonus_number: int
    multiplier: Optional[int] = None
    jackpot_amount: Optional[float] = None


class LotteryAnalysisResponse(BaseModel):
    """Lottery number analysis"""

    game: str
    total_draws_analyzed: int
    hot_numbers: List[Dict[str, Any]]
    cold_numbers: List[Dict[str, Any]]
    overdue_numbers: List[Dict[str, Any]]
    frequency_distribution: Dict[str, int]
    # Historical combination profile (sum/odd-even/low-high/spread stats + top
    # co-occurring pairs). Optional so older clients keep working.
    profile: Optional[Dict[str, Any]] = None


class LotteryCombination(BaseModel):
    """One generated lottery line with its feature breakdown."""

    numbers: List[int]
    bonus_number: int
    score: float
    strategy: str
    rationale: str
    features: Dict[str, float]


class LotteryRecommendationsResponse(BaseModel):
    """A set of generated lottery combinations.

    NOTE: `score` ranks lines by statistical-profile fit and expected value
    (jackpot-share avoidance); it is NOT a probability of winning. Lottery draws
    are random — see `disclaimer`.
    """

    game: str
    strategy: str
    total_draws_analyzed: int
    generated_at: datetime
    combinations: List[LotteryCombination]
    # Non-empty when the ranking degraded (e.g. too few current-era draws for
    # hot/due/profile statistics — the ranking is then effectively EV-only).
    warnings: List[str] = []
    disclaimer: str


class LotteryTrackedLine(BaseModel):
    """One line the daily lottery_pipeline persisted for honest backtesting:
    generated pre-draw, stamped with its target draw, settled against the
    actual numbers once they land. Hit rates are expected to track pure
    chance — the ledger exists to prove that, not to find an edge."""

    line_id: str
    game: str
    strategy: str
    numbers: List[int]
    bonus_number: int
    score: Optional[float] = None
    target_draw_date: Optional[date] = None
    created_at: datetime
    # Settlement (NULL until the target draw's numbers are ingested).
    matched_main: Optional[int] = None
    matched_bonus: Optional[bool] = None
    prize_tier: Optional[str] = None
    settled_at: Optional[datetime] = None


class LotteryEVResponse(BaseModel):
    """Per-ticket expected-value verdict for the next draw.

    The only decision-relevant lottery output: EV as a function of the
    advertised jackpot, cash value, taxes, and expected co-winner sharing.
    The verdict is almost always "don't play" — that is the honest answer.
    """

    game: str
    ticket_price: float
    advertised_jackpot: float
    cash_value: float
    cash_ratio: float
    tax_rate: float
    tickets_estimated: float
    tickets_source: str
    expected_co_winners: float
    share_factor: float
    jackpot_odds: float
    ev_ex_jackpot: float
    ev_jackpot_term: float
    ev_total: float
    ev_per_dollar: float
    expected_loss_pct: float
    breakeven_advertised_jackpot: Optional[float] = None
    expected_multiplier: float
    jackpot_source: str
    next_draw_date: Optional[date] = None
    verdict: str
    disclaimer: str


class DashboardResponse(BaseModel):
    """User dashboard summary"""

    total_bets: int
    active_bets: int
    total_staked: float
    total_returns: float
    profit_loss: float
    roi_pct: float
    win_rate: float
    best_streak: int
    active_recommendations: int
    upcoming_matches: int


class ErrorResponse(BaseModel):
    """Error response"""

    error: str
    detail: str
    timestamp: datetime
