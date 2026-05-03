"""Response schemas"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


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

    match_info: MatchInfo
    predicted_outcome: str
    probabilities: Dict[str, float]
    confidence: float
    model_version: str
    timestamp: datetime
    explanation: Optional[Dict[str, Any]] = None
    alternate_models: Optional[List[Dict[str, Any]]] = None


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
