"""Request schemas"""

from typing import List, Optional

from pydantic import BaseModel, Field, validator


class LoginRequest(BaseModel):
    """Login request. `username` may be either the username or the email."""

    username: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8, max_length=255)


class PredictionRequest(BaseModel):
    """Request for match prediction"""

    match_id: str = Field(...)
    include_explanation: bool = True
    include_alternate_models: bool = False


class BulkPredictionRequest(BaseModel):
    """Request for multiple predictions"""

    match_ids: List[str] = Field(..., min_length=1, max_length=50)
    include_explanation: bool = False


class UserPreferencesUpdate(BaseModel):
    """Update user preferences"""

    bankroll: Optional[float] = Field(None, ge=0)
    max_stake_per_bet: Optional[float] = Field(None, ge=0)
    risk_tolerance: Optional[str] = Field(None, pattern="^(LOW|MEDIUM|HIGH)$")
    confidence_threshold: Optional[float] = Field(None, ge=0, le=1)
    kelly_fraction: Optional[float] = Field(None, ge=0, le=1)
    favorite_sports: Optional[List[str]] = None
    notification_enabled: Optional[bool] = None


class AccumulatorRequest(BaseModel):
    """Request to build accumulator"""

    min_legs: int = Field(2, ge=2, le=10)
    max_legs: int = Field(5, ge=2, le=10)
    min_total_odds: float = Field(2.0, ge=1.5)
    max_total_odds: float = Field(10.0, ge=2.0)
    confidence_threshold: float = Field(0.65, ge=0.5, le=0.95)

    @validator("max_legs")
    def validate_legs(cls, v, values):
        if "min_legs" in values and v < values["min_legs"]:
            raise ValueError("max_legs must be >= min_legs")
        return v


class BettingHistoryRecord(BaseModel):
    """Record a placed bet"""

    recommendation_id: Optional[str] = None
    match_id: Optional[str] = None
    bookmaker: str = Field(..., min_length=1, max_length=50)
    bet_type: str = Field(..., min_length=1, max_length=50)
    selection: str = Field(..., min_length=1, max_length=100)
    odds: float = Field(..., gt=1.0)
    stake: float = Field(..., gt=0)
    notes: Optional[str] = None
