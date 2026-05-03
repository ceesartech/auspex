"""SQLAlchemy ORM models mapping to existing database tables"""

from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, Date, Text,
    ForeignKey, Numeric, ARRAY, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from database import Base


class League(Base):
    __tablename__ = "leagues"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    country = Column(String(100), nullable=False)
    sport = Column(String(50), nullable=False, default="soccer")
    tier = Column(Integer, default=1)
    external_ids = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    teams = relationship("Team", back_populates="league")
    matches = relationship("Match", back_populates="league")

    __table_args__ = (
        UniqueConstraint("name", "country", "sport"),
    )


class Team(Base):
    __tablename__ = "teams"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    normalized_name = Column(String(255), nullable=False)
    league_id = Column(UUID(as_uuid=True), ForeignKey("leagues.id", ondelete="SET NULL"))
    country = Column(String(100))
    sport = Column(String(50), nullable=False, default="soccer")
    external_ids = Column(JSONB, default={})
    metadata_ = Column("metadata", JSONB, default={})
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    league = relationship("League", back_populates="teams")

    __table_args__ = (
        UniqueConstraint("normalized_name", "sport"),
    )


class Match(Base):
    __tablename__ = "matches"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    league_id = Column(UUID(as_uuid=True), ForeignKey("leagues.id", ondelete="CASCADE"))
    home_team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"))
    away_team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"))
    match_date = Column(DateTime(timezone=True), nullable=False)
    status = Column(String(20), default="scheduled")
    home_score = Column(Integer)
    away_score = Column(Integer)
    half_time_home = Column(Integer)
    half_time_away = Column(Integer)
    season = Column(String(10), nullable=False)
    round = Column(String(50))
    venue = Column(String(255))
    attendance = Column(Integer)
    referee_id = Column(UUID(as_uuid=True), ForeignKey("referees.id", ondelete="SET NULL"))
    external_ids = Column(JSONB, default={})
    metadata_ = Column("metadata", JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    league = relationship("League", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])
    referee = relationship("Referee")
    stats = relationship("MatchStats", back_populates="match")
    odds = relationship("Odds", back_populates="match")
    predictions = relationship("Prediction", back_populates="match")

    __table_args__ = (
        UniqueConstraint("home_team_id", "away_team_id", "match_date"),
        CheckConstraint(
            "status IN ('scheduled', 'live', 'finished', 'postponed', 'cancelled')"
        ),
    )


class MatchStats(Base):
    __tablename__ = "match_stats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    match_id = Column(UUID(as_uuid=True), ForeignKey("matches.id", ondelete="CASCADE"))
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"))
    possession = Column(Numeric(5, 2))
    shots = Column(Integer, default=0)
    shots_on_target = Column(Integer, default=0)
    expected_goals = Column(Numeric(5, 3))
    expected_goals_against = Column(Numeric(5, 3))
    passes = Column(Integer, default=0)
    pass_accuracy = Column(Numeric(5, 2))
    corners = Column(Integer, default=0)
    fouls = Column(Integer, default=0)
    yellow_cards = Column(Integer, default=0)
    red_cards = Column(Integer, default=0)
    offsides = Column(Integer, default=0)
    tackles = Column(Integer, default=0)
    interceptions = Column(Integer, default=0)
    saves = Column(Integer, default=0)
    clearances = Column(Integer, default=0)
    blocked_shots = Column(Integer, default=0)
    crosses = Column(Integer, default=0)
    cross_accuracy = Column(Numeric(5, 2))
    long_balls = Column(Integer, default=0)
    through_balls = Column(Integer, default=0)
    aerials_won = Column(Integer, default=0)
    dribbles_completed = Column(Integer, default=0)
    progressive_carries = Column(Integer, default=0)
    progressive_passes = Column(Integer, default=0)
    deep_completions = Column(Integer, default=0)
    ppda = Column(Numeric(5, 2))
    oppda = Column(Numeric(5, 2))
    high_press_success = Column(Numeric(5, 2))
    metadata_ = Column("metadata", JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    match = relationship("Match", back_populates="stats")
    team = relationship("Team")

    __table_args__ = (
        UniqueConstraint("match_id", "team_id"),
    )


class Odds(Base):
    __tablename__ = "odds"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    match_id = Column(UUID(as_uuid=True), ForeignKey("matches.id", ondelete="CASCADE"))
    bookmaker = Column(String(50), nullable=False)
    market_type = Column(String(50), nullable=False)
    selection = Column(String(100), nullable=False)
    odds_decimal = Column(Numeric(10, 4), nullable=False)
    odds_american = Column(Integer)
    line = Column(Numeric(5, 2))
    implied_probability = Column(Numeric(5, 4))
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow)
    is_opening = Column(Boolean, default=False)
    is_live = Column(Boolean, default=False)
    metadata_ = Column("metadata", JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    match = relationship("Match", back_populates="odds")


class Player(Base):
    __tablename__ = "players"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    normalized_name = Column(String(255), nullable=False)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id", ondelete="SET NULL"))
    position = Column(String(50))
    nationality = Column(String(100))
    date_of_birth = Column(Date)
    height_cm = Column(Integer)
    weight_kg = Column(Integer)
    preferred_foot = Column(String(10))
    market_value = Column(Numeric(15, 2))
    external_ids = Column(JSONB, default={})
    metadata_ = Column("metadata", JSONB, default={})
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    team = relationship("Team")


class Referee(Base):
    __tablename__ = "referees"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    normalized_name = Column(String(255), nullable=False)
    nationality = Column(String(100))
    external_ids = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    match_id = Column(UUID(as_uuid=True), ForeignKey("matches.id", ondelete="CASCADE"))
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    prediction_type = Column(String(50), nullable=False)
    predicted_outcome = Column(String(100), nullable=False)
    confidence = Column(Numeric(5, 4), nullable=False)
    probabilities = Column(JSONB, nullable=False)
    features_used = Column(JSONB, default={})
    feature_importance = Column(JSONB, default={})
    expected_value = Column(Numeric(10, 4))
    kelly_fraction = Column(Numeric(10, 6))
    recommended_stake = Column(Numeric(10, 2))
    actual_outcome = Column(String(100))
    is_correct = Column(Boolean)
    profit_loss = Column(Numeric(10, 2))
    metadata_ = Column("metadata", JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    match = relationship("Match", back_populates="predictions")
    recommendations = relationship("BettingRecommendation", back_populates="prediction")


class BettingRecommendation(Base):
    __tablename__ = "betting_recommendations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="CASCADE"))
    match_id = Column(UUID(as_uuid=True), ForeignKey("matches.id", ondelete="CASCADE"))
    bet_type = Column(String(50), nullable=False)
    selection = Column(String(100), nullable=False)
    odds_at_recommendation = Column(Numeric(10, 4), nullable=False)
    bookmaker = Column(String(50), nullable=False)
    confidence_rating = Column(String(10))
    expected_value = Column(Numeric(10, 4))
    kelly_stake = Column(Numeric(10, 2))
    recommended_stake = Column(Numeric(10, 2))
    max_stake = Column(Numeric(10, 2))
    reasoning = Column(Text)
    risk_factors = Column(JSONB, default=[])
    status = Column(String(20), default="pending")
    actual_result = Column(String(100))
    profit_loss = Column(Numeric(10, 2))
    placed_at = Column(DateTime(timezone=True))
    settled_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    prediction = relationship("Prediction", back_populates="recommendations")
    match = relationship("Match")


class Accumulator(Base):
    __tablename__ = "accumulators"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255))
    total_odds = Column(Numeric(15, 4))
    stake = Column(Numeric(10, 2))
    potential_return = Column(Numeric(15, 2))
    status = Column(String(20), default="pending")
    confidence_rating = Column(String(10))
    reasoning = Column(Text)
    actual_return = Column(Numeric(15, 2))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    legs = relationship("AccumulatorLeg", back_populates="accumulator")


class AccumulatorLeg(Base):
    __tablename__ = "accumulator_legs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    accumulator_id = Column(UUID(as_uuid=True), ForeignKey("accumulators.id", ondelete="CASCADE"))
    recommendation_id = Column(UUID(as_uuid=True), ForeignKey("betting_recommendations.id", ondelete="CASCADE"))
    leg_order = Column(Integer, nullable=False)
    status = Column(String(20), default="pending")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    accumulator = relationship("Accumulator", back_populates="legs")
    recommendation = relationship("BettingRecommendation")

    __table_args__ = (
        UniqueConstraint("accumulator_id", "leg_order"),
    )


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    preference_key = Column(String(100), nullable=False, unique=True)
    preference_value = Column(JSONB, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class BettingHistory(Base):
    __tablename__ = "betting_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recommendation_id = Column(UUID(as_uuid=True), ForeignKey("betting_recommendations.id", ondelete="SET NULL"))
    match_id = Column(UUID(as_uuid=True), ForeignKey("matches.id", ondelete="SET NULL"))
    bookmaker = Column(String(50), nullable=False)
    bet_type = Column(String(50), nullable=False)
    selection = Column(String(100), nullable=False)
    odds = Column(Numeric(10, 4), nullable=False)
    stake = Column(Numeric(10, 2), nullable=False)
    potential_return = Column(Numeric(15, 2))
    actual_return = Column(Numeric(15, 2))
    status = Column(String(20), default="pending")
    notes = Column(Text)
    placed_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    settled_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    recommendation = relationship("BettingRecommendation")
    match = relationship("Match")


class LotteryDraw(Base):
    __tablename__ = "lottery_draws"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    game = Column(String(50), nullable=False)
    draw_date = Column(Date, nullable=False)
    draw_number = Column(Integer)
    numbers = Column(ARRAY(Integer), nullable=False)
    bonus_number = Column(Integer, nullable=False)
    multiplier = Column(Integer)
    jackpot_amount = Column(Numeric(15, 2))
    winners_by_tier = Column(JSONB, default={})
    metadata_ = Column("metadata", JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("game", "draw_date"),
    )


class ModelPerformanceLog(Base):
    __tablename__ = "model_performance_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    sport = Column(String(50), nullable=False)
    league_id = Column(UUID(as_uuid=True), ForeignKey("leagues.id", ondelete="SET NULL"))
    evaluation_date = Column(Date, nullable=False)
    metrics = Column(JSONB, nullable=False)
    sample_size = Column(Integer)
    date_range_start = Column(Date)
    date_range_end = Column(Date)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class ModelRetrainingLog(Base):
    __tablename__ = "model_retraining_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    old_version = Column(String(50))
    new_version = Column(String(50), nullable=False)
    trigger_reason = Column(String(50))
    training_data_start = Column(Date)
    training_data_end = Column(Date)
    training_samples = Column(Integer)
    validation_metrics = Column(JSONB)
    test_metrics = Column(JSONB)
    hyperparameters = Column(JSONB)
    training_duration_seconds = Column(Integer)
    status = Column(String(20))
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    completed_at = Column(DateTime(timezone=True))


class FeaturesCache(Base):
    __tablename__ = "features_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    match_id = Column(UUID(as_uuid=True), ForeignKey("matches.id", ondelete="CASCADE"))
    feature_set = Column(String(100), nullable=False)
    features = Column(JSONB, nullable=False)
    feature_version = Column(String(50), nullable=False)
    computed_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    expires_at = Column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("match_id", "feature_set", "feature_version"),
    )


class ApiRequestLog(Base):
    __tablename__ = "api_request_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer)
    response_time_ms = Column(Integer)
    request_body = Column(JSONB)
    response_summary = Column(JSONB)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)


class SystemNotification(Base):
    __tablename__ = "system_notifications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type = Column(String(50), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    source = Column(String(100))
    is_read = Column(Boolean, default=False)
    metadata_ = Column("metadata", JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
