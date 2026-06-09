import { MatchInfo } from './prediction';

export interface Recommendation {
  recommendation_id: string;
  match_info: MatchInfo;
  market_type: string;
  outcome: string;
  recommended_odds: number;
  recommended_stake: number;
  expected_value: number;
  confidence_level: string;
  reasoning: string;
  expires_at: string;
  // Sport key — "horse_racing" frames the card as a single runner vs the
  // field; null/absent → a team or 1v1 match (default rendering).
  sport?: string | null;
  // Model's predicted probability for the selection (0–1). Shown as the
  // "Model" stat so the prediction behind the bet is visible.
  model_probability?: number | null;
}

export interface Accumulator {
  accumulator_id: string;
  legs: Recommendation[];
  total_odds: number;
  recommended_stake: number;
  potential_return: number;
  combined_probability: number;
  expected_value: number;
  confidence_level: string;
}

export interface AccumulatorRequest {
  min_legs: number;
  max_legs: number;
  min_total_odds: number;
  max_total_odds: number;
  confidence_threshold: number;
}

export interface ModelPerformance {
  model_name: string;
  model_version: string;
  evaluation_date: string;
  total_predictions: number;
  accuracy: number;
  log_loss: number;
  brier_score: number;
  roi: number;
  sharpe_ratio: number | null;
  by_sport: Record<string, Record<string, number>> | null;
}

export interface OddsMovement {
  match_id: string;
  market_type: string;
  movements: Record<string, OddsDataPoint[]>;
  summary: Record<string, OddsSummary>;
}

export interface OddsDataPoint {
  odds_decimal: number;
  implied_probability: number | null;
  timestamp: string | null;
  bookmaker: string;
  is_opening: boolean;
}

export interface OddsSummary {
  opening_odds: number;
  current_odds: number;
  change: number;
  change_pct: number;
  direction: 'shortening' | 'drifting' | 'stable';
  data_points: number;
}
