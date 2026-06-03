// Horse-racing types. Parallel to prediction.ts / recommendation.ts
// but operating on the multi-runner schema (migration 013): races,
// race_entrants, race_predictions, race_recommendations.

export interface RaceSummary {
  race_id: string;
  race_date: string;
  track_name: string;
  race_number: number | null;
  distance_meters: number | null;
  surface: string | null;
  track_condition: string | null;
  race_class: string | null;
  purse_currency: string | null;
  purse_amount: number | null;
  field_size: number;
  runners: number;
  status: 'scheduled' | 'live' | 'finished' | 'cancelled' | 'abandoned';
  // How many value-bet recommendations the rec engine fired for this
  // race. Lets the index badge "3 picks" without a second roundtrip.
  recommendation_count: number;
}

export interface RaceEntrantRecommendation {
  recommendation_id: string;
  odds: number | null;
  bookmaker: string | null;
  expected_value: number | null;
  recommended_stake: number | null;
  confidence_rating: string | null;
  // Lifecycle status mirrors race_recommendations.status:
  // pending / placed / won / lost / void / cancelled.
  status: string;
  profit_loss: number | null;
}

export interface RaceEntrant {
  entrant_id: string;
  program_number: number | null;
  post_position: number | null;
  weight_carried_lbs: number | null;
  morning_line_odds: number | null;
  starting_price: number | null;
  // NULL for entrants without finish_position — typically scratched or
  // non-runners. The grader uses this to leave actual_outcome NULL.
  finish_position: number | null;
  disqualified: boolean;
  scratched: boolean;
  horse_name: string;
  jockey_name: string | null;
  trainer_name: string | null;
  // Devigged market-consensus win probability. NULL when precompute
  // hasn't run for this race yet.
  consensus_prob: number | null;
  // {entrant_id → prob} dict, JSONB on the server. Keys are the same
  // entrant_ids in this race; values sum to ~1.0.
  consensus_field_probs: Record<string, number> | null;
  // Grading fields — populated by grade_completed_races once the race
  // finishes. actual_outcome = 1.0 if THIS entrant won, 0.0 if lost.
  actual_outcome: number | null;
  // Race-level fact replicated to every row: TRUE iff the consensus
  // favourite won.
  is_correct: boolean | null;
  recommendation: RaceEntrantRecommendation | null;
}

export interface RaceDetail {
  race: Omit<RaceSummary, 'runners' | 'recommendation_count'>;
  entrants: RaceEntrant[];
}
