export interface MatchInfo {
  match_id: string;
  league_name: string;
  home_team: string;
  away_team: string;
  match_date: string;
  venue: string | null;
}

export interface Prediction {
  match_info: MatchInfo;
  predicted_outcome: string;
  probabilities: Record<string, number>;
  confidence: number;
  model_version: string;
  timestamp: string;
  explanation: Record<string, any> | null;
  alternate_models: Record<string, any>[] | null;
  // Populated by the per-match endpoint and by NHL upcoming rows so the
  // card UI can label which task this row represents (moneyline,
  // regulation, puck_line, total, match_result). Optional — soccer
  // rows from /upcoming may omit it.
  market?: string;
}

export interface PredictionRequest {
  match_id: string;
  include_explanation?: boolean;
  include_alternate_models?: boolean;
}

export interface BulkPredictionRequest {
  match_ids: string[];
  include_explanation?: boolean;
}

export interface UpcomingMatch {
  match_id: string;
  match_date: string;
  league_name: string;
  league_country: string;
  home_team: string;
  away_team: string;
  venue: string | null;
  odds: {
    home: number | null;
    draw: number | null;
    away: number | null;
    bookmaker: string | null;
  };
}
