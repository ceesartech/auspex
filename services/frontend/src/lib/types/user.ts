export interface User {
  username: string;
  email?: string;
  role: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface UserPreferences {
  [key: string]: {
    value: any;
    description: string;
  };
}

export interface UserPreferencesUpdate {
  bankroll?: number;
  max_stake_per_bet?: number;
  risk_tolerance?: 'LOW' | 'MEDIUM' | 'HIGH';
  confidence_threshold?: number;
  kelly_fraction?: number;
  favorite_sports?: string[];
  notification_enabled?: boolean;
}

export interface BettingHistoryEntry {
  id: string;
  bookmaker: string;
  bet_type: string;
  selection: string;
  odds: number;
  stake: number;
  potential_return: number | null;
  actual_return: number | null;
  status: 'pending' | 'won' | 'lost' | 'void' | 'cashout';
  notes: string | null;
  placed_at: string | null;
  settled_at: string | null;
  match: {
    home_team: string;
    away_team: string;
    match_date: string | null;
    league: string;
  } | null;
}

export interface DashboardStats {
  total_bets: number;
  active_bets: number;
  total_staked: number;
  total_returns: number;
  profit_loss: number;
  roi_pct: number;
  win_rate: number;
  active_recommendations: number;
  upcoming_matches: number;
}

export interface LotteryDraw {
  draw_id: string;
  game: string;
  draw_date: string;
  numbers: number[];
  bonus_number: number;
  multiplier: number | null;
  jackpot_amount: number | null;
}

// One line from the daily backtest ledger: generated pre-draw, settled once the
// target draw's numbers land. Settled iff settled_at is non-null; prize_tier null
// on a settled line means no prize (the expected outcome).
export interface LotteryTrackedLine {
  line_id: string;
  game: string;
  strategy: string;
  numbers: number[];
  bonus_number: number;
  score: number | null;
  target_draw_date: string | null;
  created_at: string;
  matched_main: number | null;
  matched_bonus: boolean | null;
  prize_tier: string | null;
  settled_at: string | null;
}

export interface LotteryAnalysis {
  game: string;
  total_draws_analyzed: number;
  hot_numbers: { number: number; frequency: number; percentage: number }[];
  cold_numbers: { number: number; frequency: number; percentage: number }[];
  overdue_numbers: { number: number; draws_since_last: number; frequency: number }[];
  frequency_distribution: Record<string, number>;
  profile?: Record<string, unknown>;
}

export interface LotteryCombination {
  numbers: number[];
  bonus_number: number;
  score: number;
  strategy: string;
  rationale: string;
  features: Record<string, number>;
}

export interface LotteryRecommendations {
  game: string;
  strategy: string;
  total_draws_analyzed: number;
  generated_at: string;
  combinations: LotteryCombination[];
  // Non-empty when draw history is too thin and the ranking degrades to EV-only.
  warnings: string[];
  disclaimer: string;
}

export interface LotteryEV {
  game: string;
  ticket_price: number;
  advertised_jackpot: number;
  cash_value: number;
  cash_ratio: number;
  tax_rate: number;
  tickets_estimated: number;
  tickets_source: string;
  expected_co_winners: number;
  share_factor: number;
  // 1-in-N jackpot odds (the N, e.g. 292201338).
  jackpot_odds: number;
  ev_ex_jackpot: number;
  ev_jackpot_term: number;
  ev_total: number;
  ev_per_dollar: number;
  // Already in percent units (0-100).
  expected_loss_pct: number;
  breakeven_advertised_jackpot: number | null;
  expected_multiplier: number;
  jackpot_source: 'live' | 'user';
  next_draw_date: string | null;
  verdict: string;
  disclaimer: string;
}
