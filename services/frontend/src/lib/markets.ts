// Market dropdown options for the predictions page, per sport.
//
// The market dropdown only renders when the selected sport has an entry
// here; tennis / MMA / horse racing are single-market (moneyline / win) so
// they get no dropdown, and neither does "All Sports" because a market value
// only makes sense within one sport's schema.
//
// Soccer exposes the markets derived from the Dixon-Coles scoreline model
// (docs: soccer market derivation). The '' default asks the API for the
// headline market only (match_result / 1X2, one row per match); every other
// value is passed through verbatim as the `prediction_type` filter, which is
// why these values must match the prediction_type strings the precompute job
// writes.
//
// NHL exposes the 4 markets shipped in Phase 3 — moneyline (headline) →
// regulation → puck_line → total, locked to fixed canonical lines
// (±1.5 / 5.5).
//
// NBA exposes the 3 markets shipped in Phase 6 — moneyline → spread →
// total. Spread + total use LINE-AS-FEATURE: one trained model handles every
// line the book offers (variable -3.5 to -13.5 spreads, 210-245 totals), so
// labels are intentionally generic — the displayed line per pick comes from
// the prediction row itself.
//
// NFL (Phase 10) mirrors the NBA shape — same 3 markets, same
// line-as-feature design. NFL spreads typically live on key numbers
// (3, 7, 10, 14) and totals in the 40-50 range; labels stay generic.

export interface MarketOption {
  value: string;
  label: string;
}

// Soccer prediction types the API stores and grades but never displays:
// a mirror of the 'ensemble:' entries of HIDDEN_PREDICTION_PAIRS in
// services/api/src/services/prediction_service.py (the frontend cannot
// import Python; an api unit test fails if the two lists differ). odd_even
// lost to a simple per-league constant on 2026-11-01
// (scripts/ab_soccer_derived_vs_constant.py). Offering a hidden market would
// always return an empty list, so the dropdown leaves it out. Remove an entry
// here and in the API together.
export const HIDDEN_SOCCER_MARKETS: readonly string[] = ['odd_even'];

// Every soccer market the precompute job writes, in display order.
const SOCCER_MARKET_CATALOGUE: MarketOption[] = [
  { value: '', label: 'Default (1X2)' },
  { value: 'over_under', label: 'Over/Under' },
  { value: 'btts', label: 'BTTS' },
  { value: 'asian_handicap', label: 'Asian Handicap' },
  { value: 'correct_score', label: 'Correct Score' },
  { value: 'double_chance', label: 'Double Chance' },
  { value: 'draw_no_bet', label: 'Draw No Bet' },
  { value: 'total_goals', label: 'Total Goals' },
  { value: 'winning_margin', label: 'Winning Margin' },
  { value: 'team_total', label: 'Team Total' },
  { value: 'clean_sheet', label: 'Clean Sheet' },
  { value: 'win_to_nil', label: 'Win To Nil' },
  { value: 'odd_even', label: 'Odd/Even' },
  { value: 'over_under_ht', label: 'HT Over/Under' },
  { value: 'match_result_ht', label: 'HT Result' },
  { value: 'btts_ht', label: 'HT BTTS' },
  { value: 'result_btts', label: 'Result & BTTS' },
  { value: 'result_over_under', label: 'Result & O/U' },
  { value: 'ht_ft_double_result', label: 'HT/FT' },
];

export const marketOptionsBySport: Record<string, MarketOption[]> = {
  soccer: SOCCER_MARKET_CATALOGUE.filter((option) => !HIDDEN_SOCCER_MARKETS.includes(option.value)),
  nhl: [
    { value: '', label: 'Default (Moneyline)' },
    { value: 'moneyline', label: 'Moneyline' },
    { value: 'regulation', label: 'Regulation (60 min)' },
    { value: 'puck_line', label: 'Puck Line' },
    { value: 'total', label: 'Total Goals O/U 5.5' },
  ],
  nba: [
    { value: '', label: 'Default (Moneyline)' },
    { value: 'moneyline', label: 'Moneyline' },
    { value: 'spread', label: 'Spread' },
    { value: 'total', label: 'Total Points' },
  ],
  nfl: [
    { value: '', label: 'Default (Moneyline)' },
    { value: 'moneyline', label: 'Moneyline' },
    { value: 'spread', label: 'Spread' },
    { value: 'total', label: 'Total Points' },
  ],
};
