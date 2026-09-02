'use client';

import { Select } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { usePredictionsStore } from '@/lib/store/predictions-store';
import { X } from 'lucide-react';

const sportOptions = [
  { value: '', label: 'All Sports' },
  { value: 'soccer', label: 'Soccer' },
  { value: 'nfl', label: 'NFL' },
  { value: 'nhl', label: 'NHL' },
  { value: 'nba', label: 'NBA' },
  { value: 'tennis', label: 'Tennis' },
  { value: 'horse_racing', label: 'Horse Racing' },
  { value: 'mma', label: 'MMA' },
];

// Markets available per sport. The market dropdown only renders when the
// selected sport has an entry here; tennis / MMA / horse racing are
// single-market (moneyline / win) so they get no dropdown, and neither
// does "All Sports" because a market value only makes sense within one
// sport's schema.
//
// Soccer exposes the ~19 markets derived from the Dixon-Coles scoreline
// model (docs: soccer market derivation). The '' default asks the API for
// the headline market only (match_result / 1X2, one row per match); every
// other value is passed through verbatim as the `prediction_type` filter,
// which is why these values must match the prediction_type strings the
// precompute job writes.
//
// NHL exposes the 4 markets shipped in Phase 3 — moneyline (headline) →
// regulation → puck_line → total, locked to fixed canonical lines
// (±1.5 / 5.5).
//
// NBA exposes the 3 markets shipped in Phase 6 — moneyline → spread →
// total. Spread + total use LINE-AS-FEATURE: one trained model handles
// every line the book offers (variable -3.5 to -13.5 spreads, 210-245
// totals), so labels are intentionally generic — the displayed line
// per pick comes from the prediction row itself.
//
// NFL (Phase 10) mirrors the NBA shape — same 3 markets, same
// line-as-feature design. NFL spreads typically live on key numbers
// (3, 7, 10, 14) and totals in the 40-50 range; labels stay generic.
const marketOptionsBySport: Record<string, { value: string; label: string }[]> = {
  soccer: [
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
  ],
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

interface PredictionFiltersProps {
  // League names present in the current result set, for the league
  // dropdown. Derived from the fetched predictions so the options always
  // reflect what's actually on screen.
  leagueOptions?: string[];
}

export function PredictionFilters({ leagueOptions = [] }: PredictionFiltersProps) {
  const {
    selectedSport,
    setSelectedSport,
    selectedMarket,
    setSelectedMarket,
    selectedLeague,
    setSelectedLeague,
    clearFilters,
  } = usePredictionsStore();

  const marketOptions = selectedSport ? marketOptionsBySport[selectedSport] : undefined;

  return (
    <div className="flex flex-wrap items-center gap-3">
      <Select
        options={sportOptions}
        value={selectedSport || ''}
        onChange={(e) => setSelectedSport(e.target.value || null)}
        className="w-[180px]"
      />
      {leagueOptions.length > 0 && (
        <Select
          options={[
            { value: '', label: 'All Leagues' },
            ...leagueOptions.map((league) => ({ value: league, label: league })),
          ]}
          value={selectedLeague || ''}
          onChange={(e) => setSelectedLeague(e.target.value || null)}
          className="w-[200px]"
        />
      )}
      {marketOptions && (
        <Select
          options={marketOptions}
          value={selectedMarket || ''}
          onChange={(e) => setSelectedMarket(e.target.value || null)}
          className="w-[220px]"
        />
      )}
      {(selectedSport || selectedMarket || selectedLeague) && (
        <Button variant="ghost" size="sm" onClick={clearFilters}>
          <X className="mr-1 h-3 w-3" /> Clear
        </Button>
      )}
    </div>
  );
}
