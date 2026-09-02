'use client';

import { useState } from 'react';
import { Select } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { X } from 'lucide-react';

interface FilterState {
  confidence_level?: string;
  market_type?: string;
  min_odds?: number;
  max_odds?: number;
  // Client-side only — the recommendations API has no league param, so the
  // page filters on match_info.league_name instead of forwarding this.
  league?: string;
}

interface RecommendationFiltersProps {
  onFilterChange: (filters: FilterState) => void;
  // League names present in the current result set.
  leagueOptions?: string[];
}

const confidenceOptions = [
  { value: '', label: 'All Confidence' },
  { value: 'HIGH', label: 'High' },
  { value: 'MEDIUM', label: 'Medium' },
  { value: 'LOW', label: 'Low' },
];

// Values must equal the `bet_type` written by generate_recommendations.py
// (which mirrors the odds `market_type`). Only markets we actually price are
// listed here so the filter never yields an empty result set.
const marketOptions = [
  { value: '', label: 'All Markets' },
  // Soccer (1x2-derived) markets
  { value: '1x2', label: '1X2' },
  { value: 'double_chance', label: 'Double Chance' },
  { value: 'draw_no_bet', label: 'Draw No Bet' },
  { value: 'over_under', label: 'Over/Under' },
  { value: 'btts', label: 'BTTS' },
  { value: 'asian_handicap', label: 'Asian Handicap' },
  { value: 'correct_score', label: 'Correct Score' },
  // Team / 1v1 sports (NFL/NBA/NHL/tennis/MMA)
  { value: 'moneyline', label: 'Moneyline' },
  { value: 'spread', label: 'Spread' },
  { value: 'total', label: 'Total' },
  { value: 'puck_line', label: 'Puck Line' },
  // Horse racing
  { value: 'win', label: 'Win (Horse Racing)' },
];

export function RecommendationFilters({ onFilterChange, leagueOptions = [] }: RecommendationFiltersProps) {
  const [filters, setFilters] = useState<FilterState>({});

  const updateFilter = (key: keyof FilterState, value: string | number | undefined) => {
    const newFilters = { ...filters, [key]: value || undefined };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  const clearFilters = () => {
    setFilters({});
    onFilterChange({});
  };

  const hasFilters = Object.values(filters).some(Boolean);

  return (
    <div className="flex flex-wrap items-center gap-3">
      <Select
        options={confidenceOptions}
        value={filters.confidence_level || ''}
        onChange={(e) => updateFilter('confidence_level', e.target.value)}
        className="w-[160px]"
      />
      <Select
        options={marketOptions}
        value={filters.market_type || ''}
        onChange={(e) => updateFilter('market_type', e.target.value)}
        className="w-[160px]"
      />
      {leagueOptions.length > 0 && (
        <Select
          options={[
            { value: '', label: 'All Leagues' },
            ...leagueOptions.map((league) => ({ value: league, label: league })),
          ]}
          value={filters.league || ''}
          onChange={(e) => updateFilter('league', e.target.value)}
          className="w-[180px]"
        />
      )}
      <Input
        type="number"
        placeholder="Min odds"
        value={filters.min_odds || ''}
        onChange={(e) => updateFilter('min_odds', e.target.value ? Number(e.target.value) : undefined)}
        className="w-[120px]"
        step="0.1"
        min="1"
      />
      <Input
        type="number"
        placeholder="Max odds"
        value={filters.max_odds || ''}
        onChange={(e) => updateFilter('max_odds', e.target.value ? Number(e.target.value) : undefined)}
        className="w-[120px]"
        step="0.1"
        min="1"
      />
      {hasFilters && (
        <Button variant="ghost" size="sm" onClick={clearFilters}>
          <X className="mr-1 h-3 w-3" /> Clear
        </Button>
      )}
    </div>
  );
}
