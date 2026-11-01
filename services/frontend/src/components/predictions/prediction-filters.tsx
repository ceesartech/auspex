'use client';

import { Select } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { usePredictionsStore } from '@/lib/store/predictions-store';
import { marketOptionsBySport } from '@/lib/markets';
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

// Per-sport market dropdown options (and the soccer markets the API hides)
// live in @/lib/markets.

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
