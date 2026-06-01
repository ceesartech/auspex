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
  { value: 'tennis', label: 'Tennis' },
  { value: 'horse_racing', label: 'Horse Racing' },
  { value: 'boxing', label: 'Boxing' },
  { value: 'mma', label: 'MMA' },
];

// Markets available per sport. Soccer is single-market so we don't show
// the market dropdown at all when selectedSport is 'soccer' or empty.
// NHL exposes the 4 markets shipped in Phase 3 — keep the order
// matching the most-decisive-to-least: moneyline (headline) → regulation
// → puck_line → total.
const marketOptionsBySport: Record<string, { value: string; label: string }[]> = {
  nhl: [
    { value: '', label: 'Default (Moneyline)' },
    { value: 'moneyline', label: 'Moneyline' },
    { value: 'regulation', label: 'Regulation (60 min)' },
    { value: 'puck_line', label: 'Puck Line' },
    { value: 'total', label: 'Total Goals O/U 5.5' },
  ],
};

export function PredictionFilters() {
  const {
    selectedSport,
    setSelectedSport,
    selectedMarket,
    setSelectedMarket,
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
      {marketOptions && (
        <Select
          options={marketOptions}
          value={selectedMarket || ''}
          onChange={(e) => setSelectedMarket(e.target.value || null)}
          className="w-[220px]"
        />
      )}
      {(selectedSport || selectedMarket) && (
        <Button variant="ghost" size="sm" onClick={clearFilters}>
          <X className="mr-1 h-3 w-3" /> Clear
        </Button>
      )}
    </div>
  );
}
