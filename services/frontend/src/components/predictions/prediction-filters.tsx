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

export function PredictionFilters() {
  const { selectedSport, setSelectedSport, clearFilters } = usePredictionsStore();

  return (
    <div className="flex flex-wrap items-center gap-3">
      <Select
        options={sportOptions}
        value={selectedSport || ''}
        onChange={(e) => setSelectedSport(e.target.value || null)}
        className="w-[180px]"
      />
      {selectedSport && (
        <Button variant="ghost" size="sm" onClick={clearFilters}>
          <X className="mr-1 h-3 w-3" /> Clear
        </Button>
      )}
    </div>
  );
}
