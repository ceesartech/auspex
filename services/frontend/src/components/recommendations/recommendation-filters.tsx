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
}

interface RecommendationFiltersProps {
  onFilterChange: (filters: FilterState) => void;
}

const confidenceOptions = [
  { value: '', label: 'All Confidence' },
  { value: 'HIGH', label: 'High' },
  { value: 'MEDIUM', label: 'Medium' },
  { value: 'LOW', label: 'Low' },
];

const marketOptions = [
  { value: '', label: 'All Markets' },
  { value: '1x2', label: '1X2' },
  { value: 'over_under', label: 'Over/Under' },
  { value: 'btts', label: 'BTTS' },
  { value: 'handicap', label: 'Handicap' },
  { value: 'correct_score', label: 'Correct Score' },
];

export function RecommendationFilters({ onFilterChange }: RecommendationFiltersProps) {
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
