'use client';

import { useQuery } from '@tanstack/react-query';
import { accuracyApi } from '@/lib/api/accuracy';

export function useAccuracySummary(params?: {
  sport?: string;
  market?: string;
  days?: number;
}) {
  return useQuery({
    queryKey: ['accuracy', 'summary', params],
    queryFn: () => accuracyApi.getSummary(params),
    // Phase 5 grading runs every 15 min via the DAG; refreshing the
    // widget every 5 min is enough to surface newly-settled picks
    // without hammering the endpoint.
    refetchInterval: 5 * 60 * 1000,
    staleTime: 60 * 1000,
  });
}
