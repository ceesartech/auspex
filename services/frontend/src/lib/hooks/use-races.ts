'use client';

import { useQuery } from '@tanstack/react-query';
import { racesApi, ListRacesParams } from '@/lib/api/races';

// Horse-racing hooks. Parallel to use-predictions.ts but on the
// races (multi-runner) schema. Same TanStack Query conventions:
// 60s polling for the index (race-card list moves slowly across the
// day), no polling for the detail page (open once, refresh on user
// action).

export function useRaces(params?: ListRacesParams) {
  return useQuery({
    queryKey: ['races', params],
    queryFn: () => racesApi.listRaces(params),
    refetchInterval: 60000,
  });
}

export function useRaceDetail(raceId: string) {
  return useQuery({
    queryKey: ['race', raceId],
    queryFn: () => racesApi.getRaceDetail(raceId),
    enabled: !!raceId,
  });
}
