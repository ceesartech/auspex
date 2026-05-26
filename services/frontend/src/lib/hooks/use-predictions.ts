'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { predictionsApi } from '@/lib/api/predictions';
import { Prediction } from '@/lib/types/prediction';

export function useUpcomingPredictions(params?: {
  sport?: string;
  league?: string;
  limit?: number;
}) {
  return useQuery({
    queryKey: ['predictions', 'upcoming', params],
    queryFn: () => predictionsApi.getUpcomingPredictions(params),
    refetchInterval: 60000,
  });
}

export function useLivePredictions() {
  return useQuery({
    queryKey: ['predictions', 'live'],
    queryFn: () => predictionsApi.getLivePredictions(),
    refetchInterval: 30000,
  });
}

export function usePrediction(matchId: string) {
  return useQuery({
    queryKey: ['prediction', matchId],
    queryFn: () => predictionsApi.getPrediction(matchId),
    enabled: !!matchId,
  });
}

export function useBulkPredictions() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (matchIds: string[]) => predictionsApi.getBulkPredictions(matchIds),
    onSuccess: (predictions: Prediction[]) => {
      predictions.forEach((prediction) => {
        queryClient.setQueryData(
          ['prediction', prediction.match_info.match_id],
          prediction
        );
      });
    },
  });
}

export function useUpcomingMatches(params?: {
  sport?: string;
  league?: string;
  days_ahead?: number;
  limit?: number;
}) {
  return useQuery({
    queryKey: ['matches', 'upcoming', params],
    queryFn: () => predictionsApi.getUpcomingMatches(params),
    refetchInterval: 120000,
  });
}

export function useMatchDetail(matchId: string) {
  return useQuery({
    queryKey: ['match', matchId],
    queryFn: () => predictionsApi.getMatchDetail(matchId),
    enabled: !!matchId,
  });
}

export function useMatchStats(matchId: string) {
  return useQuery({
    queryKey: ['match', matchId, 'stats'],
    queryFn: () => predictionsApi.getMatchStats(matchId),
    enabled: !!matchId,
  });
}

export function useMatchOdds(matchId: string) {
  return useQuery({
    queryKey: ['match', matchId, 'odds'],
    queryFn: () => predictionsApi.getMatchOdds(matchId),
    enabled: !!matchId,
  });
}

export function useOddsMovements(matchId: string, marketType = '1x2') {
  return useQuery({
    queryKey: ['odds', 'movements', matchId, marketType],
    queryFn: () => predictionsApi.getOddsMovements(matchId, marketType),
    enabled: !!matchId,
  });
}
