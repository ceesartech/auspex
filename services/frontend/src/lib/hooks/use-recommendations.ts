'use client';

import { useQuery, useMutation } from '@tanstack/react-query';
import { recommendationsApi } from '@/lib/api/recommendations';
import { AccumulatorRequest } from '@/lib/types/recommendation';

export function useRecommendations(params?: {
  confidence_level?: string;
  market_type?: string;
  min_odds?: number;
  max_odds?: number;
  limit?: number;
}) {
  return useQuery({
    queryKey: ['recommendations', params],
    queryFn: () => recommendationsApi.getRecommendations(params),
    refetchInterval: 120000,
  });
}

export function useHighValueBets(params?: {
  min_ev?: number;
  limit?: number;
}) {
  return useQuery({
    queryKey: ['recommendations', 'high-value', params],
    queryFn: () => recommendationsApi.getHighValueBets(params),
    refetchInterval: 120000,
  });
}

export function useBuildAccumulator() {
  return useMutation({
    mutationFn: (request: AccumulatorRequest) =>
      recommendationsApi.buildAccumulator(request),
  });
}

export function useModelPerformance(params?: {
  model_name?: string;
  sport?: string;
  limit?: number;
}) {
  return useQuery({
    queryKey: ['models', 'performance', params],
    queryFn: () => recommendationsApi.getModelPerformance(params),
  });
}

export function useActiveModels() {
  return useQuery({
    queryKey: ['models', 'active'],
    queryFn: () => recommendationsApi.getActiveModels(),
  });
}

export function useModelComparison(modelNames: string[], sport?: string) {
  return useQuery({
    queryKey: ['models', 'comparison', modelNames, sport],
    queryFn: () => recommendationsApi.compareModels(modelNames, sport),
    enabled: modelNames.length > 0,
  });
}

export function useLotteryDraws(game?: string, limit = 20) {
  return useQuery({
    queryKey: ['lottery', 'draws', game, limit],
    queryFn: () => recommendationsApi.getLotteryDraws(game, limit),
  });
}

export function useLotteryAnalysis(game: string, numDraws = 100) {
  return useQuery({
    queryKey: ['lottery', 'analysis', game, numDraws],
    queryFn: () => recommendationsApi.getLotteryAnalysis(game, numDraws),
    enabled: !!game,
  });
}
