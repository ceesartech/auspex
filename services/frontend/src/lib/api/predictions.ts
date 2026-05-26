import { apiClient } from './client';
import { Prediction, UpcomingMatch } from '@/lib/types/prediction';

export const predictionsApi = {
  async getPrediction(matchId: string, includeExplanation = true): Promise<Prediction> {
    const { data } = await apiClient.post('/api/v1/predictions/', {
      match_id: matchId,
      include_explanation: includeExplanation,
    });
    return data;
  },

  async getBulkPredictions(matchIds: string[]): Promise<Prediction[]> {
    const { data } = await apiClient.post('/api/v1/predictions/bulk', {
      match_ids: matchIds,
    });
    return data;
  },

  async getUpcomingPredictions(params?: {
    sport?: string;
    league?: string;
    limit?: number;
  }): Promise<Prediction[]> {
    const { data } = await apiClient.get('/api/v1/predictions/upcoming', { params });
    return data;
  },

  async getLivePredictions(): Promise<Prediction[]> {
    const { data } = await apiClient.get('/api/v1/predictions/live');
    return data;
  },

  async getUpcomingMatches(params?: {
    sport?: string;
    league?: string;
    days_ahead?: number;
    limit?: number;
  }): Promise<UpcomingMatch[]> {
    const { data } = await apiClient.get('/api/v1/matches/upcoming', { params });
    return data;
  },

  async getMatchDetail(matchId: string) {
    const { data } = await apiClient.get(`/api/v1/matches/${matchId}`);
    return data;
  },

  async getMatchStats(matchId: string) {
    const { data } = await apiClient.get(`/api/v1/matches/${matchId}/stats`);
    return data;
  },

  async getMatchOdds(matchId: string) {
    const { data } = await apiClient.get(`/api/v1/matches/${matchId}/odds`);
    return data;
  },

  async getOddsMovements(matchId: string, marketType = '1x2') {
    const { data } = await apiClient.get(`/api/v1/odds/movements/${matchId}`, {
      params: { market_type: marketType },
    });
    return data;
  },
};
