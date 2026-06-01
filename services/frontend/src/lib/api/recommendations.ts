import { apiClient } from './client';
import { Recommendation, Accumulator, AccumulatorRequest, ModelPerformance } from '@/lib/types/recommendation';
import { LotteryDraw, LotteryAnalysis, LotteryRecommendations } from '@/lib/types/user';

export const recommendationsApi = {
  async getRecommendations(params?: {
    confidence_level?: string;
    market_type?: string;
    min_odds?: number;
    max_odds?: number;
    limit?: number;
  }): Promise<Recommendation[]> {
    const { data } = await apiClient.get('/api/v1/recommendations/', { params });
    return data;
  },

  async getHighValueBets(params?: {
    min_ev?: number;
    limit?: number;
  }): Promise<Recommendation[]> {
    const { data } = await apiClient.get('/api/v1/recommendations/high-value', { params });
    return data;
  },

  async buildAccumulator(request: AccumulatorRequest): Promise<Accumulator> {
    const { data } = await apiClient.post('/api/v1/recommendations/accumulator', request);
    return data;
  },

  async getModelPerformance(params?: {
    model_name?: string;
    sport?: string;
    limit?: number;
  }): Promise<ModelPerformance[]> {
    const { data } = await apiClient.get('/api/v1/models/performance', { params });
    return data;
  },

  async getActiveModels() {
    const { data } = await apiClient.get('/api/v1/models/active');
    return data;
  },

  async compareModels(modelNames: string[], sport?: string) {
    const { data } = await apiClient.get('/api/v1/models/comparison', {
      params: { model_names: modelNames.join(','), sport },
    });
    return data;
  },

  async getLotteryDraws(game?: string, limit = 20): Promise<LotteryDraw[]> {
    const { data } = await apiClient.get('/api/v1/lottery/draws', {
      params: { game, limit },
    });
    return data;
  },

  async getLotteryAnalysis(game: string, numDraws = 100): Promise<LotteryAnalysis> {
    const { data } = await apiClient.get('/api/v1/lottery/analysis', {
      params: { game, num_draws: numDraws },
    });
    return data;
  },

  async getLotteryRecommendations(
    game: string,
    strategy = 'blend',
    numSets = 5,
  ): Promise<LotteryRecommendations> {
    const { data } = await apiClient.post('/api/v1/lottery/recommendations', null, {
      params: { game, strategy, num_sets: numSets },
    });
    return data;
  },
};
