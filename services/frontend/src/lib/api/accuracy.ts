import { apiClient } from './client';

export interface MarketAccuracy {
  sport: string;
  prediction_type: string;
  model_name: string;
  total: number;
  graded: number;
  correct: number;
  accuracy: number;
}

export interface RecsPnL {
  settled: number;
  won: number;
  lost: number;
  void: number;
  total_staked: number;
  total_profit_loss: number;
  roi_pct: number;
}

export interface AccuracySummary {
  days: number;
  sport_filter: string | null;
  market_filter: string | null;
  total: number;
  graded: number;
  correct: number;
  accuracy: number;
  by_market: MarketAccuracy[];
  recs: RecsPnL;
}

export const accuracyApi = {
  async getSummary(params?: {
    sport?: string;
    market?: string;
    days?: number;
  }): Promise<AccuracySummary> {
    const { data } = await apiClient.get('/api/v1/accuracy/summary', { params });
    return data;
  },
};
