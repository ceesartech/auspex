import { apiClient } from './client';
import {
  AuthResponse, LoginRequest, UserPreferences,
  UserPreferencesUpdate, DashboardStats,
} from '@/lib/types/user';

export const userApi = {
  async login(request: LoginRequest): Promise<AuthResponse> {
    const { data } = await apiClient.post('/api/v1/user/login', request);
    return data;
  },

  async getPreferences(): Promise<UserPreferences> {
    const { data } = await apiClient.get('/api/v1/user/preferences');
    return data;
  },

  async updatePreferences(updates: UserPreferencesUpdate) {
    const { data } = await apiClient.put('/api/v1/user/preferences', updates);
    return data;
  },

  async getBettingHistory(params?: {
    status?: string;
    bookmaker?: string;
    limit?: number;
    offset?: number;
  }) {
    const { data } = await apiClient.get('/api/v1/user/betting-history', { params });
    return data;
  },

  async recordBet(bet: {
    bookmaker: string;
    bet_type: string;
    selection: string;
    odds: number;
    stake: number;
    recommendation_id?: string;
    match_id?: string;
    notes?: string;
  }) {
    const { data } = await apiClient.post('/api/v1/user/betting-history', bet);
    return data;
  },

  async getDashboard(): Promise<DashboardStats> {
    const { data } = await apiClient.get('/api/v1/user/dashboard');
    return data;
  },

  async healthCheck() {
    const { data } = await apiClient.get('/health');
    return data;
  },
};
