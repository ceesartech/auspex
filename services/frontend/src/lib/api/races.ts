import { apiClient } from './client';
import { RaceDetail, RaceSummary } from '@/lib/types/race';

// Horse-racing API client. Mirrors lib/api/predictions.ts but on the
// /api/v1/races namespace (multi-runner schema, migration 013).

export interface ListRacesParams {
  // Defaults to 'scheduled' server-side. 'finished' switches to a
  // lookback (graded races); 'live' is the small window where races
  // have flipped status but not yet recorded results.
  status?: 'scheduled' | 'live' | 'finished';
  days_ahead?: number;
  limit?: number;
}

export const racesApi = {
  async listRaces(params?: ListRacesParams): Promise<RaceSummary[]> {
    const { data } = await apiClient.get('/api/v1/races', { params });
    return data;
  },

  async getRaceDetail(raceId: string): Promise<RaceDetail> {
    const { data } = await apiClient.get(`/api/v1/races/${raceId}`);
    return data;
  },
};
