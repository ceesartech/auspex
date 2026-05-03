import { act } from '@testing-library/react';
import { useAuthStore } from '@/lib/store/auth-store';
import { useSettingsStore } from '@/lib/store/settings-store';
import { usePredictionsStore } from '@/lib/store/predictions-store';

// Reset stores between tests
beforeEach(() => {
  useAuthStore.setState({
    user: null,
    token: null,
    isAuthenticated: false,
  });
  useSettingsStore.setState({
    theme: 'system',
    currency: 'USD',
    oddsFormat: 'decimal',
    notifications: true,
    autoRefresh: true,
  });
  usePredictionsStore.setState({
    livePredictions: [],
    selectedSport: null,
    selectedLeague: null,
  });
});

describe('AuthStore', () => {
  it('initializes with unauthenticated state', () => {
    const state = useAuthStore.getState();
    expect(state.isAuthenticated).toBe(false);
    expect(state.user).toBeNull();
    expect(state.token).toBeNull();
  });

  it('login sets user and token', () => {
    act(() => {
      useAuthStore.getState().login('test-token', { username: 'ceesar', role: 'admin' });
    });

    const state = useAuthStore.getState();
    expect(state.isAuthenticated).toBe(true);
    expect(state.token).toBe('test-token');
    expect(state.user?.username).toBe('ceesar');
  });

  it('logout clears state', () => {
    act(() => {
      useAuthStore.getState().login('test-token', { username: 'ceesar', role: 'admin' });
    });

    act(() => {
      useAuthStore.getState().logout();
    });

    const state = useAuthStore.getState();
    expect(state.isAuthenticated).toBe(false);
    expect(state.user).toBeNull();
    expect(state.token).toBeNull();
  });
});

describe('SettingsStore', () => {
  it('initializes with defaults', () => {
    const state = useSettingsStore.getState();
    expect(state.theme).toBe('system');
    expect(state.currency).toBe('USD');
    expect(state.oddsFormat).toBe('decimal');
    expect(state.notifications).toBe(true);
    expect(state.autoRefresh).toBe(true);
  });

  it('setTheme updates theme', () => {
    act(() => {
      useSettingsStore.getState().setTheme('dark');
    });
    expect(useSettingsStore.getState().theme).toBe('dark');
  });

  it('setOddsFormat updates format', () => {
    act(() => {
      useSettingsStore.getState().setOddsFormat('american');
    });
    expect(useSettingsStore.getState().oddsFormat).toBe('american');
  });

  it('toggleNotifications flips value', () => {
    expect(useSettingsStore.getState().notifications).toBe(true);
    act(() => {
      useSettingsStore.getState().toggleNotifications();
    });
    expect(useSettingsStore.getState().notifications).toBe(false);
  });

  it('setCurrency updates currency', () => {
    act(() => {
      useSettingsStore.getState().setCurrency('EUR');
    });
    expect(useSettingsStore.getState().currency).toBe('EUR');
  });
});

describe('PredictionsStore', () => {
  const mockPrediction = {
    match_info: {
      match_id: '123',
      league_name: 'EPL',
      home_team: 'Arsenal',
      away_team: 'Chelsea',
      match_date: '2024-01-15',
      venue: 'Emirates',
    },
    predicted_outcome: 'Home Win',
    probabilities: { 'Home Win': 0.6, Draw: 0.25, 'Away Win': 0.15 },
    confidence: 0.75,
    model_version: 'ensemble_v1',
    timestamp: '2024-01-14T10:00:00Z',
    explanation: null,
    alternate_models: null,
  };

  it('initializes empty', () => {
    const state = usePredictionsStore.getState();
    expect(state.livePredictions).toEqual([]);
    expect(state.selectedSport).toBeNull();
  });

  it('setLivePredictions updates list', () => {
    act(() => {
      usePredictionsStore.getState().setLivePredictions([mockPrediction]);
    });
    expect(usePredictionsStore.getState().livePredictions).toHaveLength(1);
  });

  it('updatePrediction updates existing', () => {
    act(() => {
      usePredictionsStore.getState().setLivePredictions([mockPrediction]);
    });

    const updated = { ...mockPrediction, confidence: 0.8 };
    act(() => {
      usePredictionsStore.getState().updatePrediction(updated);
    });

    const state = usePredictionsStore.getState();
    expect(state.livePredictions).toHaveLength(1);
    expect(state.livePredictions[0].confidence).toBe(0.8);
  });

  it('updatePrediction adds new if not found', () => {
    act(() => {
      usePredictionsStore.getState().setLivePredictions([mockPrediction]);
    });

    const newPrediction = {
      ...mockPrediction,
      match_info: { ...mockPrediction.match_info, match_id: '456' },
    };
    act(() => {
      usePredictionsStore.getState().updatePrediction(newPrediction);
    });

    expect(usePredictionsStore.getState().livePredictions).toHaveLength(2);
  });

  it('setSelectedSport updates filter', () => {
    act(() => {
      usePredictionsStore.getState().setSelectedSport('soccer');
    });
    expect(usePredictionsStore.getState().selectedSport).toBe('soccer');
  });

  it('clearFilters resets filters', () => {
    act(() => {
      usePredictionsStore.getState().setSelectedSport('soccer');
      usePredictionsStore.getState().setSelectedLeague('EPL');
    });
    act(() => {
      usePredictionsStore.getState().clearFilters();
    });
    expect(usePredictionsStore.getState().selectedSport).toBeNull();
    expect(usePredictionsStore.getState().selectedLeague).toBeNull();
  });
});
