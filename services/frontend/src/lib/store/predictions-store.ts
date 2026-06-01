import { create } from 'zustand';
import { Prediction } from '@/lib/types/prediction';

interface PredictionsState {
  livePredictions: Prediction[];
  selectedSport: string | null;
  selectedLeague: string | null;
  selectedMarket: string | null;
  setLivePredictions: (predictions: Prediction[]) => void;
  updatePrediction: (prediction: Prediction) => void;
  setSelectedSport: (sport: string | null) => void;
  setSelectedLeague: (league: string | null) => void;
  setSelectedMarket: (market: string | null) => void;
  clearFilters: () => void;
}

export const usePredictionsStore = create<PredictionsState>()((set) => ({
  livePredictions: [],
  selectedSport: null,
  selectedLeague: null,
  selectedMarket: null,

  setLivePredictions: (predictions) => set({ livePredictions: predictions }),

  updatePrediction: (prediction) =>
    set((state) => {
      const index = state.livePredictions.findIndex(
        (p) => p.match_info.match_id === prediction.match_info.match_id
      );
      if (index >= 0) {
        const updated = [...state.livePredictions];
        updated[index] = prediction;
        return { livePredictions: updated };
      }
      return { livePredictions: [...state.livePredictions, prediction] };
    }),

  // Changing sport resets the market filter because a market that was
  // valid for the previous sport (e.g. 'puck_line' under NHL) makes no
  // sense after switching (soccer has no puck-line market).
  setSelectedSport: (sport) =>
    set((state) =>
      state.selectedSport === sport
        ? { selectedSport: sport }
        : { selectedSport: sport, selectedMarket: null }
    ),
  setSelectedLeague: (league) => set({ selectedLeague: league }),
  setSelectedMarket: (market) => set({ selectedMarket: market }),
  clearFilters: () => set({ selectedSport: null, selectedLeague: null, selectedMarket: null }),
}));
