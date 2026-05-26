import { create } from 'zustand';
import { Prediction } from '@/lib/types/prediction';

interface PredictionsState {
  livePredictions: Prediction[];
  selectedSport: string | null;
  selectedLeague: string | null;
  setLivePredictions: (predictions: Prediction[]) => void;
  updatePrediction: (prediction: Prediction) => void;
  setSelectedSport: (sport: string | null) => void;
  setSelectedLeague: (league: string | null) => void;
  clearFilters: () => void;
}

export const usePredictionsStore = create<PredictionsState>()((set) => ({
  livePredictions: [],
  selectedSport: null,
  selectedLeague: null,

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

  setSelectedSport: (sport) => set({ selectedSport: sport }),
  setSelectedLeague: (league) => set({ selectedLeague: league }),
  clearFilters: () => set({ selectedSport: null, selectedLeague: null }),
}));
