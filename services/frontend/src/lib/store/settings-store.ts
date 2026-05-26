import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface SettingsState {
  theme: 'light' | 'dark' | 'system';
  currency: string;
  oddsFormat: 'decimal' | 'american' | 'fractional';
  notifications: boolean;
  autoRefresh: boolean;
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
  setCurrency: (currency: string) => void;
  setOddsFormat: (format: 'decimal' | 'american' | 'fractional') => void;
  toggleNotifications: () => void;
  toggleAutoRefresh: () => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      theme: 'system',
      currency: 'USD',
      oddsFormat: 'decimal',
      notifications: true,
      autoRefresh: true,

      setTheme: (theme) => set({ theme }),
      setCurrency: (currency) => set({ currency }),
      setOddsFormat: (format) => set({ oddsFormat: format }),
      toggleNotifications: () => set((s) => ({ notifications: !s.notifications })),
      toggleAutoRefresh: () => set((s) => ({ autoRefresh: !s.autoRefresh })),
    }),
    { name: 'settings-storage' }
  )
);
