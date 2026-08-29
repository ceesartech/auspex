import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { User } from '@/lib/types/user';

interface AuthState {
  user: User | null;
  token: string | null;
  // Session expiry in ms epoch, decoded from the JWT `exp` claim at login.
  // null means "no usable expiry" — treated as an invalid session so a
  // malformed or legacy persisted token fails closed instead of looping 401s.
  expiresAt: number | null;
  isAuthenticated: boolean;
  login: (token: string, user: User) => void;
  logout: () => void;
}

export function decodeTokenExpiry(token: string): number | null {
  try {
    const segment = token.split('.')[1];
    if (!segment) return null;
    const base64 = segment.replace(/-/g, '+').replace(/_/g, '/');
    const payload = JSON.parse(atob(base64));
    return typeof payload.exp === 'number' ? payload.exp * 1000 : null;
  } catch {
    return null;
  }
}

export function isSessionValid(
  state: Pick<AuthState, 'isAuthenticated' | 'expiresAt'>
): boolean {
  return (
    state.isAuthenticated &&
    state.expiresAt !== null &&
    Date.now() < state.expiresAt
  );
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      expiresAt: null,
      isAuthenticated: false,

      login: (token, user) => {
        if (typeof window !== 'undefined') {
          localStorage.setItem('auth_token', token);
        }
        set({
          token,
          user,
          expiresAt: decodeTokenExpiry(token),
          isAuthenticated: true,
        });
      },

      logout: () => {
        if (typeof window !== 'undefined') {
          localStorage.removeItem('auth_token');
        }
        set({ token: null, user: null, expiresAt: null, isAuthenticated: false });
      },
    }),
    { name: 'auth-storage' }
  )
);
