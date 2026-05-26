'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';
import { userApi } from '@/lib/api/user';
import { useAuthStore } from '@/lib/store/auth-store';
import { LoginRequest } from '@/lib/types/user';

export function useLogin() {
  const router = useRouter();
  const { login } = useAuthStore();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: LoginRequest) => userApi.login(request),
    onSuccess: (data) => {
      login(data.access_token, data.user);
      queryClient.invalidateQueries({ queryKey: ['user'] });
      router.push('/');
    },
  });
}

export function useLogout() {
  const router = useRouter();
  const { logout } = useAuthStore();
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      logout();
      queryClient.clear();
    },
    onSuccess: () => {
      router.push('/login');
    },
  });
}

export function useUserPreferences() {
  const { isAuthenticated } = useAuthStore();

  return useQuery({
    queryKey: ['user', 'preferences'],
    queryFn: () => userApi.getPreferences(),
    enabled: isAuthenticated,
  });
}

export function useUpdatePreferences() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: userApi.updatePreferences,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['user', 'preferences'] });
    },
  });
}

export function useDashboard() {
  const { isAuthenticated } = useAuthStore();

  return useQuery({
    queryKey: ['user', 'dashboard'],
    queryFn: () => userApi.getDashboard(),
    enabled: isAuthenticated,
    refetchInterval: 120000,
  });
}

export function useBettingHistory(params?: {
  status?: string;
  bookmaker?: string;
  limit?: number;
  offset?: number;
}) {
  const { isAuthenticated } = useAuthStore();

  return useQuery({
    queryKey: ['user', 'betting-history', params],
    queryFn: () => userApi.getBettingHistory(params),
    enabled: isAuthenticated,
  });
}

export function useRecordBet() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: userApi.recordBet,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['user', 'betting-history'] });
      queryClient.invalidateQueries({ queryKey: ['user', 'dashboard'] });
    },
  });
}
