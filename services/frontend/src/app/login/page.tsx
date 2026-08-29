'use client';

import { Suspense, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { loginSchema, type LoginFormData } from '@/lib/utils/validation';
import { useLogin } from '@/lib/hooks/use-auth';
import { isSessionValid, useAuthStore } from '@/lib/store/auth-store';
import { toast } from '@/components/ui/toast';

function ExpiredSessionNotice() {
  const searchParams = useSearchParams();
  if (searchParams.get('expired') !== '1') return null;
  return (
    <div
      role="alert"
      className="mb-4 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-600 dark:text-amber-400"
    >
      Your session has expired — please sign in again.
    </div>
  );
}

export default function LoginPage() {
  const login = useLogin();
  const router = useRouter();
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const expiresAt = useAuthStore((s) => s.expiresAt);
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
  });

  // A live session has no business on the login form — go to the dashboard.
  useEffect(() => {
    if (isSessionValid({ isAuthenticated, expiresAt })) {
      router.replace('/');
    }
  }, [isAuthenticated, expiresAt, router]);

  const onSubmit = async (data: LoginFormData) => {
    try {
      await login.mutateAsync(data);
    } catch {
      toast('Login failed. Please check your credentials.');
    }
  };

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden p-4">
      {/* Ambient gradient backdrop */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 -z-10
                   bg-[radial-gradient(60%_50%_at_50%_0%,rgba(124,58,237,0.18),transparent_70%),
                       radial-gradient(40%_40%_at_80%_80%,rgba(6,182,212,0.12),transparent_70%)]"
      />
      <Card className="w-full max-w-md border-border/60 shadow-xl shadow-violet-500/5 backdrop-blur">
        <CardHeader className="items-center text-center">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo.svg" alt="" width={48} height={48} className="mb-3 h-12 w-12" />
          <CardTitle className="text-2xl tracking-tight">
            <span className="bg-gradient-to-br from-violet-500 to-cyan-500 bg-clip-text text-transparent">
              Auspex
            </span>
          </CardTitle>
          <CardDescription>Sign in to your prediction dashboard</CardDescription>
        </CardHeader>
        <CardContent>
          <Suspense fallback={null}>
            <ExpiredSessionNotice />
          </Suspense>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            <div>
              <label className="text-sm font-medium">Username or email</label>
              <Input
                {...register('username')}
                placeholder="admin or you@example.com"
                error={errors.username?.message}
              />
            </div>
            <div>
              <label className="text-sm font-medium">Password</label>
              <Input
                type="password"
                {...register('password')}
                placeholder="Enter your password"
                error={errors.password?.message}
              />
            </div>
            <Button type="submit" className="w-full" loading={login.isPending}>
              Sign In
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
