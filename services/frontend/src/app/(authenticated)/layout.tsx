'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Header } from '@/components/layout/header';
import { Sidebar } from '@/components/layout/sidebar';
import { Footer } from '@/components/layout/footer';
import { LoadingSpinner } from '@/components/shared/loading';
import { isSessionValid, useAuthStore } from '@/lib/store/auth-store';

export default function AuthenticatedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const expiresAt = useAuthStore((s) => s.expiresAt);
  const logout = useAuthStore((s) => s.logout);

  // The persisted store only reflects localStorage on the client; gating on
  // `mounted` keeps SSR and the first client render identical, so the
  // redirect decision never races rehydration (and React never sees a
  // hydration mismatch on this subtree).
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const sessionValid = mounted && isSessionValid({ isAuthenticated, expiresAt });

  // Both effects below can end the session; whichever fires first wins.
  // Without the guard, the state change from logout() re-fires the other
  // effect and its plain '/login' replace clobbers the '?expired=1' URL.
  const sessionEndedRef = useRef(false);
  const endSession = useCallback(
    (expired: boolean) => {
      if (sessionEndedRef.current) return;
      sessionEndedRef.current = true;
      logout();
      router.replace(expired ? '/login?expired=1' : '/login');
    },
    [logout, router]
  );

  useEffect(() => {
    if (!mounted || sessionValid) return;
    // A persisted flag without a live token means the session expired (or
    // predates expiry tracking) — say so on the login page rather than
    // rendering a dashboard whose API calls all 401.
    endSession(isAuthenticated);
  }, [mounted, sessionValid, isAuthenticated, endSession]);

  // End the session at the moment the token expires while the app is open,
  // instead of waiting for the next API call to 401.
  useEffect(() => {
    if (!sessionValid || !expiresAt) return;
    const timer = setTimeout(
      () => endSession(true),
      expiresAt - Date.now()
    );
    return () => clearTimeout(timer);
  }, [sessionValid, expiresAt, endSession]);

  if (!sessionValid) {
    // The redirect to /login is client-side, so if the JS bundle fails to
    // load (e.g. stale cached HTML referencing chunks replaced by a fresh
    // deploy) this spinner is all the user ever sees. The plain anchor is
    // server-rendered HTML and works even with dead JS.
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-3">
        <LoadingSpinner size="lg" />
        <p className="text-sm text-muted-foreground">Loading your session…</p>
        {/* eslint-disable-next-line @next/next/no-html-link-for-pages */}
        <a href="/login" className="text-sm text-primary underline-offset-4 hover:underline">
          Taking too long? Continue to sign in
        </a>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <div className="flex flex-1">
        <Sidebar />
        <main className="flex-1 container py-6">{children}</main>
      </div>
      <Footer />
    </div>
  );
}
