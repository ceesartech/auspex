'use client';

// Route-level error boundary for the horse-racing race-detail page.
// Production Next.js hides client-render exceptions behind a generic
// "Application error" — this surfaces the actual message + stack so a
// crash here is diagnosable instead of opaque. Added 2026-06-09 to
// debug a reported client-side exception on race-tile clicks.

import { useEffect } from 'react';
import { Button } from '@/components/ui/button';

export default function RaceDetailError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Also log to the browser console for the full stack.
    // eslint-disable-next-line no-console
    console.error('Race detail render error:', error);
  }, [error]);

  return (
    <div className="space-y-4 rounded-lg border border-destructive/40 bg-destructive/5 p-6">
      <h2 className="text-lg font-semibold text-destructive">
        Race detail failed to render
      </h2>
      <p className="text-sm text-muted-foreground">
        The error below is shown so we can fix it. Please copy it.
      </p>
      <div className="space-y-2">
        <div className="text-xs font-medium uppercase text-muted-foreground">
          message
        </div>
        <pre className="overflow-auto rounded bg-muted p-3 text-xs">
          {error?.message || '(no message)'}
        </pre>
        {error?.digest && (
          <>
            <div className="text-xs font-medium uppercase text-muted-foreground">
              digest
            </div>
            <pre className="overflow-auto rounded bg-muted p-3 text-xs">
              {error.digest}
            </pre>
          </>
        )}
        {error?.stack && (
          <>
            <div className="text-xs font-medium uppercase text-muted-foreground">
              stack
            </div>
            <pre className="max-h-64 overflow-auto rounded bg-muted p-3 text-[10px] leading-relaxed">
              {error.stack}
            </pre>
          </>
        )}
      </div>
      <Button onClick={reset} variant="outline" size="sm">
        Try again
      </Button>
    </div>
  );
}
