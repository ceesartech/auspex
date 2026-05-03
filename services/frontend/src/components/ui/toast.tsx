'use client';

import { Toaster as SonnerToaster, toast } from 'sonner';
import { useSettingsStore } from '@/lib/store/settings-store';

function Toaster() {
  const { theme } = useSettingsStore();

  return (
    <SonnerToaster
      theme={theme === 'system' ? undefined : theme}
      position="bottom-right"
      toastOptions={{
        classNames: {
          toast: 'group border bg-background text-foreground',
          description: 'text-muted-foreground',
          actionButton: 'bg-primary text-primary-foreground',
          cancelButton: 'bg-muted text-muted-foreground',
        },
      }}
    />
  );
}

export { Toaster, toast };
