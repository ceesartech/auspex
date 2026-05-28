import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import '@/styles/globals.css';
import { ThemeProvider } from '@/components/providers/theme-provider';
import { QueryProvider } from '@/components/providers/query-provider';
import { Toaster } from '@/components/ui/toast';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: {
    default: 'Auspex',
    template: '%s · Auspex',
  },
  description: 'Auspex — sports prediction & recommendation analytics.',
  manifest: '/manifest.json',
  icons: {
    icon: [{ url: '/logo.svg', type: 'image/svg+xml' }],
  },
};

export const viewport: Viewport = {
  themeColor: '#7c3aed',
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
          <QueryProvider>
            {children}
            <Toaster />
          </QueryProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
