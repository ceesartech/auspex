'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Menu, Moon, Sun, Bell, BellOff, LogOut } from 'lucide-react';
import { cn } from '@/lib/utils/cn';
import { Button } from '@/components/ui/button';
import { useAuthStore } from '@/lib/store/auth-store';
import { useSettingsStore } from '@/lib/store/settings-store';
import { useLogout } from '@/lib/hooks/use-auth';
import { useState } from 'react';

const navItems = [
  { href: '/', label: 'Dashboard' },
  { href: '/predictions', label: 'Predictions' },
  { href: '/recommendations', label: 'Recommendations' },
  { href: '/races', label: 'Race cards' },
  { href: '/accumulator', label: 'Accumulator' },
  { href: '/analytics', label: 'Analytics' },
];

export function Header() {
  const pathname = usePathname();
  const { isAuthenticated } = useAuthStore();
  const { theme, setTheme, notifications, toggleNotifications } = useSettingsStore();
  const logout = useLogout();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  return (
    <header className="sticky top-0 z-40 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center">
        <Link href="/" className="mr-6 flex items-center gap-2">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo.svg" alt="" width={28} height={28} className="h-7 w-7" />
          <span className="text-xl font-semibold tracking-tight bg-gradient-to-br from-violet-500 to-cyan-500 bg-clip-text text-transparent">
            Auspex
          </span>
        </Link>

        <nav className="hidden md:flex items-center space-x-6 text-sm font-medium">
          {isAuthenticated &&
            navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  'transition-colors hover:text-foreground/80',
                  pathname === item.href ? 'text-foreground' : 'text-foreground/60'
                )}
              >
                {item.label}
              </Link>
            ))}
        </nav>

        <div className="flex flex-1 items-center justify-end space-x-2">
          <Button variant="ghost" size="icon" onClick={toggleTheme} aria-label="Toggle theme">
            {theme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          </Button>

          {isAuthenticated && (
            <>
              <Button variant="ghost" size="icon" onClick={toggleNotifications} aria-label="Toggle notifications">
                {notifications ? <Bell className="h-5 w-5" /> : <BellOff className="h-5 w-5" />}
              </Button>

              <Link href="/settings">
                <Button variant="ghost" size="sm">Settings</Button>
              </Link>

              <Button
                variant="ghost"
                size="icon"
                onClick={() => logout.mutate()}
                aria-label="Logout"
              >
                <LogOut className="h-5 w-5" />
              </Button>
            </>
          )}

          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle menu"
          >
            <Menu className="h-5 w-5" />
          </Button>
        </div>
      </div>

      {mobileMenuOpen && isAuthenticated && (
        <div className="border-t md:hidden">
          <nav className="container flex flex-col space-y-2 py-4">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  'rounded-md px-3 py-2 text-sm transition-colors hover:bg-accent',
                  pathname === item.href ? 'bg-accent text-foreground' : 'text-foreground/60'
                )}
                onClick={() => setMobileMenuOpen(false)}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
      )}
    </header>
  );
}
