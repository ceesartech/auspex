'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  TrendingUp,
  Star,
  Layers,
  BarChart3,
  Settings,
  Ticket,
  Award,
} from 'lucide-react';
import { cn } from '@/lib/utils/cn';

const sidebarItems = [
  { href: '/', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/predictions', label: 'Predictions', icon: TrendingUp },
  { href: '/recommendations', label: 'Recommendations', icon: Star },
  // Horse racing uses its own schema (migration 013) — separate
  // page rather than shoehorning multi-runner shapes into the
  // matches-based predictions/recommendations pages.
  { href: '/races', label: 'Race cards', icon: Award },
  { href: '/accumulator', label: 'Accumulator', icon: Layers },
  { href: '/analytics', label: 'Analytics', icon: BarChart3 },
  { href: '/lottery', label: 'Lottery', icon: Ticket },
  { href: '/settings', label: 'Settings', icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden lg:flex lg:flex-col lg:w-64 lg:border-r bg-background">
      <nav className="flex flex-col gap-1 p-4">
        {sidebarItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;

          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-all hover:bg-accent',
                isActive
                  ? 'bg-accent text-accent-foreground font-medium'
                  : 'text-muted-foreground'
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
