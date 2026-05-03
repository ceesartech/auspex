'use client';

import { DashboardOverview } from '@/components/dashboard/overview';
import { UpcomingMatches } from '@/components/dashboard/upcoming-matches';
import { RecentPerformance } from '@/components/dashboard/recent-performance';

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">
          Your betting analytics overview
        </p>
      </div>

      <DashboardOverview />

      <div className="grid gap-6 lg:grid-cols-2">
        <UpcomingMatches />
        <RecentPerformance />
      </div>
    </div>
  );
}
