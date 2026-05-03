'use client';

import {
  DollarSign,
  TrendingUp,
  TrendingDown,
  Target,
  Activity,
  Calendar,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useDashboard } from '@/lib/hooks/use-auth';
import { formatCurrency, formatPercentage } from '@/lib/utils/format';
import { CardSkeleton } from '@/components/shared/loading';
import { ErrorDisplay } from '@/components/shared/error-display';

export function DashboardOverview() {
  const { data: stats, isLoading, error, refetch } = useDashboard();

  if (isLoading) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <CardSkeleton key={i} />
        ))}
      </div>
    );
  }

  if (error || !stats) {
    return <ErrorDisplay message="Failed to load dashboard stats" onRetry={refetch} />;
  }

  const cards = [
    {
      title: 'Total P/L',
      value: formatCurrency(stats.profit_loss),
      icon: stats.profit_loss >= 0 ? TrendingUp : TrendingDown,
      description: `ROI: ${formatPercentage(stats.roi_pct)}`,
      color: stats.profit_loss >= 0 ? 'text-green-600' : 'text-red-600',
    },
    {
      title: 'Win Rate',
      value: formatPercentage(stats.win_rate),
      icon: Target,
      description: `${stats.total_bets} total bets`,
      color: 'text-blue-600',
    },
    {
      title: 'Active Bets',
      value: stats.active_bets.toString(),
      icon: Activity,
      description: `${formatCurrency(stats.total_staked)} staked`,
      color: 'text-orange-600',
    },
    {
      title: 'Upcoming',
      value: stats.upcoming_matches.toString(),
      icon: Calendar,
      description: `${stats.active_recommendations} recommendations`,
      color: 'text-purple-600',
    },
  ];

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {cards.map((card) => {
        const Icon = card.icon;
        return (
          <Card key={card.title}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{card.title}</CardTitle>
              <Icon className={`h-4 w-4 ${card.color}`} />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${card.color}`}>{card.value}</div>
              <p className="text-xs text-muted-foreground">{card.description}</p>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
