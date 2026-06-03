'use client';

import { useRaces } from '@/lib/hooks/use-races';
import { RaceCard } from '@/components/races/race-card';
import { LoadingPage } from '@/components/shared/loading';
import { ErrorDisplay } from '@/components/shared/error-display';
import { EmptyState } from '@/components/shared/empty-state';
import { Badge } from '@/components/ui/badge';
import { Award } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';

type RaceStatus = 'scheduled' | 'finished';

export default function RacesPage() {
  const [statusFilter, setStatusFilter] = useState<RaceStatus>('scheduled');
  const { data: races, isLoading, error, refetch } = useRaces({
    status: statusFilter,
    days_ahead: statusFilter === 'finished' ? 7 : 2,
    limit: 200,
  });

  if (isLoading) return <LoadingPage message="Loading race cards..." />;
  if (error) return <ErrorDisplay message="Failed to load race cards" onRetry={refetch} />;

  const totalPicks = races?.reduce((acc, r) => acc + r.recommendation_count, 0) ?? 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Race cards</h1>
          <p className="text-muted-foreground">
            Horse racing — UK & Ireland, market-consensus predictions
          </p>
        </div>
        {totalPicks > 0 && (
          <Badge variant="success" className="flex items-center gap-1">
            <Award className="h-3 w-3" />
            {totalPicks} value picks across {races?.length ?? 0} races
          </Badge>
        )}
      </div>

      {/* Status switcher: scheduled (today + tomorrow) vs finished
          (past 7 days). The detail page also shows grading info, so
          the user can drill into a finished race to see who won and
          how the picks settled. */}
      <div className="flex gap-2">
        <Button
          variant={statusFilter === 'scheduled' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setStatusFilter('scheduled')}
        >
          Upcoming
        </Button>
        <Button
          variant={statusFilter === 'finished' ? 'default' : 'outline'}
          size="sm"
          onClick={() => setStatusFilter('finished')}
        >
          Recent results
        </Button>
      </div>

      {!races?.length ? (
        <EmptyState
          icon={<Award className="h-12 w-12" />}
          title={
            statusFilter === 'scheduled'
              ? 'No upcoming races'
              : 'No recent results'
          }
          description={
            statusFilter === 'scheduled'
              ? 'Race cards will appear once the day’s entries are ingested.'
              : 'Finished races from the last 7 days will appear here once graded.'
          }
        />
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {races.map((race) => (
            <RaceCard key={race.race_id} race={race} />
          ))}
        </div>
      )}
    </div>
  );
}
