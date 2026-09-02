'use client';

import { useMemo } from 'react';
import { useUpcomingPredictions, useLivePredictions } from '@/lib/hooks/use-predictions';
import { usePredictionsStore } from '@/lib/store/predictions-store';
import { PredictionCard } from '@/components/predictions/prediction-card';
import { PredictionFilters } from '@/components/predictions/prediction-filters';
import { AccuracyWidget } from '@/components/predictions/accuracy-widget';
import { LoadingPage } from '@/components/shared/loading';
import { ErrorDisplay } from '@/components/shared/error-display';
import { EmptyState } from '@/components/shared/empty-state';
import { Badge } from '@/components/ui/badge';
import { Prediction } from '@/lib/types/prediction';
import { formatDate } from '@/lib/utils/format';
import { TrendingUp } from 'lucide-react';

// One row per market per match, so the key must include the market —
// match_id alone collides when a soccer sub-market (e.g. over_under) is
// selected and the API returns several rows for the same fixture.
function predictionKey(p: Prediction): string {
  return `${p.match_info.match_id}-${p.market ?? 'headline'}`;
}

interface DateGroup {
  date: string;
  predictions: Prediction[];
}

// Bucket predictions by kickoff date, preserving the API's match_date ASC
// ordering so the groups come out chronologically.
function groupByKickoffDate(predictions: Prediction[]): DateGroup[] {
  const groups: DateGroup[] = [];
  const index = new Map<string, DateGroup>();
  for (const p of predictions) {
    const date = formatDate(p.match_info.match_date);
    let group = index.get(date);
    if (!group) {
      group = { date, predictions: [] };
      index.set(date, group);
      groups.push(group);
    }
    group.predictions.push(p);
  }
  return groups;
}

export default function PredictionsPage() {
  const { selectedSport, selectedMarket, selectedLeague } = usePredictionsStore();
  // Fetch the full upcoming slate for the sport. With `market` omitted the
  // API returns ONE headline row per match for every sport, so 2000 rows
  // covers the entire slate (~700 matches today) with headroom. The old
  // limit=50 counted soccer's 19 market rows per match and silently
  // dropped every match past the first ~3 by kickoff — entire leagues like
  // the Premier League never rendered. League filtering happens
  // client-side so the league dropdown's options never collapse while
  // filtering.
  const { data: upcoming, isLoading, error, refetch } = useUpcomingPredictions({
    sport: selectedSport || undefined,
    market: selectedMarket || undefined,
    limit: 2000,
  });
  const { data: live } = useLivePredictions();

  const fetched = useMemo(() => upcoming ?? [], [upcoming]);
  const leagueOptions = useMemo(
    () => Array.from(new Set(fetched.map((p) => p.match_info.league_name))).sort(),
    [fetched]
  );
  const filtered = useMemo(
    () => (selectedLeague ? fetched.filter((p) => p.match_info.league_name === selectedLeague) : fetched),
    [fetched, selectedLeague]
  );
  const filteredLeagueCount = useMemo(
    () => new Set(filtered.map((p) => p.match_info.league_name)).size,
    [filtered]
  );
  const groups = useMemo(() => groupByKickoffDate(filtered), [filtered]);

  if (isLoading) return <LoadingPage message="Loading predictions..." />;
  if (error) return <ErrorDisplay message="Failed to load predictions" onRetry={refetch} />;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Predictions</h1>
          <p className="text-muted-foreground">AI-powered match predictions</p>
        </div>
        {live && live.length > 0 && (
          <Badge variant="success" className="animate-pulse">
            {live.length} Live
          </Badge>
        )}
      </div>

      <div className="space-y-2">
        <PredictionFilters leagueOptions={leagueOptions} />
        <p className="text-sm text-muted-foreground" data-testid="predictions-result-count">
          Showing {filtered.length} of {fetched.length} upcoming predictions across {filteredLeagueCount}{' '}
          {filteredLeagueCount === 1 ? 'league' : 'leagues'}
        </p>
      </div>

      {/* Phase 5: accuracy / ROI panel — scopes to the same filters
          as the predictions list so the headline numbers match what
          the user is currently viewing. */}
      <AccuracyWidget />

      {live && live.length > 0 && (
        <div>
          <h2 className="mb-3 text-lg font-semibold flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            Live Predictions
          </h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {live.map((prediction) => (
              <PredictionCard key={predictionKey(prediction)} prediction={prediction} />
            ))}
          </div>
        </div>
      )}

      <div>
        <h2 className="mb-3 text-lg font-semibold">Upcoming</h2>
        {groups.length === 0 ? (
          <EmptyState
            icon={<TrendingUp className="h-12 w-12" />}
            title="No predictions available"
            description="Predictions will appear when upcoming matches are analyzed"
          />
        ) : (
          <div className="space-y-6">
            {groups.map((group) => (
              <section key={group.date}>
                <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide text-muted-foreground">
                  {group.date}
                  <span className="ml-2 font-normal normal-case tracking-normal">
                    ({group.predictions.length})
                  </span>
                </h3>
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  {group.predictions.map((prediction) => (
                    <PredictionCard key={predictionKey(prediction)} prediction={prediction} />
                  ))}
                </div>
              </section>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
