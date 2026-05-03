'use client';

import { useUpcomingPredictions, useLivePredictions } from '@/lib/hooks/use-predictions';
import { usePredictionsStore } from '@/lib/store/predictions-store';
import { PredictionCard } from '@/components/predictions/prediction-card';
import { PredictionFilters } from '@/components/predictions/prediction-filters';
import { LoadingPage } from '@/components/shared/loading';
import { ErrorDisplay } from '@/components/shared/error-display';
import { EmptyState } from '@/components/shared/empty-state';
import { Badge } from '@/components/ui/badge';
import { TrendingUp } from 'lucide-react';

export default function PredictionsPage() {
  const { selectedSport } = usePredictionsStore();
  const { data: upcoming, isLoading, error, refetch } = useUpcomingPredictions({
    sport: selectedSport || undefined,
    limit: 50,
  });
  const { data: live } = useLivePredictions();

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

      <PredictionFilters />

      {live && live.length > 0 && (
        <div>
          <h2 className="mb-3 text-lg font-semibold flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
            Live Predictions
          </h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {live.map((prediction) => (
              <PredictionCard key={prediction.match_info.match_id} prediction={prediction} />
            ))}
          </div>
        </div>
      )}

      <div>
        <h2 className="mb-3 text-lg font-semibold">Upcoming</h2>
        {!upcoming?.length ? (
          <EmptyState
            icon={<TrendingUp className="h-12 w-12" />}
            title="No predictions available"
            description="Predictions will appear when upcoming matches are analyzed"
          />
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {upcoming.map((prediction) => (
              <PredictionCard key={prediction.match_info.match_id} prediction={prediction} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
