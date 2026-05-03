'use client';

import { useState } from 'react';
import { useRecommendations, useHighValueBets } from '@/lib/hooks/use-recommendations';
import { RecommendationCard } from '@/components/recommendations/recommendation-card';
import { RecommendationFilters } from '@/components/recommendations/recommendation-filters';
import { LoadingPage } from '@/components/shared/loading';
import { ErrorDisplay } from '@/components/shared/error-display';
import { EmptyState } from '@/components/shared/empty-state';
import { Badge } from '@/components/ui/badge';
import { Star, Sparkles } from 'lucide-react';

export default function RecommendationsPage() {
  const [filters, setFilters] = useState<any>({});
  const { data: recommendations, isLoading, error, refetch } = useRecommendations(filters);
  const { data: highValue } = useHighValueBets({ min_ev: 0.05, limit: 5 });

  if (isLoading) return <LoadingPage message="Loading recommendations..." />;
  if (error) return <ErrorDisplay message="Failed to load recommendations" onRetry={refetch} />;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Recommendations</h1>
        <p className="text-muted-foreground">AI-curated betting recommendations</p>
      </div>

      <RecommendationFilters onFilterChange={setFilters} />

      {highValue && highValue.length > 0 && (
        <div>
          <h2 className="mb-3 text-lg font-semibold flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-yellow-500" />
            High Value Bets
            <Badge variant="warning">{highValue.length}</Badge>
          </h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {highValue.map((rec) => (
              <RecommendationCard key={rec.recommendation_id} recommendation={rec} />
            ))}
          </div>
        </div>
      )}

      <div>
        <h2 className="mb-3 text-lg font-semibold">All Recommendations</h2>
        {!recommendations?.length ? (
          <EmptyState
            icon={<Star className="h-12 w-12" />}
            title="No recommendations"
            description="Recommendations will appear when value bets are identified"
          />
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {recommendations.map((rec) => (
              <RecommendationCard key={rec.recommendation_id} recommendation={rec} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
