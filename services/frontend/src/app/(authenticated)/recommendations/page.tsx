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

interface RecommendationFilterState {
  confidence_level?: string;
  market_type?: string;
  min_odds?: number;
  max_odds?: number;
  league?: string;
}

export default function RecommendationsPage() {
  const [filters, setFilters] = useState<RecommendationFilterState>({});
  // `league` is filtered client-side (the API has no league param); every
  // other filter goes to the server. The old default limit of 20 hid the
  // long tail of leagues entirely; 2000 is the route's ceiling and well
  // above the ~250 active recs a full slate produces.
  const { league, ...serverFilters } = filters;
  const { data: recommendations, isLoading, error, refetch } = useRecommendations({
    ...serverFilters,
    limit: 2000,
  });
  const { data: highValue } = useHighValueBets({ min_ev: 0.05, limit: 5 });

  const fetched = recommendations ?? [];
  const leagueOptions = Array.from(new Set(fetched.map((r) => r.match_info.league_name))).sort();
  const filtered = league ? fetched.filter((r) => r.match_info.league_name === league) : fetched;

  if (isLoading) return <LoadingPage message="Loading recommendations..." />;
  if (error) return <ErrorDisplay message="Failed to load recommendations" onRetry={refetch} />;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Recommendations</h1>
        <p className="text-muted-foreground">AI-curated betting recommendations</p>
      </div>

      <div className="space-y-2">
        <RecommendationFilters onFilterChange={setFilters} leagueOptions={leagueOptions} />
        <p className="text-sm text-muted-foreground" data-testid="recommendations-result-count">
          Showing {filtered.length} of {fetched.length} recommendations
        </p>
      </div>

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
        {filtered.length === 0 ? (
          <EmptyState
            icon={<Star className="h-12 w-12" />}
            title="No recommendations"
            description="Recommendations will appear when value bets are identified"
          />
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {filtered.map((rec) => (
              <RecommendationCard key={rec.recommendation_id} recommendation={rec} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
