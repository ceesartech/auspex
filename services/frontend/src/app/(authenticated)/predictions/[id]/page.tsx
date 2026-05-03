'use client';

import { useParams } from 'next/navigation';
import { usePrediction, useMatchStats, useMatchOdds, useOddsMovements } from '@/lib/hooks/use-predictions';
import { PredictionDetails } from '@/components/predictions/prediction-details';
import { ProbabilityChart } from '@/components/charts/probability-chart';
import { LoadingPage } from '@/components/shared/loading';
import { ErrorDisplay } from '@/components/shared/error-display';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { formatOdds, formatDateTime } from '@/lib/utils/format';
import { useSettingsStore } from '@/lib/store/settings-store';
import { ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import Link from 'next/link';

export default function PredictionDetailPage() {
  const params = useParams();
  const matchId = params.id as string;
  const { oddsFormat } = useSettingsStore();

  const { data: prediction, isLoading, error, refetch } = usePrediction(matchId);
  const { data: stats } = useMatchStats(matchId);
  const { data: odds } = useMatchOdds(matchId);
  const { data: movements } = useOddsMovements(matchId);

  if (isLoading) return <LoadingPage message="Loading prediction details..." />;
  if (error || !prediction) return <ErrorDisplay message="Failed to load prediction" onRetry={refetch} />;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Link href="/predictions">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="h-5 w-5" />
          </Button>
        </Link>
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            {prediction.match_info.home_team} vs {prediction.match_info.away_team}
          </h1>
          <p className="text-muted-foreground">{prediction.match_info.league_name}</p>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <PredictionDetails prediction={prediction} />
        </div>
        <div className="space-y-6">
          <ProbabilityChart probabilities={prediction.probabilities} />

          {odds && Array.isArray(odds) && odds.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Current Odds</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {odds.map((o: any, i: number) => (
                    <div key={i} className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">{o.bookmaker || 'N/A'}</span>
                      <div className="flex gap-2">
                        {o.outcome && <Badge variant="outline">{o.outcome}</Badge>}
                        <span className="font-medium">
                          {formatOdds(o.odds_decimal || o.odds, oddsFormat)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {stats && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Match Stats</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  {stats.team_form && (
                    <div>
                      <p className="text-muted-foreground">Home Form</p>
                      <p className="font-medium">{JSON.stringify(stats.team_form)}</p>
                    </div>
                  )}
                  {stats.h2h && (
                    <div>
                      <p className="text-muted-foreground">Head to Head</p>
                      <p className="font-medium">{JSON.stringify(stats.h2h)}</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
