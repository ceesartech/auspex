'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Prediction } from '@/lib/types/prediction';
import { formatPercentage, formatDateTime, formatConfidenceLevel, getOutcomeColor } from '@/lib/utils/format';
import { cn } from '@/lib/utils/cn';

interface PredictionDetailsProps {
  prediction: Prediction;
}

export function PredictionDetails({ prediction }: PredictionDetailsProps) {
  const { match_info, predicted_outcome, probabilities, confidence, model_version, timestamp, explanation } = prediction;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-2">
            <CardTitle className="min-w-0 break-words">{match_info.home_team} vs {match_info.away_team}</CardTitle>
            <Badge variant="outline" className="shrink-0">{match_info.league_name}</Badge>
          </div>
          <div className="flex gap-2 text-sm text-muted-foreground">
            <span>{formatDateTime(match_info.match_date)}</span>
            {match_info.venue && <span>at {match_info.venue}</span>}
          </div>
        </CardHeader>
        <CardContent>
          {/* min-w-0 on both columns: grid tracks default to minmax(auto,1fr),
              so long outcome names (e.g. tennis players) would otherwise
              force the column wider than the card. */}
          <div className="grid gap-6 md:grid-cols-2">
            <div className="space-y-4 min-w-0">
              <div>
                <p className="text-sm text-muted-foreground">Predicted Outcome</p>
                <p className={cn('text-3xl font-bold break-words', getOutcomeColor(predicted_outcome))}>
                  {predicted_outcome}
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Confidence</p>
                <div className="flex items-center gap-2">
                  <span className="text-2xl font-bold">{formatPercentage(confidence)}</span>
                  <Badge variant={confidence >= 0.7 ? 'success' : confidence >= 0.5 ? 'warning' : 'secondary'}>
                    {formatConfidenceLevel(confidence)}
                  </Badge>
                </div>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Model</p>
                <p className="text-sm">{model_version}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Generated</p>
                <p className="text-sm">{formatDateTime(timestamp)}</p>
              </div>
            </div>

            <div className="min-w-0">
              <p className="mb-3 text-sm font-medium">Probabilities</p>
              <div className="space-y-3">
                {Object.entries(probabilities).map(([outcome, prob]) => (
                  <div key={outcome}>
                    <div className="flex justify-between gap-2 text-sm">
                      <span className="min-w-0 break-words">{outcome}</span>
                      <span className="shrink-0 font-medium">{formatPercentage(prob)}</span>
                    </div>
                    <div className="mt-1 h-2 rounded-full bg-muted">
                      <div
                        className={cn(
                          'h-2 rounded-full transition-all',
                          outcome === predicted_outcome ? 'bg-primary' : 'bg-muted-foreground/30'
                        )}
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {explanation && Object.keys(explanation).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Prediction Explanation</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(explanation).map(([key, value]) => (
                <div key={key} className="flex justify-between gap-4 border-b pb-2 last:border-0">
                  <span className="min-w-0 break-words text-sm text-muted-foreground">
                    {key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                  </span>
                  <span className="min-w-0 break-words text-right text-sm font-medium">
                    {typeof value === 'number' ? value.toFixed(4) : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
