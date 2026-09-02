'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Prediction } from '@/lib/types/prediction';
import { formatPercentage, formatDateTime, formatConfidenceLevel, getOutcomeColor } from '@/lib/utils/format';
import { cn } from '@/lib/utils/cn';

interface PredictionDetailsProps {
  prediction: Prediction;
}

/**
 * Markets with more outcomes than this (soccer asian_handicap has 51,
 * correct_score 13, over_under 12) are sorted by probability and capped
 * to a scrolling box. Small markets keep their natural order (home /
 * draw / away reads better than draw / home / away).
 */
const LARGE_MARKET_THRESHOLD = 8;

function formatExplanationValue(value: unknown): string {
  if (typeof value === 'number') return value.toFixed(4);
  if (value === null || value === undefined) return '—';
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

export function PredictionDetails({ prediction }: PredictionDetailsProps) {
  const { match_info, predicted_outcome, probabilities, confidence, model_version, timestamp, explanation } = prediction;

  const entries = Object.entries(probabilities ?? {});
  const isLargeMarket = entries.length > LARGE_MARKET_THRESHOLD;
  const outcomes = isLargeMarket ? [...entries].sort((a, b) => b[1] - a[1]) : entries;

  return (
    <div className="min-w-0 space-y-6">
      <Card className="min-w-0 overflow-hidden">
        <CardHeader>
          <div className="flex min-w-0 items-start justify-between gap-2">
            <CardTitle className="min-w-0 break-words">
              {match_info.home_team} vs {match_info.away_team}
            </CardTitle>
            <Badge variant="outline" className="min-w-0 max-w-[45%] shrink-0" title={match_info.league_name}>
              <span className="truncate">{match_info.league_name}</span>
            </Badge>
          </div>
          <div className="flex min-w-0 flex-wrap gap-x-2 text-sm text-muted-foreground">
            <span className="min-w-0 break-words">{formatDateTime(match_info.match_date)}</span>
            {match_info.venue && <span className="min-w-0 break-words">at {match_info.venue}</span>}
          </div>
        </CardHeader>
        <CardContent className="min-w-0">
          {/* min-w-0 on both columns: grid tracks default to minmax(auto,1fr),
              so long outcome names (e.g. tennis players) would otherwise
              force the column wider than the card. */}
          <div className="grid min-w-0 gap-6 md:grid-cols-2">
            <div className="min-w-0 space-y-4">
              <div className="min-w-0">
                <p className="text-sm text-muted-foreground">Predicted Outcome</p>
                <p className={cn('min-w-0 break-words text-3xl font-bold', getOutcomeColor(predicted_outcome))}>
                  {predicted_outcome}
                </p>
              </div>
              <div className="min-w-0">
                <p className="text-sm text-muted-foreground">Confidence</p>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-2xl font-bold">{formatPercentage(confidence)}</span>
                  <Badge variant={confidence >= 0.7 ? 'success' : confidence >= 0.5 ? 'warning' : 'secondary'}>
                    {formatConfidenceLevel(confidence)}
                  </Badge>
                </div>
              </div>
              <div className="min-w-0">
                <p className="text-sm text-muted-foreground">Model</p>
                {/* break-all: version strings are one unbroken token. */}
                <p className="min-w-0 break-all text-sm">{model_version}</p>
              </div>
              <div className="min-w-0">
                <p className="text-sm text-muted-foreground">Generated</p>
                <p className="min-w-0 break-words text-sm">{formatDateTime(timestamp)}</p>
              </div>
            </div>

            <div className="min-w-0">
              <p className="mb-3 text-sm font-medium">
                Probabilities
                {isLargeMarket && (
                  <span className="ml-1 font-normal text-muted-foreground">({entries.length}, sorted)</span>
                )}
              </p>
              <div className={cn('space-y-3', isLargeMarket && 'max-h-80 overflow-y-auto pr-2')}>
                {outcomes.map(([outcome, prob]) => {
                  const isPick = outcome === predicted_outcome;
                  return (
                    <div key={outcome} className="min-w-0">
                      <div className="flex min-w-0 justify-between gap-2 text-sm">
                        <span className={cn('min-w-0 break-words', isPick && 'font-semibold')}>{outcome}</span>
                        <span className="shrink-0 font-medium tabular-nums">{formatPercentage(prob)}</span>
                      </div>
                      <div className="mt-1 h-2 rounded-full bg-muted">
                        <div
                          className={cn(
                            'h-2 rounded-full transition-all',
                            isPick ? 'bg-primary' : 'bg-muted-foreground/30'
                          )}
                          style={{ width: `${Math.min(100, Math.max(0, prob * 100))}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {explanation && Object.keys(explanation).length > 0 && (
        <Card className="min-w-0 overflow-hidden">
          <CardHeader>
            <CardTitle className="text-lg">Prediction Explanation</CardTitle>
          </CardHeader>
          <CardContent className="min-w-0">
            <div className="space-y-3">
              {Object.entries(explanation).map(([key, value]) => (
                <div key={key} className="flex min-w-0 justify-between gap-4 border-b pb-2 last:border-0">
                  <span className="min-w-0 break-words text-sm text-muted-foreground">
                    {key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                  </span>
                  {/* break-all: values are often unbroken tokens (ids, JSON). */}
                  <span className="min-w-0 break-all text-right text-sm font-medium">
                    {formatExplanationValue(value)}
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
