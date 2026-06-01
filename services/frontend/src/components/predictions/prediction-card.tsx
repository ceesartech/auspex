'use client';

import Link from 'next/link';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Prediction } from '@/lib/types/prediction';
import { formatPercentage, formatDateTime, formatConfidenceLevel, getOutcomeColor } from '@/lib/utils/format';
import { cn } from '@/lib/utils/cn';

interface PredictionCardProps {
  prediction: Prediction;
  compact?: boolean;
}

// Human-readable label for the API's `market` value. The API uses
// snake_case (puck_line, match_result) — translate to display strings
// here so the card stays presentational.
const MARKET_LABELS: Record<string, string> = {
  moneyline: 'Moneyline',
  regulation: 'Regulation',
  puck_line: 'Puck Line',
  total: 'Total O/U',
  match_result: '1X2',
};

export function PredictionCard({ prediction, compact = false }: PredictionCardProps) {
  const { match_info, predicted_outcome, probabilities, confidence, timestamp, market } = prediction;
  const outcomeColor = getOutcomeColor(predicted_outcome);
  const confidenceLevel = formatConfidenceLevel(confidence);

  const confidenceVariant = confidence >= 0.7 ? 'success' : confidence >= 0.5 ? 'warning' : 'secondary';
  const marketLabel = market ? (MARKET_LABELS[market] ?? market) : null;

  return (
    <Link href={`/predictions/${match_info.match_id}`}>
      <Card className="transition-all hover:shadow-md hover:border-primary/50">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-xs">
                {match_info.league_name}
              </Badge>
              {marketLabel && (
                <Badge variant="secondary" className="text-xs">
                  {marketLabel}
                </Badge>
              )}
            </div>
            <Badge variant={confidenceVariant} className="text-xs">
              {confidenceLevel}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="text-center">
              <p className="text-sm font-medium">{match_info.home_team}</p>
              <p className="text-xs text-muted-foreground">vs</p>
              <p className="text-sm font-medium">{match_info.away_team}</p>
            </div>

            <div className="text-center">
              <p className={cn('text-lg font-bold', outcomeColor)}>
                {predicted_outcome}
              </p>
              <p className="text-xs text-muted-foreground">
                {formatPercentage(confidence)} confidence
              </p>
            </div>

            {!compact && (
              <div className="flex justify-between text-xs">
                {Object.entries(probabilities).map(([outcome, prob]) => (
                  <div key={outcome} className="text-center">
                    <p className="text-muted-foreground">{outcome}</p>
                    <p className="font-medium">{formatPercentage(prob)}</p>
                  </div>
                ))}
              </div>
            )}

            <p className="text-xs text-muted-foreground text-center">
              {formatDateTime(timestamp)}
            </p>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
