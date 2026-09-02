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
// here so the card stays presentational. Soccer sub-markets arrive as
// the raw prediction_type, so those are listed too; anything unknown
// falls back to the raw string.
const MARKET_LABELS: Record<string, string> = {
  moneyline: 'Moneyline',
  regulation: 'Regulation',
  puck_line: 'Puck Line',
  spread: 'Spread',
  total: 'Total O/U',
  match_result: '1X2',
  // Soccer markets derived from the Dixon-Coles scoreline model.
  over_under: 'Over/Under',
  btts: 'BTTS',
  asian_handicap: 'Asian Handicap',
  correct_score: 'Correct Score',
  double_chance: 'Double Chance',
  draw_no_bet: 'Draw No Bet',
  total_goals: 'Total Goals',
  winning_margin: 'Winning Margin',
  team_total: 'Team Total',
  clean_sheet: 'Clean Sheet',
  win_to_nil: 'Win To Nil',
  odd_even: 'Odd/Even',
  over_under_ht: 'HT Over/Under',
  match_result_ht: 'HT Result',
  btts_ht: 'HT BTTS',
  result_btts: 'Result & BTTS',
  result_over_under: 'Result & O/U',
  ht_ft_double_result: 'HT/FT',
};

// Soccer sub-markets carry up to 51 outcome keys (asian_handicap); the
// card only has room for a handful, so show the top-N by probability and
// summarise the rest as a chip.
const MAX_VISIBLE_PROBABILITIES = 4;

interface ProbabilityEntry {
  outcome: string;
  prob: number;
}

// Pick the entries to render: sorted by probability desc, capped at
// MAX_VISIBLE_PROBABILITIES, and always including the predicted outcome
// (swapping it in for the last slot if it would otherwise be cut off).
function selectVisibleProbabilities(
  probabilities: Record<string, number>,
  predictedOutcome: string
): { visible: ProbabilityEntry[]; hiddenCount: number } {
  const sorted: ProbabilityEntry[] = Object.entries(probabilities)
    .map(([outcome, prob]) => ({ outcome, prob }))
    .sort((a, b) => b.prob - a.prob);

  if (sorted.length <= MAX_VISIBLE_PROBABILITIES) {
    return { visible: sorted, hiddenCount: 0 };
  }

  const visible = sorted.slice(0, MAX_VISIBLE_PROBABILITIES);
  const predicted = sorted.find((e) => e.outcome === predictedOutcome);
  if (predicted && !visible.some((e) => e.outcome === predictedOutcome)) {
    visible[visible.length - 1] = predicted;
  }
  return { visible, hiddenCount: sorted.length - visible.length };
}

export function PredictionCard({ prediction, compact = false }: PredictionCardProps) {
  const { match_info, predicted_outcome, probabilities, confidence, timestamp, market } = prediction;
  const outcomeColor = getOutcomeColor(predicted_outcome);
  const confidenceLevel = formatConfidenceLevel(confidence);

  const confidenceVariant = confidence >= 0.7 ? 'success' : confidence >= 0.5 ? 'warning' : 'secondary';
  const marketLabel = market ? (MARKET_LABELS[market] ?? market) : null;
  const { visible, hiddenCount } = selectVisibleProbabilities(probabilities, predicted_outcome);

  return (
    <Link href={`/predictions/${match_info.match_id}`} className="block min-w-0">
      <Card className="h-full min-w-0 overflow-hidden transition-all hover:shadow-md hover:border-primary/50">
        <CardHeader className="pb-2">
          <div className="flex items-start justify-between gap-2">
            <div className="flex min-w-0 flex-wrap items-center gap-2">
              {/* Badge is inline-flex, so `truncate` must sit on a block-level
                  child — text-overflow only applies to block containers. */}
              <Badge variant="outline" className="min-w-0 max-w-full text-xs" title={match_info.league_name}>
                <span className="truncate">{match_info.league_name}</span>
              </Badge>
              {marketLabel && (
                <Badge variant="secondary" className="min-w-0 max-w-full text-xs" title={marketLabel}>
                  <span className="truncate">{marketLabel}</span>
                </Badge>
              )}
            </div>
            <Badge variant={confidenceVariant} className="shrink-0 text-xs">
              {confidenceLevel}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="text-center">
              <p className="text-sm font-medium break-words">{match_info.home_team}</p>
              <p className="text-xs text-muted-foreground">vs</p>
              <p className="text-sm font-medium break-words">{match_info.away_team}</p>
            </div>

            <div className="text-center">
              <p className={cn('text-lg font-bold break-words', outcomeColor)}>
                {predicted_outcome}
              </p>
              <p className="text-xs text-muted-foreground">
                {formatPercentage(confidence)} confidence
              </p>
            </div>

            {!compact && (
              <div className="flex min-w-0 flex-wrap items-end justify-center gap-x-3 gap-y-1 text-xs">
                {visible.map(({ outcome, prob }) => (
                  <div
                    key={outcome}
                    className={cn(
                      'min-w-0 max-w-full text-center',
                      outcome === predicted_outcome && 'font-semibold'
                    )}
                  >
                    <p className="truncate text-muted-foreground" title={outcome}>
                      {outcome}
                    </p>
                    <p className="font-medium">{formatPercentage(prob)}</p>
                  </div>
                ))}
                {hiddenCount > 0 && (
                  <span
                    className="shrink-0 self-center rounded-full bg-muted px-2 py-0.5 text-[11px] text-muted-foreground"
                    title={`${hiddenCount} more outcomes on the detail page`}
                  >
                    +{hiddenCount} more
                  </span>
                )}
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
