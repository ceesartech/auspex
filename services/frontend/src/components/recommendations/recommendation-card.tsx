'use client';

import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Recommendation } from '@/lib/types/recommendation';
import { formatCurrency, formatOdds, formatPercentage, formatDateTime } from '@/lib/utils/format';
import { useSettingsStore } from '@/lib/store/settings-store';
import { isValueBet, calculateExpectedValue } from '@/lib/utils/calculations';
import { TrendingUp, Clock } from 'lucide-react';

interface RecommendationCardProps {
  recommendation: Recommendation;
  onAddToAccumulator?: (rec: Recommendation) => void;
}

export function RecommendationCard({ recommendation, onAddToAccumulator }: RecommendationCardProps) {
  const { oddsFormat, currency } = useSettingsStore();
  const {
    match_info,
    market_type,
    outcome,
    recommended_odds,
    recommended_stake,
    expected_value,
    confidence_level,
    reasoning,
    expires_at,
    sport,
    model_probability,
  } = recommendation;

  const isHorseRacing = sport === 'horse_racing';
  // confidence_rating arrives lowercase for horse racing (high/medium)
  // and uppercase for team sports (HIGH/MEDIUM) — normalise both so the
  // badge colour + label are consistent across sports.
  const conf = (confidence_level || '').toUpperCase();
  const confidenceVariant =
    conf === 'HIGH' || conf === 'VERY_HIGH'
      ? 'success'
      : conf === 'MEDIUM'
        ? 'warning'
        : 'secondary';

  return (
    <Card className="transition-all hover:shadow-md">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <Badge variant="outline">
            {isHorseRacing ? `🏇 ${match_info.league_name}` : match_info.league_name}
          </Badge>
          <Badge variant={confidenceVariant}>{conf.replace('_', ' ') || 'N/A'}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <p className="text-sm font-medium">
            {isHorseRacing
              ? match_info.home_team
              : `${match_info.home_team} vs ${match_info.away_team}`}
          </p>
          <p className="text-xs text-muted-foreground">{formatDateTime(match_info.match_date)}</p>
        </div>

        <div className="flex items-center justify-between rounded-lg bg-muted p-3">
          <div>
            <p className="text-xs text-muted-foreground">
              {isHorseRacing ? 'Win' : market_type}
            </p>
            <p className="text-lg font-bold">{outcome}</p>
            {model_probability != null && (
              <p className="text-xs text-muted-foreground">
                Model {formatPercentage(model_probability)}
              </p>
            )}
          </div>
          <div className="text-right">
            <p className="text-xs text-muted-foreground">Odds</p>
            <p className="text-lg font-bold">{formatOdds(recommended_odds, oddsFormat)}</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <p className="text-xs text-muted-foreground">Stake</p>
            <p className="font-medium">{formatCurrency(recommended_stake, currency)}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Expected Value</p>
            <p className={`font-medium ${expected_value >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {expected_value >= 0 ? '+' : ''}{formatPercentage(expected_value)}
            </p>
          </div>
        </div>

        {/* Reasoning strings can be long ("Spread home_-5.00: model 80%
            (capped from raw 87%)…"). break-words forces wrap inside the
            fixed-width card so it doesn't bleed past the right edge. */}
        <p className="text-xs text-muted-foreground break-words">{reasoning}</p>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <Clock className="h-3 w-3" />
            Expires {formatDateTime(expires_at)}
          </div>
          {onAddToAccumulator && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => onAddToAccumulator(recommendation)}
            >
              <TrendingUp className="mr-1 h-3 w-3" />
              Add to Acca
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
