import { Recommendation } from '@/lib/types/recommendation';
import { Badge } from '@/components/ui/badge';
import { formatOdds, formatDateTime } from '@/lib/utils/format';
import { useSettingsStore } from '@/lib/store/settings-store';

interface AccumulatorLegProps {
  leg: Recommendation;
  index: number;
}

export function AccumulatorLeg({ leg, index }: AccumulatorLegProps) {
  const { oddsFormat } = useSettingsStore();

  return (
    <div className="flex items-center justify-between rounded-lg border p-3">
      <div className="flex items-center gap-3">
        <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground">
          {index + 1}
        </span>
        <div>
          <p className="text-sm font-medium">
            {leg.match_info.home_team} vs {leg.match_info.away_team}
          </p>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>{leg.market_type}</span>
            <span>{formatDateTime(leg.match_info.match_date)}</span>
          </div>
        </div>
      </div>
      <div className="text-right">
        <Badge variant="outline">{leg.outcome}</Badge>
        <p className="mt-1 text-sm font-bold">{formatOdds(leg.recommended_odds, oddsFormat)}</p>
      </div>
    </div>
  );
}
