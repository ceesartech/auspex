'use client';

import { useAccuracySummary } from '@/lib/hooks/use-accuracy';
import { usePredictionsStore } from '@/lib/store/predictions-store';

/**
 * Compact accuracy + recommendations P/L panel. Reads the same sport
 * + market filters the predictions list uses, so the headline numbers
 * track whatever the user is currently viewing.
 *
 * Backed by Phase 5 grading (scripts/grade_completed_matches.py) —
 * the numbers reflect actual settled outcomes, not pre-publication
 * validation metrics. ROI excludes still-pending picks.
 */
export function AccuracyWidget() {
  const { selectedSport, selectedMarket } = usePredictionsStore();
  const { data, isLoading } = useAccuracySummary({
    sport: selectedSport || undefined,
    market: selectedMarket || undefined,
    days: 30,
  });

  if (isLoading || !data) {
    return (
      <div className="rounded-lg border bg-card p-4">
        <div className="text-sm text-muted-foreground">Loading performance…</div>
      </div>
    );
  }

  const accuracyPct = data.graded > 0 ? (data.accuracy * 100).toFixed(1) : '—';
  const roiPct = data.recs.total_staked > 0 ? data.recs.roi_pct.toFixed(1) : '—';
  const roiTone =
    data.recs.total_profit_loss > 0
      ? 'text-green-600'
      : data.recs.total_profit_loss < 0
        ? 'text-red-600'
        : 'text-muted-foreground';

  const scopeLabel = [
    data.sport_filter ? data.sport_filter.toUpperCase() : 'All sports',
    data.market_filter || null,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <div className="rounded-lg border bg-card p-4">
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-sm font-medium">Performance (last {data.days} days)</div>
        <div className="text-xs text-muted-foreground">{scopeLabel}</div>
      </div>

      <div className="mt-3 grid grid-cols-3 gap-4">
        <div>
          <div className="text-2xl font-bold tabular-nums">
            {accuracyPct}
            {accuracyPct !== '—' && <span className="text-base font-normal">%</span>}
          </div>
          <div className="text-xs text-muted-foreground">
            Model accuracy ({data.correct}/{data.graded} graded)
          </div>
        </div>
        <div>
          <div className={`text-2xl font-bold tabular-nums ${roiTone}`}>
            {roiPct !== '—' && data.recs.total_profit_loss > 0 ? '+' : ''}
            {roiPct}
            {roiPct !== '—' && <span className="text-base font-normal">%</span>}
          </div>
          <div className="text-xs text-muted-foreground">
            Recs ROI ({data.recs.settled} settled)
          </div>
        </div>
        <div>
          <div className={`text-2xl font-bold tabular-nums ${roiTone}`}>
            {data.recs.total_profit_loss > 0 ? '+' : ''}
            {data.recs.total_profit_loss.toFixed(2)}
          </div>
          <div className="text-xs text-muted-foreground">
            P/L ({data.recs.won}W · {data.recs.lost}L · {data.recs.void}V)
          </div>
        </div>
      </div>

      {data.by_market.length > 1 && (
        <div className="mt-4 border-t pt-3">
          <div className="text-xs font-medium text-muted-foreground">By market</div>
          <div className="mt-2 space-y-1">
            {data.by_market.map((row) => (
              <div
                key={`${row.sport}:${row.prediction_type}:${row.model_name}`}
                className="flex items-center justify-between text-xs"
              >
                <span className="text-muted-foreground">
                  {row.sport.toUpperCase()} · {row.prediction_type}
                </span>
                <span className="tabular-nums">
                  {row.graded > 0 ? `${(row.accuracy * 100).toFixed(1)}%` : '—'}
                  <span className="ml-1 text-muted-foreground">
                    ({row.correct}/{row.graded})
                  </span>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
