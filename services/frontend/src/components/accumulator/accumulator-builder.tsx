'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { useBuildAccumulator } from '@/lib/hooks/use-recommendations';
import { Accumulator, AccumulatorRequest } from '@/lib/types/recommendation';
import { formatCurrency, formatOdds, formatPercentage } from '@/lib/utils/format';
import { useSettingsStore } from '@/lib/store/settings-store';
import { toast } from '@/components/ui/toast';
import { Layers, Trash2, Zap } from 'lucide-react';
import { AccumulatorLeg } from './accumulator-leg';

export function AccumulatorBuilder() {
  const { oddsFormat, currency } = useSettingsStore();
  const buildAccumulator = useBuildAccumulator();

  const [params, setParams] = useState<AccumulatorRequest>({
    min_legs: 2,
    max_legs: 5,
    min_total_odds: 2.0,
    max_total_odds: 50.0,
    confidence_threshold: 0.6,
  });

  const [result, setResult] = useState<Accumulator | null>(null);

  const handleBuild = async () => {
    try {
      const acca = await buildAccumulator.mutateAsync(params);
      setResult(acca);
      toast('Accumulator built successfully!');
    } catch {
      toast('Failed to build accumulator. Try adjusting your criteria.');
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-5 w-5" />
            Build Accumulator
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <div>
              <label className="text-sm font-medium">Min Legs</label>
              <Input
                type="number"
                min={2}
                max={10}
                value={params.min_legs}
                onChange={(e) => setParams({ ...params, min_legs: Number(e.target.value) })}
              />
            </div>
            <div>
              <label className="text-sm font-medium">Max Legs</label>
              <Input
                type="number"
                min={2}
                max={10}
                value={params.max_legs}
                onChange={(e) => setParams({ ...params, max_legs: Number(e.target.value) })}
              />
            </div>
            <div>
              <label className="text-sm font-medium">Min Total Odds</label>
              <Input
                type="number"
                min={1}
                step={0.5}
                value={params.min_total_odds}
                onChange={(e) => setParams({ ...params, min_total_odds: Number(e.target.value) })}
              />
            </div>
            <div>
              <label className="text-sm font-medium">Max Total Odds</label>
              <Input
                type="number"
                min={1}
                step={0.5}
                value={params.max_total_odds}
                onChange={(e) => setParams({ ...params, max_total_odds: Number(e.target.value) })}
              />
            </div>
            <div>
              <label className="text-sm font-medium">Min Confidence</label>
              <Input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={params.confidence_threshold}
                onChange={(e) => setParams({ ...params, confidence_threshold: Number(e.target.value) })}
              />
            </div>
          </div>
        </CardContent>
        <CardFooter>
          <Button onClick={handleBuild} loading={buildAccumulator.isPending}>
            <Zap className="mr-2 h-4 w-4" />
            Build Accumulator
          </Button>
        </CardFooter>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">Your Accumulator</CardTitle>
              <div className="flex gap-2">
                <Badge variant={result.confidence_level === 'HIGH' ? 'success' : 'warning'}>
                  {result.confidence_level}
                </Badge>
                <Badge variant="outline">{result.legs.length} legs</Badge>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              {result.legs.map((leg, i) => (
                <AccumulatorLeg key={leg.recommendation_id} leg={leg} index={i} />
              ))}
            </div>

            <div className="rounded-lg bg-muted p-4">
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <div>
                  <p className="text-xs text-muted-foreground">Total Odds</p>
                  <p className="text-lg font-bold">{formatOdds(result.total_odds, oddsFormat)}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Stake</p>
                  <p className="text-lg font-bold">{formatCurrency(result.recommended_stake, currency)}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Potential Return</p>
                  <p className="text-lg font-bold text-green-600">
                    {formatCurrency(result.potential_return, currency)}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">EV</p>
                  <p className={`text-lg font-bold ${result.expected_value >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {result.expected_value >= 0 ? '+' : ''}{formatPercentage(result.expected_value)}
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
          <CardFooter>
            <Button variant="outline" onClick={() => setResult(null)}>
              <Trash2 className="mr-2 h-4 w-4" />
              Clear
            </Button>
          </CardFooter>
        </Card>
      )}
    </div>
  );
}
