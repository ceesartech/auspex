'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useModelPerformance } from '@/lib/hooks/use-recommendations';
import { formatPercentage } from '@/lib/utils/format';
import { CardSkeleton } from '@/components/shared/loading';
import { EmptyState } from '@/components/shared/empty-state';
import { BarChart3 } from 'lucide-react';

export function RecentPerformance() {
  const { data: models, isLoading } = useModelPerformance({ limit: 5 });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Model Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <CardSkeleton />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Model Performance</CardTitle>
      </CardHeader>
      <CardContent>
        {!models?.length ? (
          <EmptyState
            icon={<BarChart3 className="h-8 w-8" />}
            title="No performance data"
            description="Model performance will appear after predictions are evaluated"
          />
        ) : (
          <div className="space-y-4">
            {models.map((model) => (
              <div key={`${model.model_name}-${model.model_version}`} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">{model.model_name}</span>
                    <Badge variant="outline" className="text-xs">
                      v{model.model_version}
                    </Badge>
                  </div>
                  <span className="text-sm text-muted-foreground">
                    {model.total_predictions} predictions
                  </span>
                </div>
                <div className="flex gap-4 text-xs">
                  <div>
                    <span className="text-muted-foreground">Accuracy: </span>
                    <span className="font-medium">{formatPercentage(model.accuracy)}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">ROI: </span>
                    <span className={`font-medium ${model.roi >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {formatPercentage(model.roi)}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Brier: </span>
                    <span className="font-medium">{model.brier_score.toFixed(4)}</span>
                  </div>
                </div>
                <div className="h-2 rounded-full bg-muted">
                  <div
                    className="h-2 rounded-full bg-primary transition-all"
                    style={{ width: `${Math.min(model.accuracy * 100, 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
