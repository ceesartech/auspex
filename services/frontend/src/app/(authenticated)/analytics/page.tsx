'use client';

import { useModelPerformance, useActiveModels } from '@/lib/hooks/use-recommendations';
import { PerformanceChart } from '@/components/charts/performance-chart';
import { ROIChart } from '@/components/charts/roi-chart';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '@/components/ui/table';
import { LoadingPage } from '@/components/shared/loading';
import { ErrorDisplay } from '@/components/shared/error-display';
import { formatPercentage } from '@/lib/utils/format';

export default function AnalyticsPage() {
  const { data: performance, isLoading, error, refetch } = useModelPerformance({ limit: 20 });
  const { data: activeModels } = useActiveModels();

  if (isLoading) return <LoadingPage message="Loading analytics..." />;
  if (error) return <ErrorDisplay message="Failed to load analytics" onRetry={refetch} />;

  const roiData = (performance || []).map((m) => ({
    name: m.model_name,
    roi: m.roi,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Analytics</h1>
        <p className="text-muted-foreground">Model performance and betting analytics</p>
      </div>

      {performance && performance.length > 0 && (
        <>
          <div className="grid gap-6 lg:grid-cols-2">
            <PerformanceChart data={performance} />
            <ROIChart data={roiData} />
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Model Details</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Model</TableHead>
                    <TableHead>Version</TableHead>
                    <TableHead>Predictions</TableHead>
                    <TableHead>Accuracy</TableHead>
                    <TableHead>ROI</TableHead>
                    <TableHead>Brier Score</TableHead>
                    <TableHead>Log Loss</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {performance.map((model) => (
                    <TableRow key={`${model.model_name}-${model.model_version}`}>
                      <TableCell className="font-medium">{model.model_name}</TableCell>
                      <TableCell>
                        <Badge variant="outline">v{model.model_version}</Badge>
                      </TableCell>
                      <TableCell>{model.total_predictions}</TableCell>
                      <TableCell>{formatPercentage(model.accuracy)}</TableCell>
                      <TableCell className={model.roi >= 0 ? 'text-green-600' : 'text-red-600'}>
                        {formatPercentage(model.roi)}
                      </TableCell>
                      <TableCell>{model.brier_score.toFixed(4)}</TableCell>
                      <TableCell>{model.log_loss.toFixed(4)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </>
      )}

      {activeModels && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Active Models</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {Array.isArray(activeModels) &&
                activeModels.map((model: any) => (
                  <Badge key={model.model_name || model.name} variant="secondary">
                    {model.model_name || model.name} v{model.model_version || model.version}
                  </Badge>
                ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
