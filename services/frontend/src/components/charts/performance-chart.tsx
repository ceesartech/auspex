'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ModelPerformance } from '@/lib/types/recommendation';

interface PerformanceChartProps {
  data: ModelPerformance[];
  title?: string;
}

export function PerformanceChart({ data, title = 'Model Performance' }: PerformanceChartProps) {
  const chartData = data.map((d) => ({
    name: `${d.model_name} v${d.model_version}`,
    date: d.evaluation_date,
    accuracy: Number((d.accuracy * 100).toFixed(1)),
    roi: Number((d.roi * 100).toFixed(1)),
    brier: Number(d.brier_score.toFixed(4)),
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis dataKey="date" className="text-xs" tick={{ fontSize: 12 }} />
            <YAxis className="text-xs" tick={{ fontSize: 12 }} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'hsl(var(--background))',
                border: '1px solid hsl(var(--border))',
                borderRadius: '8px',
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="accuracy"
              stroke="hsl(var(--primary))"
              name="Accuracy %"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
            <Line
              type="monotone"
              dataKey="roi"
              stroke="hsl(142, 76%, 36%)"
              name="ROI %"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
