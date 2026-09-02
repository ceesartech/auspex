'use client';

import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils/cn';

interface ProbabilityChartProps {
  probabilities?: Record<string, number> | null;
  title?: string;
  /** Outcome the model picked — rendered bold / highlighted. */
  predictedOutcome?: string;
  /** Human label for the market the probabilities belong to (e.g. "Match Result"). */
  marketLabel?: string;
}

// Six distinct swatches. Any market with more outcomes than this is
// rendered as a bar list instead of a donut, so the palette never wraps.
const COLORS = [
  'hsl(var(--primary))',
  'hsl(217, 91%, 60%)',
  'hsl(38, 92%, 50%)',
  'hsl(142, 71%, 45%)',
  'hsl(0, 84%, 60%)',
  'hsl(var(--muted-foreground))',
];

/** Above this many outcomes a donut is unreadable — use bars instead. */
const MAX_DONUT_OUTCOMES = 6;

interface Slice {
  name: string;
  value: number; // percent, 1dp
}

function toSlices(probabilities?: Record<string, number> | null): Slice[] {
  if (!probabilities) return [];
  return Object.entries(probabilities)
    .filter(([, p]) => typeof p === 'number' && Number.isFinite(p))
    .map(([name, p]) => ({ name, value: Number((p * 100).toFixed(1)) }));
}

/**
 * Headline-market probability card for the prediction detail sidebar.
 *
 * Deliberately NO recharts outside-labels and NO recharts <Legend/>:
 * both lay text out in absolute coordinates and, for anything beyond a
 * 3-way market, overlap into an unreadable jumble that also escapes
 * the card. The legend is plain DOM below the chart so it wraps and
 * truncates like everything else on the page.
 */
export function ProbabilityChart({
  probabilities,
  title = 'Outcome Probabilities',
  predictedOutcome,
  marketLabel,
}: ProbabilityChartProps) {
  const slices = toSlices(probabilities);
  const heading = marketLabel ? `${title} · ${marketLabel}` : title;

  return (
    <Card className="min-w-0 overflow-hidden">
      <CardHeader>
        <CardTitle className="min-w-0 break-words text-lg">{heading}</CardTitle>
      </CardHeader>
      <CardContent className="min-w-0">
        {slices.length === 0 ? (
          <p className="text-sm text-muted-foreground">No probabilities available.</p>
        ) : slices.length <= MAX_DONUT_OUTCOMES ? (
          <DonutWithLegend slices={slices} predictedOutcome={predictedOutcome} />
        ) : (
          <BarList slices={slices} predictedOutcome={predictedOutcome} />
        )}
      </CardContent>
    </Card>
  );
}

function DonutWithLegend({ slices, predictedOutcome }: { slices: Slice[]; predictedOutcome?: string }) {
  return (
    <div className="space-y-3">
      {/* Fixed pixel height: ResponsiveContainer with a % height inside a
          flex/grid parent collapses to 0 on first paint. */}
      <ResponsiveContainer width="100%" height={200}>
        <PieChart>
          <Pie
            data={slices}
            cx="50%"
            cy="50%"
            innerRadius={55}
            outerRadius={85}
            paddingAngle={slices.length > 1 ? 4 : 0}
            dataKey="value"
            nameKey="name"
            isAnimationActive={false}
          >
            {slices.map((s, index) => (
              <Cell
                key={s.name}
                fill={COLORS[index % COLORS.length]}
                stroke={s.name === predictedOutcome ? 'hsl(var(--foreground))' : undefined}
                strokeWidth={s.name === predictedOutcome ? 2 : 0}
              />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--background))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '8px',
            }}
            formatter={(value: number) => [`${value}%`, 'Probability']}
          />
        </PieChart>
      </ResponsiveContainer>

      <ul className="space-y-1.5 text-sm">
        {slices.map((s, index) => {
          const isPick = s.name === predictedOutcome;
          return (
            <li key={s.name} className="flex min-w-0 items-center gap-2">
              <span
                className="h-3 w-3 shrink-0 rounded-sm"
                style={{ backgroundColor: COLORS[index % COLORS.length] }}
                aria-hidden
              />
              <span className={cn('min-w-0 flex-1 truncate', isPick && 'font-semibold')} title={s.name}>
                {s.name}
              </span>
              <span className={cn('shrink-0 tabular-nums', isPick ? 'font-semibold' : 'text-muted-foreground')}>
                {s.value}%
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function BarList({ slices, predictedOutcome }: { slices: Slice[]; predictedOutcome?: string }) {
  const sorted = [...slices].sort((a, b) => b.value - a.value);
  return (
    <div className="max-h-72 space-y-2 overflow-y-auto pr-2">
      {sorted.map((s) => {
        const isPick = s.name === predictedOutcome;
        return (
          <div key={s.name} className="min-w-0">
            <div className="flex min-w-0 justify-between gap-2 text-sm">
              <span className={cn('min-w-0 truncate', isPick && 'font-semibold')} title={s.name}>
                {s.name}
              </span>
              <span className={cn('shrink-0 tabular-nums', isPick ? 'font-semibold' : 'text-muted-foreground')}>
                {s.value}%
              </span>
            </div>
            <div className="mt-1 h-2 rounded-full bg-muted">
              <div
                className={cn('h-2 rounded-full', isPick ? 'bg-primary' : 'bg-muted-foreground/30')}
                style={{ width: `${Math.min(100, Math.max(0, s.value))}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
