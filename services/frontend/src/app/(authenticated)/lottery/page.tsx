'use client';

import { useState } from 'react';
import { isAxiosError } from 'axios';
import {
  useLotteryDraws,
  useLotteryAnalysis,
  useLotteryEV,
  useLotteryRecommendations,
} from '@/lib/hooks/use-recommendations';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '@/components/ui/table';
import { LoadingPage } from '@/components/shared/loading';
import { ErrorDisplay } from '@/components/shared/error-display';
import { EmptyState } from '@/components/shared/empty-state';
import { formatCurrency, formatDate, formatNumber } from '@/lib/utils/format';
import { Ticket } from 'lucide-react';

const FUN_CAPTION = 'For fun — draws are random';

const strategyOptions = [
  { value: 'blend', label: 'Balanced blend' },
  { value: 'ev', label: 'EV / unpopular' },
  { value: 'statistical', label: 'Statistical profile · for fun' },
  { value: 'hot', label: 'Hot numbers · for fun' },
  { value: 'due', label: 'Overdue numbers · for fun' },
  { value: 'random', label: 'Quick pick' },
];

const strategyCaptions: Record<string, string> = {
  ev: 'Reduces expected jackpot splitting — does not improve odds of winning',
  statistical: FUN_CAPTION,
  hot: FUN_CAPTION,
  due: FUN_CAPTION,
};

// $786M / $1.2B style.
const compactUSD = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  notation: 'compact',
  maximumFractionDigits: 1,
});

export default function LotteryPage() {
  const [game, setGame] = useState('powerball');
  const [strategy, setStrategy] = useState('blend');
  const [numSets, setNumSets] = useState(5);
  const [jackpotInput, setJackpotInput] = useState('');
  const [taxInput, setTaxInput] = useState('');
  const [evParams, setEvParams] = useState<{ jackpot?: number; stateTax?: number }>({});
  const generate = useLotteryRecommendations();
  const { data: draws, isLoading, error, refetch } = useLotteryDraws(game, 20);
  const { data: analysis } = useLotteryAnalysis(game);
  const { data: ev, isLoading: evLoading, error: evError } = useLotteryEV(game, evParams);

  // 422 = live jackpot estimate unavailable; prompt for a manual jackpot.
  const evUnavailable = isAxiosError(evError) && evError.response?.status === 422;

  const applyEvInputs = () => {
    const jackpot = Number(jackpotInput);
    const taxPct = Number(taxInput);
    setEvParams({
      jackpot: jackpotInput && Number.isFinite(jackpot) && jackpot > 0 ? jackpot : undefined,
      // API accepts 0 <= state_tax < 0.6; input is in percent.
      stateTax:
        taxInput && Number.isFinite(taxPct) && taxPct >= 0
          ? Math.min(taxPct, 59.9) / 100
          : undefined,
    });
  };

  if (isLoading) return <LoadingPage message="Loading lottery data..." />;
  if (error) return <ErrorDisplay message="Failed to load lottery data" onRetry={refetch} />;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Lottery</h1>
          <p className="text-muted-foreground">Draw history and number analysis</p>
        </div>
        <Select
          options={[
            { value: 'powerball', label: 'Powerball' },
            { value: 'mega_millions', label: 'Mega Millions' },
          ]}
          value={game}
          onChange={(e) => {
            setGame(e.target.value);
            // Jackpot/tax overrides are per-game; reset to the live estimate.
            setJackpotInput('');
            setTaxInput('');
            setEvParams({});
          }}
          className="w-[180px]"
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Expected Value</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {evLoading ? (
            <p className="text-sm text-muted-foreground">Computing expected value…</p>
          ) : evUnavailable ? (
            <div className="rounded-md border border-amber-300/50 bg-amber-50 p-3 text-xs text-amber-900 dark:border-amber-500/30 dark:bg-amber-950/40 dark:text-amber-200">
              Live jackpot estimate unavailable — enter the advertised jackpot below and apply to
              compute EV.
            </div>
          ) : evError ? (
            <p className="text-sm text-destructive">Failed to compute expected value.</p>
          ) : ev ? (
            <div className="space-y-4">
              <p
                className={`text-lg font-semibold ${
                  ev.ev_total >= ev.ticket_price ? 'text-green-600 dark:text-green-400' : ''
                }`}
              >
                {ev.verdict}
              </p>
              <dl className="grid gap-x-8 gap-y-2 text-sm sm:grid-cols-2">
                {(
                  [
                    [
                      'Advertised jackpot',
                      `${compactUSD.format(ev.advertised_jackpot)} · ${
                        ev.jackpot_source === 'live' ? 'live estimate' : 'manual'
                      }`,
                    ],
                    ['Cash value', compactUSD.format(ev.cash_value)],
                    [
                      'EV per ticket',
                      `${formatCurrency(ev.ev_total)} back per ${formatCurrency(ev.ticket_price)} ticket`,
                    ],
                    ['Expected loss', `${ev.expected_loss_pct.toFixed(1)}%`],
                    ['Jackpot odds', `1 in ${formatNumber(ev.jackpot_odds)}`],
                    [
                      'Co-winner sharing',
                      `${ev.expected_co_winners.toFixed(2)} expected co-winners · keep ${(
                        ev.share_factor * 100
                      ).toFixed(0)}% of jackpot`,
                    ],
                    [
                      'Breakeven jackpot',
                      ev.breakeven_advertised_jackpot != null
                        ? compactUSD.format(ev.breakeven_advertised_jackpot)
                        : 'effectively never',
                    ],
                    ['Next draw', ev.next_draw_date ? formatDate(ev.next_draw_date) : '—'],
                  ] as [string, string][]
                ).map(([label, value]) => (
                  <div key={label} className="flex items-baseline justify-between gap-4">
                    <dt className="text-muted-foreground">{label}</dt>
                    <dd className="text-right font-medium">{value}</dd>
                  </div>
                ))}
              </dl>
              <p className="text-xs text-muted-foreground">{ev.disclaimer}</p>
            </div>
          ) : null}

          <div className="flex flex-wrap items-end gap-3">
            <div className="space-y-1">
              <label htmlFor="ev-jackpot" className="text-xs text-muted-foreground">
                Jackpot override ($)
              </label>
              <Input
                id="ev-jackpot"
                type="number"
                min="1"
                placeholder="e.g. 500000000"
                value={jackpotInput}
                onChange={(e) => setJackpotInput(e.target.value)}
                className="w-[180px]"
              />
            </div>
            <div className="space-y-1">
              <label htmlFor="ev-state-tax" className="text-xs text-muted-foreground">
                State tax (%)
              </label>
              <Input
                id="ev-state-tax"
                type="number"
                min="0"
                max="59.9"
                step="0.1"
                placeholder="0"
                value={taxInput}
                onChange={(e) => setTaxInput(e.target.value)}
                className="w-[110px]"
              />
            </div>
            <Button variant="outline" onClick={applyEvInputs}>
              Apply
            </Button>
          </div>
        </CardContent>
      </Card>

      {analysis && (
        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium text-green-600">Hot Numbers</CardTitle>
              <p className="text-xs text-muted-foreground">{FUN_CAPTION}</p>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {analysis.hot_numbers?.slice(0, 10).map((n) => (
                  <Badge key={n.number} variant="success">
                    {n.number} ({n.frequency})
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium text-blue-600">Cold Numbers</CardTitle>
              <p className="text-xs text-muted-foreground">{FUN_CAPTION}</p>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {analysis.cold_numbers?.slice(0, 10).map((n) => (
                  <Badge key={n.number} variant="secondary">
                    {n.number} ({n.frequency})
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-sm font-medium text-orange-600">Overdue Numbers</CardTitle>
              <p className="text-xs text-muted-foreground">{FUN_CAPTION}</p>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {analysis.overdue_numbers?.slice(0, 10).map((n) => (
                  <Badge key={n.number} variant="warning">
                    {n.number} ({n.draws_since_last}d)
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Generate Combinations</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-3">
            <Select
              options={strategyOptions}
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="w-[200px]"
            />
            <Input
              type="number"
              min="1"
              max="20"
              value={numSets}
              onChange={(e) => setNumSets(Math.min(20, Math.max(1, Number(e.target.value) || 1)))}
              className="w-[100px]"
            />
            <Button
              onClick={() => generate.mutate({ game, strategy, numSets })}
              disabled={generate.isPending}
            >
              {generate.isPending ? 'Generating…' : `Generate ${numSets}`}
            </Button>
          </div>

          {strategyCaptions[strategy] && (
            <p className="text-xs text-muted-foreground">{strategyCaptions[strategy]}</p>
          )}

          {generate.isError && (
            <p className="text-sm text-destructive">Failed to generate combinations.</p>
          )}

          {generate.data && (
            <div className="space-y-3">
              {(generate.data.warnings?.length ?? 0) > 0 && (
                <div className="rounded-md border border-amber-300/50 bg-amber-50 p-3 text-xs text-amber-900 dark:border-amber-500/30 dark:bg-amber-950/40 dark:text-amber-200">
                  {generate.data.warnings.map((w, i) => (
                    <p key={i}>{w}</p>
                  ))}
                </div>
              )}
              <div className="rounded-md border border-amber-300/50 bg-amber-50 p-3 text-xs text-amber-900 dark:border-amber-500/30 dark:bg-amber-950/40 dark:text-amber-200">
                {generate.data.disclaimer}
              </div>
              {generate.data.combinations.map((combo, i) => (
                <div key={i} className="flex flex-wrap items-center gap-2 rounded-lg border p-3">
                  {combo.numbers.map((n) => (
                    <span
                      key={n}
                      className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground"
                    >
                      {n}
                    </span>
                  ))}
                  <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-destructive text-xs font-bold text-destructive-foreground">
                    {combo.bonus_number}
                  </span>
                  <Badge variant="secondary">score {combo.score.toFixed(2)}</Badge>
                  <span className="text-xs text-muted-foreground">{combo.rationale}</span>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Recent Draws</CardTitle>
        </CardHeader>
        <CardContent>
          {!draws?.length ? (
            <EmptyState
              icon={<Ticket className="h-8 w-8" />}
              title="No draws available"
            />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Date</TableHead>
                  <TableHead>Numbers</TableHead>
                  <TableHead>Bonus</TableHead>
                  <TableHead>Multiplier</TableHead>
                  <TableHead>Jackpot</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {draws.map((draw) => (
                  <TableRow key={draw.draw_id}>
                    <TableCell>{formatDate(draw.draw_date)}</TableCell>
                    <TableCell>
                      <div className="flex gap-1">
                        {draw.numbers?.map((n: number) => (
                          <span key={n} className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground">
                            {n}
                          </span>
                        ))}
                      </div>
                    </TableCell>
                    <TableCell>
                      <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-destructive text-xs font-bold text-destructive-foreground">
                        {draw.bonus_number}
                      </span>
                    </TableCell>
                    <TableCell>{draw.multiplier || '-'}</TableCell>
                    <TableCell>
                      {draw.jackpot_amount ? formatCurrency(draw.jackpot_amount) : '-'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
