'use client';

import { useParams } from 'next/navigation';
import { useMatchPredictions, useMatchStats, useMatchOdds } from '@/lib/hooks/use-predictions';
import { PredictionDetails } from '@/components/predictions/prediction-details';
import { ProbabilityChart } from '@/components/charts/probability-chart';
import { LoadingPage } from '@/components/shared/loading';
import { ErrorDisplay } from '@/components/shared/error-display';
import { EmptyState } from '@/components/shared/empty-state';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { formatOdds, formatDate } from '@/lib/utils/format';
import { useSettingsStore } from '@/lib/store/settings-store';
import { cn } from '@/lib/utils/cn';
import { Prediction } from '@/lib/types/prediction';
import { ArrowLeft, TrendingUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import Link from 'next/link';

// ---------------------------------------------------------------------------
// Response shapes for the two sidebar endpoints. These mirror the API
// exactly (GET /api/v1/matches/{id}/odds and /stats); the previous version
// of this page read `outcome`, `team_form` and `h2h`, none of which exist,
// so the cards rendered empty / label-less.
// ---------------------------------------------------------------------------

interface OddsRow {
  odds_id?: string;
  bookmaker: string | null;
  market_type: string | null;
  selection: string | null;
  odds_decimal: number | null;
  odds_american?: number | null;
  line: number | null;
  implied_probability?: number | null;
  timestamp?: string;
  is_opening?: boolean;
  is_live?: boolean;
}

interface FormRow {
  match_id: string;
  match_date: string;
  is_home: boolean;
  goals_for: number | null;
  goals_against: number | null;
  result: 'W' | 'D' | 'L' | string;
  points?: number | null;
}

interface H2HRow {
  match_id: string;
  match_date: string;
  home_team_id?: string;
  away_team_id?: string;
  home_score: number | null;
  away_score: number | null;
  league_name: string | null;
}

interface MatchStatsResponse {
  match_id?: string;
  match_stats?: Record<string, Record<string, number | null>> | null;
  home_form?: FormRow[] | null;
  away_form?: FormRow[] | null;
  head_to_head?: H2HRow[] | null;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const HEADLINE_MARKETS = new Set(['match_result', 'moneyline']);

function isHeadline(p: Prediction): boolean {
  return !!p.market && HEADLINE_MARKETS.has(p.market);
}

/**
 * Headline market first, everything else in API order. The API is being
 * changed to return the headline row first too; this sort is the guard
 * so the chart never gets soccer's 51-outcome asian_handicap again if
 * the server-side ordering regresses.
 */
function sortHeadlineFirst(predictions: Prediction[]): Prediction[] {
  return [...predictions].sort((a, b) => Number(isHeadline(b)) - Number(isHeadline(a)));
}

function humanizeMarket(market?: string | null): string | undefined {
  if (!market) return undefined;
  const special: Record<string, string> = {
    match_result: 'Match Result (1X2)',
    moneyline: 'Moneyline',
    puck_line: 'Puck Line',
    btts: 'Both Teams To Score',
    btts_ht: 'BTTS (1st Half)',
    ht_ft_double_result: 'HT/FT',
    match_result_ht: '1st Half Result',
    over_under_ht: 'Over/Under (1st Half)',
    draw_no_bet: 'Draw No Bet',
  };
  if (special[market]) return special[market];
  return market.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase());
}

/** 1x2 / moneyline group first, then alphabetical. */
function marketGroupRank(marketType: string): number {
  const m = marketType.toLowerCase();
  if (m === '1x2' || m === 'moneyline' || m === 'h2h' || m === 'match_result') return 0;
  return 1;
}

function formatLine(line: number | null): string {
  if (line === null || line === undefined || !Number.isFinite(line)) return '';
  // Asian handicaps carry a sign, totals do not — render whatever the API sent.
  return String(line);
}

function selectionLabel(o: OddsRow): string {
  const sel = (o.selection ?? '?').replace(/_/g, ' ');
  const line = formatLine(o.line);
  return line ? `${sel} ${line}` : sel;
}

function selectionOrder(sel: string | null): number {
  const order = ['home', 'draw', 'away', 'over', 'under', 'yes', 'no'];
  const idx = order.indexOf((sel ?? '').toLowerCase());
  return idx === -1 ? order.length : idx;
}

interface OddsGroup {
  marketType: string;
  rows: OddsRow[];
  /** key = `${selection}|${line}` → highest odds_decimal in that cluster */
  best: Map<string, number>;
}

function groupOdds(odds: OddsRow[]): OddsGroup[] {
  const byMarket = new Map<string, OddsRow[]>();
  for (const o of odds) {
    if (typeof o.odds_decimal !== 'number' || !Number.isFinite(o.odds_decimal)) continue;
    const mt = o.market_type || 'other';
    const arr = byMarket.get(mt);
    if (arr) arr.push(o);
    else byMarket.set(mt, [o]);
  }
  const groups: OddsGroup[] = [];
  for (const [marketType, rows] of byMarket) {
    const best = new Map<string, number>();
    for (const r of rows) {
      const key = `${r.selection ?? ''}|${r.line ?? ''}`;
      const cur = best.get(key);
      if (cur === undefined || (r.odds_decimal as number) > cur) best.set(key, r.odds_decimal as number);
    }
    // Cluster rows by (selection, line) so the bold best price sits at
    // the top of each cluster, then bookmakers descending by price.
    rows.sort((a, b) => {
      const s = selectionOrder(a.selection) - selectionOrder(b.selection);
      if (s !== 0) return s;
      const sa = a.selection ?? '';
      const sb = b.selection ?? '';
      if (sa !== sb) return sa.localeCompare(sb);
      const la = a.line ?? 0;
      const lb = b.line ?? 0;
      if (la !== lb) return la - lb;
      return (b.odds_decimal as number) - (a.odds_decimal as number);
    });
    groups.push({ marketType, rows, best });
  }
  groups.sort((a, b) => {
    const r = marketGroupRank(a.marketType) - marketGroupRank(b.marketType);
    if (r !== 0) return r;
    return a.marketType.localeCompare(b.marketType);
  });
  return groups;
}

function formVariant(result: string): 'success' | 'warning' | 'destructive' | 'secondary' {
  switch ((result || '').toUpperCase()) {
    case 'W':
      return 'success';
    case 'D':
      return 'warning';
    case 'L':
      return 'destructive';
    default:
      return 'secondary';
  }
}

function humanizeStatKey(key: string): string {
  const special: Record<string, string> = {
    expected_goals: 'xG',
    shots_on_target: 'Shots on target',
    pass_accuracy: 'Pass accuracy',
    yellow_cards: 'Yellow cards',
    red_cards: 'Red cards',
  };
  return special[key] ?? key.replace(/_/g, ' ').replace(/^\w/, (l) => l.toUpperCase());
}

function formatStatValue(key: string, v: number): string {
  if (key === 'possession' || key === 'pass_accuracy') return `${v}%`;
  if (key === 'expected_goals') return v.toFixed(2);
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}

// ---------------------------------------------------------------------------
// Sidebar sub-components
// ---------------------------------------------------------------------------

function TeamForm({ label, form }: { label: string; form: FormRow[] }) {
  return (
    <div className="min-w-0">
      <p className="mb-1 min-w-0 truncate text-muted-foreground" title={label}>
        {label}
      </p>
      <div className="flex flex-wrap gap-1.5">
        {form.map((f) => {
          const gf = f.goals_for ?? '–';
          const ga = f.goals_against ?? '–';
          const title = `${formatDate(f.match_date)} · ${f.is_home ? 'home' : 'away'}`;
          return (
            <Badge
              key={f.match_id}
              variant={formVariant(f.result)}
              className="gap-1 whitespace-nowrap px-2 font-mono text-[11px]"
              title={title}
            >
              <span className="font-bold">{(f.result || '?').toUpperCase()}</span>
              <span className="font-normal">
                {gf}-{ga}
              </span>
            </Badge>
          );
        })}
      </div>
    </div>
  );
}

function HeadToHead({ rows }: { rows: H2HRow[] }) {
  return (
    <div className="min-w-0">
      <p className="mb-1 text-muted-foreground">
        Head to Head <span className="text-xs">(score listed home–away)</span>
      </p>
      <div className="max-h-60 overflow-auto rounded-md border">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-muted text-muted-foreground">
            <tr>
              <th className="px-2 py-1 text-left font-medium">Date</th>
              <th className="px-2 py-1 text-left font-medium">League</th>
              <th className="px-2 py-1 text-right font-medium">Score</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((h) => (
              <tr key={h.match_id} className="border-t">
                <td className="whitespace-nowrap px-2 py-1">{formatDate(h.match_date)}</td>
                <td className="max-w-[9rem] truncate px-2 py-1" title={h.league_name ?? ''}>
                  {h.league_name ?? '—'}
                </td>
                <td className="whitespace-nowrap px-2 py-1 text-right font-medium tabular-nums">
                  {h.home_score ?? '–'}–{h.away_score ?? '–'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TeamStatsTable({ team, stats }: { team: string; stats: Record<string, number | null> }) {
  const rows = Object.entries(stats).filter(
    (e): e is [string, number] => typeof e[1] === 'number' && Number.isFinite(e[1])
  );
  if (rows.length === 0) return null;
  return (
    <div className="min-w-0">
      <p className="mb-1 min-w-0 truncate font-medium" title={team}>
        {team}
      </p>
      <table className="w-full text-xs">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k} className="border-t first:border-0">
              <td className="min-w-0 break-words py-0.5 pr-2 text-muted-foreground">{humanizeStatKey(k)}</td>
              <td className="whitespace-nowrap py-0.5 text-right tabular-nums">{formatStatValue(k, v)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

/**
 * Per-match detail page. Lists EVERY market's prediction for the
 * selected match — moneyline + spread + total for NBA, the 4 NHL
 * markets, all ~19 derived markets for soccer.
 *
 * Why the GET /api/v1/predictions/match/{id} endpoint instead of the
 * legacy POST /api/v1/predictions/ (re-runs inference)? Three reasons:
 *   1. Reads existing DB rows — no chance of failing due to missing
 *      features / unloaded models / stale closing lines.
 *   2. Returns ALL markets at once; the old single-prediction shape
 *      only surfaced the headline and hid NBA's spread/total picks.
 *   3. Cheap. No model load, no predict, no re-write.
 *
 * Empty list (no predictions yet) shows an EmptyState. Network /
 * server error surfaces the actual error message so operators can
 * debug without checking server logs.
 */
export default function PredictionDetailPage() {
  const params = useParams();
  const matchId = params.id as string;
  const { oddsFormat } = useSettingsStore();

  const { data: predictions, isLoading, error, refetch } = useMatchPredictions(matchId);
  const { data: statsRaw } = useMatchStats(matchId);
  const { data: oddsRaw } = useMatchOdds(matchId);

  if (isLoading) return <LoadingPage message="Loading match predictions..." />;
  if (error) {
    // Surface the real error so it's diagnosable — the old generic
    // "Failed to load prediction" hid 404 vs 500 vs network noise.
    const msg = error instanceof Error ? error.message : 'Failed to load match predictions';
    return <ErrorDisplay message={msg} onRetry={refetch} />;
  }

  // Empty list = match exists but no predictions yet (DAG hasn't run
  // since fixtures landed, or all task ensembles failed to predict).
  // Falling through to the per-market loop without a guard would
  // show an empty page with the back button only.
  if (!predictions || predictions.length === 0) {
    return (
      <div className="space-y-6">
        <Link href="/predictions">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="h-5 w-5" />
          </Button>
        </Link>
        <EmptyState
          icon={<TrendingUp className="h-12 w-12" />}
          title="No predictions for this match yet"
          description="Predictions are generated by the scheduled pipeline. Check back after the next pipeline run."
        />
      </div>
    );
  }

  const sorted = sortHeadlineFirst(predictions as Prediction[]);
  const headline = sorted[0];
  // All predictions are for the same match — pull match_info from the
  // headline row to render the header.
  const headerMatch = headline.match_info;

  const odds: OddsRow[] = Array.isArray(oddsRaw) ? (oddsRaw as OddsRow[]) : [];
  const oddsGroups = groupOdds(odds);

  const stats: MatchStatsResponse | null =
    statsRaw && typeof statsRaw === 'object' ? (statsRaw as MatchStatsResponse) : null;
  const homeForm = stats?.home_form ?? [];
  const awayForm = stats?.away_form ?? [];
  const h2h = stats?.head_to_head ?? [];
  const teamStats = Object.entries(stats?.match_stats ?? {}).filter(
    ([, s]) => s && Object.values(s).some((v) => typeof v === 'number' && Number.isFinite(v))
  );
  const hasStats = homeForm.length > 0 || awayForm.length > 0 || h2h.length > 0 || teamStats.length > 0;

  return (
    <div className="min-w-0 space-y-6">
      <div className="flex min-w-0 items-center gap-4">
        <Link href="/predictions" className="shrink-0">
          <Button variant="ghost" size="icon">
            <ArrowLeft className="h-5 w-5" />
          </Button>
        </Link>
        <div className="min-w-0">
          <h1 className="break-words text-2xl font-bold tracking-tight">
            {headerMatch.home_team} vs {headerMatch.away_team}
          </h1>
          <p className="break-words text-muted-foreground">{headerMatch.league_name}</p>
        </div>
      </div>

      {/* min-w-0 on both columns: grid tracks default to minmax(auto,1fr),
          so long content (bookmaker lists, JSON blobs) would otherwise
          stretch the column past the viewport. */}
      <div className="grid min-w-0 gap-6 lg:grid-cols-3">
        <div className="min-w-0 space-y-4 lg:col-span-2">
          {/* One card per market, headline first. PredictionDetails reads
              probabilities + predicted_outcome + confidence from each row,
              so it handles soccer's 3-way 1X2, NHL's mix of 2-class and
              3-class regulation, and soccer's 51-outcome asian_handicap —
              no per-sport branching needed here. */}
          {sorted.map((prediction, idx) => (
            <div key={`${prediction.market || 'headline'}-${idx}`} className="min-w-0">
              {prediction.market && (
                <Badge
                  variant={isHeadline(prediction) ? 'default' : 'outline'}
                  className="mb-2 max-w-full"
                  title={prediction.market}
                >
                  <span className="truncate">{humanizeMarket(prediction.market)}</span>
                </Badge>
              )}
              <PredictionDetails prediction={prediction} />
            </div>
          ))}
        </div>

        <div className="min-w-0 space-y-6">
          {/* Chart for the headline market only. Showing 19 charts for
              soccer would crowd the sidebar. */}
          <ProbabilityChart
            probabilities={headline.probabilities}
            predictedOutcome={headline.predicted_outcome}
            marketLabel={humanizeMarket(headline.market)}
          />

          {oddsGroups.length > 0 && (
            <Card className="min-w-0 overflow-hidden">
              <CardHeader>
                <CardTitle className="text-lg">Current Odds</CardTitle>
              </CardHeader>
              <CardContent className="min-w-0">
                {/* Scrollable: one row per bookmaker × market × selection ×
                    line runs to 100+ rows for soccer; without a cap the
                    sidebar stretches past the main column. */}
                <div className="max-h-80 space-y-4 overflow-y-auto pr-1">
                  {oddsGroups.map((g) => (
                    <div key={g.marketType} className="min-w-0">
                      <p className="sticky top-0 mb-1 bg-card text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        {humanizeMarket(g.marketType)}
                      </p>
                      <div className="space-y-1.5">
                        {g.rows.map((o, i) => {
                          const key = `${o.selection ?? ''}|${o.line ?? ''}`;
                          const isBest = g.best.get(key) === o.odds_decimal;
                          return (
                            <div
                              key={o.odds_id ?? `${g.marketType}-${i}`}
                              className="flex min-w-0 items-center gap-2 text-sm"
                            >
                              <Badge
                                variant="outline"
                                className="max-w-[45%] shrink-0 whitespace-nowrap px-2 capitalize"
                                title={selectionLabel(o)}
                              >
                                <span className="truncate">{selectionLabel(o)}</span>
                              </Badge>
                              <span
                                className="min-w-0 flex-1 truncate text-muted-foreground"
                                title={o.bookmaker ?? undefined}
                              >
                                {o.bookmaker || 'N/A'}
                              </span>
                              <span className={cn('shrink-0 text-right tabular-nums', isBest && 'font-bold')}>
                                {formatOdds(o.odds_decimal as number, oddsFormat)}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {hasStats && (
            <Card className="min-w-0 overflow-hidden">
              <CardHeader>
                <CardTitle className="text-lg">Match Stats</CardTitle>
              </CardHeader>
              <CardContent className="min-w-0">
                <div className="min-w-0 space-y-4 text-sm">
                  {homeForm.length > 0 && <TeamForm label={`${headerMatch.home_team} form`} form={homeForm} />}
                  {awayForm.length > 0 && <TeamForm label={`${headerMatch.away_team} form`} form={awayForm} />}
                  {h2h.length > 0 && <HeadToHead rows={h2h} />}
                  {teamStats.length > 0 && (
                    <div className="min-w-0 space-y-3">
                      <p className="text-muted-foreground">Team Stats</p>
                      {teamStats.map(([team, s]) => (
                        <TeamStatsTable key={team} team={team} stats={s} />
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
