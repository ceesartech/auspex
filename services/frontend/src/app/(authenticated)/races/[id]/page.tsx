'use client';

import { useRaceDetail } from '@/lib/hooks/use-races';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { LoadingPage } from '@/components/shared/loading';
import { ErrorDisplay } from '@/components/shared/error-display';
import { formatDateTime, formatPercentage } from '@/lib/utils/format';
import { cn } from '@/lib/utils/cn';
import { Trophy, AlertTriangle } from 'lucide-react';
import { RaceEntrant } from '@/lib/types/race';

interface PageProps {
  // Next 14 client components receive params as a PLAIN object (not a
  // Promise — that's the Next 15 shape). The earlier version typed
  // this as Promise + called React's use(params), which throws a
  // client-side exception on React 18 because use() only accepts a
  // Promise/Context, not a plain object — that was the "Application
  // error" crash on race-tile clicks. The sibling predictions/[id]
  // page reads params.id directly, which is the correct Next 14 form.
  params: { id: string };
}

export default function RaceDetailPage({ params }: PageProps) {
  const { id } = params;
  const { data, isLoading, error, refetch } = useRaceDetail(id);

  if (isLoading) return <LoadingPage message="Loading race detail..." />;
  if (error) return <ErrorDisplay message="Failed to load race" onRetry={refetch} />;
  if (!data) return null;

  const { race, entrants } = data;
  // Sort: best consensus prob first (lowest implied odds = favourite),
  // with scratched horses pushed to the bottom regardless.
  const sortedEntrants = [...entrants].sort((a, b) => {
    if (a.scratched && !b.scratched) return 1;
    if (!a.scratched && b.scratched) return -1;
    const pa = a.consensus_prob ?? 0;
    const pb = b.consensus_prob ?? 0;
    return pb - pa;
  });

  const winner = entrants.find((e) => e.finish_position === 1);
  const furlongs = race.distance_meters
    ? (race.distance_meters / 201.168).toFixed(1)
    : null;

  return (
    <div className="space-y-6">
      {/* Race header card */}
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between">
            <div>
              <CardTitle className="text-2xl">
                {race.track_name}
                {race.race_number && (
                  <span className="ml-2 text-base font-normal text-muted-foreground">
                    Race {race.race_number}
                  </span>
                )}
              </CardTitle>
              <div className="mt-1 text-sm text-muted-foreground">
                {formatDateTime(race.race_date)}
              </div>
            </div>
            <div className="flex flex-col items-end gap-1">
              <Badge variant={race.status === 'finished' ? 'secondary' : 'outline'}>
                {race.status}
              </Badge>
              {winner && (
                <div className="flex items-center gap-1 text-sm text-muted-foreground">
                  <Trophy className="h-4 w-4 text-amber-500" />
                  Winner: <span className="font-medium text-foreground">{winner.horse_name}</span>
                </div>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
          {furlongs && (
            <Stat label="Distance" value={`${furlongs}f`} />
          )}
          {race.surface && <Stat label="Surface" value={race.surface} />}
          {race.track_condition && (
            <Stat label="Going" value={race.track_condition} />
          )}
          {race.race_class && <Stat label="Class" value={race.race_class} />}
          {race.field_size && (
            <Stat label="Field" value={`${race.field_size} runners`} />
          )}
          {race.purse_amount && (
            <Stat
              label="Purse"
              value={`${race.purse_currency ?? '£'}${Number(
                race.purse_amount,
              ).toLocaleString()}`}
            />
          )}
        </CardContent>
      </Card>

      {/* Entrants table */}
      <Card>
        <CardHeader>
          <CardTitle>Runners</CardTitle>
        </CardHeader>
        <CardContent className="px-0">
          <div className="divide-y">
            {sortedEntrants.map((entrant) => (
              <EntrantRow key={entrant.entrant_id} entrant={entrant} />
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-medium">{value}</div>
    </div>
  );
}

function EntrantRow({ entrant }: { entrant: RaceEntrant }) {
  const isWinner = entrant.finish_position === 1;
  const isPicked = entrant.recommendation !== null;
  const prob = entrant.consensus_prob ?? 0;

  return (
    <div
      className={cn(
        'px-4 py-3 flex items-center gap-4',
        entrant.scratched && 'opacity-50',
        isWinner && 'bg-amber-50 dark:bg-amber-950/20',
      )}
    >
      <div className="w-8 text-center text-sm font-mono text-muted-foreground">
        {entrant.program_number ?? '—'}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 truncate">
          <span className="font-medium truncate">{entrant.horse_name}</span>
          {isWinner && (
            <Trophy className="h-4 w-4 text-amber-500 flex-shrink-0" />
          )}
          {entrant.scratched && (
            <Badge variant="secondary" className="text-xs">scratched</Badge>
          )}
          {entrant.disqualified && (
            <Badge variant="destructive" className="text-xs flex items-center gap-1">
              <AlertTriangle className="h-3 w-3" /> DQ
            </Badge>
          )}
        </div>
        <div className="text-xs text-muted-foreground truncate">
          {entrant.jockey_name && <span>J: {entrant.jockey_name}</span>}
          {entrant.trainer_name && (
            <span className="ml-2">T: {entrant.trainer_name}</span>
          )}
        </div>
      </div>

      {/* Consensus probability — gradient-styled so the favourite
          stands out without an extra label */}
      {entrant.consensus_prob !== null && (
        <div className="text-right">
          <div className="text-xs text-muted-foreground">Consensus</div>
          <div className="font-mono text-sm">
            {formatPercentage(prob)}
          </div>
        </div>
      )}

      {/* Pre-race price (morning line if known, else starting price) */}
      {(entrant.morning_line_odds || entrant.starting_price) && (
        <div className="text-right">
          <div className="text-xs text-muted-foreground">
            {entrant.morning_line_odds ? 'ML' : 'SP'}
          </div>
          <div className="font-mono text-sm">
            {(entrant.morning_line_odds ?? entrant.starting_price)?.toFixed(2)}
          </div>
        </div>
      )}

      {/* Value-bet rec badge — when one fired for this horse */}
      {isPicked && entrant.recommendation && (
        <div className="text-right min-w-[80px]">
          <Badge variant="success" className="text-xs">
            VALUE
          </Badge>
          <div className="text-xs text-muted-foreground mt-1">
            EV {(entrant.recommendation.expected_value
              ? entrant.recommendation.expected_value * 100
              : 0
            ).toFixed(0)}%
          </div>
          {entrant.recommendation.confidence_rating && (
            <div className="text-xs text-muted-foreground capitalize">
              {entrant.recommendation.confidence_rating} confidence
            </div>
          )}
        </div>
      )}

      {/* Finish position (graded races) */}
      {entrant.finish_position !== null && !isWinner && (
        <div className="text-right">
          <div className="text-xs text-muted-foreground">Finish</div>
          <div className="font-mono text-sm">{entrant.finish_position}</div>
        </div>
      )}
    </div>
  );
}
