'use client';

import Link from 'next/link';
import { Calendar, ArrowRight } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { useUpcomingMatches } from '@/lib/hooks/use-predictions';
import { formatDateTime, formatOdds } from '@/lib/utils/format';
import { useSettingsStore } from '@/lib/store/settings-store';
import { CardSkeleton } from '@/components/shared/loading';
import { EmptyState } from '@/components/shared/empty-state';

export function UpcomingMatches() {
  const { data: matches, isLoading } = useUpcomingMatches({ limit: 6 });
  const { oddsFormat } = useSettingsStore();

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Upcoming Matches</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <CardSkeleton key={i} />
          ))}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-lg">Upcoming Matches</CardTitle>
        <Link href="/predictions">
          <Button variant="ghost" size="sm">
            View All <ArrowRight className="ml-1 h-4 w-4" />
          </Button>
        </Link>
      </CardHeader>
      <CardContent>
        {!matches?.length ? (
          <EmptyState
            icon={<Calendar className="h-8 w-8" />}
            title="No upcoming matches"
            description="Check back later for upcoming fixtures"
          />
        ) : (
          <div className="space-y-3">
            {matches.map((match) => (
              <Link
                key={match.match_id}
                href={`/predictions/${match.match_id}`}
                className="flex items-center justify-between rounded-lg border p-3 transition-colors hover:bg-accent"
              >
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-xs">
                      {match.league_name}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {formatDateTime(match.match_date)}
                    </span>
                  </div>
                  <p className="mt-1 text-sm font-medium">
                    {match.home_team} vs {match.away_team}
                  </p>
                </div>
                {match.odds?.home && (
                  <div className="flex gap-2 text-xs">
                    <span className="rounded bg-muted px-2 py-1">
                      {formatOdds(match.odds.home, oddsFormat)}
                    </span>
                    {match.odds.draw && (
                      <span className="rounded bg-muted px-2 py-1">
                        {formatOdds(match.odds.draw, oddsFormat)}
                      </span>
                    )}
                    <span className="rounded bg-muted px-2 py-1">
                      {formatOdds(match.odds.away!, oddsFormat)}
                    </span>
                  </div>
                )}
              </Link>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
