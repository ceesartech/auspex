'use client';

import Link from 'next/link';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { RaceSummary } from '@/lib/types/race';
import { formatDateTime } from '@/lib/utils/format';
import { Trophy } from 'lucide-react';

interface RaceCardProps {
  race: RaceSummary;
}

// Display label for surface — racecards distinguish turf / dirt /
// synthetic and we want the card to read at a glance which it is.
const SURFACE_LABELS: Record<string, string> = {
  turf: 'Turf',
  dirt: 'Dirt',
  synthetic: 'AW', // all-weather (Wolverhampton, Newcastle, Southwell)
  all_weather: 'AW',
};

export function RaceCard({ race }: RaceCardProps) {
  // 1 furlong = 201.168 m. The schema stores meters; render in
  // furlongs because that's how punters read distance in UK/IRE.
  const furlongs = race.distance_meters
    ? (race.distance_meters / 201.168).toFixed(1)
    : null;

  const surfaceLabel = race.surface
    ? SURFACE_LABELS[race.surface.toLowerCase()] ?? race.surface
    : null;

  const recCount = race.recommendation_count;

  return (
    <Link href={`/races/${race.race_id}`}>
      <Card className="transition-all hover:shadow-md hover:border-primary/50">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-xs">
                {race.track_name}
              </Badge>
              {race.race_number && (
                <Badge variant="secondary" className="text-xs">
                  R{race.race_number}
                </Badge>
              )}
              {surfaceLabel && (
                <Badge variant="outline" className="text-xs">
                  {surfaceLabel}
                </Badge>
              )}
            </div>
            {recCount > 0 && (
              <Badge variant="success" className="text-xs flex items-center gap-1">
                <Trophy className="h-3 w-3" />
                {recCount} {recCount === 1 ? 'pick' : 'picks'}
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-baseline justify-between">
            <div className="text-sm font-medium">
              {formatDateTime(race.race_date)}
            </div>
            <div className="text-xs text-muted-foreground">
              {race.runners} runners
            </div>
          </div>
          <div className="mt-1 flex items-center gap-3 text-xs text-muted-foreground">
            {furlongs && <span>{furlongs}f</span>}
            {race.track_condition && <span>· {race.track_condition}</span>}
            {race.race_class && <span>· {race.race_class}</span>}
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
