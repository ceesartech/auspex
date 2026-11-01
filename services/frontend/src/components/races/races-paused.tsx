import { PauseCircle } from 'lucide-react';
import { EmptyState } from '@/components/shared/empty-state';

// What every /races page shows while HORSE_RACING_ENABLED
// (@/lib/sport-switches, mirroring the DAG switch in
// services/data-ingestion/dags/sport_switches.py) is false. The pages render
// this instead of fetching race lists, which can only come back empty while
// the pipeline is off.
export function RacesPaused() {
  return (
    <div className="space-y-6" data-testid="races-paused">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Race cards</h1>
        <p className="text-muted-foreground">Horse racing — UK & Ireland</p>
      </div>
      <EmptyState
        icon={<PauseCircle className="h-12 w-12" />}
        title="Horse racing is paused"
        description="Race cards, predictions and value picks are switched off for now, so there is nothing to show here. They come back when horse racing is turned on again."
      />
    </div>
  );
}
