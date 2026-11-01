// Per-sport display switches for the frontend.
//
// HORSE_RACING_ENABLED mirrors HORSE_RACING_ENABLED in
// services/data-ingestion/dags/sport_switches.py, the ONE switch that turns
// the horse-racing pipeline on or off (False since 2026-11-01: dormant). Keep
// the two equal: a repo-root unit test (tests/unit/test_frontend_sport_switches.py)
// fails when they differ. While it is false the Race cards nav entry is
// hidden and the /races pages show a "paused" notice instead of fetching
// lists that can only come back empty.
//
// Re-enable: set this to true together with the DAG switch. The frontend
// image changes only when CI rebuilds it (push to main).
export const HORSE_RACING_ENABLED: boolean = false;

// Routes that belong to a switched-off sport.
const RACES_ROUTE = '/races';

// False for a nav href under a switched-off sport. `horseRacingEnabled` is a
// parameter only so tests can exercise both states of the constant.
export function isRouteEnabled(href: string, horseRacingEnabled: boolean = HORSE_RACING_ENABLED): boolean {
  if (href === RACES_ROUTE || href.startsWith(`${RACES_ROUTE}/`)) {
    return horseRacingEnabled;
  }
  return true;
}
