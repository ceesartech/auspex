// Per-sport display switches for the frontend.
//
// HORSE_RACING_ENABLED mirrors HORSE_RACING_ENABLED in
// services/data-ingestion/dags/sport_switches.py, the ONE switch that turns
// the horse-racing pipeline on or off (False since 2026-11-01: dormant). Keep
// the two equal: a repo-root unit test (tests/unit/test_frontend_sport_switches.py)
// fails when they differ. While it is false:
//   * the Race cards nav entry is hidden (sidebar and header);
//   * the /races pages show a "paused" notice instead of fetching lists that
//     can only come back empty;
//   * the predictions sport filter drops "Horse Racing" and the
//     recommendations market filter drops "Win (Horse Racing)".
//
// Re-enable: set this to true together with the DAG switch, and push to main.
// That is the whole change: no test pins either value (the mirror test only
// keeps them equal, and the dormant and enabled states are each tested with
// the constant forced). The frontend image changes only when CI rebuilds it.
export const HORSE_RACING_ENABLED: boolean = false;

// Routes that belong to a switched-off sport.
const RACES_ROUTE = '/races';
const HORSE_RACING = 'horse_racing';

// False for a sport key (as the API spells it) whose switch is off. The
// `horseRacingEnabled` parameter exists only so tests can exercise both
// states of the constant.
export function isSportEnabled(sport: string, horseRacingEnabled: boolean = HORSE_RACING_ENABLED): boolean {
  return sport === HORSE_RACING ? horseRacingEnabled : true;
}

// False for a nav href under a switched-off sport.
export function isRouteEnabled(href: string, horseRacingEnabled: boolean = HORSE_RACING_ENABLED): boolean {
  if (href === RACES_ROUTE || href.startsWith(`${RACES_ROUTE}/`)) {
    return isSportEnabled(HORSE_RACING, horseRacingEnabled);
  }
  return true;
}
