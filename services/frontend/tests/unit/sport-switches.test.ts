import { HORSE_RACING_ENABLED, isRouteEnabled, isSportEnabled } from '@/lib/sport-switches';

// No test here pins the committed value of HORSE_RACING_ENABLED: re-enabling
// horse racing is the two constants (this one and the DAG's) and nothing
// else. tests/unit/test_frontend_sport_switches.py (repo root) keeps the two
// equal; each state is exercised below by passing it explicitly.

describe('sport switches', () => {
  it('is a plain boolean, and the helpers default to it', () => {
    expect(typeof HORSE_RACING_ENABLED).toBe('boolean');
    expect(isRouteEnabled('/races')).toBe(HORSE_RACING_ENABLED);
    expect(isSportEnabled('horse_racing')).toBe(HORSE_RACING_ENABLED);
  });

  it('dormant: switches off the races routes and the horse_racing sport, nothing else', () => {
    expect(isRouteEnabled('/races', false)).toBe(false);
    expect(isRouteEnabled('/races/0b7c6a6e-1f1e-4a55-9d0c-2f3d1c9b8a77', false)).toBe(false);
    for (const href of ['/', '/predictions', '/recommendations', '/accumulator', '/analytics', '/lottery', '/settings']) {
      expect(isRouteEnabled(href, false)).toBe(true);
    }
    // A prefix that is not the races route is never caught by it.
    expect(isRouteEnabled('/racesx', false)).toBe(true);
    expect(isSportEnabled('horse_racing', false)).toBe(false);
    for (const sport of ['soccer', 'nfl', 'nhl', 'nba', 'tennis', 'mma', 'lottery']) {
      expect(isSportEnabled(sport, false)).toBe(true);
    }
  });

  it('enabled: the races routes and the sport come back', () => {
    expect(isRouteEnabled('/races', true)).toBe(true);
    expect(isRouteEnabled('/races/abc', true)).toBe(true);
    expect(isRouteEnabled('/predictions', true)).toBe(true);
    expect(isSportEnabled('horse_racing', true)).toBe(true);
  });
});
