import { HORSE_RACING_ENABLED, isRouteEnabled } from '@/lib/sport-switches';

describe('sport switches', () => {
  it('horse racing is dormant, mirroring sport_switches.HORSE_RACING_ENABLED', () => {
    // tests/unit/test_frontend_sport_switches.py (repo root) checks the two agree.
    expect(HORSE_RACING_ENABLED).toBe(false);
  });

  it('switches off the races routes and nothing else while horse racing is dormant', () => {
    expect(isRouteEnabled('/races')).toBe(false);
    expect(isRouteEnabled('/races/0b7c6a6e-1f1e-4a55-9d0c-2f3d1c9b8a77')).toBe(false);
    for (const href of ['/', '/predictions', '/recommendations', '/accumulator', '/analytics', '/lottery', '/settings']) {
      expect(isRouteEnabled(href)).toBe(true);
    }
    // A prefix that is not the races route is never caught by it.
    expect(isRouteEnabled('/racesx')).toBe(true);
  });

  it('turns the races routes back on with the one switch', () => {
    expect(isRouteEnabled('/races', true)).toBe(true);
    expect(isRouteEnabled('/races/abc', true)).toBe(true);
    expect(isRouteEnabled('/predictions', true)).toBe(true);
  });
});
