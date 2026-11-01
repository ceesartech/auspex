import { test, expect, Page } from '@playwright/test';
import { HORSE_RACING_ENABLED } from '../../../src/lib/sport-switches';

// While horse racing is dormant (HORSE_RACING_ENABLED false in src/lib/sport-switches.ts,
// mirroring services/data-ingestion/dags/sport_switches.py): no Race cards nav
// entry, and the /races pages show a paused notice without calling the API.
// These flows follow the committed constant: with horse racing switched back
// on they are skipped (the enabled pages are covered by the Jest tests in
// tests/unit/components/horse-racing-switch.test.tsx), so re-enabling needs
// no edit here.

async function signIn(page: Page) {
  await page.goto('/login');
  await page.evaluate(() => {
    localStorage.setItem('auth-storage', JSON.stringify({
      state: {
        user: { username: 'ceesar', role: 'admin' },
        token: 'e2e-token',
        expiresAt: Date.now() + 60 * 60 * 1000,
        isAuthenticated: true,
      },
    }));
  });
}

test.describe('Horse racing paused', () => {
  test.skip(HORSE_RACING_ENABLED, 'horse racing is switched on (src/lib/sport-switches.ts)');

  test('/races shows the paused notice and the nav offers no Race cards entry', async ({ page }) => {
    // The races API is /api/v1/races (src/lib/api/races.ts); a paused page never asks it.
    const raceCalls: string[] = [];
    page.on('request', (req) => {
      if (new URL(req.url()).pathname.startsWith('/api/v1/races')) raceCalls.push(req.url());
    });
    await signIn(page);
    await page.goto('/races');
    await expect(page.getByText('Horse racing is paused')).toBeVisible();
    await expect(page.getByRole('link', { name: /race cards/i })).toHaveCount(0);
    await expect(page.getByRole('link', { name: 'Predictions' }).first()).toBeVisible();
    expect(raceCalls).toEqual([]);
  });

  test('a race detail deep link lands on the same notice', async ({ page }) => {
    await signIn(page);
    await page.goto('/races/0b7c6a6e-1f1e-4a55-9d0c-2f3d1c9b8a77');
    await expect(page.getByText('Horse racing is paused')).toBeVisible();
  });
});
