import { test, expect } from '@playwright/test';

test.describe('Authentication Flow', () => {
  test('redirects unauthenticated users to login', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL(/\/login/);
  });

  test('login page renders correctly', async ({ page }) => {
    await page.goto('/login');
    await expect(page.getByText('Auspex')).toBeVisible();
    await expect(page.getByPlaceholder('admin or you@example.com')).toBeVisible();
    await expect(page.getByPlaceholder('Enter your password')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Sign In' })).toBeVisible();
  });

  test('shows validation errors for empty form', async ({ page }) => {
    await page.goto('/login');
    await page.getByRole('button', { name: 'Sign In' }).click();
    // Form validation should prevent submission
    await expect(page.getByPlaceholder('admin or you@example.com')).toBeVisible();
  });
});

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    // Set auth token in localStorage to simulate authenticated state
    await page.goto('/login');
    await page.evaluate(() => {
      localStorage.setItem('auth-storage', JSON.stringify({
        state: {
          user: { username: 'ceesar', role: 'admin' },
          token: 'test-token',
          isAuthenticated: true,
        },
      }));
    });
  });

  test('dashboard loads for authenticated users', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
  });

  test('navigation links are visible', async ({ page }) => {
    await page.goto('/');
    const header = page.getByRole('banner');
    await expect(header.getByRole('link', { name: 'Predictions' })).toBeVisible();
    await expect(header.getByRole('link', { name: 'Recommendations' })).toBeVisible();
    await expect(header.getByRole('link', { name: 'Accumulator' })).toBeVisible();
    await expect(header.getByRole('link', { name: 'Analytics' })).toBeVisible();
  });
});
