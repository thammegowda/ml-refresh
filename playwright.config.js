import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: '.',
  testMatch: '**/*.browser.spec.js',
  testIgnore: ['**/node_modules/**', 'dist/**', '.venv/**', 'jupyter/.cache/**'],
  use: {
    baseURL: process.env.REFRESH_BASE_URL ?? 'http://localhost:1414/app/refresh/',
    browserName: 'chromium',
    viewport: { width: 1440, height: 900 },
    trace: 'retain-on-failure',
  },
});
