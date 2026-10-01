import { defineConfig } from '@playwright/test';
import { book } from './app.js';

export default defineConfig({
  testDir: '.',
  testMatch: '**/*.browser.spec.js',
  testIgnore: ['**/node_modules/**', 'dist/**', '.venv/**', 'jupyter/.cache/**'],
  retries: process.env.CI ? 1 : 0,
  use: {
    baseURL: process.env.REFRESH_BASE_URL ?? `http://localhost:1414${new URL(book.url).pathname}`,
    browserName: 'chromium',
    viewport: { width: 1440, height: 900 },
    trace: 'retain-on-failure',
  },
});
