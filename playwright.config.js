import { defineConfig } from '@playwright/test';
import { book } from './app.js';

export default defineConfig({
  testDir: '.',
  testMatch: '**/*.browser.spec.js',
  testIgnore: ['**/node_modules/**', 'dist/**', '.venv/**', 'jupyter/.cache/**'],
  // Tests are independent, so CI can split any file across workers and shards (--shard=i/n).
  fullyParallel: true,
  workers: process.env.CI ? 2 : undefined,
  retries: process.env.CI ? 1 : 0,
  webServer: process.env.REFRESH_BASE_URL ? undefined : {
    command: 'npm run preview',
    url: `http://localhost:1414${new URL(book.url).pathname}`,
    reuseExistingServer: true,
  },
  use: {
    baseURL: process.env.REFRESH_BASE_URL ?? `http://localhost:1414${new URL(book.url).pathname}`,
    browserName: 'chromium',
    viewport: { width: 1440, height: 900 },
    trace: 'on-first-retry',
  },
});
