import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './chapters',
  testMatch: '**/*.browser.spec.js',
  use: {
    baseURL: process.env.REFRESH_BASE_URL ?? 'http://localhost:1414/app/refresh/',
    browserName: 'chromium',
    viewport: { width: 1440, height: 900 },
    trace: 'retain-on-failure',
  },
});