import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  fullyParallel: true,
  workers: process.env.CI ? 2 : 3,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: 'http://127.0.0.1:4173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'chromium', testIgnore: /webgpu\.spec\.ts$/, use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 1000 } } },
    {
      // The Transformer page's WebGPU path. Locally, full Chromium runs it on this machine's GPU. CI runners
      // have no GPU, so the headless shell's software WebGPU (SwiftShader) stands in; the Linux flags are unverified.
      name: 'webgpu',
      testMatch: /webgpu\.spec\.ts$/,
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1440, height: 1000 },
        ...(process.env.CI
          ? { launchOptions: { args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader'] } }
          : { channel: 'chromium' }),
      },
    },
  ],
  webServer: {
    command: 'npm run preview -- --port 4173 --strictPort',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
