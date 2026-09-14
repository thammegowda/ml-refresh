import { test, expect } from '@playwright/test';

async function ready(page) {
  await expect(page.locator('.py-runtime')).toHaveText('Python + NumPy ready', { timeout: 20000 });
  const example = await page.getByLabel('Calculus example').inputValue();
  await expect(page.locator('.py-chart svg')).toHaveCount(example === 'comparison' || example.startsWith('composition:') ? 3 : 1);
}
async function setSource(page, source) {
  const tab = page.getByRole('tab', { name: 'Code', exact: true });
  if (await tab.isVisible()) await tab.click();
  await page.getByRole('textbox', { name: 'Python source' }).focus();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(source);
}
test.beforeEach(async ({ page }) => { await page.goto('calculus.html'); await ready(page); });

test('framework starts folded and can be revealed without changing execution', async ({ page }) => {
  const editor = page.getByRole('textbox', { name: 'Python source' });
  await expect(page.getByLabel('Show framework')).not.toBeChecked();
  await expect(editor).toContainText('def F(x)');
  await expect(editor).not.toContainText('def connected(x)');
  await page.getByLabel('Show framework').check();
  await expect(editor).toContainText('def connected(x)');
  await page.getByLabel('Show framework').uncheck();
  await expect(editor).not.toContainText('def connected(x)');
  await page.locator('.cm-foldPlaceholder').click();
  await expect(page.getByLabel('Show framework')).toBeChecked();
  await page.getByRole('button', { name: 'Run Python', exact: true }).click();
  await ready(page);
  await expect(page.locator('.py-output-log')).toContainText('Slope at 1 = 2');
});

test('dragging the plot point updates Python tangents and keeps navigation', async ({ page }) => {
  const source = await (await page.request.get('calculus.py')).text();
  await setSource(page, source + '\nprint("setup complete")\n');
  await page.getByRole('button', { name: 'Run Python', exact: true }).click();
  await expect(page.locator('.py-output-log')).toContainText('setup complete');
  await ready(page);
  const handle = page.getByRole('slider', { name: 'Plot evaluation point', exact: true });
  const tangent = page.locator('.py-curve[data-series="Tangent"]').first();
  const before = await tangent.getAttribute('d');
  await page.getByRole('button', { name: 'Zoom in all plots' }).click();
  await page.getByLabel("Function: f'(x)", { exact: true }).uncheck();
  await handle.scrollIntoViewIfNeeded();
  const bounds = await handle.boundingBox();
  const target = await handle.evaluate(element => {
    const ticks = [...element.ownerSVGElement.querySelector('.py-plot-axis').querySelectorAll('.tick')];
    const first = ticks[0], last = ticks.at(-1);
    return first.getScreenCTM().e + (2 - first.__data__) * (last.getScreenCTM().e - first.getScreenCTM().e) / (last.__data__ - first.__data__);
  });
  await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
  await page.mouse.down();
  await page.mouse.move(target, bounds.y + bounds.height / 2, { steps: 8 });
  await page.mouse.up();
  const selected = Number(await page.getByRole('spinbutton', { name: 'Evaluation point', exact: true }).inputValue());
  expect(selected).toBeCloseTo(2, 1);
  await expect(page.locator('.py-output-log')).toContainText(`Slope at ${selected} = ${2 * selected}`);
  await expect(page.locator('.py-output-log')).not.toContainText('setup complete');
  await ready(page);
  await expect(tangent).not.toHaveAttribute('d', before);
  expect(await tangent.evaluate((element, selected) => element.__data__.every(([position, value]) => Math.abs(value - (2 * selected * position - selected ** 2)) < 1e-4), selected)).toBe(true);
  await expect(page.getByLabel('Plot zoom level')).toHaveText('150%');
  await expect(page.getByLabel("Function: f'(x)", { exact: true })).not.toBeChecked();
  await expect(page.getByRole('textbox', { name: 'Python source' })).toContainText(`point = ${selected}`);
  await handle.focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('spinbutton', { name: 'Evaluation point', exact: true })).toHaveValue(String(Number((selected + 0.01).toFixed(2))));
  await expect(page.getByLabel('Plot zoom level')).toHaveText('150%');
});

test('point changes keep plot dimensions and axes fixed while the tangent moves', async ({ page }) => {
  await page.getByLabel('Calculus example').selectOption('log');
  await ready(page);
  const input = page.getByRole('spinbutton', { name: 'Evaluation point', exact: true });
  const axes = () => page.locator('.py-plot-axis').evaluateAll(elements => elements.map(element => element.outerHTML));
  const dimensions = () => page.locator('.py-chart svg').evaluateAll(elements => elements.map(element => element.getAttribute('viewBox')));
  const curveStart = () => page.locator('.py-curve[data-series="f(x)"]').evaluate(element => {
    const point = element.getPointAtLength(0);
    return { x: point.x, y: point.y };
  });
  const initialAxes = await axes();
  const initialDimensions = await dimensions();
  const initialStart = await curveStart();
  const tangent = page.locator('.py-curve[data-series="Tangent"]');
  const initialTangent = await tangent.getAttribute('d');
  for (const [point, slope] of [['0.1', '10'], ['5', '0.2']]) {
    await input.fill(point);
    await expect(page.locator('.py-output-log')).toContainText(`Slope at ${point} = ${slope}`);
    await ready(page);
    expect(await axes()).toEqual(initialAxes);
    expect(await dimensions()).toEqual(initialDimensions);
    expect(await curveStart()).toEqual(initialStart);
    await expect(tangent).not.toHaveAttribute('d', initialTangent);
  }
  await input.fill('0.1');
  await expect(page.locator('.py-output-log')).toContainText('Slope at 0.1 = 10');
  await ready(page);
  await page.getByRole('button', { name: 'Run Python', exact: true }).click();
  await ready(page);
  expect(await axes()).not.toEqual(initialAxes);
});

test('point controls use edited equations and reject nonliteral assignments', async ({ page }) => {
  const source = await (await page.request.get('calculus.py')).text();
  await setSource(page, source.replace('return coefficient * x ** exponent', 'return 3 * x').replace('return coefficient * exponent * x ** (exponent - 1)', 'return np.full_like(x, 3)'));
  const input = page.getByRole('spinbutton', { name: 'Evaluation point', exact: true });
  await input.fill('-2');
  await expect(page.locator('.py-output-log')).toContainText('f(-2) = -6');
  await expect(page.locator('.py-output-log')).toContainText('Slope at -2 = 3');
  await ready(page);
  await setSource(page, source.replace('point = 1', 'point = np.pi'));
  await expect(input).toBeDisabled();
  await expect(page.getByRole('slider', { name: 'Plot evaluation point' })).toHaveAttribute('aria-disabled', 'true');
  await setSource(page, 'text = """\npoint = 1\n"""\nINPUTS = {}\nOUTPUTS = {}');
  await expect(input).toBeDisabled();
});

test('point changes during execution apply the newest value and Stop cancels queued runs', async ({ page }) => {
  const source = await (await page.request.get('calculus.py')).text();
  await setSource(page, source.replace('def update(point):', 'def update(point):\n    import time\n    started = time.monotonic()\n    while time.monotonic() - started < 0.5:\n        pass'));
  const input = page.getByRole('spinbutton', { name: 'Evaluation point', exact: true });
  await input.fill('2');
  await expect(page.getByRole('button', { name: 'Stop Python' })).toBeVisible();
  await input.fill('3');
  await expect(page.locator('.py-output-log')).toContainText('Slope at 3 = 6');
  await ready(page);
  await expect(page.locator('.py-stale')).toHaveCount(0);
  await input.fill('1');
  await expect(page.getByRole('button', { name: 'Stop Python' })).toBeVisible();
  await input.fill('2');
  await page.getByRole('button', { name: 'Stop Python' }).click();
  await expect(page.getByRole('alert')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Run Python', exact: true })).toBeVisible();
  await expect(page.locator('.py-stale')).toBeVisible();
  await page.getByRole('button', { name: 'Reset Python code' }).click();
  await ready(page);
  await expect(input).toHaveValue('1');
});

test('Python equations supply all plot data and editor changes update curves', async ({ page }) => {
  await expect(page.getByRole('textbox', { name: 'Python source' })).toContainText('def f(x)');
  await expect(page.locator('.py-chart svg').first().locator('.py-curve').first()).toHaveAttribute('d', /M/);
  await expect(page.locator('.py-output-log')).toContainText('Slope at 1 = 2');
  const curve = page.locator('.py-curve[data-series="f(x)"]');
  const initial = await curve.getAttribute('d');
  const response = await page.request.get('calculus.py');
  const source = await response.text();
  await setSource(page, source.replace('return coefficient * x ** exponent', 'return 3 * x + 2'));
  await expect(page.locator('.py-stale')).toBeVisible();
  await page.getByRole('button', { name: 'Run Python', exact: true }).click();
  await ready(page);
  await expect(curve).not.toHaveAttribute('d', initial);
  expect(await curve.evaluate(element => element.__data__.find(point => point[0] === 0)[1])).toBe(2);
  await expect(page.locator('.py-output-log')).toContainText('f(1) = 5');
});

test('presets include activations and composition with real Python output', async ({ page }) => {
  await page.getByLabel('Calculus example').selectOption('sigmoid');
  await ready(page);
  await expect(page.locator('.py-output-log')).toContainText('f(1) = 0.731059');
  await page.getByLabel('Calculus example').selectOption('composition:chain');
  await ready(page);
  await expect(page.locator('.py-output-log')).toContainText("h'(1) = 1.0806");
  await expect(page.locator('.py-plot-panel h3')).toHaveText(['f(x)', 'g(x)', 'h(x)']);
  await page.getByRole('spinbutton', { name: 'Evaluation point', exact: true }).fill('2');
  await expect(page.locator('.py-output-log')).toContainText("h'(2) = -2.61457");
  await page.getByLabel('Calculus example').selectOption('comparison');
  await ready(page);
  await expect(page.locator('.py-curve')).toHaveCount(15);
  await expect(page.locator('.py-output-log')).toContainText('Sigmoid: f(0) = 0.5, slope = 0.25');
  const curves = await page.locator('.py-curve').evaluateAll(elements => elements.map(element => element.getAttribute('d')));
  await page.getByRole('spinbutton', { name: 'Evaluation point', exact: true }).fill('2');
  await expect(page.locator('.py-output-log')).toContainText('Sigmoid: f(2) = 0.880797');
  expect(await page.locator('.py-curve').evaluateAll(elements => elements.map(element => element.getAttribute('d')))).toEqual(curves);
});

test('failed Python package startup can be retried without reloading', async ({ page }) => {
  await page.route('**/*numpy*.whl', route => route.abort());
  await page.reload();
  await expect(page.getByRole('alert')).toBeVisible({ timeout: 20000 });
  await page.unroute('**/*numpy*.whl');
  await page.getByRole('button', { name: 'Run Python', exact: true }).click();
  await ready(page);
});

test('preset switches replace edited drafts without prompting and legacy links populate Python', async ({ page }) => {
  await page.goto('calculus.html#family=linear&coefficient=2&point=3&lower=0');
  await page.reload();
  await ready(page);
  await expect(page.locator('.py-output-log')).toContainText('f(3) = 6');
  await setSource(page, 'INPUTS = {}\nOUTPUTS = {"draft": 1}');
  const dialogs = [];
  page.on('dialog', async dialog => { dialogs.push(dialog.message()); await dialog.dismiss(); });
  await page.getByLabel('Calculus example').selectOption('sin');
  await ready(page);
  await expect(page.getByLabel('Calculus example')).toHaveValue('sin');
  await expect(page.locator('.py-output-log')).toContainText('f(1) = 0.841471');
  expect(dialogs).toEqual([]);
});

test('all plots share zoom and preserve finite gaps', async ({ page }) => {
  await page.getByRole('button', { name: 'Zoom in all plots' }).click();
  await expect(page.getByLabel('Plot zoom level')).toHaveText('150%');
  for (const plot of await page.locator('.py-chart svg').all()) await expect(plot).toHaveAttribute('data-zoom', '1.5');
  const plot = page.locator('.py-chart svg').first();
  await plot.focus();
  await page.keyboard.press('+');
  await expect(page.getByLabel('Plot zoom level')).toHaveText('225%');
  await page.keyboard.press('0');
  await expect(page.getByLabel('Plot zoom level')).toHaveText('100%');
  await page.getByLabel('Function: Tangent', { exact: true }).uncheck();
  await expect(page.locator('.py-curve[data-series="Tangent"]')).toHaveCount(0);
  await page.getByLabel('Calculus example').selectOption('reciprocal');
  await ready(page);
  const path = page.locator('.py-curve[data-series="f(x)"]');
  expect((await path.getAttribute('d')).match(/M/g).length).toBe(2);
  await page.getByRole('spinbutton', { name: 'Evaluation point', exact: true }).fill('0');
  const handle = page.getByRole('slider', { name: 'Plot evaluation point' });
  await expect(handle).toHaveAttribute('aria-valuetext', /undefined/);
  await handle.focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.locator('.py-output-log')).toContainText('f(0.01) = 100');
});

test('wheel, drag and pinch synchronize all sampled plots', async ({ page }) => {
  await page.getByLabel('Calculus example').selectOption('composition:chain');
  await ready(page);
  const first = page.locator('.py-chart svg').first();
  await first.scrollIntoViewIfNeeded();
  const bounds = await first.boundingBox();
  const center = { x: bounds.x + bounds.width / 2, y: bounds.y + bounds.height / 2 };
  await page.mouse.move(center.x, center.y);
  await page.mouse.wheel(0, -200);
  await expect.poll(async () => Number(await first.getAttribute('data-zoom'))).toBeGreaterThan(1);
  const zoom = await first.getAttribute('data-zoom');
  for (const plot of await page.locator('.py-chart svg').all()) await expect(plot).toHaveAttribute('data-zoom', zoom);
  const before = await page.locator('.py-curve').evaluateAll(elements => elements.map(element => element.getAttribute('d')));
  await page.mouse.down();
  await page.mouse.move(center.x + 35, center.y + 15, { steps: 5 });
  await page.mouse.up();
  const after = await page.locator('.py-curve').evaluateAll(elements => elements.map(element => element.getAttribute('d')));
  expect(after.every((path, index) => path !== before[index])).toBe(true);
  await page.getByRole('button', { name: 'Reset all plot views' }).click();
  await first.scrollIntoViewIfNeeded();
  const session = await page.context().newCDPSession(page);
  try {
    await session.send('Emulation.setTouchEmulationEnabled', { enabled: true, maxTouchPoints: 2 });
    await page.reload();
    await ready(page);
    await first.scrollIntoViewIfNeeded();
    const touchBounds = await first.boundingBox();
    const centerX = touchBounds.x + touchBounds.width / 2;
    const centerY = touchBounds.y + touchBounds.height / 2;
    await session.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: centerX - 25, y: centerY, id: 0 }, { x: centerX + 25, y: centerY, id: 1 }] });
    await session.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: centerX - 50, y: centerY, id: 0 }, { x: centerX + 50, y: centerY, id: 1 }] });
    await session.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
    await expect(page.getByLabel('Plot zoom level')).toHaveText('200%');
    for (const plot of await page.locator('.py-chart svg').all()) await expect(plot).toHaveAttribute('data-zoom', '2');
  } finally { await session.detach(); }
});

test('combined plot shades signed area and updates bounds without rescaling', async ({ page }, testInfo) => {
  await page.getByLabel('Calculus example').selectOption('sin');
  await ready(page);
  const input = page.getByRole('spinbutton', { name: 'Evaluation point', exact: true });
  const integral = page.getByLabel('Function: Signed integral', { exact: true });
  await input.fill('5');
  await expect(integral).toHaveText('Integral [0, 5] = 0.716338');
  await expect(page.locator('.py-chart svg')).toHaveCount(1);
  await expect(page.locator('.py-curve')).toHaveCount(3);
  await expect(page.locator('.py-area')).toHaveCount(2);
  await expect(page.locator('.py-area-bound')).toHaveCount(2);
  for (const area of await page.locator('.py-area').all()) await expect(area).toHaveAttribute('d', /M.*Z/);
  await page.screenshot({ path: testInfo.outputPath('signed-area.png'), fullPage: true });
  await input.fill('0');
  await expect(integral).toHaveText('Integral [0, 0] = 0');
  await expect(page.locator('.py-area')).toHaveCount(0);
  const source = await (await page.request.get('calculus.py')).text();
  await setSource(page, source.replace('lower = 0', 'lower = 2').replace('point = 1', 'point = 0'));
  await page.getByRole('button', { name: 'Run Python', exact: true }).click();
  await ready(page);
  await expect(integral).toHaveText('Integral [2, 0] = -2.66667');
  const negativeClip = await page.locator('.py-area-negative').getAttribute('clip-path');
  expect(await page.locator('clipPath').evaluateAll((elements, clip) => {
    const rectangle = elements.find(element => `url(#${element.id})` === clip).firstElementChild;
    return Number(rectangle.getAttribute('height'));
  }, negativeClip)).toBeGreaterThan(0);
  await page.getByLabel('Calculus example').selectOption('reciprocal');
  await ready(page);
  await input.fill('-1');
  await expect(integral).toHaveText('Integral [1, -1] = undefined');
  await expect(page.locator('.py-area')).toHaveCount(0);
});

test('plots and workspace fit desktop and mobile', async ({ page }, testInfo) => {
  await expect(page.locator('input[type="number"]')).toHaveCount(1);
  for (const width of [1440, 320]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await expect(page.locator('.py-chart svg')).toHaveCount(1);
    const chart = page.locator('.py-chart svg');
    const bounds = await chart.boundingBox();
    expect(bounds.height / bounds.width).toBeCloseTo(9 / 16, 2);
    expect(bounds.width).toBeLessThanOrEqual(800);
    for (const curve of await page.locator('.py-curve').all()) await expect(curve).toHaveAttribute('d', /M/);
    await page.screenshot({ path: testInfo.outputPath(`calculus-python-${width}.png`), fullPage: true });
  }
  const handle = page.getByRole('slider', { name: 'Plot evaluation point' });
  await handle.scrollIntoViewIfNeeded();
  const bounds = await handle.boundingBox();
  const center = { x: bounds.x + bounds.width / 2, y: bounds.y + bounds.height / 2 };
  const session = await page.context().newCDPSession(page);
  try {
    await session.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [center] });
    for (const offset of [12, 24, 36]) await session.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: center.x + offset, y: center.y }] });
    await session.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
    const selected = Number(await page.getByRole('spinbutton', { name: 'Evaluation point', exact: true }).inputValue());
    expect(selected).toBeGreaterThan(3);
    await expect(page.locator('.py-output-log')).toContainText(`Slope at ${selected} = ${2 * selected}`);
    await expect(page.getByLabel('Plot zoom level')).toHaveText('100%');
  } finally { await session.detach(); }
});