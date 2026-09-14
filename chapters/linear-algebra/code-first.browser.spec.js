import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';

const original = readFileSync(new URL('./lesson.py', import.meta.url), 'utf8');
const seeded = original.replace('np.random.seed(42)', 'np.random.seed(7)');
const cell = (page, name, group = 'outputs') => page.locator(`[data-group="${group}"]`).getByLabel(name, { exact: true });
const value = async (page, name, group = 'outputs') => Number((await cell(page, name, group).getAttribute('title')).split(' = ')[1]);
async function losses(page) {
  const output = await page.locator('.py-output-log').textContent();
  return [...output.matchAll(/Update \d+: MSE = ([\d.]+)/g)].map((match) => Number(match[1]));
}
async function showTab(page, name) {
  const tab = page.getByRole('tab', { name, exact: true });
  if (await tab.isVisible()) await tab.click();
}
async function setSource(page, source) {
  await showTab(page, 'Code');
  await page.getByRole('textbox', { name: 'Python source' }).focus();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(source);
}
async function run(page, source) {
  await setSource(page, source);
  await page.getByRole('button', { name: 'Run Python', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Run Python', exact: true })).toBeVisible();
  expect(await page.locator('.py-error').allTextContents()).toEqual([]);
  await expect(page.locator('.py-runtime')).toHaveText('Python + NumPy ready');
}

test.beforeEach(async ({ page }) => {
  await page.goto('linear-algebra.html');
  await expect(page.locator('.py-runtime')).toHaveText('Python + NumPy ready', { timeout: 20000 });
  await expect(cell(page, 'Y_pred[0,0]')).toBeVisible();
});

test('defaults populate Python and render only declared inputs and outputs', async ({ page }) => {
  await expect(page.locator('.site-header h1')).toContainText('Matrix operations & backprop');
  await expect(page.locator('.py-toolbar')).toHaveCount(0);
  await expect(page.locator('.py-editor-toolbar .py-runtime')).toHaveText('Python + NumPy ready');
  const header = await page.locator('.site-header').boundingBox();
  const toolbar = await page.locator('.py-editor-toolbar').boundingBox();
  expect(toolbar.y).toBeCloseTo(header.y + header.height, 0);
  await expect(page.getByRole('textbox', { name: 'Python source' })).toContainText('import numpy as np');
  await expect(page.locator('[data-group="inputs"] .py-variable')).toHaveCount(2);
  await expect(page.locator('[data-group="outputs"] .py-variable')).toHaveCount(4);
  await expect(page.locator('.py-variable-meta > span:first-child')).toHaveText(Array(6).fill('float32'));
  await expect(cell(page, 'X[0,0]', 'inputs')).toHaveText('1');
  await expect(cell(page, 'Y[0,0]', 'inputs')).toHaveText('1');
  const prediction = await value(page, 'W[0,0]') + 2 * await value(page, 'W[1,0]') - await value(page, 'W[2,0]') + await value(page, 'b[0,0]');
  expect(await value(page, 'Y_pred[0,0]')).toBeCloseTo(prediction, 5);
  const history = await losses(page);
  expect(history).toHaveLength(11);
  expect(history[10]).toBeGreaterThanOrEqual(0);
  expect(history[10]).toBeLessThan(history[0] * 0.1);
  await expect(page.locator('[data-variable^="loss"]')).toHaveCount(0);
  for (const heading of await page.locator('.py-result-heading, .py-value-group > header').all()) {
    expect((await heading.boundingBox()).height).toBeLessThanOrEqual(16);
    await expect(heading).toHaveCSS('margin-bottom', '0px');
    await expect(heading).toHaveCSS('padding-top', '0px');
    await expect(heading).toHaveCSS('padding-bottom', '0px');
  }
  for (const name of ['X', 'Y']) {
    expect((await page.locator(`[data-variable="${name}"]`).boundingBox()).width).toBeLessThan(250);
    expect((await cell(page, `${name}[0,0]`, 'inputs').boundingBox()).width).toBe(72);
  }
  await expect(page.locator('[data-group="inputs"] [data-variable="dY"]')).toHaveCount(0);
  await expect(page.getByRole('spinbutton', { name: 'Learning rate' })).toHaveValue('0.1');
  await expect(page.getByRole('combobox')).toHaveCount(0);
  const geometry = await page.evaluate(() => {
    const editor = document.querySelector('.py-code').getBoundingClientRect();
    const terminal = document.querySelector('.py-output-panel').getBoundingClientRect();
    const workbench = document.querySelector('.py-workbench').getBoundingClientRect();
    const visualization = document.querySelector('.py-visualization').getBoundingClientRect();
    return { editorRight: editor.right, terminalLeft: terminal.left, editorTop: editor.top, terminalTop: terminal.top, height: workbench.height, viewportHeight: innerHeight, bottom: workbench.bottom, visualizationTop: visualization.top };
  });
  expect(geometry.editorRight).toBeLessThan(geometry.terminalLeft);
  expect(geometry.editorTop).toBe(geometry.terminalTop);
  expect(geometry.height).toBeCloseTo(Math.max(360, geometry.viewportHeight * 0.50), 0);
  expect(geometry.visualizationTop).toBeGreaterThanOrEqual(geometry.bottom);
  await expect(page.getByRole('region', { name: 'Python terminal', exact: true })).toBeVisible();
  await expect(page.getByRole('tablist', { name: 'Python workspace' })).toBeHidden();
  expect((await cell(page, 'Y_pred[0,0]').boundingBox()).height).toBeLessThanOrEqual(26);
});

test('code alone changes input values, equations, dimensions, and displayed labels', async ({ page }) => {
  const forwardOnly = seeded.replace('learning_rate = 0.1', 'learning_rate = 0.0');
  await run(page, forwardOnly);
  const prediction = await value(page, 'Y_pred[0,0]');
  const weight = await value(page, 'W[0,0]');
  await run(page, forwardOnly.replace('[1., 2., -1.]', '[50., 2., -1.]'));
  await expect(cell(page, 'X[0,0]', 'inputs')).toHaveText('50');
  expect(await value(page, 'Y_pred[0,0]')).toBeCloseTo(prediction + 49 * weight, 5);
  await run(page, 'import numpy as np\nX = np.arange(12.).reshape(4, 3)\nW = np.ones((3, 5))\nb = np.arange(5.)\nY = (X @ W + b).T\nINPUTS = {"Features": X, "Bias": b}\nOUTPUTS = {"Transposed": Y}');
  await expect(page.locator('[data-group="inputs"] .py-variable')).toHaveCount(2);
  await expect(page.locator('[data-group="outputs"] .py-variable')).toHaveCount(1);
  await expect(page.locator('[data-variable="Features"] .py-shape')).toHaveText('(4, 3)');
  await expect(page.locator('[data-variable="Bias"] .py-shape')).toHaveText('(5,)');
  await expect(page.locator('[data-variable="Transposed"] .py-shape')).toHaveText('(5, 4)');
  await expect(cell(page, 'Transposed[0,3]')).toHaveText('30');
  await expect(page.locator('[data-variable="Y_pred"]')).toHaveCount(0);
});

test('ground-truth edits change learned predictions but not initialization', async ({ page }) => {
  await run(page, seeded);
  const before = await value(page, 'Y_before[0,0]');
  const prediction = await value(page, 'Y_pred[0,0]');
  await run(page, seeded.replace('[1., -0.5]', '[2., -0.5]'));
  await expect(cell(page, 'Y[0,0]', 'inputs')).toHaveText('2');
  expect(await value(page, 'Y_before[0,0]')).toBe(before);
  expect(await value(page, 'Y_pred[0,0]')).not.toBe(prediction);
  const history = await losses(page);
  expect(history[10]).toBeLessThan(history[0]);
});

test('SGD respects the global seed and data dimensions', async ({ page }) => {
  const weight = await value(page, 'W[0,0]');
  const bias = await value(page, 'b[0,0]');
  await run(page, original);
  expect(await value(page, 'W[0,0]')).toBe(weight);
  expect(await value(page, 'b[0,0]')).toBe(bias);
  await run(page, seeded);
  expect(await value(page, 'W[0,0]')).not.toBe(weight);
  expect(await value(page, 'b[0,0]')).not.toBe(bias);
  await run(page, original.replace('bound =', 'X = np.ones((4, 5), dtype=np.float32)\nY = np.zeros((4, 3), dtype=np.float32)\nbound ='));
  await expect(page.locator('.py-variable-meta > span:first-child')).toHaveText(Array(6).fill('float32'));
  await expect(page.locator('[data-variable="W"] .py-shape')).toHaveText('(5, 3)');
  await expect(page.locator('[data-variable="b"] .py-shape')).toHaveText('(1, 3)');
  await expect(page.locator('[data-variable="Y_pred"] .py-shape')).toHaveText('(4, 3)');
  const history = await losses(page);
  expect(history[10]).toBeLessThan(history[0]);
});

test('zero learning rate stops learning and ten updates are printed', async ({ page }) => {
  await showTab(page, 'Output');
  await expect(page.locator('.py-output-log')).toContainText('Update 00: MSE');
  await expect(page.locator('.py-output-log')).toContainText('Update 10: MSE');
  expect((await page.locator('.py-output-log').textContent()).match(/Update \d\d:/g)).toHaveLength(11);
  await run(page, original + '\nprint("initialized")\n');
  const trained = await value(page, 'Y_pred[0,0]');
  const rate = page.getByRole('spinbutton', { name: 'Learning rate' });
  await rate.fill('0');
  await rate.press('Tab');
  await expect(cell(page, 'Y_pred[0,0]')).toHaveText(await cell(page, 'Y_before[0,0]').textContent());
  await expect(page.locator('.py-output-log')).not.toContainText('initialized');
  const history = await losses(page);
  expect(history).toHaveLength(11);
  expect(history.every((loss) => loss === history[0])).toBe(true);
  expect(await value(page, 'Y_pred[0,0]')).toBe(await value(page, 'Y_before[0,0]'));
  await rate.fill('0.1');
  await rate.press('Tab');
  await expect.poll(() => value(page, 'Y_pred[0,0]')).toBe(trained);
  await expect(page.locator('.py-output-log')).not.toContainText('initialized');
});

test('display rounds to three decimal places without changing underlying values', async ({ page }) => {
  await run(page, 'import numpy as np\nINPUTS = {"Values": np.array([1.23456, -2.34567, 0.0004, -0.0004, 2.0, 0.0006])}\nOUTPUTS = {"Scalar": 1.23456, "Squared": 1.23456 ** 2}');
  await expect(page.locator('[data-variable="Values"] output')).toHaveText(['1.235', '-2.346', '0', '0', '2', '0.001']);
  await expect(cell(page, 'Scalar')).toHaveText('1.235');
  await expect(cell(page, 'Scalar')).toHaveAttribute('title', 'Scalar = 1.23456');
  expect(await value(page, 'Squared')).toBeCloseTo(1.23456 ** 2, 12);
});

test('scalars, vectors, empty groups, and previews adapt without numeric input restrictions', async ({ page }) => {
  await run(page, 'import numpy as np\nX = np.array([10000., -100000.])\nINPUTS = {"X": X}\nOUTPUTS = {"Count": X.size, "Squared": X ** 2, "Empty": np.empty((0, 2)), "Large": np.ones((20, 30))}');
  await expect(cell(page, 'X[1]', 'inputs')).toHaveText('-100000');
  await expect(cell(page, 'Count')).toHaveText('2');
  await expect(cell(page, 'Squared[1]')).toHaveAttribute('title', 'Squared[1] = 10000000000');
  await expect(page.locator('[data-variable="Empty"]')).toContainText('Empty array');
  await expect(page.locator('[data-variable="Large"]')).toContainText('Preview 12');
  await expect(page.locator('[data-variable="Large"] output')).toHaveCount(144);
  await run(page, 'INPUTS = {}\nOUTPUTS = {}\nprint("only output")');
  await expect(page.locator('.py-variable')).toHaveCount(0);
  await showTab(page, 'Output');
  await expect(page.locator('.py-output-log')).toContainText('only output');
});

test('drafts and invalid exports keep the last successful visualization clearly marked', async ({ page }) => {
  const prediction = await cell(page, 'Y_pred[0,0]').textContent();
  for (const [source, message] of [
    ['def broken(:', 'SyntaxError'],
    ['X = 1', 'Define INPUTS'],
    ['INPUTS = {}\nOUTPUTS = {"Y": float("nan")}', 'finite'],
    ['import numpy as np\nY = np.ones((2, 3)) @ np.ones((4, 2))', 'matmul'],
  ]) {
    await setSource(page, source);
    await expect(page.locator('.py-stale')).toContainText('current code not applied');
    await page.getByRole('button', { name: 'Run Python', exact: true }).click();
    await expect(page.getByRole('alert')).toContainText(message);
    await expect(cell(page, 'Y_pred[0,0]')).toHaveText(prediction);
  }
  await page.getByRole('button', { name: 'Reset Python code', exact: true }).click();
  await expect(page.locator('.py-runtime')).toHaveText('Python + NumPy ready');
  await expect(page.locator('.py-stale')).toHaveCount(0);
  await page.getByRole('textbox', { name: 'Python source' }).press('ControlOrMeta+Home');
  await expect(page.getByRole('textbox', { name: 'Python source' })).toContainText('np.random.seed(42)');
});

test('code edited during a run cannot publish outdated results', async ({ page }) => {
  const prediction = await cell(page, 'Y_pred[0,0]').textContent();
  await setSource(page, 'import time\ntime.sleep(1)\nINPUTS = {}\nOUTPUTS = {"Old": 5}');
  await page.getByRole('button', { name: 'Run Python', exact: true }).click();
  await expect(page.locator('.py-runtime')).toContainText('Running Python');
  await setSource(page, 'INPUTS = {}\nOUTPUTS = {"New": 10}');
  await expect(page.locator('.py-runtime')).toContainText('Code changed during run');
  await expect(page.locator('[data-variable="Old"]')).toHaveCount(0);
  await expect(cell(page, 'Y_pred[0,0]')).toHaveText(prediction);
  await page.getByRole('button', { name: 'Run Python', exact: true }).click();
  await expect(cell(page, 'New')).toHaveText('10');
});

test('editor uses gray comments and bright text on dark selections', async ({ page }, testInfo) => {
  const editor = page.getByRole('textbox', { name: 'Python source' });
  const comment = page.locator('.cm-line span').filter({ hasText: '# Edit these arrays' });
  await expect(comment).toHaveCSS('color', 'rgb(112, 112, 112)');
  await setSource(page, '# Edit these arrays\nimport numpy as np\nclass Layer:\n    def forward(self, values):\n        label = "prediction"\n        return np.array(values) + 42\n');
  for (const [text, color] of [
    ['import', 'rgb(166, 38, 70)'],
    ['forward', 'rgb(36, 95, 165)'],
    ['array', 'rgb(36, 95, 165)'],
    ['Layer', 'rgb(135, 82, 11)'],
    ['"prediction"', 'rgb(38, 112, 68)'],
    ['42', 'rgb(161, 83, 18)'],
  ]) {
    await expect(page.locator('.cm-line').getByText(text, { exact: true }).first()).toHaveCSS('color', color);
  }
  await editor.focus();
  await page.keyboard.press('ControlOrMeta+A');
  expect(await page.evaluate(() => getSelection().toString())).toContain('# Edit these arrays');
  for (const token of [comment, page.locator('.cm-line span').filter({ hasText: /^import$/ }).first()]) {
    const selection = await token.evaluate(element => {
      const style = getComputedStyle(element, '::selection');
      return { color: style.color, background: style.backgroundColor };
    });
    expect(selection).toEqual({ color: 'rgb(255, 255, 255)', background: 'rgb(38, 50, 56)' });
  }
  await expect(page.locator('.cm-selectionBackground').first()).toHaveCSS('background-color', 'rgb(38, 50, 56)');
  await page.screenshot({ path: testInfo.outputPath('editor-selection.png') });
  await page.keyboard.press('ArrowRight');
  await page.screenshot({ path: testInfo.outputPath('editor-palette.png') });
  await page.setViewportSize({ width: 390, height: 900 });
  await expect(comment).toHaveCSS('color', 'rgb(112, 112, 112)');
  await page.screenshot({ path: testInfo.outputPath('editor-palette-mobile.png') });
});

test('secondary code selections retain bright text on the dark selection layer', async ({ page }) => {
  await setSource(page, 'repeated = 1\nrepeated = 2\nINPUTS = {}\nOUTPUTS = {}');
  await page.keyboard.press('ControlOrMeta+Home');
  await page.keyboard.press('ControlOrMeta+d');
  await page.keyboard.press('ControlOrMeta+d');
  await expect(page.locator('.cm-secondary-selection').first()).toHaveCSS('color', 'rgb(255, 255, 255)');
  await expect(page.locator('.cm-selectionBackground').first()).toHaveCSS('background-color', 'rgb(38, 50, 56)');
});

test('editor highlights Python, numbers lines, and preserves drafts across keyboard tabs', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 900 });
  const editor = page.getByRole('textbox', { name: 'Python source' });
  await editor.scrollIntoViewIfNeeded();
  await expect(page.locator('.cm-lineNumbers')).toBeVisible();
  const colors = await page.locator('.cm-content .cm-line span').evaluateAll((tokens) => [...new Set(tokens.map((token) => getComputedStyle(token).color))]);
  expect(colors.length).toBeGreaterThan(1);
  await setSource(page, 'INPUTS = {}\nOUTPUTS = {"Answer": 42}');
  await page.getByRole('tab', { name: 'Code', exact: true }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: 'Output', exact: true })).toBeFocused();
  await page.keyboard.press('Home');
  await expect(editor).toContainText('Answer');
  await editor.press('ControlOrMeta+Enter');
  await expect(cell(page, 'Answer')).toHaveText('42');
  await expect(page.getByRole('tab', { name: 'Code', exact: true })).toHaveAttribute('aria-selected', 'true');
  await setSource(page, 'raise ValueError("mobile error")');
  await page.getByRole('button', { name: 'Run Python', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('mobile error');
  await expect(page.getByRole('tab', { name: 'Output', exact: true })).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('.py-code')).toBeHidden();
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(page.locator('.py-code')).toBeVisible();
  await expect(page.getByRole('alert')).toBeVisible();
});

test('Output streams stdout and stderr before failure and remains clearable', async ({ page }) => {
  await setSource(page, "import sys, time\nprint('before failure')\nprint('a warning', file=sys.stderr)\ntime.sleep(2)\nraise ValueError('example failure')");
  await page.getByRole('button', { name: 'Run Python', exact: true }).click();
  await showTab(page, 'Output');
  await expect(page.locator('[data-stream="stdout"] pre')).toContainText('before failure');
  await expect(page.locator('[data-stream="stderr"] pre')).toContainText('a warning');
  await expect(page.getByRole('button', { name: 'Stop Python', exact: true })).toBeVisible();
  await expect(page.getByRole('alert')).toContainText('example failure');
  await expect(page.locator('[data-stream="stdout"] pre')).toContainText('before failure');
  await page.getByRole('button', { name: 'Clear output', exact: true }).click();
  await expect(page.getByRole('alert')).toHaveCount(0);
  await expect(page.locator('.py-output-entry')).toHaveCount(0);
});

test('clearing a failed rerun does not mark old results as current', async ({ page }) => {
  await run(page, 'import builtins\nbuiltins.refresh_runs = getattr(builtins, "refresh_runs", 0) + 1\nINPUTS = {}\nOUTPUTS = {"Count": builtins.refresh_runs}\nif builtins.refresh_runs > 1:\n    raise ValueError("second run failed")');
  await page.getByRole('button', { name: 'Run Python', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('second run failed');
  await page.getByRole('button', { name: 'Clear output' }).click();
  await expect(page.getByRole('alert')).toHaveCount(0);
  await expect(page.locator('.py-stale')).toContainText('Last successful run');
  await expect(cell(page, 'Count')).toHaveText('1');
});

test('Stop and timeouts retain bounded output and recover', async ({ page }) => {
  await setSource(page, "print('x' * 10000)\nwhile True:\n    print('more')");
  await page.getByRole('button', { name: 'Run Python', exact: true }).click();
  await showTab(page, 'Output');
  await expect(page.locator('.py-output-log')).toContainText('Output truncated');
  expect((await page.locator('.py-output-entry pre').textContent()).length).toBeLessThan(8100);
  await page.getByRole('button', { name: 'Stop Python', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('stopped');
  await page.getByRole('button', { name: 'Run Python', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('exceeded 5 seconds', { timeout: 15000 });
  await page.getByRole('button', { name: 'Reset Python code', exact: true }).click();
  await expect(page.locator('.py-runtime')).toHaveText('Python + NumPy ready', { timeout: 20000 });
});

test('default runtime stays local and chapter links and Python download work', async ({ page, context }) => {
  const external = [];
  const origin = new URL(page.url()).origin;
  await context.route('**/*', async (route) => {
    if (new URL(route.request().url()).origin !== origin) {
      external.push(route.request().url());
      await route.abort();
    } else await route.continue();
  });
  await page.reload();
  await expect(page.locator('.py-runtime')).toHaveText('Python + NumPy ready', { timeout: 20000 });
  await expect(cell(page, 'Y_pred[0,0]')).toBeVisible();
  expect(external).toEqual([]);
  await expect(page.locator('.chapter-pagination a[rel="prev"]')).toHaveAttribute('href', './calculus.html');
  const lesson = await page.request.get('linear-algebra.py');
  expect(lesson.ok()).toBe(true);
  expect(await lesson.text()).toBe(original);
  await page.getByRole('link', { name: 'Reference', exact: true }).click();
  await expect(page.locator('#_keep_in_mind')).toBeInViewport();
});

test('code panel resizes with pointer and keyboard without losing source', async ({ page }, testInfo) => {
  const widthHandle = page.getByRole('separator', { name: 'Code panel width' });
  const heightHandle = page.getByRole('separator', { name: 'Workspace height' });
  const panel = page.locator('.py-code');
  const editor = page.locator('.py-workbench');
  const initialWidth = (await panel.boundingBox()).width;
  const divider = await widthHandle.boundingBox();
  await page.mouse.move(divider.x + divider.width / 2, divider.y + 50);
  await page.mouse.down();
  await page.mouse.move(divider.x + 150, divider.y + 50);
  await page.mouse.up();
  expect((await panel.boundingBox()).width).toBeGreaterThan(initialWidth + 100);
  await widthHandle.press('Home');
  await expect(widthHandle).toHaveAttribute('aria-valuenow', '30');
  await widthHandle.press('End');
  await expect(widthHandle).toHaveAttribute('aria-valuenow', '80');
  await widthHandle.dblclick();
  await expect(widthHandle).toHaveAttribute('aria-valuenow', '65');
  const initialHeight = (await editor.boundingBox()).height;
  await heightHandle.scrollIntoViewIfNeeded();
  const bottom = await heightHandle.boundingBox();
  await page.mouse.move(bottom.x + bottom.width / 2, bottom.y + bottom.height / 2);
  await page.mouse.down();
  await page.mouse.move(bottom.x + bottom.width / 2, bottom.y + 100);
  await page.mouse.up();
  expect((await editor.boundingBox()).height).toBeGreaterThan(initialHeight + 80);
  await expect(page.locator('.py-code')).toBeVisible();
  await expect(page.locator('.py-output-panel')).toBeVisible();
  expect((await editor.boundingBox()).height).toBeGreaterThan(initialHeight + 80);
  await expect(page.getByRole('textbox', { name: 'Python source' })).toContainText('np.random.seed(42)');
  await page.screenshot({ path: testInfo.outputPath('resized-desktop.png'), fullPage: true });
  await page.setViewportSize({ width: 390, height: 900 });
  await expect(widthHandle).toBeHidden();
  await showTab(page, 'Output');
  await expect(page.locator('.py-code')).toBeHidden();
  await showTab(page, 'Code');
  expect((await editor.boundingBox()).height).toBeGreaterThan(initialHeight + 80);
  await heightHandle.press('Home');
  expect((await editor.boundingBox()).height).toBe(360);
  await expect(heightHandle).toHaveAttribute('aria-valuenow', '360');
  await heightHandle.press('ArrowDown');
  expect((await editor.boundingBox()).height).toBeGreaterThan(360);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('resized-mobile.png'), fullPage: true });
  await heightHandle.dblclick();
  await expect(heightHandle).toHaveAttribute('aria-valuenow', '450');
  await page.getByRole('button', { name: 'Run Python', exact: true }).click();
  await expect(page.locator('.py-runtime')).toHaveText('Python + NumPy ready');
  await expect(cell(page, 'Y_pred[0,0]')).toBeVisible();
});

test('code-first workspace and variable-sized previews fit desktop and mobile', async ({ page }, testInfo) => {
  for (const width of [1440, 768, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await expect(page.locator('.cm-lineNumbers')).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath(`code-first-${width}.png`), fullPage: true });
    if (width <= 800) {
      await showTab(page, 'Output');
      await expect(page.locator('.py-output-log')).toContainText('Update 10:');
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      await page.screenshot({ path: testInfo.outputPath(`terminal-${width}.png`), fullPage: true });
      await showTab(page, 'Code');
    }
  }
  await run(page, 'import numpy as np\nINPUTS = {"Long label " * 6: np.ones((20, 30))}\nOUTPUTS = {"Vector": np.arange(12)}');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await expect(page.locator('.py-variable-meta')).toContainText(['Preview 12', 'int']);
  await page.screenshot({ path: testInfo.outputPath('code-first-large-320.png'), fullPage: true });
});