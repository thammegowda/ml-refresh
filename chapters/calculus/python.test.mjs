import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { families, rules, functionCode, compositionCode, comparisonCode } from './expressions.js';
import { runPython } from '../../python/runtime-fixture.mjs';

const template = await readFile(new URL('./lesson.py.in', import.meta.url), 'utf8');
const composition = await readFile(new URL('./composition.py.in', import.meta.url), 'utf8');
const comparison = await readFile(new URL('./comparison.py.in', import.meta.url), 'utf8');
for (const key of Object.keys(families)) {
  test(`${key}: NumPy expressions agree with finite differences`, () => {
    runPython(`${functionCode(template, key, { coefficient: 1.7, exponent: 3 })}
with np.errstate(all="ignore"):
    for value in [x, y, derivative, tangent, f(point), df(point), F(point), F(x)]:
        assert np.asarray(value).dtype == np.float32
    for position in [-2.3, -0.7, 0.6, 1.8]:
        if not np.isfinite(f(position)):
            continue
        step = 0.005
        np.testing.assert_allclose(df(position), (f(position + step) - f(position - step)) / (2 * step), rtol=5e-4, atol=2e-4)
        np.testing.assert_allclose(f(position), (F(position + step) - F(position - step)) / (2 * step), rtol=5e-4, atol=2e-4)
`);
  });
}

test('known, reversed, coincident and invalid integrals preserve their mathematical meaning', () => {
  for (const [key, lower, point, expected] of [['power', 0, 3, 9], ['linear', 2, 0, -2], ['sin', 0, 2 * Math.PI, 0], ['reciprocal', -2, -1, -Math.log(2)], ['relu', -2, 2, 2], ['tanh', -3, 3, 0], ['sigmoid', -3, 3, 3], ['power', 1, 1, 0]]) {
    runPython(`${functionCode(template, key, { lower, point })}\nnp.testing.assert_allclose(total, ${expected}, atol=1e-5)`);
  }
  for (const [key, lower, point] of [['reciprocal', -1, 1], ['reciprocal', 0, 0], ['log', -1, 2]]) {
    runPython(`${functionCode(template, key, { lower, point })}\nassert np.isnan(total)`);
  }
});

test('poles, ReLU kink and stable saturated activations export correctly', () => {
  const integral = runPython(functionCode(template, 'linear', { lower: 2, point: 0 })).plots[0].series[0];
  assert.deepEqual(integral.area, { lower: 2, upper: 0, value: -2 });
  assert.equal(integral.points.find(([position]) => position === 1)[1], 1);
  const invalid = runPython(functionCode(template, 'reciprocal', { lower: -1, point: 1 })).plots[0].series[0];
  assert.equal(invalid.area.value, null);
  for (const key of ['reciprocal', 'relu']) {
    const response = runPython(functionCode(template, key));
    const series = response.plots[0].series[2].points;
    assert.equal(series.find(([position]) => position === 0)[1], null);
  }
  runPython(`${functionCode(template, 'sigmoid')}\nassert f(-1000) == 0\nassert f(1000) == 1\nassert df(0) == 0.25`);
  runPython(`${functionCode(template, 'softplus')}\nassert f(1000) == 1000\nnp.testing.assert_allclose(F(3) + F(-3), 4.5, atol=1e-5)`);
  runPython(`${functionCode(template, 'silu')}\nassert df(-2) < 0\nnp.testing.assert_allclose(F(3) + F(-3), 4.5, atol=1e-5)`);
});

for (const rule of Object.keys(rules)) {
  test(`${rule}: composition uses Python values and derivatives`, () => {
    for (const outer of ['sin', 'power', 'sigmoid']) {
      runPython(`${compositionCode(composition, rule, outer)}
assert x.dtype == np.float32
assert all(values.dtype == np.float32 for series in PLOTS.values() for values in series.values())
for position in [-1.7, 0.3, 1.2]:
    step = 0.005
    np.testing.assert_allclose(dh(position), (h(position + step) - h(position - step)) / (2 * step), rtol=5e-4, atol=2e-4)
`);
    }
  });
}

test('editing expressions changes generated curves and malformed sampling fails', () => {
  const code = functionCode(template, 'power');
  const response = runPython(code.replace('return coefficient * x ** exponent', 'return 3 * x + 2'));
  assert.equal(response.plots[0].series[0].points.find(([position]) => position === 0)[1], 2);
  assert.throws(() => runPython(code.replace('samples = 1001', 'samples = 5000')), /samples must/);
  assert.throws(() => runPython(code.replace('x_min, x_max = -10.0, 10.0', 'x_min, x_max = 1, -1')), /finite bounds/);
});

test('activation comparison reuses the same function, derivative and integral expressions', () => {
  const response = runPython(comparisonCode(comparison) + '\nassert all(values.dtype == np.float32 for series in PLOTS.values() for values in series.values())');
  assert.equal(response.plots.length, 3);
  for (const plot of response.plots) assert.equal(plot.series.length, 5);
  const sigmoid = response.plots[0].series.find(series => series.name === 'Sigmoid');
  assert.equal(sigmoid.points.find(([position]) => position === 0)[1], 0.5);
  for (const family of Object.values(families).filter(family => family.activation)) {
    const key = Object.keys(families).find(key => families[key] === family);
    const individual = runPython(functionCode(template, key) + '\nPLOTS = {"f": {"f": np.column_stack((x, y))}, "df": {"df": np.column_stack((x, derivative))}, "F": {"F": np.column_stack((x, F(x) - F(lower)))}}');
    for (const [index, plot] of response.plots.entries()) {
      assert.deepEqual(plot.series.find(series => series.name === family.name).points, individual.plots[index].series[0].points);
    }
  }
});