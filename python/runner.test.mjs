import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { runPython, createPythonSession } from './runtime-fixture.mjs';

const code = await readFile(new URL('../chapters/linear-algebra/lesson.py', import.meta.url), 'utf8');
const run = (source = code, emit) => runPython(source, emit);
const variable = (group, name) => group.find((entry) => entry.name === name);

test('ten SGD updates reduce loss and display final float32 predictions', () => {
  const response = run();
  assert.deepEqual(response.inputs.map((entry) => entry.name), ['X', 'Y']);
  assert.deepEqual(response.outputs.map((entry) => entry.name), ['Y_before', 'Y_pred', 'W', 'b']);
  for (const entry of [...response.inputs, ...response.outputs]) assert.equal(entry.dtype, 'float32');
  assert.deepEqual(variable(response.inputs, 'Y').values, [[1, -0.5], [0.5, 2]]);
  for (const [name, shape] of [['Y_before', [2, 2]], ['Y_pred', [2, 2]], ['W', [3, 2]], ['b', [1, 2]]]) {
    assert.deepEqual(variable(response.outputs, name).shape, shape);
  }
  assert.equal(response.stdout.trim().split('\n').length, 11);
  assert.match(response.stdout, /Update 10: MSE/);
  run(`${code}
assert step == 9
assert len(loss_history) == 11
assert loss_history[0] == loss_before
assert loss_history[-1] == loss
assert loss < loss_before * 0.1
np.testing.assert_allclose(Y_pred, X @ W + b)
np.testing.assert_allclose(loss, np.sum((Y_pred - Y) ** 2) / Y.size)
`);
});

test('SGD sampled-example gradients agree with finite differences', () => {
  run(`${code.replace('learning_rate = 0.1', 'learning_rate = 0.0')}
batch_X = batch_X.astype(np.float64)
batch_Y = batch_Y.astype(np.float64)
W = W.astype(np.float64)
b = b.astype(np.float64)
epsilon = 1e-5
for parameter, gradient in [(W, dW), (b, db)]:
    numerical = np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        original = parameter[index]
        parameter[index] = original + epsilon
        upper = np.mean((batch_X @ W + b - batch_Y) ** 2)
        parameter[index] = original - epsilon
        lower = np.mean((batch_X @ W + b - batch_Y) ** 2)
        parameter[index] = original
        numerical[index] = (upper - lower) / (2 * epsilon)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=2e-7)
`);
});

test('zero learning rate preserves predictions and matching targets need no update', () => {
  const unchanged = run(code.replace('learning_rate = 0.1', 'learning_rate = 0.0'));
  assert.deepEqual(variable(unchanged.outputs, 'Y_pred').values, variable(unchanged.outputs, 'Y_before').values);
  run(`${code.replace('learning_rate = 0.1', 'learning_rate = 0.0')}\nassert all(value == loss_before for value in loss_history)`);
  run(`${code.replace('loss_before =', 'Y = Y_before.copy()\nloss_before =')}\nassert all(value == 0 for value in loss_history)`);
});

test('SGD uses the global seed and data-derived parameter dimensions', () => {
  const first = run();
  const second = run();
  const unseeded = code.replace('np.random.seed(42)', 'np.random.seed(None)');
  const fresh = run(unseeded);
  const next = run(unseeded);
  for (const name of ['W', 'b']) {
    assert.deepEqual(variable(first.outputs, name).values, variable(second.outputs, name).values);
    assert.notDeepEqual(variable(fresh.outputs, name).values, variable(next.outputs, name).values);
  }
  const resized = run(code.replace('bound =', 'X = np.ones((4, 5), dtype=np.float32)\nY = np.zeros((4, 3), dtype=np.float32)\nbound ='));
  for (const entry of [...resized.inputs, ...resized.outputs]) assert.equal(entry.dtype, 'float32');
  assert.deepEqual(variable(resized.outputs, 'W').shape, [5, 3]);
  assert.deepEqual(variable(resized.outputs, 'b').shape, [1, 3]);
  assert.deepEqual(variable(resized.outputs, 'Y_pred').shape, [4, 3]);
});

test('scripts freely change labels, dimensions, broadcasting, and equations', () => {
  const response = run('import numpy as np\nfeatures = np.arange(12.).reshape(4, 3)\nweights = np.ones((3, 5))\noffset = np.arange(5.)\noutput = features @ weights + offset\nINPUTS = {"Features": features, "Offset": offset}\nOUTPUTS = {"Transposed result": output.T}');
  assert.deepEqual(variable(response.inputs, 'Features').shape, [4, 3]);
  assert.deepEqual(variable(response.inputs, 'Offset').shape, [5]);
  assert.deepEqual(response.outputs[0].shape, [5, 4]);
  assert.deepEqual(response.outputs[0].values[0], [3, 12, 21, 30]);
  assert.equal(response.inputs.length, 2);
});

test('scalars, vectors, empty arrays, and larger values reflect Python', () => {
  const response = run('import numpy as np\nX = np.array([50., -10000., 100000.])\nINPUTS = {"X": X}\nOUTPUTS = {"Squared": X ** 2, "Empty": np.empty((0, 3)), "Count": 3, "Empty vector": np.array([])}');
  assert.deepEqual(response.inputs[0].values, [[50, -10000, 100000]]);
  assert.deepEqual(response.outputs[0].values, [[2500, 100000000, 10000000000]]);
  assert.deepEqual(response.outputs[1].shape, [0, 3]);
  assert.deepEqual(response.outputs[2].values, [[3]]);
  assert.deepEqual(response.outputs[3].shape, [0]);
});

test('previews are bounded without changing computed data', () => {
  const response = run('import numpy as np\nbig = np.arange(600.).reshape(20, 30)\nINPUTS = {"big": big}\nOUTPUTS = {"total": big.sum()}');
  assert.deepEqual(response.inputs[0].shape, [20, 30]);
  assert.equal(response.inputs[0].values.length, 12);
  assert.equal(response.inputs[0].values[0].length, 12);
  assert.equal(response.inputs[0].truncated, true);
  assert.deepEqual(response.outputs[0].values, [[179700]]);
});

test('export dictionaries are explicit and validated', () => {
  assert.throws(() => run('X = 1'), /Define INPUTS/);
  assert.throws(() => run('INPUTS = {}'), /Define OUTPUTS/);
  assert.throws(() => run('INPUTS = []\nOUTPUTS = {}'), /dictionary/);
  assert.throws(() => run('INPUTS = {1: 2}\nOUTPUTS = {}'), /labels/);
  assert.throws(() => run('INPUTS = {"": 2}\nOUTPUTS = {}'), /labels/);
  assert.throws(() => run('INPUTS = {str(index): index for index in range(33)}\nOUTPUTS = {}'), /at most 32/);
  assert.throws(() => run('INPUTS = {}\nOUTPUTS = {"text": "abc"}'), /real numeric/);
  assert.throws(() => run('import numpy as np\nINPUTS = {}\nOUTPUTS = {"cube": np.ones((2, 2, 2))}'), /scalar, vector, or matrix/);
  assert.deepEqual(run('INPUTS = {}\nOUTPUTS = {}').outputs, []);
});

test('plot exports preserve sampled curves and turn undefined ordinates into gaps', () => {
  const response = run('import numpy as np\nINPUTS = {}\nOUTPUTS = {}\nPLOTS = {"Curve": {"f(x)": np.array([[-1, -1], [0, np.nan], [1, np.inf]])}}');
  assert.deepEqual(response.plots, [{ title: 'Curve', series: [{ name: 'f(x)', points: [[-1, -1], [0, null], [1, null]] }] }]);
  assert.deepEqual(run('INPUTS = {}\nOUTPUTS = {}').plots, []);
  const large = run('import numpy as np\nINPUTS = {}\nOUTPUTS = {}\nx = np.linspace(-5, 5, 1001)\nPLOTS = {"Curve": {"f(x)": np.column_stack((x, x ** 2))}}');
  assert.equal(large.plots[0].series[0].points.length, 1001);
});

test('plot contracts reject malformed and oversized data', () => {
  const prefix = 'import numpy as np\nINPUTS = {}\nOUTPUTS = {}\n';
  for (const expression of ['[]', '{"": {}}', '{"Curve": {}}', '{"Curve": {"": [[0, 0], [1, 1]]}}', '{"Curve": {"f": [1, 2]}}', '{"Curve": {"f": np.zeros((4098, 2))}}', '{"Curve": {"f": [[np.nan, 0], [1, 1]]}}', '{"Curve": {"f": [[1j, 0], [1, 1]]}}']) {
    assert.throws(() => run(`${prefix}PLOTS = ${expression}`), /PLOTS|Plot/);
  }
});

test('shaded plot series validate numeric bounds and preserve signed totals', () => {
  const prefix = 'import numpy as np\nINPUTS = {}\nOUTPUTS = {}\n';
  const response = run(prefix + 'PLOTS = {"Integral": {"f": {"points": [[0, 0], [1, 1]], "area": {"lower": 1, "upper": 0, "value": -0.5}}}}');
  assert.deepEqual(response.plots[0].series[0].area, { lower: 1, upper: 0, value: -0.5 });
  for (const area of ['{}', '{"lower": 0, "upper": np.inf, "value": 1}', '{"lower": "0", "upper": 1, "value": 1}', '{"lower": 0, "upper": 1, "value": 1j}']) {
    assert.throws(() => run(prefix + `PLOTS = {"Integral": {"f": {"points": [[0, 0], [1, 1]], "area": ${area}}}}`), /Plot area/);
  }
});

test('invalid syntax, incompatible equations, and nonfinite values fail explicitly', () => {
  assert.throws(() => run('def broken(:'), /SyntaxError/);
  assert.throws(() => run('import numpy as np\nY = np.ones((2, 3)) @ np.ones((4, 2))'), /matmul/);
  assert.throws(() => run('INPUTS = {}\nOUTPUTS = {"Y": float("nan")}'), /finite/);
  assert.throws(() => run('INPUTS = {}\nOUTPUTS = {"Y": 1j}'), /real numeric/);
});

test('fresh namespaces and explicit exports hide intermediate values', () => {
  run('secret = 123\nINPUTS = {}\nOUTPUTS = {}');
  const response = run('print("secret" in globals())\nintermediate = 42\nINPUTS = {}\nOUTPUTS = {"Answer": intermediate}');
  assert.equal(response.stdout, 'False\n');
  assert.deepEqual(response.outputs.map((entry) => entry.name), ['Answer']);
});

test('live sessions retain initialization, omit static plots, and invalidate stale or failed updates', () => {
  const session = createPythonSession();
  const code = `
print("initialize")
PARAMETERS = {"value": 1}
history = []
def update(value):
    history.append(value)
    if value < 0:
        raise ValueError("negative value")
    return {"INPUTS": {}, "OUTPUTS": {"value": value, "calls": len(history)}}
globals().update(update(**PARAMETERS))
PLOTS = {"Static": {"line": [[0, 0], [1, 1]]}}
`;
  try {
    const initial = session.run({ code, revision: 1 });
    assert.equal(initial.stdout, 'initialize\n');
    assert.equal(initial.plots.length, 1);
    const updated = session.run({ parameters: { value: 2 }, revision: 1 });
    assert.equal(updated.stdout, '');
    assert.equal(updated.plots, undefined);
    assert.deepEqual(updated.outputs.map(entry => entry.values), [[[2]], [[2]]]);
    assert.throws(() => session.run({ parameters: { value: 3 }, revision: 0 }), /session expired/);
    session.run({ code, revision: 2 });
    assert.throws(() => session.run({ parameters: { value: -1 }, revision: 2 }), /negative value/);
    assert.throws(() => session.run({ parameters: { value: 4 }, revision: 2 }), /session expired/);
    const reset = session.run({ code, revision: 3 });
    assert.deepEqual(variable(reset.outputs, 'calls').values, [[1]]);
    assert.throws(() => session.run({ parameters: { unknown: 1 }, revision: 3 }), /declared PARAMETERS/);
  } finally { session.destroy(); }
});

test('plot deltas preserve static samples, detect in-place edits, and replace changed structures', () => {
  const session = createPythonSession();
  const code = `
import numpy as np
PARAMETERS = {"value": 1}
fixed = np.array([[0., 0.], [1., 1.]])
moving = fixed.copy()
def update(value):
    moving[1, 1] = value
    plots = {"Curve": {
        "fixed": fixed,
        "moving": moving,
        "area": {"points": fixed, "area": {"lower": 0, "upper": value, "value": value}},
    }}
    if value == 3:
        plots = {"Replacement": {"line": fixed}}
    return {"INPUTS": {}, "OUTPUTS": {}, "PLOTS": plots}
globals().update(update(**PARAMETERS))
`;
  try {
    session.run({ code, revision: 1 });
    const changed = session.run({ parameters: { value: 2 }, revision: 1 });
    assert.equal(changed.plots, undefined);
    assert.deepEqual(changed.plotUpdates, [{ title: 'Curve', series: [
      { name: 'moving', area: null, points: [[0, 0], [1, 2]] },
      { name: 'area', area: { lower: 0, upper: 2, value: 2 } },
    ] }]);
    const unchanged = session.run({ parameters: { value: 2 }, revision: 1 });
    assert.equal(unchanged.plots, undefined);
    assert.equal(unchanged.plotUpdates, undefined);
    const replaced = session.run({ parameters: { value: 3 }, revision: 1 });
    assert.equal(replaced.plots[0].title, 'Replacement');
    assert.equal(replaced.plotUpdates, undefined);
  } finally { session.destroy(); }
});

test('stdout and stderr stream in order, survive exceptions, and are bounded', () => {
  const events = [];
  const emit = (stream, text) => events.push({ stream, text });
  const script = "import sys\nsys.stdout.write('first')\nsys.stderr.write('warning')\nsys.stdout.write('last')\nINPUTS = {}\nOUTPUTS = {}";
  const response = run(script, emit);
  assert.equal(response.stdout, 'firstlast');
  assert.equal(response.stderr, 'warning');
  assert.deepEqual(events, [{ stream: 'stdout', text: 'first' }, { stream: 'stderr', text: 'warning' }, { stream: 'stdout', text: 'last' }]);
  events.length = 0;
  assert.throws(() => run(`${script}\nraise ValueError('broken')`, emit), /broken/);
  assert.equal(events.map((entry) => entry.text).join(''), 'firstwarninglast');
  events.length = 0;
  const large = run("import sys\nsys.stderr.write('a' * 12000)\nINPUTS = {}\nOUTPUTS = {}", emit);
  assert.equal(large.stderr.length, 8000);
  assert.match(events[1].text, /truncated/);
});