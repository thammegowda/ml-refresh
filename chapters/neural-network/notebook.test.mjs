import test from 'node:test';
import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

test('neural network notebook checks derivatives, learns, and memorizes noisy labels', async () => {
  const source = await readFile(new URL('./neural-network.ipynb', import.meta.url), 'utf8');
  const python = fileURLToPath(new URL('../../.venv/bin/python', import.meta.url));
  execFileSync(python, ['-c', `
import sys
import nbformat
from nbclient import NotebookClient
notebook = nbformat.reads(sys.stdin.read(), as_version=4)
nbformat.validate(notebook)
assert all(cell.metadata.get('language') in ('python', 'markdown') for cell in notebook.cells)
assert all(cell.metadata.get('id') == cell.id for cell in notebook.cells)
assert all(not cell.get('outputs') for cell in notebook.cells)
notebook.cells.append(nbformat.v4.new_code_cell('''
assert clean_history['train_loss'][-1] < 0.1 * clean_history['train_loss'][0]
assert clean_history['validation_accuracy'][-1] > 0.85
assert memorized_history['train_accuracy'][-1] >= 0.95
assert memorized_history['train_accuracy'][-1] - memorized_history['validation_accuracy'][-1] > 0.5
assert memorized_history['validation_loss'][-1] > memorized_history['validation_loss'][0]
assert not np.shares_memory(clean_state['m']['W1'], memorized_state['m']['W1'])
assert sum(value.size for value in clean_params.values()) == 104
assert X_train.shape == (320, 5) and X_validation.shape == (160, 5)
np.testing.assert_allclose(X_train.mean(axis=0), 0, atol=1e-6)
assert all(value.dtype == np.float32 for value in clean_params.values())
probabilities, baseline = softmax_cross_entropy(np.zeros((4, 4)), classes)
np.testing.assert_allclose(baseline, np.log(4))
probabilities, extreme_loss = softmax_cross_entropy(np.array([[1000, -1000, 0, 1]], dtype=np.float32), np.array([1]))
assert np.isfinite(extreme_loss) and extreme_loss > 1000
np.testing.assert_allclose(probabilities.sum(axis=1), 1)
demo, _ = softmax_cross_entropy(np.array([[0.8, -0.3, 1.2, 0.1]]), np.array([2]))
probability = demo[0]
target = np.eye(4)[2]
jacobian = np.diag(probability) - np.outer(probability, probability)
np.testing.assert_allclose(jacobian.T @ (-target / probability), probability - target, atol=1e-12)
epsilon = 1e-5
check_params = initialize(seed=31, dtype=np.float64)
check_inputs = X_train[[0, 80, 160, 240]].astype(np.float64).copy()
check_labels = y_train[[0, 80, 160, 240]]
_, check_cache = forward(check_inputs, check_labels, check_params)
analytic, analytic_inputs = backward(check_cache, check_labels, check_params)
reference_mask = check_cache['Z1'] > 0
for name, values in {**check_params, 'X': check_inputs}.items():
  numerical = np.zeros_like(values)
  for index in np.ndindex(values.shape):
    original = values[index]
    try:
      values[index] = original + epsilon
      loss_plus, cache_plus = forward(check_inputs, check_labels, check_params)
      values[index] = original - epsilon
      loss_minus, cache_minus = forward(check_inputs, check_labels, check_params)
    finally:
      values[index] = original
    assert np.array_equal(reference_mask, cache_plus['Z1'] > 0)
    assert np.array_equal(reference_mask, cache_minus['Z1'] > 0)
    numerical[index] = (loss_plus - loss_minus) / (2 * epsilon)
  expected = analytic_inputs if name == 'X' else analytic[name]
  assert expected.shape == values.shape
  np.testing.assert_allclose(expected, numerical, atol=1e-8, rtol=1e-5)
probe = {'weight': np.array([1.0, -2.0])}
probe_gradients = {'weight': np.array([0.5, -0.25])}
probe_state = adam_state(probe)
probe_before = probe['weight'].copy()
adam_step(probe, probe_gradients, probe_state)
np.testing.assert_allclose(probe['weight'], probe_before - 0.01 * probe_gradients['weight'] / (np.abs(probe_gradients['weight']) + 1e-8))
adam_step(probe, probe_gradients, probe_state)
assert probe_state['t'] == 2
np.testing.assert_allclose(probe_state['m']['weight'], (1 - 0.9**2) * probe_gradients['weight'])
np.testing.assert_allclose(probe_state['v']['weight'], (1 - 0.999**2) * probe_gradients['weight']**2)
assert adam_state(probe)['t'] == 0
assert np.all(adam_state(probe)['m']['weight'] == 0)
assert np.isin(y_validation, classes).all()
for params in [clean_params, memorized_params]:
  _, cache = forward(X_validation, y_validation, params)
  np.testing.assert_allclose(cache['P'].sum(axis=1), 1, atol=1e-6)
  assert np.isfinite(cache['P']).all()
short_params, short_history, short_state = train(X_train, y_train, X_validation, y_validation, epochs=2, batch_size=30)
assert short_state['t'] == 22
repeat_params, repeat_history, repeat_state = train(X_train, y_train, X_validation, y_validation, epochs=2, batch_size=30)
for name in short_params:
    np.testing.assert_array_equal(short_params[name], repeat_params[name])
single_loss, single_cache = forward(X_train[:1], y_train[:1], clean_params)
single_gradients, _ = backward(single_cache, y_train[:1], clean_params)
repeated_inputs = np.repeat(X_train[:1], 3, axis=0)
repeated_labels = np.repeat(y_train[:1], 3)
_, repeated_cache = forward(repeated_inputs, repeated_labels, clean_params)
repeated_gradients, _ = backward(repeated_cache, repeated_labels, clean_params)
for name in single_gradients:
    np.testing.assert_allclose(single_gradients[name], repeated_gradients[name], atol=1e-7, rtol=1e-5)
print('Network regression checks passed.')
'''))
NotebookClient(notebook, timeout=120, kernel_name='python3').execute()
assert 'Network regression checks passed.' in notebook.cells[-1].outputs[0].text
figures = [output for cell in notebook.cells for output in cell.get('outputs', []) if 'application/vnd.plotly.v1+json' in output.get('data', {})]
assert len(figures) == 3
`], { input: source, env: { ...process.env, PATH: `${fileURLToPath(new URL('../../.venv/bin', import.meta.url))}:${process.env.PATH}` }, timeout: 180000 });
});