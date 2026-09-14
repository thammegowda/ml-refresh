import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { runPython } from '../../python/runtime-fixture.mjs';

const source = await readFile(new URL('./lesson.py', import.meta.url), 'utf8');
test('unit circle, projection, and vector products agree at key angles', () => {
  for (const angle of [0, 45, 90, 180, 270, -90, 360]) {
    runPython(source.replace('angle = 45.0', `angle = ${angle}`) + `
cosine, sine = OUTPUTS["Unit point"]
dot = OUTPUTS["Dot product"]
cross_z = OUTPUTS["Cross z"]
projection = OUTPUTS["Projection"]
u, v = INPUTS["u"], INPUTS["v"]
np.testing.assert_allclose(cosine ** 2 + sine ** 2, 1, atol=1e-6)
np.testing.assert_allclose(dot, length_u * length_v * cosine, atol=1e-6)
np.testing.assert_allclose(cross_z, length_u * length_v * sine, atol=1e-6)
np.testing.assert_allclose(projection, [v[0], 0], atol=1e-6)
assert u.dtype == v.dtype == degrees.dtype == np.float32
if abs(${angle}) % 180 == 90:
    assert OUTPUTS["Tangent defined"] == 0
assert np.isnan(tan_curve[np.where(degrees == 90)[0][0]])
`);
  }
  runPython(source.replace('length_u = 1.5', 'length_u = 0.0') + '\nassert OUTPUTS["Dot product"] == OUTPUTS["Cross z"] == 0');
});