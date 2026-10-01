import numpy as np

from scratch.arrays import unbroadcast
from scratch.gradcheck import check_gradient, numerical_gradient, relative_error
from scratch.precision import round_to_bfloat16


def test_numerical_gradient_matches_known_derivatives_without_mutating_input():
    x = np.array([[0.3, -1.2], [2.0, 0.5]], dtype=np.float32)
    before = x.copy()
    gradient = numerical_gradient(lambda z: float(np.sum(np.sin(z) * z)), x)
    exact = np.cos(x.astype(np.float64)) * x + np.sin(x.astype(np.float64))
    np.testing.assert_allclose(gradient, exact, rtol=1e-8, atol=1e-10)
    np.testing.assert_array_equal(x, before)
    assert gradient.dtype == np.float64


def test_check_gradient_accepts_correct_and_rejects_wrong_gradients():
    x = np.array([0.5, -1.5, 2.0])
    assert check_gradient(lambda z: float(np.sum(z ** 3)), x, 3 * x ** 2) < 1e-8
    try:
        check_gradient(lambda z: float(np.sum(z ** 3)), x, 2 * x ** 2)
    except AssertionError as error:
        assert "relative error" in str(error)
    else:
        raise AssertionError("a wrong gradient passed the check")
    assert relative_error(0.0, 0.0) == 0.0


def test_unbroadcast_sums_over_stretched_axes():
    rng = np.random.default_rng(0)
    for shape, target in [((4, 3), (3,)), ((4, 3), (1, 3)), ((2, 4, 3), (4, 1)), ((5,), ()), ((2, 3), (2, 3))]:
        gradient = rng.standard_normal(shape)
        reduced = unbroadcast(gradient, target)
        assert reduced.shape == target
        # The adjoint identity: <broadcast(v), g> == <v, unbroadcast(g)> for every v.
        v = rng.standard_normal(target)
        np.testing.assert_allclose(np.sum(np.broadcast_to(v, shape) * gradient), np.sum(v * reduced))


def test_bfloat16_rounding_keeps_eight_significant_bits():
    x = np.array([1.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, 1.0 + 2 ** -7, 3.14159265, -2.5e-3, np.inf, np.nan], dtype=np.float32)
    y = round_to_bfloat16(x)
    assert y.dtype == np.float32
    np.testing.assert_array_equal(y[:4], [1.0, 1.0, 1.0 + 2 ** -6, 1.0 + 2 ** -7])  # ties round to even
    assert y[4] == np.float32(3.140625)
    assert abs(y[5] - x[5]) <= abs(x[5]) * 2 ** -8
    assert np.isinf(y[6]) and np.isnan(y[7])
    bits = y[:6].view(np.uint32)
    assert np.all(bits & 0xFFFF == 0)
