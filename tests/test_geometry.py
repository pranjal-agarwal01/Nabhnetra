import numpy as np

from nabhnetra.geometry import enu_to_rae, rae_jacobian, rae_to_enu, rae_to_enu_jacobian, wrap_angle


def test_round_trip():
    rng = np.random.default_rng(0)
    d = rng.uniform(-5000, 5000, (100, 3))
    assert np.allclose(rae_to_enu(enu_to_rae(d)), d)


def test_azimuth_is_clockwise_from_north():
    assert np.isclose(enu_to_rae([0, 100, 0])[1], 0.0)            # north
    assert np.isclose(enu_to_rae([100, 0, 0])[1], np.pi / 2)      # east
    assert np.isclose(enu_to_rae([0, 0, 100])[2], np.pi / 2)      # straight up


def _numeric_jacobian(f, x, eps=1e-4, wrap_row=None):
    x = np.asarray(x, float)
    cols = []
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = eps
        diff = f(x + dx) - f(x - dx)
        if wrap_row is not None:
            diff[wrap_row] = wrap_angle(diff[wrap_row])
        cols.append(diff / (2 * eps))
    return np.stack(cols, axis=1)


def test_rae_jacobian_matches_finite_differences():
    for d in ([1200.0, -800.0, 150.0], [-30.0, 2500.0, 600.0], [400.0, 400.0, -20.0]):
        num = _numeric_jacobian(enu_to_rae, d, wrap_row=1)
        assert np.allclose(rae_jacobian(d), num, atol=1e-7)


def test_inverse_jacobian_matches_finite_differences():
    for rae in ([1500.0, 0.3, 0.05], [800.0, -2.0, 0.4]):
        assert np.allclose(rae_to_enu_jacobian(rae), _numeric_jacobian(rae_to_enu, rae), atol=1e-4)
