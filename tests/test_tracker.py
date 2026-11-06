import numpy as np

from nabhnetra.common import Detection
from nabhnetra.geometry import enu_to_rae, wrap_angle
from nabhnetra.identity import IdentityFusion
from nabhnetra.tracker import Tracker


def _straight_line(n=60, dt=1.0):
    p0, v = np.array([2000.0, 1500.0, 120.0]), np.array([-12.0, -9.0, 0.0])
    return [p0 + v * k * dt for k in range(n)], v


def _run(cfg, use_camera, seed=0):
    rng = np.random.default_rng(seed)
    tr = Tracker(cfg["tracker"], IdentityFusion(cfg["identity"]), bearing_init=False)
    origin = np.array([0.0, 0.0, 10.0])
    r_sig = np.array([5.0, np.deg2rad(0.5), np.deg2rad(0.8)])
    c_sig = 0.3e-3
    truth, v = _straight_line()
    errors = []
    for k, p in enumerate(truth):
        t = float(k)
        rae = enu_to_rae(p - origin)
        z = rae + rng.normal(0, r_sig)
        z[1] = wrap_angle(z[1])
        tr.process([Detection(t, "radar", "rae", z, np.diag(r_sig ** 2), origin)], t)
        if use_camera:
            zc = rae[1:] + rng.normal(0, c_sig, 2)
            tr.process([Detection(t, "eo", "ae", zc, np.eye(2) * c_sig ** 2, origin)], t)
        tr.maintain(t)
        conf = tr.confirmed()
        if k >= 20 and conf:
            errors.append(np.linalg.norm(conf[0].x[:3] - p))
    return tr, errors, v


def test_radar_track_is_confirmed_and_accurate(cfg):
    tr, errors, v = _run(cfg, use_camera=False)
    assert len(tr.confirmed()) == 1
    assert np.sqrt(np.mean(np.square(errors))) < 40.0
    assert np.linalg.norm(tr.confirmed()[0].x[3:] - v) < 5.0


def test_camera_fusion_reduces_error(cfg):
    _, radar_only, _ = _run(cfg, use_camera=False, seed=3)
    _, fused, _ = _run(cfg, use_camera=True, seed=3)
    assert np.sqrt(np.mean(np.square(fused))) < 0.7 * np.sqrt(np.mean(np.square(radar_only)))


def test_direction_only_reports_cannot_start_tracks_when_radar_is_present(cfg):
    tr = Tracker(cfg["tracker"], IdentityFusion(cfg["identity"]), bearing_init=False)
    d = Detection(0.0, "eo", "ae", np.array([0.3, 0.05]), np.eye(2) * 1e-7, np.zeros(3))
    tr.process([d], 0.0)
    assert tr.tracks == []
