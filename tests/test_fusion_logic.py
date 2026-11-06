import numpy as np

from nabhnetra.common import CLASSES, Detection
from nabhnetra.identity import IdentityFusion
from nabhnetra.metrics import gospa
from nabhnetra.threat import assess
from nabhnetra.tracker import Track


def _track():
    x = np.array([500.0, 500.0, 80.0, -10.0, -10.0, 0.0])
    return Track(tid=1, x=x, P=np.eye(6), t=0.0, born=0.0, last_update=0.0, init_sensor="radar",
                 hits=10)


def test_drone_reports_raise_drone_probability(cfg):
    fusion = IdentityFusion(cfg["identity"])
    tr = _track()
    p0, _ = fusion.posterior(tr, 0.0)
    lik = np.array([0.9, 0.05, 0.02, 0.03])
    for k in range(10):
        fusion.update(tr, Detection(float(k), "eo", "ae", np.zeros(2), np.eye(2), np.zeros(3), lik))
    p1, explanation = fusion.posterior(tr, 9.0)
    assert p1[CLASSES.index("drone")] > p0[CLASSES.index("drone")]
    assert p1.argmax() == CLASSES.index("drone")
    assert "eo" in explanation


def test_evidence_fades_without_new_reports(cfg):
    fusion = IdentityFusion(cfg["identity"])
    tr = _track()
    fusion.update(tr, Detection(0.0, "eo", "ae", np.zeros(2), np.eye(2), np.zeros(3),
                                np.array([0.9, 0.05, 0.02, 0.03])))
    soon, _ = fusion.posterior(tr, 1.0)
    later, _ = fusion.posterior(tr, 120.0)
    assert later[0] < soon[0]


def test_gospa_cases():
    X = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
    assert gospa(X, X)["gospa"] == 0.0
    one_missing = gospa(X, X[:1], c=100.0, p=2.0)
    assert one_missing["missed"] == 1 and one_missing["false"] == 0
    assert np.isclose(one_missing["gospa"], np.sqrt(100.0 ** 2 / 2))
    far = gospa(X[:1], X[:1] + 500.0, c=100.0, p=2.0)
    assert far["missed"] == 1 and far["false"] == 1 and far["pairs"] == []


def test_threat_ranks_close_inbound_drone_above_distant_bird(cfg):
    drone = np.array([0.9, 0.05, 0.02, 0.03])
    bird = np.array([0.05, 0.9, 0.02, 0.03])
    asset = np.zeros(3)
    near_drone = assess([300, 0, 50], [-15, 0, 0], drone, False, asset, cfg["threat"])
    loiter = assess([150, 0, 50], [0, 12, 0], drone, False, asset, cfg["threat"])
    far_bird = assess([2500, 0, 50], [-10, 0, 0], bird, False, asset, cfg["threat"])
    assert near_drone.level == "HIGH" and loiter.level == "HIGH"
    assert far_bird.level == "NONE"
    assert near_drone.score > far_bird.score
