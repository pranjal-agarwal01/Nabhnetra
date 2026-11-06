import numpy as np
import pytest

from nabhnetra import SCENARIOS, build_scenario, run
from nabhnetra.confusion import ConfusionModel


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenarios_build(name):
    sc = build_scenario(name, seed=0, dt=0.5, duration_s=60)
    assert sc.states.shape == (len(sc.specs), len(sc.times), 6)
    assert any(s.cls == "drone" for s in sc.specs)
    for i in range(len(sc.specs)):
        alive = (~np.isnan(sc.states[i, :, 0])).astype(int)
        changes = np.diff(np.concatenate([[0], alive, [0]]))
        assert (changes == 1).sum() <= 1  # each target appears once and never comes back


def test_full_pipeline_runs_and_finds_drones(cfg):
    sc = build_scenario("mixed", seed=2, dt=0.5, duration_s=150)
    res = run(sc, cfg, seed=2, keep_frames=True)
    assert len(res.frames) == len(sc.times)
    s = res.summary
    assert 0.0 <= s["completeness"] <= 1.0
    assert s["matched_pairs"] > 0
    assert s["drones_identified"] > 0


def test_confusion_files_are_valid():
    for name in ("eo", "ir", "acoustic", "radar_microdoppler"):
        cm = ConfusionModel.load(f"configs/confusion/{name}.json")
        for m in cm.matrices.values():
            assert np.allclose(m.sum(axis=1), 1.0)
