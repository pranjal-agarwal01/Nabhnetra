"""Run a scenario through the sensors, tracker, identity fusion and threat ranking."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .common import CLASSES
from .identity import IdentityFusion
from .metrics import Evaluator
from .scenario import Scenario
from .sensors import build_sensors
from .threat import Threat, assess
from .tracker import Track, Tracker

RECENT_S = 3.0  # a sensor "supports" a track if it updated it within this many seconds


@dataclass
class TrackView:
    """What the operator display (and the evaluator) sees for one confirmed track."""

    tid: int
    pos: np.ndarray
    vel: np.ndarray
    pos_sigma: float
    probs: np.ndarray
    label: str
    p_label: float
    threat: Threat
    sensors: tuple[str, ...]
    cooperative_id: str | None
    explanation: dict = field(repr=False, default_factory=dict)


@dataclass
class Frame:
    t: float
    truth_ids: np.ndarray
    tracks: list[TrackView]
    failed: tuple[str, ...]


@dataclass
class RunResult:
    scenario: Scenario
    sensors: list[str]
    frames: list[Frame]
    summary: dict
    fused_confusion: dict
    confusion_sources: dict


def _view(tr: Track, t: float, identity: IdentityFusion, asset, threat_cfg) -> TrackView:
    probs, explanation = identity.posterior(tr, t)
    best = int(np.argmax(probs))
    threat = assess(tr.x[:3], tr.x[3:], probs, tr.cooperative_id is not None, asset, threat_cfg)
    recent = tuple(sorted(s for s, ts in tr.sensor_last.items() if t - ts <= RECENT_S))
    return TrackView(tr.tid, tr.x[:3].copy(), tr.x[3:].copy(), tr.pos_sigma, probs, CLASSES[best],
                     float(probs[best]), threat, recent, tr.cooperative_id, explanation)


def run(scenario: Scenario, cfg: dict, sensor_names=None, seed: int = 0,
        keep_frames: bool = True) -> RunResult:
    rng = np.random.default_rng((seed, 7919))
    sensors = build_sensors(cfg, sensor_names, scenario.dt)
    identity = IdentityFusion(cfg["identity"])
    mode = cfg["tracker"].get("bearing_init", "auto")
    if mode == "auto":
        bearing_init = not any(s.starts_noncooperative_tracks for s in sensors)
    else:
        bearing_init = bool(mode)
    tracker = Tracker(cfg["tracker"], identity, bearing_init)
    evaluator = Evaluator(scenario, cfg["evaluation"])
    asset = np.asarray(cfg["site"]["asset_enu"], dtype=float)

    frames: list[Frame] = []
    for k, t in enumerate(scenario.times):
        t = float(t)
        start = time.perf_counter()
        failed = tuple(s.name for s in sensors if scenario.sensor_failed(s.name, t))
        for s in sensors:
            if s.due(k) and s.name not in failed:
                tracker.process(s.observe(k, t, scenario, rng), t)
        tracker.maintain(t)
        views = [_view(tr, t, identity, asset, cfg["threat"]) for tr in tracker.confirmed()]
        evaluator.step(k, t, views, (time.perf_counter() - start) * 1000.0)
        if keep_frames:
            frames.append(Frame(t, evaluator.in_volume(k), views, failed))

    sources = {s.name: (s.cm.source if s.cm is not None else "") for s in sensors}
    return RunResult(scenario, [s.name for s in sensors], frames, evaluator.summary(),
                     evaluator.fused_confusion(), sources)
