"""Evaluation: how good is the fused air picture compared with the ground truth?

GOSPA (generalised optimal sub-pattern assignment, alpha = 2) is the standard multi-target
tracking metric. With cut-off c it splits the error into localisation error, missed targets
and false tracks:

    GOSPA = ( sum over matched pairs of d^p  +  c^p / 2 * (missed + false) ) ^ (1/p)

A truth and a track only count as matched if they are closer than c.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import CLASS_INDEX, CLASSES, N_CLASSES

MIN_SAMPLES = 50  # matched truth-track samples needed before a rate is reported


def gospa(X: np.ndarray, Y: np.ndarray, c: float = 100.0, p: float = 2.0) -> dict:
    """X: truth positions (n, 3); Y: track positions (m, 3)."""
    n, m = len(X), len(Y)
    pairs: list[tuple[int, int]] = []
    loc = 0.0
    if n and m:
        D = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(np.minimum(D, c) ** p)
        for i, j in zip(rows, cols):
            if D[i, j] < c:
                pairs.append((int(i), int(j)))
                loc += D[i, j] ** p
    missed, false = n - len(pairs), m - len(pairs)
    total = loc + (c ** p) / 2.0 * (missed + false)
    return dict(gospa=total ** (1.0 / p), loc_sum=loc, missed=missed, false=false, pairs=pairs)


class Evaluator:
    def __init__(self, scenario, cfg: dict):
        self.sc = scenario
        self.c = float(cfg["gospa_c_m"])
        self.p = float(cfg["gospa_p"])
        self.radius = float(cfg["volume_radius_m"])
        self.id_threshold = float(cfg["id_threshold"])
        self.gospa_values: list[float] = []
        self.loc_sum = 0.0
        self.missed = 0
        self.false = 0
        self.truth_steps = 0
        self.pairs = 0
        self.correct = 0
        self.bird_pairs = 0
        self.bird_as_drone = 0
        self.bird_alarm = 0
        self.fused_cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        self.tracks_seen: set[int] = set()
        self.tracks_matched: set[int] = set()
        self.entered: dict[int, float] = {}
        self.identified: dict[int, float] = {}
        self.step_ms: list[float] = []

    def in_volume(self, k: int) -> np.ndarray:
        pos = self.sc.states[:, k, :3]
        with np.errstate(invalid="ignore"):
            inside = np.hypot(pos[:, 0], pos[:, 1]) <= self.radius
        return np.flatnonzero(self.sc.alive(k) & inside)

    def step(self, k: int, t: float, views: list, step_ms: float = 0.0):
        self.step_ms.append(step_ms)
        ids = self.in_volume(k)
        X = self.sc.states[ids, k, :3]
        Y = np.array([v.pos for v in views], dtype=float).reshape(-1, 3)
        g = gospa(X, Y, self.c, self.p)
        self.gospa_values.append(g["gospa"])
        self.loc_sum += g["loc_sum"]
        self.missed += g["missed"]
        self.false += g["false"]
        self.truth_steps += len(ids)
        for tid in ids:
            self.entered.setdefault(int(tid), t)
        for v in views:
            self.tracks_seen.add(v.tid)
        for i, j in g["pairs"]:
            truth, v = int(ids[i]), views[j]
            cls = self.sc.specs[truth].cls
            self.pairs += 1
            self.tracks_matched.add(v.tid)
            self.fused_cm[CLASS_INDEX[cls], CLASS_INDEX[v.label]] += 1
            self.correct += int(v.label == cls)
            if cls == "bird":
                self.bird_pairs += 1
                self.bird_as_drone += int(v.label == "drone")
                self.bird_alarm += int(v.threat.level in ("HIGH", "MEDIUM"))
            if (cls == "drone" and truth not in self.identified and v.label == "drone"
                    and v.p_label >= self.id_threshold):
                self.identified[truth] = t

    def summary(self) -> dict:
        hours = (self.sc.times[-1] - self.sc.times[0]) / 3600.0
        drones = [i for i, s in enumerate(self.sc.specs) if s.cls == "drone" and i in self.entered]
        delays = [self.identified[i] - self.entered[i] for i in drones if i in self.identified]

        def ratio(a, b, min_count=1):
            # rates over fewer than min_count samples are noise, so report them as missing
            return float(a / b) if b >= min_count else float("nan")

        return {
            "gospa_m": float(np.mean(self.gospa_values)),
            "loc_rmse_m": float(np.sqrt(self.loc_sum / self.pairs)) if self.pairs else float("nan"),
            "missed_per_step": ratio(self.missed, len(self.gospa_values)),
            "false_per_step": ratio(self.false, len(self.gospa_values)),
            "completeness": ratio(self.pairs, self.truth_steps),
            "false_tracks_per_hour": float(len(self.tracks_seen - self.tracks_matched) / hours),
            "class_accuracy": ratio(self.correct, self.pairs, MIN_SAMPLES),
            "bird_as_drone": ratio(self.bird_as_drone, self.bird_pairs, MIN_SAMPLES),
            "bird_alarm_rate": ratio(self.bird_alarm, self.bird_pairs, MIN_SAMPLES),
            "drones_identified": ratio(len(delays), len(drones)),
            "drone_id_delay_s": float(np.median(delays)) if delays else float("nan"),
            "step_ms_mean": float(np.mean(self.step_ms)),
            "step_ms_p95": float(np.percentile(self.step_ms, 95)),
            "matched_pairs": int(self.pairs),
        }

    def fused_confusion(self) -> dict:
        return {"classes": list(CLASSES), "rows_true_cols_fused": self.fused_cm.tolist()}
