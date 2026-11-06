"""Identity fusion: what is each track? (drone, bird, aircraft or helicopter)

Every classifier report carries a likelihood vector L[c] = P(reported label | true class c),
taken from that sensor's confusion matrix. For a track we keep a running log-evidence

    E_c(t) = exp(-dt / tau) * E_c(t_prev) + w_sensor * log L_c

and read out the posterior P(c | evidence) = softmax(log prior_c + E_c + kinematics_c).

- tau (forgetting) and w_sensor (tempering) stop correlated reports from making the system
  overconfident. Naive Bayes over 100 frames of the same blurry drone is not 100 independent votes.
- kinematics_c compares the track's estimated speed with each class's typical speed. It is
  recomputed each time, never accumulated, so radar-only tracks still get a weak opinion.
- contrib[sensor] keeps each sensor's share of the evidence so an operator can see why the
  system thinks a track is a drone.
"""
from __future__ import annotations

import numpy as np

from .common import CLASSES, N_CLASSES


def _softmax(v):
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


class IdentityFusion:
    def __init__(self, cfg: dict):
        prior = np.array([float(cfg["prior"][c]) for c in CLASSES])
        self.log_prior = np.log(prior / prior.sum())
        self.tau = float(cfg["tau_s"])
        self.floor = float(cfg["floor"])
        self.cap = float(cfg["max_abs_log_evidence"])
        self.weights = {k: float(v) for k, v in cfg["sensor_weights"].items()}
        kin = cfg["kinematics"]
        self.kin_enabled = bool(kin["enabled"])
        self.kin_weight = float(kin["weight"])
        self.kin_min_hits = int(kin["min_hits"])
        self.kin_max_sigma = float(kin["max_vel_sigma_mps"])
        self.speed_mu = np.array([kin["speed_mps"][c][0] for c in CLASSES], dtype=float)
        self.speed_sd = np.array([kin["speed_mps"][c][1] for c in CLASSES], dtype=float)

    def _decay_factor(self, tr, t):
        return float(np.exp(-max(0.0, t - tr.evidence_t) / self.tau))

    def update(self, tr, det):
        f = self._decay_factor(tr, det.t)
        tr.log_evidence *= f
        for k in tr.contrib:
            tr.contrib[k] *= f
        tr.evidence_t = det.t
        ll = np.log(np.clip(det.likelihood, self.floor, 1.0))
        ll -= ll.max()
        w = self.weights.get(det.sensor, 1.0)
        tr.log_evidence = np.clip(tr.log_evidence + w * ll, -self.cap, 0.0)
        tr.contrib[det.sensor] = tr.contrib.get(det.sensor, np.zeros(N_CLASSES)) + w * ll

    def kinematics(self, tr) -> np.ndarray:
        """Log-likelihood of the track's speed under each class's speed model."""
        if not self.kin_enabled or tr.hits < self.kin_min_hits:
            return np.zeros(N_CLASSES)
        v, Pv = tr.x[3:], tr.P[3:, 3:]
        speed = float(np.linalg.norm(v))
        var = float(v @ Pv @ v) / speed ** 2 if speed > 1e-3 else float(np.trace(Pv)) / 3.0
        if np.sqrt(var) > self.kin_max_sigma:
            return np.zeros(N_CLASSES)
        s2 = self.speed_sd ** 2 + var
        ll = -0.5 * (speed - self.speed_mu) ** 2 / s2 - 0.5 * np.log(s2)
        return self.kin_weight * (ll - ll.max())

    def posterior(self, tr, t: float):
        """Class probabilities at time t, plus each source's evidence for the explanation panel."""
        f = self._decay_factor(tr, t)
        kin = self.kinematics(tr)
        probs = _softmax(self.log_prior + f * tr.log_evidence + kin)
        explanation = {k: f * v for k, v in tr.contrib.items()}
        if kin.any():
            explanation["kinematics"] = kin
        return probs, explanation
