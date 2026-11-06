"""Multi-sensor, multi-target tracker: one extended Kalman filter per track, one shared track list.

State per track: x = [east, north, up, v_east, v_north, v_up] in a local frame (metres, m/s).
Motion model: nearly constant velocity (white-noise acceleration with spectral density q).
Each sensor scan is processed in turn ("sequential fusion"):
  1. predict every track to the scan time
  2. gate each detection against each track using the Mahalanobis distance
  3. assign detections to tracks with the Hungarian algorithm (global nearest neighbour)
  4. EKF-update assigned tracks; start tentative tracks from unassigned detections that carry range
Radar contributes range, cameras contribute very precise direction. Neither alone gives a good
3-D track of a small drone; together they do.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from .common import N_CLASSES, Detection
from .geometry import enu_to_rae, rae_jacobian, rae_to_enu, rae_to_enu_jacobian, wrap_angle


def cv_model(dt: float, q: float):
    """Transition matrix F and process noise Q for the constant-velocity model."""
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    Q = np.zeros((6, 6))
    for i in range(3):
        Q[i, i] = q * dt ** 3 / 3.0
        Q[i, i + 3] = Q[i + 3, i] = q * dt ** 2 / 2.0
        Q[i + 3, i + 3] = q * dt
    return F, Q


def innovation(x: np.ndarray, det: Detection):
    """Measurement residual y = z - h(x) and Jacobian H for any detection kind."""
    H = np.zeros((len(det.z), 6))
    if det.kind == "pos":
        H[:, :3] = np.eye(3)
        return det.z - x[:3], H
    d = x[:3] - det.origin
    J = rae_jacobian(d)
    h = enu_to_rae(d)
    if det.kind == "rae":
        H[:, :3] = J
        y = det.z - h
        y[1] = wrap_angle(y[1])
        return y, H
    if det.kind == "ae":
        H[:, :3] = J[1:]
        y = det.z - h[1:]
        y[0] = wrap_angle(y[0])
        return y, H
    raise ValueError(det.kind)


@dataclass
class Track:
    tid: int
    x: np.ndarray
    P: np.ndarray
    t: float
    born: float
    last_update: float
    init_sensor: str
    hits: int = 1
    confirmed: bool = False
    cooperative_id: str | None = None
    sensor_last: dict = field(default_factory=dict)
    # identity evidence, maintained by IdentityFusion
    log_evidence: np.ndarray = field(default_factory=lambda: np.zeros(N_CLASSES))
    contrib: dict = field(default_factory=dict)
    evidence_t: float = 0.0

    @property
    def pos_sigma(self) -> float:
        return float(np.sqrt(np.trace(self.P[:3, :3])))


class Tracker:
    def __init__(self, cfg: dict, identity, bearing_init: bool):
        self.q = float(cfg["q"])
        self.gates = {m: chi2.ppf(float(cfg["gate_prob"]), m) for m in (2, 3)}
        self.confirm_hits = int(cfg["confirm_hits"])
        self.tentative_timeout = float(cfg["tentative_timeout_s"])
        self.coast_s = float(cfg["coast_s"])
        self.max_pos_sigma = float(cfg["max_pos_sigma_m"])
        self.merge_dist = float(cfg["merge_dist_m"])
        self.merge_vel = float(cfg["merge_vel_mps"])
        self.init_vel_sigma = float(cfg["init_vel_sigma_mps"])
        self.bearing_init = bearing_init
        self.r0 = float(cfg["bearing_init_range_m"])
        self.r0_sigma = float(cfg["bearing_init_range_sigma_m"])
        self.identity = identity
        self.tracks: list[Track] = []
        self._next_id = 1

    # --- filter steps ------------------------------------------------------------------------

    def predict(self, t: float):
        for tr in self.tracks:
            dt = t - tr.t
            if dt > 1e-9:
                F, Q = cv_model(dt, self.q)
                tr.x = F @ tr.x
                tr.P = F @ tr.P @ F.T + Q
                tr.t = t

    def _update(self, tr: Track, det: Detection, y, H, S):
        K = tr.P @ H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ y
        I_KH = np.eye(6) - K @ H
        tr.P = I_KH @ tr.P @ I_KH.T + K @ det.R @ K.T  # Joseph form: stays symmetric positive
        tr.P = 0.5 * (tr.P + tr.P.T)
        tr.hits += 1
        tr.last_update = det.t
        tr.sensor_last[det.sensor] = det.t
        if det.cooperative_id:
            tr.cooperative_id = det.cooperative_id
        if not tr.confirmed and tr.hits >= self.confirm_hits:
            tr.confirmed = True
        if det.likelihood is not None:
            self.identity.update(tr, det)

    def _gate(self, tr: Track, det: Detection, scale: float = 1.0):
        y, H = innovation(tr.x, det)
        S = H @ tr.P @ H.T + det.R
        d2 = float(y @ np.linalg.solve(S, y))
        if d2 > scale * self.gates[len(y)]:
            return None
        return d2 + float(np.linalg.slogdet(S)[1]), (y, H, S)

    # --- one sensor scan ---------------------------------------------------------------------

    def process(self, dets: list[Detection], t: float):
        if not dets:
            return
        self.predict(t)
        pending = []
        for d in dets:
            # Reports carrying an identity (ADS-B) go straight to the track with that identity.
            if d.cooperative_id is not None:
                tr = next((x for x in self.tracks if x.cooperative_id == d.cooperative_id), None)
                if tr is not None:
                    g = self._gate(tr, d, scale=4.0)
                    if g is not None:
                        self._update(tr, d, *g[1])
                        continue
            pending.append(d)
        if not pending:
            return

        n, m = len(pending), len(self.tracks)
        cost = np.full((n, m), np.inf)
        cache = {}
        for i, d in enumerate(pending):
            for j, tr in enumerate(self.tracks):
                if d.cooperative_id and tr.cooperative_id and d.cooperative_id != tr.cooperative_id:
                    continue
                if not tr.confirmed and d.kind == "ae" and not self.bearing_init:
                    # A direction-only report must not confirm a tentative track: radar clutter
                    # plus a camera seeing a real target on a similar bearing would make a ghost.
                    continue
                g = self._gate(tr, d)
                if g is not None:
                    cost[i, j], cache[(i, j)] = g
        used = set()
        finite = np.isfinite(cost)
        if finite.any():
            shifted = np.where(finite, cost - cost[finite].min() + 1.0, 1e12)
            for i, j in zip(*linear_sum_assignment(shifted)):
                if finite[i, j]:
                    self._update(self.tracks[j], pending[i], *cache[(i, j)])
                    used.add(i)
        for i, d in enumerate(pending):
            if i not in used:
                self._initiate(d, t)

    def _initiate(self, d: Detection, t: float):
        if d.kind == "pos":
            p, Pp = d.z.copy(), d.R.copy()
        elif d.kind == "rae":
            J = rae_to_enu_jacobian(d.z)
            p, Pp = d.origin + rae_to_enu(d.z), J @ d.R @ J.T
        elif d.kind == "ae":
            if not self.bearing_init:
                return  # a direction alone cannot place a target in 3-D
            rae = np.array([self.r0, d.z[0], d.z[1]])
            J = rae_to_enu_jacobian(rae)
            Rr = np.diag([self.r0_sigma ** 2, d.R[0, 0], d.R[1, 1]])
            p, Pp = d.origin + rae_to_enu(rae), J @ Rr @ J.T
        else:
            raise ValueError(d.kind)
        P = np.zeros((6, 6))
        P[:3, :3] = Pp + np.eye(3)
        P[3:, 3:] = np.eye(3) * self.init_vel_sigma ** 2
        tr = Track(tid=self._next_id, x=np.concatenate([p, np.zeros(3)]), P=P, t=t, born=t,
                   last_update=t, init_sensor=d.sensor, cooperative_id=d.cooperative_id,
                   evidence_t=t)
        tr.sensor_last[d.sensor] = t
        self._next_id += 1
        if d.likelihood is not None:
            self.identity.update(tr, d)
        self.tracks.append(tr)

    # --- track management --------------------------------------------------------------------

    def maintain(self, t: float):
        """Delete stale or diverged tracks and merge duplicates."""
        self.predict(t)
        alive = []
        for tr in self.tracks:
            since = t - tr.last_update
            if not tr.confirmed and since > self.tentative_timeout:
                continue
            if tr.confirmed and since > self.coast_s:
                continue
            if tr.pos_sigma > self.max_pos_sigma:
                continue
            alive.append(tr)
        alive.sort(key=lambda tr: (not tr.confirmed, -tr.hits, tr.tid))
        kept: list[Track] = []
        for tr in alive:
            duplicate = False
            for o in kept:
                if tr.cooperative_id and o.cooperative_id and tr.cooperative_id != o.cooperative_id:
                    continue
                close = np.linalg.norm(tr.x[:3] - o.x[:3]) < self.merge_dist
                similar = (not tr.confirmed) or np.linalg.norm(tr.x[3:] - o.x[3:]) < self.merge_vel
                if close and similar:
                    duplicate = True
                    o.cooperative_id = o.cooperative_id or tr.cooperative_id
                    break
            if not duplicate:
                kept.append(tr)
        self.tracks = sorted(kept, key=lambda tr: tr.tid)

    def confirmed(self) -> list[Track]:
        return [tr for tr in self.tracks if tr.confirmed]
