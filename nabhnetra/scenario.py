"""Ground-truth scenarios: who is in the sky, where, and when.

Everything is in a local east-north-up frame centred on the defended asset. No real-world
location is used, on purpose.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import wrap_angle

REGION_RADIUS_M = 9500.0  # targets beyond this horizontal range are dropped from the truth


@dataclass
class TargetSpec:
    tid: int
    cls: str
    rcs_m2: float        # radar cross-section
    size_m: float        # characteristic size seen by cameras
    acoustic_db: float   # source level at 1 m (dB re 20 uPa)
    cooperative: bool    # broadcasts ADS-B
    hostile: bool        # ground-truth intent, used only for evaluation


@dataclass
class Environment:
    light: str = "day"            # "day" or "night"
    visibility_m: float = 30000.0  # meteorological visibility; fog is ~1000 m or less


@dataclass
class Scenario:
    name: str
    seed: int
    dt: float
    times: np.ndarray
    specs: list[TargetSpec]
    states: np.ndarray  # (n_targets, n_steps, 6): east, north, up, v_e, v_n, v_u; NaN when absent
    env: Environment
    faults: list[tuple[str, float, float]] = field(default_factory=list)

    def alive(self, k: int) -> np.ndarray:
        return ~np.isnan(self.states[:, k, 0])

    def sensor_failed(self, name: str, t: float) -> bool:
        return any(n == name and t0 <= t < t1 for n, t0, t1 in self.faults)


def _log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _properties(rng, cls):
    if cls == "drone":
        return dict(rcs_m2=_log_uniform(rng, 0.005, 0.05), size_m=rng.uniform(0.3, 0.8),
                    acoustic_db=rng.uniform(78, 88), cooperative=False)
    if cls == "bird":
        return dict(rcs_m2=_log_uniform(rng, 0.001, 0.02), size_m=rng.uniform(0.2, 0.9),
                    acoustic_db=50.0, cooperative=False)
    if cls == "aircraft":
        return dict(rcs_m2=_log_uniform(rng, 2.0, 20.0), size_m=rng.uniform(12, 35),
                    acoustic_db=rng.uniform(140, 148), cooperative=bool(rng.random() < 0.95))
    if cls == "helicopter":
        return dict(rcs_m2=_log_uniform(rng, 3.0, 10.0), size_m=rng.uniform(10, 16),
                    acoustic_db=rng.uniform(125, 132), cooperative=bool(rng.random() < 0.5))
    raise ValueError(cls)


def _polar(range_m, bearing, alt):
    return np.array([range_m * np.sin(bearing), range_m * np.cos(bearing), alt], dtype=float)


# --- motion models -------------------------------------------------------------------------

def _inbound(rng, n, dt, start, alt, speed, aim_sigma=150.0, loiter_radius=150.0):
    """A drone flying at the asset with a weave, then circling it at low altitude."""
    p = start.copy()
    aim = rng.normal(0.0, aim_sigma, 2)
    heading = np.arctan2(aim[0] - p[0], aim[1] - p[1])
    weave_amp = np.deg2rad(rng.uniform(5, 30))
    weave_period = rng.uniform(15, 40)
    phase = rng.uniform(0, 2 * np.pi)
    max_turn = np.deg2rad(30.0)
    alt_target, loiter = alt, False
    out = np.empty((n, 6))
    for k in range(n):
        to_aim = aim - p[:2]
        dist = float(np.hypot(*to_aim))
        brg = np.arctan2(to_aim[0], to_aim[1])
        if dist < 1.2 * loiter_radius:
            loiter = True
        if loiter:
            pull = np.clip((dist - loiter_radius) / loiter_radius, -1, 1) * np.pi / 3
            desired = brg + np.pi / 2 - pull
            alt_target = max(30.0, 0.5 * alt)
        else:
            desired = brg + weave_amp * np.sin(2 * np.pi * k * dt / weave_period + phase)
        heading += np.clip(wrap_angle(desired - heading), -max_turn * dt, max_turn * dt)
        v = np.array([speed * np.sin(heading), speed * np.cos(heading),
                      np.clip(0.1 * (alt_target - p[2]), -3.0, 3.0)])
        out[k, :3], out[k, 3:] = p, v
        p = p + v * dt
    return out


def _wander(rng, n, dt, start, speed, alt_lo=10.0, alt_hi=250.0, turn_sigma_deg=8.0):
    """A bird: random turns, speed changes and climbs."""
    p = start.copy()
    heading = rng.uniform(-np.pi, np.pi)
    omega, climb, s = 0.0, 0.0, speed
    out = np.empty((n, 6))
    for k in range(n):
        omega = 0.95 * omega + rng.normal(0.0, np.deg2rad(turn_sigma_deg)) * np.sqrt(dt)
        heading += omega * dt
        s = float(np.clip(s + rng.normal(0.0, 0.5) * np.sqrt(dt), 0.5 * speed, 1.5 * speed))
        climb = 0.9 * climb + rng.normal(0.0, 0.5) * np.sqrt(dt)
        if p[2] < alt_lo:
            climb = abs(climb) + 0.5
        elif p[2] > alt_hi:
            climb = -abs(climb) - 0.5
        v = np.array([s * np.sin(heading), s * np.cos(heading), climb])
        out[k, :3], out[k, 3:] = p, v
        p = p + v * dt
    return out


def _transit(rng, n, dt, bearing, start_range, alt, speed, miss_m):
    """An aircraft or helicopter passing by on a straight line."""
    p = _polar(start_range, bearing, alt)
    perp = np.array([np.cos(bearing), -np.sin(bearing)]) * miss_m
    heading = np.arctan2(perp[0] - p[0], perp[1] - p[1])
    drift = 0.0
    out = np.empty((n, 6))
    for k in range(n):
        drift = 0.98 * drift + rng.normal(0.0, np.deg2rad(0.2))
        h = heading + drift
        v = np.array([speed * np.sin(h), speed * np.cos(h), 0.0])
        out[k, :3], out[k, 3:] = p, v
        p = p + v * dt
    return out


def _flock(rng, n, dt, start, speed, size):
    """Birds that move together. Close members may share one radar resolution cell."""
    leader = _wander(rng, n, dt, start, speed, alt_lo=20.0, alt_hi=200.0)
    members = []
    for _ in range(size):
        offset = np.concatenate([rng.normal(0.0, 15.0, 2), rng.normal(0.0, 4.0, 1)])
        jitter = np.zeros(3)
        traj = np.empty((n, 6))
        for k in range(n):
            jitter = 0.9 * jitter + rng.normal(0.0, 0.5, 3)
            traj[k, :3] = leader[k, :3] + offset + jitter
            traj[k, 3:] = leader[k, 3:] + rng.normal(0.0, 0.3, 3)
        members.append(traj)
    return members


# --- scenario assembly ---------------------------------------------------------------------

class _Builder:
    def __init__(self, rng, n_steps, dt):
        self.rng, self.n_steps, self.dt = rng, n_steps, dt
        self.specs: list[TargetSpec] = []
        self.tracks: list[np.ndarray] = []

    def add(self, cls, traj, birth_step=0, hostile=False):
        full = np.full((self.n_steps, 6), np.nan)
        full[birth_step:] = traj[: self.n_steps - birth_step]
        gone = np.hypot(full[:, 0], full[:, 1]) > REGION_RADIUS_M
        if gone.any():
            full[np.argmax(gone):] = np.nan
        tid = len(self.specs)
        self.specs.append(TargetSpec(tid=tid, cls=cls, hostile=hostile,
                                     **_properties(self.rng, cls)))
        self.tracks.append(full)

    def birth(self, max_s):
        return int(self.rng.integers(0, int(max_s / self.dt) + 1))

    def steps_left(self, birth):
        return self.n_steps - birth

    # common target recipes
    def drone(self, bearing=None, start_range=None, birth=None):
        rng = self.rng
        b = self.birth(40) if birth is None else birth
        bearing = rng.uniform(-np.pi, np.pi) if bearing is None else bearing
        start_range = rng.uniform(2500, 4000) if start_range is None else start_range
        start = _polar(start_range, bearing, rng.uniform(40, 150))
        self.add("drone", _inbound(rng, self.steps_left(b), self.dt, start, start[2],
                                   rng.uniform(12, 20)), b, hostile=True)

    def bird(self, start=None):
        rng = self.rng
        if start is None:
            start = _polar(rng.uniform(500, 3500), rng.uniform(-np.pi, np.pi), rng.uniform(15, 200))
        self.add("bird", _wander(rng, self.n_steps, self.dt, start, rng.uniform(8, 14)))

    def flock(self, size, start=None):
        rng = self.rng
        if start is None:
            start = _polar(rng.uniform(800, 3200), rng.uniform(-np.pi, np.pi), rng.uniform(30, 150))
        for traj in _flock(rng, self.n_steps, self.dt, start, rng.uniform(9, 13), size):
            self.add("bird", traj)

    def aircraft(self):
        rng = self.rng
        b = self.birth(60)
        self.add("aircraft", _transit(rng, self.steps_left(b), self.dt, rng.uniform(-np.pi, np.pi),
                                      9000.0, rng.uniform(600, 1500), rng.uniform(70, 110),
                                      rng.uniform(500, 3000)), b)

    def helicopter(self):
        rng = self.rng
        b = self.birth(60)
        self.add("helicopter", _transit(rng, self.steps_left(b), self.dt,
                                        rng.uniform(-np.pi, np.pi), 6000.0, rng.uniform(150, 400),
                                        rng.uniform(35, 60), rng.uniform(300, 2000)), b)

    def mixed(self):
        for _ in range(3):
            self.drone()
        for _ in range(2):
            self.flock(4)
        for _ in range(4):
            self.bird()
        self.aircraft()
        self.helicopter()


SCENARIOS = {
    "mixed": "3 inbound drones, 12 birds (two flocks), 1 aircraft, 1 helicopter; clear day",
    "night": "same traffic as mixed, at night (daylight camera nearly blind)",
    "fog": "same traffic as mixed, 800 m visibility",
    "bird_flock": "2 drones approaching through 4 bird flocks (22 birds) on their paths",
    "swarm": "20 drones from one 30-degree sector plus 4 birds",
    "radar_failure": "same traffic as mixed; radar is down from 80 s to 160 s",
}


def build_scenario(name: str, seed: int, dt: float = 0.5, duration_s: float = 240.0) -> Scenario:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario '{name}'. Choose from {sorted(SCENARIOS)}")
    rng = np.random.default_rng(seed)
    n_steps = int(round(duration_s / dt)) + 1
    b = _Builder(rng, n_steps, dt)
    env, faults = Environment(), []

    if name in ("mixed", "night", "fog", "radar_failure"):
        b.mixed()
        if name == "night":
            env.light = "night"
        elif name == "fog":
            env.visibility_m = 800.0
        elif name == "radar_failure":
            faults.append(("radar", 80.0, 160.0))
    elif name == "bird_flock":
        for _ in range(2):
            bearing = rng.uniform(-np.pi, np.pi)
            b.drone(bearing=bearing, start_range=rng.uniform(3000, 3800))
            for r in (1200.0, 2200.0):
                start = _polar(r, bearing + rng.normal(0, np.deg2rad(4)), rng.uniform(40, 140))
                b.flock(6 if r > 2000 else 5, start=start)
    elif name == "swarm":
        sector = rng.uniform(-np.pi, np.pi)
        for _ in range(20):
            b.drone(bearing=sector + rng.uniform(-np.deg2rad(15), np.deg2rad(15)),
                    start_range=rng.uniform(3000, 3800), birth=b.birth(20))
        for _ in range(4):
            b.bird()

    return Scenario(name=name, seed=seed, dt=dt, times=np.arange(n_steps) * dt, specs=b.specs,
                    states=np.stack(b.tracks), env=env, faults=faults)
