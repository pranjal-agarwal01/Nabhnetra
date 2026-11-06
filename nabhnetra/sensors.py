"""Sensor models. Each one turns ground truth into the noisy detections the real device would report.

Radar      range + direction, detects anything with a radar echo, cannot tell a bird from a drone
           unless a micro-Doppler classifier is attached.
EO / IR    very precise direction, no range; classify by appearance (confusion matrix from real data).
Acoustic   coarse direction, short range; hears rotors and engines, not birds.
ADS-B      cooperative aircraft broadcast their own position and identity.
"""
from __future__ import annotations

import numpy as np

from .common import CLASS_INDEX, CLASSES, N_CLASSES, Detection
from .confusion import ConfusionModel
from .geometry import enu_to_rae, wrap_angle


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


class Sensor:
    kind = "ae"
    starts_noncooperative_tracks = False  # True only for sensors that measure range (radar)

    def __init__(self, name: str, cfg: dict, dt: float):
        self.name = name
        self.cfg = cfg
        self.origin = np.asarray(cfg.get("origin", [0.0, 0.0, 0.0]), dtype=float)
        self.stride = max(1, int(round(float(cfg.get("period_s", dt)) / dt)))
        self.offset = int(cfg.get("offset_steps", 0)) % self.stride
        self.persistence = float(cfg.get("label_persistence", 0.0))
        self.cm: ConfusionModel | None = None
        self._last_label: dict[int, int] = {}

    def due(self, k: int) -> bool:
        return k % self.stride == self.offset

    def observe(self, k, t, scenario, rng) -> list[Detection]:
        raise NotImplementedError

    def _visible(self, scenario, k):
        """Targets present at step k, with range/azimuth/elevation as seen from this sensor."""
        idx = np.flatnonzero(scenario.alive(k))
        if idx.size == 0:
            return idx, np.zeros((0, 3))
        return idx, enu_to_rae(scenario.states[idx, k, :3] - self.origin)

    def _label(self, rng, target: int, row: np.ndarray) -> int:
        """Sample a classifier label. Real classifier mistakes are correlated over time, so with
        probability `label_persistence` the previous label for this target is repeated. This keeps
        the long-run label rates equal to the confusion matrix but stops the fusion from treating
        every frame as independent evidence."""
        prev = self._last_label.get(target)
        if prev is not None and rng.random() < self.persistence:
            return prev
        idx = int(rng.choice(len(row), p=row))
        self._last_label[target] = idx
        return idx


class Radar(Sensor):
    kind = "rae"
    starts_noncooperative_tracks = True

    def __init__(self, name, cfg, dt):
        super().__init__(name, cfg, dt)
        self.pfa = float(cfg["pfa"])
        # SNR at which a Swerling-I target is detected half the time with this false-alarm rate
        self.snr50 = np.log(self.pfa) / np.log(0.5) - 1.0
        self.r50, self.rcs_ref = float(cfg["r50_m"]), float(cfg["rcs_ref_m2"])
        self.rmin, self.rmax = float(cfg["min_range_m"]), float(cfg["max_range_m"])
        self.el_lo, self.el_hi = np.deg2rad(cfg["el_min_deg"]), np.deg2rad(cfg["el_max_deg"])
        self.sig = np.array([cfg["sigma_range_m"], np.deg2rad(cfg["sigma_az_deg"]),
                             np.deg2rad(cfg["sigma_el_deg"])])
        self.R = np.diag(self.sig ** 2)
        self.range_res, self.az_res = float(cfg["range_res_m"]), np.deg2rad(cfg["az_res_deg"])
        self.clutter = float(cfg["clutter_per_scan"])
        self.clutter_el = np.deg2rad(cfg["clutter_el_max_deg"])
        if cfg.get("classifier"):
            self.cm = ConfusionModel.load(cfg["classifier"])

    def pd(self, r, rcs):
        """Swerling-I detection probability; SNR falls as rcs / r^4 (radar range equation)."""
        snr = self.snr50 * (np.asarray(rcs) / self.rcs_ref) * (self.r50 / np.maximum(r, 1.0)) ** 4
        return self.pfa ** (1.0 / (1.0 + snr))

    def _cells(self, rae):
        """Group targets the radar cannot separate (same range and azimuth cell) into one echo."""
        groups: list[list[int]] = []
        for i in np.argsort(rae[:, 0]):
            for g in groups:
                j = g[0]
                if (abs(rae[i, 0] - rae[j, 0]) < self.range_res
                        and abs(wrap_angle(rae[i, 1] - rae[j, 1])) < self.az_res):
                    g.append(int(i))
                    break
            else:
                groups.append([int(i)])
        return groups

    def observe(self, k, t, sc, rng):
        dets = []
        idx, rae = self._visible(sc, k)
        ok = ((rae[:, 0] >= self.rmin) & (rae[:, 0] <= self.rmax)
              & (rae[:, 2] >= self.el_lo) & (rae[:, 2] <= self.el_hi))
        idx, rae = idx[ok], rae[ok]
        rcs = np.array([sc.specs[i].rcs_m2 for i in idx])
        for g in self._cells(rae):
            total = rcs[g].sum()
            w = rcs[g] / total
            ref = rae[g[0]]
            centre = np.array([w @ rae[g, 0], ref[1] + w @ wrap_angle(rae[g, 1] - ref[1]),
                               w @ rae[g, 2]])
            if rng.random() >= self.pd(centre[0], total):
                continue
            z = centre + rng.normal(0.0, self.sig)
            z[1] = wrap_angle(z[1])
            strongest = int(idx[g[int(np.argmax(rcs[g]))]])
            lik, label = self._classify(rng, strongest, sc.specs[strongest].cls)
            dets.append(Detection(t, self.name, "rae", z, self.R.copy(), self.origin, lik, label,
                                  truth_id=sc.specs[strongest].tid))
        for _ in range(rng.poisson(self.clutter)):
            r = self.rmax * np.sqrt(rng.uniform((self.rmin / self.rmax) ** 2, 1.0))
            z = np.array([r, rng.uniform(-np.pi, np.pi), rng.uniform(0.0, self.clutter_el)])
            lik, label = self._classify(rng, None, None)
            dets.append(Detection(t, self.name, "rae", z, self.R.copy(), self.origin, lik, label))
        return dets

    def _classify(self, rng, target, cls):
        if self.cm is None:
            return None, None
        b = self.cm.default_bin
        if cls is None:
            li = int(rng.integers(len(self.cm.labels)))
        else:
            li = self._label(rng, target, self.cm.row(cls, b))
        return self.cm.likelihood(li, b), self.cm.labels[li]


class Camera(Sensor):
    """Daylight (EO) or thermal (IR) camera with an ML detector behind it."""

    kind = "ae"

    def __init__(self, name, cfg, dt):
        super().__init__(name, cfg, dt)
        self.ifov = float(cfg["ifov_mrad"]) * 1e-3
        self.px50, self.px_scale = float(cfg["px50"]), float(cfg["px_scale"])
        self.sigma = float(cfg["sigma_mrad"]) * 1e-3
        self.R = np.diag([self.sigma ** 2, self.sigma ** 2])
        self.max_range = float(cfg["max_range_m"])
        self.el_lo, self.el_hi = np.deg2rad(cfg["el_min_deg"]), np.deg2rad(cfg["el_max_deg"])
        self.false_alarms = float(cfg["false_alarms_per_frame"])
        self.night_factor = float(cfg["night_factor"])
        self.fog_gain = float(cfg["fog_gain"])
        self.cm = ConfusionModel.load(cfg["confusion"])

    def pixels(self, size_m, r):
        """Pixels on target: angular size divided by the angle one pixel covers."""
        return size_m / (np.maximum(r, 1.0) * self.ifov)

    def pd(self, px, r, env):
        p = _logistic((px - self.px50) / self.px_scale)
        if env.light == "night":
            p = p * self.night_factor
        # Koschmieder: contrast falls as exp(-3.912 r / visibility). Thermal IR sees further in haze.
        return p * np.exp(-3.912 * r / (env.visibility_m * self.fog_gain))

    def observe(self, k, t, sc, rng):
        dets = []
        idx, rae = self._visible(sc, k)
        ok = (rae[:, 0] <= self.max_range) & (rae[:, 2] >= self.el_lo) & (rae[:, 2] <= self.el_hi)
        for i, (r, az, el) in zip(idx[ok], rae[ok]):
            spec = sc.specs[i]
            px = self.pixels(spec.size_m, r)
            if rng.random() >= self.pd(px, r, sc.env):
                continue
            z = np.array([az, el]) + rng.normal(0.0, self.sigma, 2)
            z[0] = wrap_angle(z[0])
            b = self.cm.bin_for(px)
            li = self._label(rng, int(i), self.cm.row(spec.cls, b))
            dets.append(Detection(t, self.name, "ae", z, self.R.copy(), self.origin,
                                  self.cm.likelihood(li, b), self.cm.labels[li], truth_id=spec.tid))
        far = self.cm.bin_for(0.0)
        for _ in range(rng.poisson(self.false_alarms)):
            z = np.array([rng.uniform(-np.pi, np.pi),
                          rng.uniform(max(self.el_lo, 0.02), np.deg2rad(30))])
            li = int(rng.integers(len(self.cm.labels)))
            dets.append(Detection(t, self.name, "ae", z, self.R.copy(), self.origin,
                                  self.cm.likelihood(li, far), self.cm.labels[li]))
        return dets


class Acoustic(Sensor):
    """Microphone array: bearing from beamforming, label from an audio classifier."""

    kind = "ae"

    def __init__(self, name, cfg, dt):
        super().__init__(name, cfg, dt)
        s_az, s_el = np.deg2rad(cfg["sigma_az_deg"]), np.deg2rad(cfg["sigma_el_deg"])
        self.sig = np.array([s_az, s_el])
        self.R = np.diag(self.sig ** 2)
        self.noise_db, self.gain_db = float(cfg["noise_db"]), float(cfg["array_gain_db"])
        self.alpha = float(cfg["absorption_db_per_m"])
        self.snr50, self.snr_scale = float(cfg["snr50_db"]), float(cfg["snr_scale_db"])
        self.max_range = float(cfg["max_range_m"])
        self.cm = ConfusionModel.load(cfg["confusion"])
        self.bg = self.cm.labels.index("background")

    def pd(self, source_db, r):
        r = np.maximum(r, 1.0)
        snr = source_db - 20.0 * np.log10(r) - self.alpha * r - self.noise_db + self.gain_db
        return _logistic((snr - self.snr50) / self.snr_scale)

    def observe(self, k, t, sc, rng):
        dets = []
        b = self.cm.default_bin
        idx, rae = self._visible(sc, k)
        ok = (rae[:, 0] <= self.max_range) & (rae[:, 2] >= 0.0)
        for i, (r, az, el) in zip(idx[ok], rae[ok]):
            spec = sc.specs[i]
            if rng.random() >= self.pd(spec.acoustic_db, r):
                continue
            li = self._label(rng, int(i), self.cm.row(spec.cls, b))
            if li == self.bg:
                continue  # the classifier heard nothing it recognises
            z = np.array([az, el]) + rng.normal(0.0, self.sig)
            z[0] = wrap_angle(z[0])
            dets.append(Detection(t, self.name, "ae", z, self.R.copy(), self.origin,
                                  self.cm.likelihood(li, b), self.cm.labels[li], truth_id=spec.tid))
        row = self.cm.background_row
        if row is not None and rng.random() < 1.0 - row[self.bg]:
            p = row.copy()
            p[self.bg] = 0.0
            li = int(rng.choice(len(p), p=p / p.sum()))
            z = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(0.0, np.deg2rad(45))])
            dets.append(Detection(t, self.name, "ae", z, self.R.copy(), self.origin,
                                  self.cm.likelihood(li, b), self.cm.labels[li]))
        return dets


class ADSB(Sensor):
    """Receiver for aircraft that broadcast their own GPS position (cooperative traffic).
    Spoofing is not modelled; treat cooperative status as a hint, not proof."""

    kind = "pos"

    def __init__(self, name, cfg, dt):
        super().__init__(name, cfg, dt)
        self.max_range = float(cfg["max_range_m"])
        self.p_report = float(cfg["pd"])
        self.sig = np.array([cfg["sigma_h_m"], cfg["sigma_h_m"], cfg["sigma_v_m"]], dtype=float)
        self.R = np.diag(self.sig ** 2)
        self.confidence = float(cfg["category_confidence"])

    def observe(self, k, t, sc, rng):
        dets = []
        for i in np.flatnonzero(sc.alive(k)):
            spec = sc.specs[i]
            if not spec.cooperative:
                continue
            pos = sc.states[i, k, :3]
            if np.hypot(*(pos[:2] - self.origin[:2])) > self.max_range or rng.random() >= self.p_report:
                continue
            lik = np.full(N_CLASSES, (1.0 - self.confidence) / (N_CLASSES - 1))
            lik[CLASS_INDEX[spec.cls]] = self.confidence
            dets.append(Detection(t, self.name, "pos", pos + rng.normal(0.0, self.sig),
                                  self.R.copy(), self.origin, lik, spec.cls,
                                  cooperative_id=f"SIM{spec.tid:03d}", truth_id=spec.tid))
        return dets


SENSOR_TYPES = {"radar": Radar, "camera": Camera, "acoustic": Acoustic, "adsb": ADSB}


def build_sensors(cfg: dict, names=None, dt: float = 0.5) -> list[Sensor]:
    """Instantiate the sensors listed in cfg['sensors'] (all of them, or only `names`)."""
    available = cfg["sensors"]
    names = list(available) if names is None else list(names)
    unknown = [n for n in names if n not in available]
    if unknown:
        raise ValueError(f"unknown sensors {unknown}; configured: {list(available)}")
    return [SENSOR_TYPES[available[n]["type"]](n, available[n], dt) for n in names]


__all__ = ["Sensor", "Radar", "Camera", "Acoustic", "ADSB", "build_sensors", "CLASSES"]
