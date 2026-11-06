"""Shared constants and the detection record every sensor emits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

CLASSES: tuple[str, ...] = ("drone", "bird", "aircraft", "helicopter")
N_CLASSES = len(CLASSES)
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}


@dataclass
class Detection:
    """One report from one sensor at one time.

    kind
      "rae": z = [range m, azimuth rad, elevation rad]   (radar)
      "ae":  z = [azimuth rad, elevation rad]            (EO, IR, acoustic: direction only, no range)
      "pos": z = [east, north, up] m                     (ADS-B self-reported position)

    Azimuth is measured from north, clockwise. Elevation is above the horizon.
    """

    t: float
    sensor: str
    kind: str
    z: np.ndarray
    R: np.ndarray
    origin: np.ndarray
    likelihood: np.ndarray | None = None  # P(reported label | true class), in CLASSES order
    label: str | None = None
    cooperative_id: str | None = None
    truth_id: int = -1  # evaluation only; the tracker never reads it
