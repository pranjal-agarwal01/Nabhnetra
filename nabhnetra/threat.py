"""Threat ranking for the operator's alert list.

This is decision support only. The score orders tracks for a human to look at; it never
triggers any action by itself.

    danger = 1 - (1 - proximity) * (1 - urgency * heading_factor)
    score  = class_weight * danger * cooperative_factor

class_weight     expected weight under the identity posterior (drone 1.0, bird 0.05, ...)
proximity        exp(-range / proximity_scale)
urgency          1 / (1 + time_to_go / 60 s) when closing on the asset, else 0
heading_factor   1.0 if the closest point of approach is within 300 m, 0.6 within 1 km, else 0.3
danger           a probabilistic OR: being close OR closing fast is enough on its own, so a
                 drone loitering over the asset (closing speed ~0) still ranks HIGH
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import CLASSES

LEVELS = ("NONE", "LOW", "MEDIUM", "HIGH")


@dataclass
class Threat:
    score: float
    level: str
    range_m: float
    closing_mps: float
    time_to_go_s: float
    cpa_m: float


def assess(pos, vel, probs, cooperative: bool, asset, cfg: dict) -> Threat:
    rel = np.asarray(pos, float) - np.asarray(asset, float)
    vel = np.asarray(vel, float)
    r = float(np.linalg.norm(rel))
    closing = -float(rel @ vel) / max(r, 1.0)
    speed2 = float(vel @ vel)
    t_cpa = max(0.0, -float(rel @ vel) / speed2) if speed2 > 1e-6 else 0.0
    cpa = float(np.linalg.norm(rel + vel * t_cpa))
    tgo = r / closing if closing > 0.5 else float("inf")

    weight = float(sum(p * cfg["class_weight"][c] for p, c in zip(probs, CLASSES)))
    proximity = float(np.exp(-r / cfg["proximity_scale_m"]))
    urgency = 1.0 / (1.0 + tgo / 60.0) if np.isfinite(tgo) else 0.0
    heading = 1.0 if cpa < 300 else (0.6 if cpa < 1000 else 0.3)
    score = weight * (1.0 - (1.0 - proximity) * (1.0 - urgency * heading))
    if cooperative:
        score *= cfg["cooperative_factor"]

    lv = cfg["levels"]
    level = ("HIGH" if score >= lv["high"] else "MEDIUM" if score >= lv["medium"]
             else "LOW" if score >= lv["low"] else "NONE")
    return Threat(score, level, r, closing, tgo, cpa)
