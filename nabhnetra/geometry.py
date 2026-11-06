"""Coordinate conversions for a local east-north-up (ENU) frame.

Range/azimuth/elevation (RAE) is what a radar measures; azimuth/elevation is what a
camera or microphone array measures. Azimuth is clockwise from north.
"""
from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def enu_to_rae(d):
    """Relative ENU vector(s) (..., 3) -> (..., 3) [range, azimuth, elevation]."""
    d = np.asarray(d, dtype=float)
    e, n, u = d[..., 0], d[..., 1], d[..., 2]
    rh = np.hypot(e, n)
    r = np.hypot(rh, u)
    return np.stack([r, np.arctan2(e, n), np.arctan2(u, rh)], axis=-1)


def rae_to_enu(rae):
    """[range, azimuth, elevation] (..., 3) -> relative ENU vector(s) (..., 3)."""
    rae = np.asarray(rae, dtype=float)
    r, az, el = rae[..., 0], rae[..., 1], rae[..., 2]
    rh = r * np.cos(el)
    return np.stack([rh * np.sin(az), rh * np.cos(az), r * np.sin(el)], axis=-1)


def rae_jacobian(d):
    """d[range, az, el] / d[e, n, u] at relative position d. Shape (3, 3).

    This is the H matrix the extended Kalman filter uses for radar and camera updates.
    """
    e, n, u = (float(v) for v in d)
    rh2 = max(e * e + n * n, 1e-9)
    rh = np.sqrt(rh2)
    r2 = rh2 + u * u
    r = np.sqrt(r2)
    return np.array(
        [
            [e / r, n / r, u / r],
            [n / rh2, -e / rh2, 0.0],
            [-u * e / (r2 * rh), -u * n / (r2 * rh), rh / r2],
        ]
    )


def rae_to_enu_jacobian(rae):
    """d[e, n, u] / d[range, az, el]. Used to turn measurement noise into position covariance."""
    r, az, el = (float(v) for v in rae)
    ce, se, ca, sa = np.cos(el), np.sin(el), np.cos(az), np.sin(az)
    return np.array(
        [
            [ce * sa, r * ce * ca, -r * se * sa],
            [ce * ca, -r * ce * sa, -r * se * ca],
            [se, 0.0, r * ce],
        ]
    )
