"""Helpers for the real-data side: the Svanstrom multi-sensor drone dataset.

Dataset: F. Svanstrom, F. Alonso-Fernandez, C. Englund, "A dataset for multi-sensor drone
detection", Data in Brief 39 (2021). CC BY 4.0. https://github.com/DroneDetectionThesis/Drone-detection-dataset

Video files look like  Data/Video_IR/IR_DRONE_001.mp4  with labels  IR_DRONE_001_LABELS.mat
Audio files look like  Data/Audio/DRONE_001.wav  (DRONE, HELICOPTER, BACKGROUND)
"""
from __future__ import annotations

import re
from math import gcd

import numpy as np

from .common import CLASS_INDEX, CLASSES

VIDEO_CLASSES = {"DRONE": "drone", "BIRD": "bird", "AIRPLANE": "aircraft", "HELICOPTER": "helicopter"}
AUDIO_LABELS = ("drone", "helicopter", "background")
AUDIO_CLASSES = {"DRONE": "drone", "HELICOPTER": "helicopter", "BACKGROUND": "background"}
_VIDEO_RE = re.compile(r"^(IR|V)_(DRONE|BIRD|AIRPLANE|HELICOPTER)_(\d+)$")
_AUDIO_RE = re.compile(r"^(DRONE|HELICOPTER|BACKGROUND)_(\d+)$")


def parse_video_name(stem: str):
    """'IR_DRONE_001' -> ('IR', 'drone', 1); None if the name does not match."""
    m = _VIDEO_RE.match(stem)
    return (m.group(1), VIDEO_CLASSES[m.group(2)], int(m.group(3))) if m else None


def parse_audio_name(stem: str):
    m = _AUDIO_RE.match(stem)
    return (AUDIO_CLASSES[m.group(1)], int(m.group(2))) if m else None


def split_by_group(items, labels, val_frac=0.2, seed=0):
    """Stratified split by whole video/file, so frames of one clip never leak across splits."""
    rng = np.random.default_rng(seed)
    train, val = [], []
    for lab in sorted(set(labels)):
        group = sorted(i for i, lb in zip(items, labels) if lb == lab)
        rng.shuffle(group)
        n_val = max(1, int(round(val_frac * len(group)))) if len(group) > 1 else 0
        val += group[:n_val]
        train += group[n_val:]
    return sorted(train), sorted(val)


def kfold_by_group(items, labels, k=5, seed=0):
    """Stratified k-fold assignment by file: returns {item: fold}."""
    rng = np.random.default_rng(seed)
    folds = {}
    for lab in sorted(set(labels)):
        group = sorted(i for i, lb in zip(items, labels) if lb == lab)
        rng.shuffle(group)
        for n, item in enumerate(group):
            folds[item] = n % k
    return folds


# --- detection matching and confusion ------------------------------------------------------

def iou_xywh(a, b) -> float:
    """Intersection over union of two (x, y, w, h) boxes with top-left corners."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    iw = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    ih = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_frame(gt_box, preds, iou_thr=0.3):
    """Label index of the best prediction overlapping the ground-truth box, or None if missed.

    preds: list of (box_xywh, class_index, confidence). Small targets get a lenient IoU (0.3)
    because a few pixels of box error is a large fraction of a 6-pixel drone.
    """
    best, best_key = None, (-1.0, -1.0)
    for box, cls_idx, conf in preds:
        iou = iou_xywh(gt_box, box)
        if iou >= iou_thr and (iou, conf) > best_key:
            best, best_key = int(cls_idx), (iou, conf)
    return best


def shrink_frame(img, box, scale: float):
    """Approximate the same scene `1/scale` times further away: shrink the frame and pad it back
    to its original size by mirroring, so the target covers `scale` times fewer pixels at the same
    sensor resolution. It ignores atmosphere and optics, so treat results as an approximation.
    Returns (image, box) with box = (x, y, w, h) in pixels."""
    if scale >= 1.0:
        return img, tuple(box)
    import cv2

    h, w = img.shape[:2]
    sw, sh = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    top, left = (h - sh) // 2, (w - sw) // 2
    pad = ((top, h - sh - top), (left, w - sw - left)) + ((0, 0),) * (img.ndim - 2)
    canvas = np.pad(small, pad, mode="reflect")
    x, y, bw, bh = box
    return canvas, (x * sw / w + left, y * sh / h + top, bw * sw / w, bh * sh / h)


def size_bin(px: float, bins_px) -> str:
    lo, hi = bins_px
    return "distant" if px < lo else ("medium" if px < hi else "close")


def confusion_by_bin(records, bins_px, n_labels=len(CLASSES)):
    """records: iterable of (true_index, predicted_index or None, pixels_on_target).

    Returns {bin: {"counts": n_classes x n_labels, "detected": [...], "total": [...]}}.
    """
    out = {b: {"counts": np.zeros((len(CLASSES), n_labels), dtype=int),
               "detected": np.zeros(len(CLASSES), dtype=int),
               "total": np.zeros(len(CLASSES), dtype=int)} for b in ("distant", "medium", "close")}
    for true_idx, pred_idx, px in records:
        cell = out[size_bin(px, bins_px)]
        cell["total"][true_idx] += 1
        if pred_idx is not None:
            cell["detected"][true_idx] += 1
            cell["counts"][true_idx, pred_idx] += 1
    return out


def normalise_rows(counts, prior=None, strength=20.0, alpha=1.0):
    """Row-normalise a count matrix.

    With `prior` (a row-stochastic matrix, e.g. the pooled matrix over all size bins), each row is
    shrunk toward the prior as if `strength` extra samples had been drawn from it:
        row = (counts + strength * prior_row) / (n + strength)
    A row with thousands of samples barely moves; a row with 1 sample is mostly the prior, instead
    of the noise that plain add-one smoothing would give. Without a prior, add-alpha smoothing.
    """
    counts = np.asarray(counts, dtype=float)
    if prior is None:
        return (counts + alpha) / (counts + alpha).sum(axis=1, keepdims=True)
    prior = np.asarray(prior, dtype=float)
    return (counts + strength * prior) / (counts.sum(axis=1, keepdims=True) + strength)


def fit_logistic_pd(px, hit):
    """Fit P(detect) = 1 / (1 + exp(-(px - px50) / scale)) by maximum likelihood (grid search)."""
    px = np.asarray(px, dtype=float)
    hit = np.asarray(hit, dtype=float)
    best, best_ll = (4.0, 1.0), -np.inf
    for px50 in np.arange(0.5, 30.01, 0.25):
        for scale in np.arange(0.25, 8.01, 0.25):
            p = np.clip(1.0 / (1.0 + np.exp(-(px - px50) / scale)), 1e-6, 1 - 1e-6)
            ll = float(np.sum(hit * np.log(p) + (1 - hit) * np.log(1 - p)))
            if ll > best_ll:
                best, best_ll = (float(px50), float(scale)), ll
    return {"px50": best[0], "px_scale": best[1]}


# --- audio ----------------------------------------------------------------------------------

def load_wav(path, target_sr=22050):
    """Mono float32 waveform resampled to target_sr (scipy only, no extra audio libraries)."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sr, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        g = gcd(int(sr), int(target_sr))
        x = resample_poly(x, target_sr // g, sr // g).astype(np.float32)
    return x, target_sr


def mel_filterbank(sr, n_fft, n_mels=64, fmin=50.0, fmax=None):
    fmax = fmax or sr / 2.0

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for i in range(n_mels):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / max(c - lo, 1e-9)
        down = (hi - bins) / max(hi - c, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


def logmel(x, sr, n_fft=1024, hop=256, n_mels=64, fmin=50.0, fmax=None):
    """Log-mel spectrogram, shape (n_mels, frames)."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    spec = np.abs(np.fft.rfft(frames * np.hanning(n_fft).astype(np.float32), axis=1)) ** 2
    mel = spec @ mel_filterbank(sr, n_fft, n_mels, fmin, fmax).T
    return np.log(mel + 1e-10).T.astype(np.float32)


def decide(probs, groups, threshold=0.0, k=1, n=1, background=2):
    """Turn per-window class probabilities into reported labels with two false-alarm guards.

    threshold: report a target class only if its probability is at least this, else background.
    k-of-n:    report a label only if it was the (thresholded) decision in at least k of the last n
               windows of the same recording. Only past windows are used, so it runs in real time.
    groups:    recording id per window; windows of each recording must be in time order.
    """
    probs = np.asarray(probs, dtype=float)
    raw = probs.argmax(axis=1)
    lab = np.where((raw != background) & (probs.max(axis=1) >= threshold), raw, background)
    if n <= 1:
        return lab
    out = lab.copy()
    groups = np.asarray(groups)
    for g in dict.fromkeys(groups.tolist()):
        idx = np.flatnonzero(groups == g)
        seq = lab[idx]
        for j in range(len(seq)):
            recent = seq[max(0, j - n + 1): j + 1]
            recent = recent[recent != background]
            if recent.size:
                vals, counts = np.unique(recent, return_counts=True)
                out[idx[j]] = vals[counts.argmax()] if counts.max() >= k else background
            else:
                out[idx[j]] = background
    return out


def windows(x, sr, win_s=1.0, hop_s=0.5):
    n, h = int(win_s * sr), int(hop_s * sr)
    return [x[i:i + n] for i in range(0, max(1, len(x) - n + 1), h)]


def class_index(name: str) -> int:
    return CLASS_INDEX[name]


def disable_ultralytics_analytics():
    """Turn off Ultralytics' anonymous usage analytics for this process only. The user's settings
    file is left unchanged; the project otherwise only downloads pretrained weights."""
    try:
        from ultralytics.hub.utils import events

        events.enabled = False
    except Exception:  # noqa: BLE001  (module layout differs between Ultralytics versions)
        pass
