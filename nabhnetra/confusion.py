"""Confusion matrices: the bridge between real-data classifiers and the simulator.

Each simulated camera or microphone samples its reported label from the confusion matrix
measured on real held-out data, so the fusion results inherit real classifier behaviour.
The matrices are written by scripts/eval_confusion.py and scripts/train_audio.py. A file whose
"source" says PLACEHOLDER holds assumed values, and numbers produced from it are not results.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import CLASS_INDEX, CLASSES, ROOT


class ConfusionModel:
    """P(reported label | true class), optionally split into pixels-on-target bins.

    JSON layout (see configs/confusion/*.json)
      labels:         labels the classifier can output (matrix columns)
      true_classes:   row order of each matrix; must cover every class in CLASSES
      bins_px:        [a, b] gives bins "distant" (< a px), "medium" (a to b), "close" (>= b);
                      omit it for a single bin
      matrices:       {bin_name: rows x labels}
      background_row: optional label distribution when no target is present (false alarms)
      source:         where the numbers came from
    """

    def __init__(self, labels, true_classes, matrices, bins_px=None, background_row=None,
                 source="", path=None):
        self.labels = list(labels)
        true_classes = list(true_classes)
        missing = [c for c in CLASSES if c not in true_classes]
        if missing:
            raise ValueError(f"confusion matrix has no rows for {missing}")
        order = [true_classes.index(c) for c in CLASSES]
        self.matrices: dict[str, np.ndarray] = {}
        for name, m in matrices.items():
            m = np.asarray(m, dtype=float)[order]
            if m.shape != (len(CLASSES), len(self.labels)):
                raise ValueError(f"matrix '{name}' has shape {m.shape}, expected "
                                 f"{(len(CLASSES), len(self.labels))}")
            if (m < 0).any() or (m.sum(axis=1) <= 0).any():
                raise ValueError(f"matrix '{name}' has negative or empty rows")
            self.matrices[name] = m / m.sum(axis=1, keepdims=True)
        self.default_bin = next(iter(self.matrices))
        self.bins_px = list(bins_px) if bins_px else None
        if self.bins_px and not {"distant", "medium", "close"} <= set(self.matrices):
            raise ValueError("bins_px needs matrices named distant, medium and close")
        self.background_row = None
        if background_row is not None:
            b = np.asarray(background_row, dtype=float)
            self.background_row = b / b.sum()
        self.source = source
        self.path = path

    @classmethod
    def load(cls, path) -> "ConfusionModel":
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(data["labels"], data["true_classes"], data["matrices"], data.get("bins_px"),
                   data.get("background_row"), data.get("source", ""), p)

    @property
    def is_placeholder(self) -> bool:
        return "PLACEHOLDER" in self.source.upper()

    def bin_for(self, px: float | None = None) -> str:
        if self.bins_px is None or px is None:
            return self.default_bin
        lo, hi = self.bins_px
        return "distant" if px < lo else ("medium" if px < hi else "close")

    def row(self, true_cls: str, bin_name: str) -> np.ndarray:
        return self.matrices[bin_name][CLASS_INDEX[true_cls]]

    def likelihood(self, label_idx: int, bin_name: str) -> np.ndarray:
        """Column of the matrix: how likely this report is under each true class."""
        return self.matrices[bin_name][:, label_idx].copy()
