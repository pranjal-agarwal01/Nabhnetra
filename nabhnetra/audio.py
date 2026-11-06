"""The audio classifier: a small CNN on log-mel spectrogram windows (drone / helicopter / background)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import ROOT
from .realdata import AUDIO_LABELS, logmel, windows

# a freshly trained model if one exists, otherwise the one shipped in weights/
AUDIO_WEIGHTS = (ROOT / "models" / "audio_cnn.pt" if (ROOT / "models" / "audio_cnn.pt").exists()
                 else ROOT / "weights" / "audio_cnn.pt")


def make_model(n_classes: int = len(AUDIO_LABELS)):
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                             nn.MaxPool2d(2))

    return nn.Sequential(block(1, 16), block(16, 32), block(32, 64), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))


def window_features(wave: np.ndarray, sr: int, win_s: float = 1.0, hop_s: float = 0.5):
    """Standardised log-mel features for each window, shape (n, 1, mels, frames)."""
    feats = []
    for w in windows(wave, sr, win_s, hop_s):
        if len(w) < int(win_s * sr):
            w = np.pad(w, (0, int(win_s * sr) - len(w)))
        m = logmel(w, sr)
        feats.append((m - m.mean()) / (m.std() + 1e-6))
    return np.stack(feats)[:, None].astype(np.float32)


class AudioClassifier:
    def __init__(self, weights: str | Path = AUDIO_WEIGHTS, device: str | None = None):
        import torch

        if not Path(weights).exists():
            raise SystemExit(f"No audio model at {weights}. Run scripts/train_audio.py --save-model")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = make_model().to(self.device)
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model.eval()

    def probabilities(self, wave: np.ndarray, sr: int, win_s: float = 1.0, hop_s: float = 0.5):
        x = self.torch.tensor(window_features(wave, sr, win_s, hop_s)).to(self.device)
        with self.torch.no_grad():
            return self.torch.softmax(self.model(x), dim=1).cpu().numpy()
