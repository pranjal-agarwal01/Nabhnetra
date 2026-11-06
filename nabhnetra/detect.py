"""Run a trained camera detector on frames and draw what it sees."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import CLASSES, ROOT
from .realdata import iou_xywh, parse_video_name

WEIGHTS = {"ir": ROOT / "runs/detect/ir/weights/best.pt", "eo": ROOT / "runs/detect/eo/weights/best.pt"}
# BGR versions of the air-picture palette: drone orange, bird aqua, manned aircraft blue
COLORS = {"drone": (52, 104, 235), "bird": (122, 175, 27), "aircraft": (214, 120, 42),
          "helicopter": (214, 120, 42)}


def pick_weights(source: str | None) -> Path:
    """Thermal model for IR_* clips, daylight model for everything else."""
    name = Path(source).stem.upper() if source else ""
    return WEIGHTS["ir" if name.startswith("IR_") else "eo"]


def load_model(weights):
    from ultralytics import YOLO

    from .realdata import disable_ultralytics_analytics

    disable_ultralytics_analytics()
    if not Path(weights).exists():
        raise SystemExit(f"No weights at {weights}. Train first: scripts/train_detector.py")
    return YOLO(str(weights))


def predict(model, frame, conf=0.25, device=None):
    """List of (box_xywh, class_name, confidence) for one BGR frame."""
    r = model.predict(frame, conf=conf, verbose=False, device=device)[0]
    return [((b[0], b[1], b[2] - b[0], b[3] - b[1]), CLASSES[int(c)], float(s))
            for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]


def load_truth(video: Path, offset: float = 0.5):
    """Per-frame ground truth for a dataset clip, if its _LABELS.mat file is next to it."""
    label = video.with_name(video.stem + "_LABELS.mat")
    parsed = parse_video_name(video.stem)
    if not label.exists() or parsed is None:
        return None
    try:
        from mcos_decoder import load_groundtruth
    except ImportError:
        return None
    boxes = load_groundtruth(str(label))
    return [None if b is None else ((b[0] - offset, b[1] - offset, b[2], b[3]), parsed[1])
            for b in boxes]


def draw(frame, preds, truth=None, title="", scale=1.0):
    """Annotated copy of the frame, upscaled by `scale` so small thermal frames stay readable."""
    img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) \
        if scale != 1.0 else frame.copy()
    if truth is not None:
        (x, y, w, h), _ = truth
        p0, p1 = (int(x * scale), int(y * scale)), (int((x + w) * scale), int((y + h) * scale))
        cv2.rectangle(img, p0, p1, (235, 235, 235), 1, cv2.LINE_AA)
    for (x, y, w, h), name, conf in preds:
        color = COLORS[name]
        p0, p1 = (int(x * scale), int(y * scale)), (int((x + w) * scale), int((y + h) * scale))
        cv2.rectangle(img, p0, p1, color, 2, cv2.LINE_AA)
        text = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = p0[1] - 6 if p0[1] - th - 8 > 0 else p1[1] + th + 6
        cv2.rectangle(img, (p0[0], ty - th - 4), (p0[0] + tw + 6, ty + 3), color, -1)
        cv2.putText(img, text, (p0[0] + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    if title:
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (20, 20, 20), -1)
        cv2.putText(img, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1,
                    cv2.LINE_AA)
    return img


class ClipScore:
    """Per-clip scoring against ground truth: was the labelled target found, with the right class?"""

    def __init__(self, iou_thr=0.3):
        self.iou_thr = iou_thr
        self.labelled = self.found = self.correct = 0
        self.unlabelled_frames = self.unlabelled_detections = 0

    def add(self, preds, truth):
        if truth is None:
            self.unlabelled_frames += 1
            self.unlabelled_detections += len(preds)
            return
        box, cls = truth
        self.labelled += 1
        best = max(preds, key=lambda p: iou_xywh(box, p[0]), default=None)
        if best is not None and iou_xywh(box, best[0]) >= self.iou_thr:
            self.found += 1
            self.correct += int(best[1] == cls)

    def summary(self) -> str:
        if not self.labelled:
            return "no labelled frames"
        return (f"target found in {self.found}/{self.labelled} labelled frames "
                f"({self.found / self.labelled:.0%}); right class in {self.correct}/{self.found} "
                f"of those ({self.correct / max(self.found, 1):.0%})")


def contact_sheet(frames: list[np.ndarray], n: int = 6, cols: int = 3) -> np.ndarray:
    """Evenly spaced frames in a grid, for a quick look at a whole clip."""
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    tiles = [frames[i] for i in idx]
    h, w = tiles[0].shape[:2]
    while len(tiles) % cols:
        tiles.append(np.zeros((h, w, 3), np.uint8))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    return np.vstack(rows)
