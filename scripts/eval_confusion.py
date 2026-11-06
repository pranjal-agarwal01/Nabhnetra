"""Measure a trained detector on held-out videos and write the confusion matrix the simulator uses.

    python scripts/eval_confusion.py --weights runs/detect/ir/weights/best.pt \
        --data data/yolo_ir --out configs/confusion/ir.json

For every validation frame with a target it records (true class, predicted class or missed,
pixels on target), then writes:
  - a confusion matrix per size bin (distant / medium / close), the same bins the simulator uses
  - a fitted detection curve P(detect | pixels on target) to copy into configs/default.yaml
  - the false-alarm rate per frame, from frames with no target
After this, the camera in the simulation behaves like your real detector.

Range extension (--scales): the dataset was filmed within about 200 m, so almost no targets are
smaller than 8 px and the "distant" bin would be empty. Each validation frame is therefore also
shrunk to 1/2 and 1/4 size (and mirror-padded back) so the same targets appear 2x and 4x further
away. This ignores atmosphere and lens effects, so it is an approximation; say so when you quote it.
Note: visible-camera boxes in this dataset are loose (1.5-3x the target), so their pixel sizes
overstate the target; the IR boxes are tighter.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # use nabhnetra without installing
import numpy as np

from nabhnetra.common import CLASSES
from nabhnetra.realdata import (confusion_by_bin, fit_logistic_pd, match_frame, normalise_rows,
                               shrink_frame)


def read_label(path: Path, w: int, h: int):
    text = path.read_text(encoding="utf-8").strip() if path.exists() else ""
    if not text:
        return None
    c, cx, cy, bw, bh = (float(v) for v in text.splitlines()[0].split())
    return int(c), ((cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="dataset folder with val/images and val/labels")
    ap.add_argument("--out", required=True, help="JSON to write, e.g. configs/confusion/ir.json")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--bins", type=float, nargs=2, default=[8, 20], metavar=("DISTANT", "CLOSE"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None, help="0 for the first GPU, cpu to force CPU")
    ap.add_argument("--prior-strength", type=float, default=20.0,
                    help="pseudo-samples pulling each bin's rows toward the pooled matrix")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.5, 0.25],
                    help="also test each frame shrunk by these factors (range extension)")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    from nabhnetra.realdata import disable_ultralytics_analytics

    disable_ultralytics_analytics()
    model = YOLO(args.weights)
    images = sorted((Path(args.data) / "val" / "images").glob("*.jpg"))
    if not images:
        raise SystemExit(f"No validation images under {args.data}/val/images")

    records, px_all, hit_all = [], [], []
    empty_frames, empty_false_alarms = 0, 0
    per_scale = {s: [0, 0] for s in args.scales}  # detected, total
    for start in range(0, len(images), args.batch):
        arrays, gts = [], []
        for path in images[start:start + args.batch]:
            img = cv2.imread(str(path))
            h, w = img.shape[:2]
            gt = read_label(path.parent.parent / "labels" / f"{path.stem}.txt", w, h)
            if gt is None:
                arrays.append(img)
                gts.append((None, None, 1.0))
                continue
            for s in args.scales:
                shrunk, box = shrink_frame(img, gt[1], s)
                arrays.append(shrunk)
                gts.append((gt[0], box, s))
        results = model.predict(arrays, conf=args.conf, verbose=False, device=args.device)
        for res, (true_idx, box, s) in zip(results, gts):
            preds = [((float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])),
                      int(c), float(sc))
                     for b, c, sc in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                                         res.boxes.conf.tolist())]
            if true_idx is None:
                empty_frames += 1
                empty_false_alarms += len(preds)
                continue
            pred_idx = match_frame(box, preds, args.iou)
            px = max(box[2], box[3])
            records.append((true_idx, pred_idx, px))
            px_all.append(px)
            hit_all.append(pred_idx is not None)
            per_scale[s][0] += int(pred_idx is not None)
            per_scale[s][1] += 1

    cells = confusion_by_bin(records, args.bins)
    pooled = sum(c["counts"] for c in cells.values())
    pooled_m = normalise_rows(pooled)
    matrices, counts_out, pd_out = {}, {}, {}
    for name, cell in cells.items():
        matrices[name] = normalise_rows(cell["counts"], prior=pooled_m,
                                        strength=args.prior_strength).round(4).tolist()
        counts_out[name] = cell["counts"].tolist()
        pd_out[name] = {c: (int(d), int(t)) for c, d, t in
                        zip(CLASSES, cell["detected"], cell["total"])}
    pd_fit = fit_logistic_pd(px_all, hit_all)
    fa_rate = empty_false_alarms / empty_frames if empty_frames else None

    out = {
        "source": (f"measured {date.today().isoformat()}: {Path(args.weights).as_posix()} on "
                   f"{len(records)} held-out target samples (frames shrunk by {args.scales} for "
                   f"range extension) and {empty_frames} empty frames (conf {args.conf}, IoU "
                   f"{args.iou}). Each bin's rows are shrunk toward the pooled matrix with "
                   f"{args.prior_strength:g} pseudo-samples, so sparse rows fall back to it."),
        "range_extension_scales": list(args.scales),
        "detected_over_total_by_scale": {str(s): v for s, v in per_scale.items()},
        "labels": list(CLASSES),
        "true_classes": list(CLASSES),
        "bins_px": list(args.bins),
        "matrices": matrices,
        "counts": counts_out,
        "detected_over_total": pd_out,
        "pd_logistic": pd_fit,
        # Upper bound only: in this dataset many unlabelled frames still contain a bird, and the
        # detector finds it. Inspect the detections before using this as a false-alarm rate.
        "detections_per_unlabelled_frame_upper_bound": fa_rate,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Detection curve: px50 = {pd_fit['px50']}, px_scale = {pd_fit['px_scale']}")
    for s, (d, t) in per_scale.items():
        print(f"  scale {s}: detected {d}/{t} ({d / max(t, 1):.1%})")
    for name, cell in cells.items():
        print(f"  bin {name:8s}: {int(cell['total'].sum())} samples")
    if fa_rate is not None:
        print(f"Detections per unlabelled frame: {fa_rate:.3f} (upper bound on false alarms: "
              "unlabelled frames can still contain a real, unlabelled target; look before using)")
    print("Copy px50 and px_scale into this camera's block in configs/default.yaml, then re-run "
          "scripts/run_ablation.py.")


if __name__ == "__main__":
    main()
