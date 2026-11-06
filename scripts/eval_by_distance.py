"""Score a trained detector by the dataset's own distance category (close / medium / distant).

    python -I scripts/eval_by_distance.py --model ir
    python -I scripts/eval_by_distance.py --model eo

The dataset's spreadsheet gives every video a distance bin. This checks, on real held-out
footage, how detection and classification fall off with range, as a cross-check on the
shrunk-frame range extension used in eval_confusion.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # use nabhnetra without installing

import cv2

from nabhnetra.common import CLASSES
from nabhnetra.detect import WEIGHTS, load_model
from nabhnetra.realdata import match_frame

SHEET = "data/raw/Drone-detection-dataset-master/Data/Video_dataset_description.xlsx"
CLASS_OF = {"DRONE": "drone", "BIRD": "bird", "AIRPLANE": "aircraft", "HELICOPTER": "helicopter"}


def distance_bins(xlsx: str) -> dict[str, str]:
    import pandas as pd

    df = pd.read_excel(xlsx)
    df.columns = [str(c).strip() for c in df.columns]
    dist_col = next(c for c in df.columns if c.upper().startswith("DISTANCE"))
    return {f"{r['SENSOR']}_{r['CLASS']}_{int(r['NUMBER']):03d}": str(r[dist_col]).strip().lower()
            for _, r in df.iterrows()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", choices=["ir", "eo"], required=True)
    ap.add_argument("--data", help="prepared dataset folder (default: data/yolo_ir or data/yolo_v)")
    ap.add_argument("--xlsx", default=SHEET)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/distance")
    args = ap.parse_args()

    data = Path(args.data or ("data/yolo_ir" if args.model == "ir" else "data/yolo_v"))
    bins = distance_bins(args.xlsx)
    model = load_model(WEIGHTS[args.model])
    images = sorted((data / "val" / "images").glob("*.jpg"))
    stats = defaultdict(lambda: {"frames": 0, "found": 0, "correct": 0})
    videos = defaultdict(set)
    for start in range(0, len(images), 64):
        chunk = images[start:start + 64]
        for path, res in zip(chunk, model.predict([str(p) for p in chunk], conf=args.conf,
                                                  verbose=False, device=args.device)):
            label = (path.parent.parent / "labels" / f"{path.stem}.txt").read_text().strip()
            if not label:
                continue
            h, w = res.orig_shape
            c, cx, cy, bw, bh = (float(v) for v in label.split())
            box = ((cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h)
            preds = [((b[0], b[1], b[2] - b[0], b[3] - b[1]), int(k), float(s))
                     for b, k, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                                        res.boxes.conf.tolist())]
            stem = path.stem.rsplit("_", 1)[0]
            dist = bins.get(stem, "unknown")
            pred = match_frame(box, preds, args.iou)
            for key in ((dist, "all"), (dist, CLASSES[int(c)])):
                s = stats[key]
                s["frames"] += 1
                s["found"] += int(pred is not None)
                s["correct"] += int(pred == int(c))
            videos[dist].add(stem)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = {f"{d}|{c}": v for (d, c), v in stats.items()}
    (out / f"{args.model}.json").write_text(json.dumps(
        {"model": args.model, "videos_per_bin": {k: len(v) for k, v in videos.items()},
         "stats": table}, indent=2), encoding="utf-8")
    name = "Thermal (IR)" if args.model == "ir" else "Daylight (EO)"
    lines = [f"### {name}", "", "| Distance bin | Videos | Frames | Target found | Right class "
             "(of found) | Found, by class |", "|---|---:|---:|---:|---:|---|"]
    for d in ("close", "medium", "distant"):
        a = stats.get((d, "all"))
        if not a or not a["frames"]:
            continue
        per = ", ".join(f"{c} {stats[(d, c)]['found'] / stats[(d, c)]['frames']:.0%}"
                        for c in CLASSES if stats.get((d, c), {}).get("frames"))
        lines.append(f"| {d} | {len(videos[d])} | {a['frames']} | {a['found'] / a['frames']:.0%} "
                     f"| {a['correct'] / max(a['found'], 1):.0%} | {per} |")
    (out / f"{args.model}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
