"""Convert the Svanstrom drone dataset videos + MATLAB labels into a YOLO detection dataset.

    python scripts/prepare_svanstrom.py --data path/to/Drone-detection-dataset/Data --modality IR
    python scripts/prepare_svanstrom.py --data path/to/Drone-detection-dataset/Data --modality V

The split is by whole video (stratified by class), so frames from one clip never appear in
both train and val. Otherwise validation scores are inflated by near-duplicate frames.

The label files are MATLAB groundTruth objects, which scipy cannot read. This script uses the
`mcos-decoder` package that the dataset's README points to (pip install mcos-decoder). It is a
young package (v0.1.0, April 2026): read its source before installing. Its README says it was
validated on the 365 IR label files. Checked here on 6 Nov 2026: all 650 files decode, and 18
(mostly bird videos) give a label count that does not match the frame count, because frames with
two boxes are skipped and the rest shift. Those videos are excluded automatically. In the other
videos the boxes sit on their targets; visible-camera boxes are loose (about 1.5-3x the target).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # use nabhnetra without installing
import cv2

from nabhnetra.common import CLASSES
from nabhnetra.realdata import parse_video_name, split_by_group


def load_boxes(label_path: Path):
    try:
        from mcos_decoder import load_groundtruth
    except ImportError as exc:
        raise SystemExit("Install the label decoder first: pip install mcos-decoder") from exc
    return load_groundtruth(str(label_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, help="the dataset's Data/ folder")
    ap.add_argument("--modality", choices=["IR", "V"], required=True)
    ap.add_argument("--out", help="output folder (default: data/yolo_<modality>)")
    ap.add_argument("--every", type=int, default=5, help="keep every Nth frame (default 5)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offset", type=float, default=0.5,
                    help="subtracted from x and y: MATLAB spatial coordinates put the image's "
                         "top-left corner at (0.5, 0.5). Check a few boxes visually.")
    args = ap.parse_args()

    folder = Path(args.data) / ("Video_IR" if args.modality == "IR" else "Video_V")
    videos = sorted(p for p in folder.glob(f"{args.modality}_*.mp4") if parse_video_name(p.stem))
    if not videos:
        raise SystemExit(f"No {args.modality} videos found in {folder}")
    out = Path(args.out or f"data/yolo_{args.modality.lower()}")

    # Decode every label file first. The decoder finds frames by byte size, so a frame with two
    # boxes is skipped and every later label shifts onto the wrong frame. A label count that
    # differs from the frame count is the symptom; those videos are left out entirely.
    labels, excluded = {}, []
    for video in videos:
        label_path = video.with_name(video.stem + "_LABELS.mat")
        if not label_path.exists():
            excluded.append({"video": video.stem, "reason": "no label file"})
            continue
        boxes = load_boxes(label_path)
        cap = cv2.VideoCapture(str(video))
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if abs(n_video - len(boxes)) > 1:
            excluded.append({"video": video.stem, "reason": f"{n_video} frames, {len(boxes)} labels"})
            continue
        labels[video.stem] = boxes
    print(f"{len(labels)} videos usable, {len(excluded)} excluded (label/frame mismatch)")

    stems = sorted(labels)
    classes = [parse_video_name(s)[1] for s in stems]
    train, val = split_by_group(stems, classes, args.val_frac, args.seed)
    split_of = {s: "train" for s in train} | {s: "val" for s in val}

    stats = {"train": {"frames": 0, "targets": 0}, "val": {"frames": 0, "targets": 0}}
    for video in videos:
        if video.stem not in labels:
            continue
        _, cls, _ = parse_video_name(video.stem)
        boxes = labels[video.stem]
        split = split_of[video.stem]
        img_dir, lab_dir = out / split / "images", out / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or i >= len(boxes):
                break
            if i % args.every == 0:
                h, w = frame.shape[:2]
                name = f"{video.stem}_{i:05d}"
                cv2.imwrite(str(img_dir / f"{name}.jpg"), frame)
                line = ""
                if boxes[i] is not None:
                    x, y, bw, bh = (float(v) for v in boxes[i])
                    x, y = x - args.offset, y - args.offset
                    x0, y0 = max(0.0, x), max(0.0, y)
                    x1, y1 = min(float(w), x + bw), min(float(h), y + bh)
                    if x1 > x0 and y1 > y0:
                        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
                        line = (f"{CLASSES.index(cls)} {cx:.6f} {cy:.6f} "
                                f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n")
                        stats[split]["targets"] += 1
                # an empty label file marks a background frame, which teaches the detector restraint
                (lab_dir / f"{name}.txt").write_text(line, encoding="utf-8")
                stats[split]["frames"] += 1
            i += 1
        cap.release()
        print(f"  {video.name:24s} -> {split:5s} ({i} frames read)")

    yaml_text = (f"path: {out.resolve().as_posix()}\ntrain: train/images\nval: val/images\n"
                 "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    (out / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    (out / "split.json").write_text(json.dumps({"train": train, "val": val, "excluded": excluded,
                                                "every": args.every, "seed": args.seed},
                                               indent=2), encoding="utf-8")
    print(f"\nWrote {out / 'dataset.yaml'}")
    print(f"train: {stats['train']}  val: {stats['val']}")
    for e in excluded:
        print(f"  excluded {e['video']}: {e['reason']}")


if __name__ == "__main__":
    main()
