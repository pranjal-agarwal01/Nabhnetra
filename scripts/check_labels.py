"""Audit the dataset's labels before training on them.

    python -I scripts/check_labels.py --data data/raw/Drone-detection-dataset-master/Data

For each camera it decodes every label file, compares the label count with the video's frame
count (a mismatch means the labels are shifted and the video must be excluded), reports target
sizes in pixels, and saves a grid of random frames with their boxes drawn so you can check the
alignment by eye. Output goes to results/label_check/.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # use nabhnetra without installing

import cv2
import numpy as np

from nabhnetra.realdata import parse_video_name


def frame_count(video: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def tile(video: Path, boxes, offset: float, size: int = 240):
    idx = [i for i, b in enumerate(boxes) if b is not None]
    if not idx:
        return None
    i = idx[len(idx) // 2]
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    x, y, w, h = boxes[i]
    x0, y0 = x - offset, y - offset
    half = int(max(w, h) * 2 + 16)
    cx, cy = int(x0 + w / 2), int(y0 + h / 2)
    H, W = frame.shape[:2]
    ox, oy = max(0, cx - half), max(0, cy - half)
    crop = frame[oy:min(H, cy + half), ox:min(W, cx + half)].copy()
    cv2.rectangle(crop, (int(x0 - ox), int(y0 - oy)), (int(x0 + w - ox), int(y0 + h - oy)),
                  (0, 255, 0), 1)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((size + 22, size, 3), np.uint8)
    out[:size] = crop
    cv2.putText(out, f"{video.stem} f{i} {w:.0f}x{h:.0f}", (3, size + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, help="the dataset's Data/ folder")
    ap.add_argument("--out", default="results/label_check")
    ap.add_argument("--samples", type=int, default=12, help="frames to draw per camera")
    ap.add_argument("--offset", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from mcos_decoder import load_groundtruth

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    report = {}
    for folder, prefix in (("Video_IR", "IR"), ("Video_V", "V")):
        videos = sorted(v for v in (Path(args.data) / folder).glob(f"{prefix}_*.mp4")
                        if parse_video_name(v.stem))
        good, bad, sizes = {}, {}, []
        for v in videos:
            boxes = load_groundtruth(str(v.with_name(v.stem + "_LABELS.mat")))
            n = frame_count(v)
            if abs(n - len(boxes)) > 1:
                bad[v.stem] = {"frames": n, "labels": len(boxes)}
                continue
            good[v.stem] = boxes
            sizes += [max(b[2], b[3]) for b in boxes if b is not None]
        sizes = np.array(sizes)
        report[prefix] = {
            "videos": len(videos), "usable": len(good), "mismatched": bad,
            "box_px_median": round(float(np.median(sizes)), 1),
            "box_px_p10": round(float(np.percentile(sizes, 10)), 1),
            "box_px_p90": round(float(np.percentile(sizes, 90)), 1),
            "share_under_8px": round(float((sizes < 8).mean()), 4),
        }
        print(f"{folder}: {len(good)}/{len(videos)} usable, {len(bad)} mismatched; box size "
              f"median {report[prefix]['box_px_median']} px, "
              f"{report[prefix]['share_under_8px']:.1%} under 8 px")
        tiles = []
        for stem in rng.sample(sorted(good), min(len(good), args.samples * 2)):
            t = tile(Path(args.data) / folder / f"{stem}.mp4", good[stem], args.offset)
            if t is not None:
                tiles.append(t)
            if len(tiles) == args.samples:
                break
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)])
        cv2.imwrite(str(out / f"boxes_{prefix}.png"), grid)
    (out / "label_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out / 'label_report.json'} and box grids; open them and check every box.")


if __name__ == "__main__":
    main()
