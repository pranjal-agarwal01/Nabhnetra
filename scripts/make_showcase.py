"""Tile the annotated sample clips into one video and one GIF, for a README or a demo video.

    python -I scripts/detect_video.py samples       # first, writes results/detections/*_detected.mp4
    python -I scripts/make_showcase.py              # then, writes results/detections/showcase.mp4 and .gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ORDER = ["IR_DRONE_001", "IR_BIRD_020", "IR_AIRPLANE_006", "IR_HELICOPTER_026",
         "V_DRONE_026", "V_BIRD_028", "V_AIRPLANE_009", "V_HELICOPTER_007"]


def read_all(path: Path, size):
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA))
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dir", default="results/detections")
    ap.add_argument("--tile", type=int, nargs=2, default=[480, 384], metavar=("W", "H"))
    ap.add_argument("--gif-every", type=int, default=6, help="keep every Nth frame in the GIF")
    args = ap.parse_args()

    d = Path(args.dir)
    clips = [read_all(d / f"{name}_detected.mp4", tuple(args.tile)) for name in ORDER
             if (d / f"{name}_detected.mp4").exists()]
    if len(clips) != len(ORDER):
        raise SystemExit("Run scripts/detect_video.py samples first.")
    n = min(len(c) for c in clips)
    grid = [np.vstack([np.hstack([c[i] for c in clips[:4]]), np.hstack([c[i] for c in clips[4:]])])
            for i in range(n)]

    h, w = grid[0].shape[:2]
    writer = cv2.VideoWriter(str(d / "showcase.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for g in grid:
        writer.write(g)
    writer.release()

    gif = [Image.fromarray(cv2.cvtColor(cv2.resize(g, (w // 2, h // 2), interpolation=cv2.INTER_AREA),
                                        cv2.COLOR_BGR2RGB)) for g in grid[:: args.gif_every]]
    gif[0].save(d / "showcase.gif", save_all=True, append_images=gif[1:], duration=200, loop=0)
    print(f"Wrote {d / 'showcase.mp4'} ({n} frames) and {d / 'showcase.gif'} ({len(gif)} frames)")


if __name__ == "__main__":
    main()
