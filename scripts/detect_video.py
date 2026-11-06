"""Run a trained detector on a video, an image, a folder of either, or a webcam.

    python -I scripts/detect_video.py samples/IR_DRONE_001.mp4      # thermal model picked from the name
    python -I scripts/detect_video.py samples                       # every clip in the folder
    python -I scripts/detect_video.py photo.jpg --model eo
    python -I scripts/detect_video.py --webcam 0                    # live daylight model; q to quit

Writes an annotated MP4 (or PNG) and a contact sheet to results/detections/. Coloured boxes are
the detector's output; a thin white box is the ground-truth label, drawn when the clip's
_LABELS.mat file sits next to it, in which case the clip is also scored.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # use nabhnetra without installing

import cv2

from nabhnetra.detect import (WEIGHTS, ClipScore, contact_sheet, draw, load_model, load_truth,
                             pick_weights, predict)

VIDEO = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE = {".jpg", ".jpeg", ".png", ".bmp"}


def run_video(model, path: Path, out: Path, args, label: str):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    truth = None if args.no_truth else load_truth(path)
    score = ClipScore()
    frames, writer, i = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        preds = predict(model, frame, args.conf, args.device)
        t = truth[i] if truth is not None and i < len(truth) else None
        if truth is not None:
            score.add(preds, t)
        scale = max(1.0, 640.0 / frame.shape[1])
        img = draw(frame, preds, t, f"{label} | {path.stem} | frame {i + 1}", scale)
        if writer is None:
            writer = cv2.VideoWriter(str(out / f"{path.stem}_detected.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, img.shape[1::-1])
        writer.write(img)
        frames.append(img)
        i += 1
    cap.release()
    if writer is None:
        print(f"  {path.name}: could not read any frames")
        return
    writer.release()
    cv2.imwrite(str(out / f"{path.stem}_sheet.jpg"), contact_sheet(frames))
    result = score.summary() if truth is not None else "no labels next to the clip, not scored"
    print(f"  {path.name}: {i} frames; {result}")


def run_webcam(model, index: int, out: Path, args):
    cap = cv2.VideoCapture(index)
    writer = None
    print("Press q in the video window to stop.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = draw(frame, predict(model, frame, args.conf, args.device), title="daylight model | webcam")
        if writer is None:
            writer = cv2.VideoWriter(str(out / "webcam_detected.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 20.0, img.shape[1::-1])
        writer.write(img)
        cv2.imshow("Nabhnetra", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("source", nargs="?", help="video, image or folder")
    ap.add_argument("--webcam", type=int, help="camera index for a live test (daylight model)")
    ap.add_argument("--model", choices=["ir", "eo"], help="force the thermal or daylight model")
    ap.add_argument("--weights", help="path to other weights")
    ap.add_argument("--conf", type=float, default=0.25, help="minimum confidence to draw a box")
    ap.add_argument("--device", default=None, help="0 for the GPU, cpu to force CPU")
    ap.add_argument("--no-truth", action="store_true", help="don't draw or score ground truth")
    ap.add_argument("--out", default="results/detections")
    args = ap.parse_args()
    if args.source is None and args.webcam is None:
        ap.error("give a video/image/folder, or --webcam 0")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.webcam is not None:
        run_webcam(load_model(args.weights or WEIGHTS[args.model or "eo"]), args.webcam, out, args)
        return

    src = Path(args.source)
    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO | IMAGE) if src.is_dir() \
        else [src]
    models = {}
    for path in paths:
        weights = Path(args.weights) if args.weights else (
            WEIGHTS[args.model] if args.model else pick_weights(str(path)))
        if weights not in models:
            models[weights] = load_model(weights)
        label = "thermal model" if weights == WEIGHTS["ir"] else (
            "daylight model" if weights == WEIGHTS["eo"] else weights.name)
        if path.suffix.lower() in VIDEO:
            run_video(models[weights], path, out, args, label)
        else:
            frame = cv2.imread(str(path))
            preds = predict(models[weights], frame, args.conf, args.device)
            scale = max(1.0, 640.0 / frame.shape[1])
            cv2.imwrite(str(out / f"{path.stem}_detected.png"),
                        draw(frame, preds, None, f"{label} | {path.name}", scale))
            print(f"  {path.name}: " + (", ".join(f"{n} {c:.2f}" for _, n, c in preds) or "nothing"))
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
