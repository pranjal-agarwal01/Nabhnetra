"""Real two-sensor fusion on your own recording: daylight camera + microphone from one video file.

    python -I scripts/fuse_recording.py my_drone_clip.mp4

Record a drone (or anything flying) with a phone, keeping sound on. The script
  1. runs the daylight detector on the frames,
  2. runs the audio classifier, with its false-alarm rule, on the soundtrack,
  3. fuses both with the same identity-fusion code the simulator uses,
and writes an annotated video with a live timeline plus a summary chart to results/recordings/.

Needs ffmpeg on PATH to read the soundtrack. Both models were trained on one Swedish dataset,
so expect weaker results on different drones, sites and phones; that is worth measuring too.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # use nabhnetra without installing

import cv2
import numpy as np

from nabhnetra import load_config
from nabhnetra.audio import AudioClassifier
from nabhnetra.common import CLASSES, ROOT, Detection
from nabhnetra.confusion import ConfusionModel
from nabhnetra.detect import WEIGHTS, draw, load_model, predict
from nabhnetra.identity import IdentityFusion
from nabhnetra.realdata import AUDIO_LABELS, decide, load_wav
from nabhnetra.tracker import Track

SR = 22050
SERIES = {"camera": (214, 120, 42), "microphone": (122, 175, 27), "fused": (52, 104, 235)}


def extract_audio(video: Path, folder: Path) -> Path | None:
    wav = folder / "audio.wav"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar",
           str(SR), "-acodec", "pcm_s16le", str(wav)]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit("ffmpeg is not on PATH; install it to read the soundtrack")
    except subprocess.CalledProcessError:
        return None  # no audio stream
    return wav if wav.exists() and wav.stat().st_size > 1000 else None


def blank_track() -> Track:
    return Track(tid=0, x=np.zeros(6), P=np.eye(6), t=0.0, born=0.0, last_update=0.0,
                 init_sensor="", hits=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--step", type=float, default=0.5, help="fusion time step in seconds")
    ap.add_argument("--out", default="results/recordings")
    args = ap.parse_args()

    video = Path(args.video)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    fusion = IdentityFusion(cfg["identity"])
    eo_cm = ConfusionModel.load(cfg["sensors"]["eo"]["confusion"])
    ac_cm = ConfusionModel.load(cfg["sensors"]["acoustic"]["confusion"])
    rule = json.loads((ROOT / cfg["sensors"]["acoustic"]["confusion"]).read_text())["decision_rule"]

    # 1. microphone: one label per window, after the false-alarm rule
    audio_reports = []
    with tempfile.TemporaryDirectory() as tmp:
        wav = extract_audio(video, Path(tmp))
        if wav is not None:
            wave, _ = load_wav(wav, SR)
            probs = AudioClassifier().probabilities(wave, SR)
            labels = decide(probs, ["rec"] * len(probs), rule["threshold"], rule["k"], rule["n"],
                            AUDIO_LABELS.index("background"))
            audio_reports = [(0.5 * i + 1.0, int(lab)) for i, lab in enumerate(labels)]
    print(f"microphone: {len(audio_reports)} windows"
          + ("" if audio_reports else " (no soundtrack found; camera only)"))

    # 2. camera: detector on every frame; 3. fuse both on a common clock
    model = load_model(WEIGHTS["eo"])
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracks = {name: blank_track() for name in SERIES}
    history = {name: [] for name in SERIES}
    times, frames_out, writer = [], [], None
    next_step, k, a_idx = 0.0, 0, 0
    step_dets = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = k / fps
        preds = predict(model, frame, args.conf)
        if preds:
            box, name, conf = max(preds, key=lambda p: p[2])
            b = eo_cm.bin_for(max(box[2], box[3]))
            step_dets.append(eo_cm.likelihood(CLASSES.index(name), b))
        if t >= next_step:
            if step_dets:  # strongest camera evidence in this step, one report per step
                lik = step_dets[-1]
                for key in ("camera", "fused"):
                    fusion.update(tracks[key], Detection(t, "eo", "ae", np.zeros(2), np.eye(2),
                                                         np.zeros(3), lik))
            while a_idx < len(audio_reports) and audio_reports[a_idx][0] <= t:
                lab = audio_reports[a_idx][1]
                if lab != AUDIO_LABELS.index("background"):
                    lik = ac_cm.likelihood(lab, ac_cm.default_bin)
                    for key in ("microphone", "fused"):
                        fusion.update(tracks[key], Detection(t, "acoustic", "ae", np.zeros(2),
                                                             np.eye(2), np.zeros(3), lik))
                a_idx += 1
            for key in SERIES:
                history[key].append(fusion.posterior(tracks[key], t)[0])
            times.append(t)
            step_dets, next_step = [], next_step + args.step

        img = draw(frame, preds, None, f"daylight model + microphone | {video.name} | {t:5.1f} s",
                   max(1.0, 640.0 / frame.shape[1]))
        img = np.vstack([img, timeline(history, img.shape[1])])
        if writer is None:
            writer = cv2.VideoWriter(str(out / f"{video.stem}_fused.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, img.shape[1::-1])
        writer.write(img)
        k += 1
    cap.release()
    if writer is None:
        raise SystemExit(f"could not read frames from {video}")
    writer.release()

    final = {key: dict(zip(CLASSES, np.round(history[key][-1], 3).tolist())) for key in SERIES}
    (out / f"{video.stem}_fused.json").write_text(json.dumps(
        {"video": str(video), "seconds": times[-1], "final_probabilities": final}, indent=2),
        encoding="utf-8")
    summary_chart(times, history, out / f"{video.stem}_fused.png", video.name)
    for key in SERIES:
        p = history[key][-1]
        print(f"{key:10s}: {CLASSES[int(np.argmax(p))]} ({p.max():.2f})")
    print(f"Saved {out / (video.stem + '_fused.mp4')} and a summary chart")


def timeline(history, width, height=120):
    """Strip under the video: P(drone) over time for camera, microphone and fused."""
    strip = np.full((height, width, 3), 24, np.uint8)
    n = len(history["fused"])
    cv2.putText(strip, "P(drone)", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    for frac in (0.0, 0.5, 1.0):
        y = int(height - 10 - frac * (height - 30))
        cv2.line(strip, (40, y), (width - 8, y), (60, 60, 60), 1)
    x0 = 40
    for i, (key, color) in enumerate(SERIES.items()):
        lx = x0 + 110 * i + 40
        cv2.line(strip, (lx, 10), (lx + 18, 10), color, 2, cv2.LINE_AA)
        cv2.putText(strip, key, (lx + 24, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        if n < 2:
            continue
        xs = np.linspace(x0, width - 8, max(n, 2))[:n]
        ys = height - 10 - np.array([h[0] for h in history[key]]) * (height - 30)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(strip, [pts], False, color, 2, cv2.LINE_AA)
    return strip


def summary_chart(times, history, path, title):
    from nabhnetra.viz import GROUP, INK, apply_style, plt

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=110)
    styles = {"camera": (GROUP["manned"], "-"), "microphone": (GROUP["bird"], "--"),
              "fused": (GROUP["drone"], "-")}
    for key, (color, ls) in styles.items():
        ax.plot(times, [h[0] for h in history[key]], color=color, ls=ls, lw=2, label=key)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("P(drone)")
    ax.grid(color=INK["grid"], lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.set_title(f"Camera, microphone and fused belief that it is a drone: {title}", loc="left",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
