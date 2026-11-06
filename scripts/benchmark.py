"""How fast does it run? Detector latency (GPU and CPU) and fusion time per update.

    python -I scripts/benchmark.py

Each sensor-fusion step covers 0.5 s of sensor data, so a step faster than 500 ms keeps up
with real time. Run it on an otherwise idle machine; other load distorts the numbers.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # use nabhnetra without installing

import cv2
import numpy as np

from nabhnetra import build_scenario, load_config, run
from nabhnetra.detect import WEIGHTS, load_model

SAMPLES = {"ir": "samples/IR_DRONE_001.mp4", "eo": "samples/V_DRONE_026.mp4"}


def first_frame(path: str):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read {path}")
    return frame


def time_detector(model, frame, device, n=100, half=False):
    for _ in range(10):
        model.predict(frame, verbose=False, device=device, half=half)
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(frame, verbose=False, device=device, half=half)
    return (time.perf_counter() - t0) * 1000.0 / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="timed frames per setting")
    ap.add_argument("--out", default="results/benchmark")
    args = ap.parse_args()

    import torch

    rows = []
    devices = (["0"] if torch.cuda.is_available() else []) + ["cpu"]
    for name in ("ir", "eo"):
        frame = first_frame(SAMPLES[name])
        for dev in devices:
            for half in ([False, True] if dev != "cpu" else [False]):
                # a fresh model per setting: Ultralytics fixes the device and precision the first
                # time a model predicts, and silently ignores later changes
                model = load_model(WEIGHTS[name])
                ms = time_detector(model, frame, dev, args.n, half)
                label = (torch.cuda.get_device_name(0) if dev != "cpu" else platform.processor()
                         or "CPU")
                rows.append({"part": f"{'thermal' if name == 'ir' else 'daylight'} detector",
                             "device": label + (" (FP16)" if half else ""),
                             "ms": round(ms, 1), "fps": round(1000.0 / ms, 1)})
                print(rows[-1])

    cfg = load_config()
    for scenario in ("mixed", "swarm"):
        sc = build_scenario(scenario, 1, cfg["sim"]["dt_s"], cfg["sim"]["duration_s"])
        s = run(sc, cfg, None, seed=1, keep_frames=False).summary
        rows.append({"part": f"fusion step, {scenario} ({len(sc.specs)} targets, all sensors)",
                     "device": "CPU, one core", "ms": round(s["step_ms_mean"], 1),
                     "p95_ms": round(s["step_ms_p95"], 1),
                     "realtime_margin": round(500.0 / s["step_ms_mean"], 1)})
        print(rows[-1])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["| Part | Hardware | Time per frame/step | Rate |", "|---|---|---:|---:|"]
    for r in rows:
        rate = f"{r['fps']} frames/s" if "fps" in r else f"{r['realtime_margin']}x faster than real time"
        extra = f" (p95 {r['p95_ms']} ms)" if "p95_ms" in r else ""
        lines.append(f"| {r['part']} | {r['device']} | {r['ms']} ms{extra} | {rate} |")
    (out / "BENCHMARK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
