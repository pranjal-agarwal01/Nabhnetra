"""Run one scenario and save the operator air picture.

    python scripts/run_demo.py --scenario mixed --seed 1
    python scripts/run_demo.py --scenario night --gif
    python scripts/run_demo.py --scenario radar_failure --sensors radar eo ir
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # use nabhnetra without installing

from nabhnetra import SCENARIOS, build_scenario, load_config, run
from nabhnetra.viz import AirPicture, busiest_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scenario", default="mixed", choices=sorted(SCENARIOS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sensors", nargs="+", help="subset of sensors (default: all configured)")
    ap.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config values, e.g. tracker.model=imm")
    ap.add_argument("--gif", action="store_true", help="also save an animated GIF")
    ap.add_argument("--at", type=float, help="time in seconds for the still (default: busiest)")
    ap.add_argument("--out", default="results/demo")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    sc = build_scenario(args.scenario, args.seed, cfg["sim"]["dt_s"], cfg["sim"]["duration_s"])
    print(f"Scenario '{sc.name}': {SCENARIOS[sc.name]}")
    print(f"{len(sc.specs)} targets, {len(sc.times)} steps of {sc.dt} s")
    result = run(sc, cfg, args.sensors, seed=args.seed)

    for name, src in result.confusion_sources.items():
        if "PLACEHOLDER" in src.upper():
            print(f"  note: {name} uses a placeholder confusion matrix; numbers are illustrative")

    print("\nSummary")
    for k, v in result.summary.items():
        print(f"  {k:24s} {v:10.3f}" if isinstance(v, float) else f"  {k:24s} {v:>10}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{sc.name}_seed{args.seed}" + ("_" + "-".join(args.sensors) if args.sensors else "")
    (out / f"summary_{tag}.json").write_text(json.dumps(
        {"scenario": sc.name, "seed": args.seed, "sensors": result.sensors,
         "summary": result.summary, "fused_confusion": result.fused_confusion,
         "confusion_sources": result.confusion_sources}, indent=2), encoding="utf-8")

    pic = AirPicture(result)
    k = busiest_frame(result) if args.at is None else int(round(args.at / sc.dt))
    png = out / f"airpicture_{tag}.png"
    pic.save_png(png, k)
    print(f"\nSaved {png}")
    if args.gif:
        gif = out / f"airpicture_{tag}.gif"
        pic.save_gif(gif)
        print(f"Saved {gif}")
    pic.close()


if __name__ == "__main__":
    main()
