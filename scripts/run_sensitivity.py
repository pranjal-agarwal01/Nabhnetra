"""Sensitivity analysis: does the fusion advantage survive if the simulation's assumptions are wrong?

    python -I scripts/run_sensitivity.py                 # all perturbations, mixed + night, 6 seeds

Every assumption that is not measured on real data (camera optics, radar performance, clutter,
noise, fusion tuning, priors) is changed one at a time to a lower and a higher value. For each
setting the script compares radar alone with radar plus both cameras on the same scenarios and
seeds, and checks whether fusion still identifies more drones, sooner, with better class accuracy.

Outputs in results/sensitivity/: results.csv, SENSITIVITY.md and sensitivity.png.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # use nabhnetra without installing

import numpy as np

from nabhnetra import build_scenario, load_config, run
from nabhnetra.config import get_key, set_key

SENSOR_SETS = {"radar": ["radar"], "radar + eo + ir": ["radar", "eo", "ir"]}

# (label, [config keys], low setting, high setting); a float multiplies, a dict sets the value
PERTURBATIONS = [
    ("Daylight camera pixel angle", ["sensors.eo.ifov_mrad"], 0.5, 2.0),
    ("Thermal camera pixel angle", ["sensors.ir.ifov_mrad"], 0.5, 2.0),
    ("Camera detection curve (px50)", ["sensors.eo.px50", "sensors.ir.px50"], 0.67, 1.5),
    ("Radar detection range (R50)", ["sensors.radar.r50_m"], 0.5, 1.5),
    ("Radar clutter per scan", ["sensors.radar.clutter_per_scan"], {"set": 0}, {"set": 9}),
    ("Radar angle noise", ["sensors.radar.sigma_az_deg", "sensors.radar.sigma_el_deg"], 0.5, 2.0),
    ("Camera label persistence", ["sensors.eo.label_persistence", "sensors.ir.label_persistence"],
     {"set": 0.3}, {"set": 0.9}),
    ("Identity memory (tau)", ["identity.tau_s"], 0.5, 2.0),
    ("Sensor evidence weights", ["identity.sensor_weights"], 0.5, 2.0),
    ("Class prior", ["identity.prior"],
     {"set": {"drone": 0.25, "bird": 0.25, "aircraft": 0.25, "helicopter": 0.25}},
     {"set": {"drone": 0.1, "bird": 0.6, "aircraft": 0.15, "helicopter": 0.15}}),
    ("Tracker manoeuvre noise", ["tracker.imm.q"], 0.5, 2.0),
    ("Speed-based identification", ["identity.kinematics.enabled"], {"set": False},
     {"set": True}),
]
METRICS = [("drones_identified", "Drones identified", 100, "%"),
           ("drone_id_delay_s", "Time to identify a drone", 1, "s"),
           ("class_accuracy", "Class accuracy", 100, "%"),
           ("bird_as_drone", "Bird time labelled drone", 100, "%"),
           ("loc_rmse_m", "Position error", 1, "m")]


def _scaled(value, factor):
    if isinstance(value, dict):
        return {k: _scaled(v, factor) for k, v in value.items()}
    if isinstance(value, list):
        return [_scaled(v, factor) for v in value]
    return value * factor


def perturbed(cfg, keys, setting):
    for key in keys:
        if isinstance(setting, dict):
            cfg = set_key(cfg, key, setting["set"])
        else:
            cfg = set_key(cfg, key, _scaled(get_key(cfg, key), setting))
    return cfg


def describe(setting):
    if isinstance(setting, dict):
        v = setting["set"]
        if isinstance(v, dict):
            return "uniform" if len(set(v.values())) == 1 else "drones rare"
        return str(v).lower()
    return f"x{setting:g}"


def _job(args):
    label, level, cfg, scenario, sensor_set, seed = args
    sc = build_scenario(scenario, seed, cfg["sim"]["dt_s"], cfg["sim"]["duration_s"])
    s = run(sc, cfg, SENSOR_SETS[sensor_set], seed=seed, keep_frames=False).summary
    return {"perturbation": label, "level": level, "scenario": scenario, "sensors": sensor_set,
            "seed": seed, **{k: s[k] for k, *_ in METRICS}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=["mixed", "night"])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results/sensitivity")
    ap.add_argument("--replot", action="store_true",
                    help="rebuild the report and chart from an existing results.csv, no new runs")
    args = ap.parse_args()

    base = load_config()
    variants = [("Baseline", "baseline", base)]
    for label, keys, low, high in PERTURBATIONS:
        for setting in (low, high):
            if setting == {"set": True} and label == "Speed-based identification":
                continue  # that is the baseline already
            variants.append((label, describe(setting), perturbed(base, keys, setting)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        with open(out / "results.csv", encoding="utf-8") as f:
            rows = [{k: (float(v) if k in {m for m, *_ in METRICS} else v) for k, v in r.items()}
                    for r in csv.DictReader(f)]
        args.seeds = len({r["seed"] for r in rows})
    else:
        jobs = [(label, level, cfg, sc, ss, seed) for label, level, cfg in variants
                for sc in args.scenarios for ss in SENSOR_SETS for seed in range(args.seeds)]
        print(f"{len(variants)} settings x {len(args.scenarios)} scenarios x {len(SENSOR_SETS)} "
              f"sensor sets x {args.seeds} seeds = {len(jobs)} runs on {args.workers} workers ...")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(_job, jobs, chunksize=2))
        with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def mean(label, level, scenario, sensors, key):
        v = np.array([r[key] for r in rows if (r["perturbation"], r["level"], r["scenario"],
                                                r["sensors"]) == (label, level, scenario, sensors)],
                     dtype=float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    lines = [f"# Sensitivity analysis ({date.today().isoformat()})", "",
             "Each unmeasured assumption is set lower and higher, one at a time. Cells show "
             f"radar only → radar + EO + IR, mean of {args.seeds} seeds. A row passes when fusion "
             "identifies at least as many drones, identifies them sooner, and has at least the "
             "same class accuracy as radar alone.", ""]
    checks, summary = [], []
    for sc in args.scenarios:
        lines += [f"## {sc}", "", "| Assumption | Setting | " +
                  " | ".join(name for _, name, _, _ in METRICS) + " | Fusion better? |",
                  "|---|---|" + "---:|" * len(METRICS) + ":---:|"]
        for label, level, _ in variants:
            cells, vals = [], {}
            for key, _, scale, unit in METRICS:
                a, b = (mean(label, level, sc, s, key) for s in SENSOR_SETS)
                vals[key] = (a, b)
                fa = "n/a" if np.isnan(a) else f"{a * scale:.0f}{unit}"
                fb = "n/a" if np.isnan(b) else f"{b * scale:.0f}{unit}"
                cells.append(f"{fa} → {fb}")
            (ia, ib), (da, db), (ca, cb) = (vals["drones_identified"], vals["drone_id_delay_s"],
                                            vals["class_accuracy"])
            ok = ib >= ia and (np.isnan(da) or db < da) and cb >= ca
            checks.append(ok)
            summary.append((sc, label, level, ia, ib, da, db))
            lines.append(f"| {label} | {level} | " + " | ".join(cells) +
                         f" | {'yes' if ok else '**no**'} |")
        lines.append("")
    lines.insert(4, f"**Fusion was better in {sum(checks)} of {len(checks)} settings.**")
    lines.insert(5, "")
    (out / "SENSITIVITY.md").write_text("\n".join(lines), encoding="utf-8")

    from nabhnetra.viz import plot_sensitivity

    plot_sensitivity(summary, args.scenarios, out / "sensitivity.png", args.seeds)
    print(f"Fusion better in {sum(checks)}/{len(checks)} settings. Wrote {out / 'SENSITIVITY.md'}")


if __name__ == "__main__":
    main()
