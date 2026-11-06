"""Sensor ablation: run every sensor combination on every scenario over several seeds.

This produces the evidence for the claim "fusion beats any single sensor".

    python scripts/run_ablation.py                         # all scenarios, 5 seeds
    python scripts/run_ablation.py --scenarios mixed night --seeds 3

Outputs in results/ablation/: results.csv (one row per run), RESULTS.md (tables), and one
chart per scenario.
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # use nabhnetra without installing
import numpy as np

from nabhnetra import SCENARIOS, build_scenario, load_config, run
from nabhnetra.viz import plot_ablation

CONFIGS = {
    "radar": ["radar"],
    "eo + ir": ["eo", "ir"],
    "acoustic": ["acoustic"],
    "radar + eo": ["radar", "eo"],
    "radar + ir": ["radar", "ir"],
    "radar + eo + ir": ["radar", "eo", "ir"],
    "all": ["radar", "eo", "ir", "acoustic", "adsb"],
}

TABLE = [  # (key, header, scale, digits)
    ("drones_identified", "Drones identified %", 100, 0),
    ("drone_id_delay_s", "Time to ID drone (s)", 1, 0),
    ("completeness", "Track coverage %", 100, 0),
    ("loc_rmse_m", "Position RMSE (m)", 1, 1),
    ("false_tracks_per_hour", "False tracks / h", 1, 0),
    ("class_accuracy", "Class accuracy %", 100, 0),
    ("bird_as_drone", "Bird as drone %", 100, 1),
    ("gospa_m", "GOSPA (m)", 1, 0),
]


def _job(args):
    scenario, config_name, sensors, seed, cfg = args
    sc = build_scenario(scenario, seed, cfg["sim"]["dt_s"], cfg["sim"]["duration_s"])
    res = run(sc, cfg, sensors, seed=seed, keep_frames=False)
    placeholder = any("PLACEHOLDER" in s.upper() for s in res.confusion_sources.values())
    return {"scenario": scenario, "config": config_name, "seed": seed,
            "placeholder": placeholder, **res.summary}


def _mean_std(rows, key):
    vals = np.array([r[key] for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=sorted(SCENARIOS))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config values, e.g. tracker.model=imm")
    ap.add_argument("--out", default="results/ablation")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    jobs = [(s, c, CONFIGS[c], seed, cfg) for s in args.scenarios for c in args.configs
            for seed in range(args.seeds)]
    print(f"{len(jobs)} runs on {args.workers} workers ...")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_job, jobs, chunksize=2))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    placeholder = any(r["placeholder"] for r in rows)
    lines = [f"# Sensor ablation results ({date.today().isoformat()})", ""]
    if placeholder:
        lines += ["> **Warning:** at least one sensor used a PLACEHOLDER confusion matrix. These "
                  "numbers show the pipeline works; they are not results. Measure the matrices "
                  "on real data (see README) and re-run before quoting anything.", ""]
    lines += [f"{args.seeds} seeds per cell. Mean ± standard deviation. "
              "Metric definitions are in docs/DESIGN.md.", ""]
    for s in args.scenarios:
        lines += [f"## {s}", "", f"_{SCENARIOS[s]}_", ""]
        lines.append("| Sensors | " + " | ".join(h for _, h, _, _ in TABLE) + " |")
        lines.append("|---|" + "---:|" * len(TABLE))
        stats = {}
        for c in args.configs:
            cell_rows = [r for r in rows if r["scenario"] == s and r["config"] == c]
            stats[c] = {k: _mean_std(cell_rows, k) for k in
                        {k for k, *_ in TABLE} | {"bird_as_drone", "loc_rmse_m"}}
            cells = []
            for key, _, scale, digits in TABLE:
                m, sd = stats[c][key]
                cells.append("n/a" if not np.isfinite(m)
                             else f"{m * scale:.{digits}f} ± {sd * scale:.{digits}f}")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines.append("")
        chart = out / f"ablation_{s}.png"
        plot_ablation(stats, s, chart, n_seeds=args.seeds, placeholder=placeholder)
        lines += [f"![{s} ablation](ablation_{s}.png)", ""]

    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out / 'RESULTS.md'}, {out / 'results.csv'} and {len(args.scenarios)} charts")
    if placeholder:
        print("Note: placeholder confusion matrices were used; numbers are illustrative only.")


if __name__ == "__main__":
    main()
