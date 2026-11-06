"""Static figures: the operator air picture (PNG or GIF) and the sensor ablation charts.

Colours come from a validated categorical palette (three slots, safe for all pairs under
colour-vision deficiency). Aircraft and helicopters share the "manned" hue and are told apart
by marker shape; every track also carries a text label, so identity never relies on colour alone.
"""
from __future__ import annotations

import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

INK = {
    "surface": "#fcfcfb",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}
GROUP = {"manned": "#2a78d6", "drone": "#eb6834", "bird": "#1baf7a"}
CLASS_STYLE = {
    "drone": (GROUP["drone"], "^"),
    "bird": (GROUP["bird"], "o"),
    "aircraft": (GROUP["manned"], "s"),
    "helicopter": (GROUP["manned"], "D"),
}
CRITICAL = "#d03b3b"  # status colour, only for the HIGH-threat ring, always with a text label
SENSOR_CODE = {"radar": "R", "eo": "E", "ir": "I", "acoustic": "A", "adsb": "B"}


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
        "axes.edgecolor": INK["axis"],
        "axes.labelcolor": INK["secondary"],
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "text.color": INK["primary"],
        "axes.titlecolor": INK["primary"],
    })


# --- air picture -----------------------------------------------------------------------------

class AirPicture:
    """Top-down plan view (east/north, km) with the track table beside it."""

    def __init__(self, result, extent_km: float = 4.5, trail_s: float = 30.0, max_rows: int = 12):
        apply_style()
        self.r = result
        self.sc = result.scenario
        self.extent = extent_km
        self.trail_steps = int(trail_s / self.sc.dt)
        self.max_rows = max_rows
        self.fig = plt.figure(figsize=(13.0, 7.4), dpi=100)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.0, 0.78], wspace=0.03,
                                   left=0.04, right=0.98, top=0.93, bottom=0.07)
        self.ax = self.fig.add_subplot(gs[0])
        self.panel = self.fig.add_subplot(gs[1])

    def _histories(self, k):
        hist = defaultdict(list)
        for f in self.r.frames[max(0, k - self.trail_steps): k + 1]:
            for v in f.tracks:
                hist[v.tid].append(v.pos[:2] / 1000.0)
        return hist

    def draw(self, k: int):
        ax, panel, frame = self.ax, self.panel, self.r.frames[k]
        ax.clear()
        panel.clear()
        panel.axis("off")
        e = self.extent

        # range rings and cardinal lines
        for rk in range(1, int(e) + 1):
            ax.add_patch(plt.Circle((0, 0), rk, fill=False, lw=0.8, color=INK["grid"]))
            ax.text(rk * 0.707 + 0.04, rk * 0.707 + 0.04, f"{rk} km", fontsize=8, color=INK["muted"])
        ax.plot([-e, e], [0, 0], lw=0.8, color=INK["grid"])
        ax.plot([0, 0], [-e, e], lw=0.8, color=INK["grid"])
        ax.text(0, e - 0.15, "N", ha="center", va="top", fontsize=9, color=INK["muted"])
        ax.plot(0, 0, marker="s", ms=7, mfc=INK["surface"], mec=INK["primary"], mew=1.5)

        # ground truth, faint
        lo = max(0, k - self.trail_steps)
        for i in np.flatnonzero(self.sc.alive(k)):
            trail = self.sc.states[i, lo:k + 1, :2] / 1000.0
            ax.plot(trail[:, 0], trail[:, 1], lw=0.8, color=INK["axis"], zorder=1)
            ax.plot(*trail[-1], marker="o", ms=4, mfc="none", mec=INK["muted"], mew=0.8, zorder=2)

        # fused tracks
        hist = self._histories(k)
        ranked = sorted(frame.tracks, key=lambda v: -v.threat.score)
        for v in frame.tracks:
            color, marker = CLASS_STYLE[v.label]
            h = np.array(hist[v.tid])
            if len(h) > 1:
                ax.plot(h[:, 0], h[:, 1], lw=1.5, color=color, alpha=0.55, zorder=3)
            p, lead = v.pos[:2] / 1000.0, v.vel[:2] * 20.0 / 1000.0
            ax.plot([p[0], p[0] + lead[0]], [p[1], p[1] + lead[1]], lw=1.5, color=color, zorder=3)
            if v.threat.level == "HIGH":
                ax.plot(*p, marker="o", ms=17, mfc="none", mec=CRITICAL, mew=1.8, zorder=4)
            ax.plot(*p, marker=marker, ms=9, mfc=color, mec=INK["surface"], mew=1.6, zorder=5)

        ax.set_xlim(-e, e)
        ax.set_ylim(-e, e)
        ax.set_aspect("equal")
        top = [v for v in ranked[:6] if v.threat.level != "NONE"]
        avoid = [v.pos[:2] / 1000.0 for v in frame.tracks]
        avoid += [np.array([rk * 0.707 + dx, rk * 0.707 + 0.08]) for rk in range(1, int(e) + 1)
                  for dx in (0.1, 0.3)]  # the ring labels
        self._place_labels(ax, top, avoid)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

        self._panel(panel, frame, ranked)

    @staticmethod
    def _place_labels(ax, views, marker_pos, fontsize=8.5):
        """Greedy label placement: try four spots around each marker, skip any that would cover
        another label or marker, and fall back to the short id if nothing fits."""
        to_pt = 72.0 / ax.figure.dpi
        markers = [ax.transData.transform(p) * to_pt for p in marker_pos]
        placed: list[tuple[float, float, float, float]] = []

        def free(box):
            x0, y0, x1, y1 = box
            for a in placed:
                if x0 < a[2] and x1 > a[0] and y0 < a[3] and y1 > a[1]:
                    return False
            return not any(x0 - 5 < mx < x1 + 5 and y0 - 5 < my < y1 + 5 for mx, my in markers)

        for v in views:
            px, py = ax.transData.transform(v.pos[:2] / 1000.0) * to_pt
            for text in (f"T{v.tid} {v.label} {v.p_label:.2f}", f"T{v.tid}"):
                w, h = 0.58 * fontsize * len(text), fontsize * 1.25
                spots = [(9, 4), (9, -4 - h), (-9 - w, 4), (-9 - w, -4 - h)]
                for dx, dy in spots:
                    box = (px + dx, py + dy, px + dx + w, py + dy + h)
                    if free(box):
                        placed.append(box)
                        ax.annotate(text, v.pos[:2] / 1000.0, xytext=(dx, dy + 2),
                                    textcoords="offset points", fontsize=fontsize,
                                    color=INK["primary"], zorder=6, va="bottom")
                        break
                else:
                    continue
                break

    def _panel(self, panel, frame, ranked):
        sc = self.sc
        y = 0.99
        panel.text(0, y, f"Air picture  t = {frame.t:5.1f} s", fontsize=14, weight="bold", va="top")
        y -= 0.055
        cond = f"{sc.name} · {sc.env.light} · visibility {sc.env.visibility_m / 1000:.1f} km"
        panel.text(0, y, cond, fontsize=9.5, color=INK["secondary"], va="top")
        y -= 0.04
        status = ", ".join(f"{s} (failed)" if s in frame.failed else s for s in self.r.sensors)
        panel.text(0, y, f"Sensors: {status}", fontsize=9.5, color=INK["secondary"], va="top")
        y -= 0.07

        cols = [(0.0, "Track"), (0.12, "Class"), (0.36, "p"), (0.46, "Threat"), (0.62, "Range"),
                (0.76, "Sensors")]
        for x, name in cols:
            panel.text(x, y, name, fontsize=9, color=INK["muted"], va="top")
        y -= 0.012
        panel.plot([0, 1], [y, y], lw=0.8, color=INK["grid"], transform=panel.transAxes)
        y -= 0.02
        for v in ranked[: self.max_rows]:
            color, marker = CLASS_STYLE[v.label]
            sensors = "".join(SENSOR_CODE.get(s, "?") for s in v.sensors)
            panel.text(0.0, y, f"T{v.tid}", fontsize=9.5, va="top", family="monospace")
            panel.plot(0.125, y - 0.012, marker=marker, ms=7, mfc=color, mec=INK["surface"],
                       transform=panel.transAxes, clip_on=False)
            panel.text(0.15, y, v.label, fontsize=9.5, va="top")
            panel.text(0.36, y, f"{v.p_label:.2f}", fontsize=9.5, va="top", family="monospace")
            panel.text(0.46, y, v.threat.level, fontsize=9.5, va="top",
                       weight="bold" if v.threat.level == "HIGH" else "normal")
            panel.text(0.62, y, f"{v.threat.range_m / 1000:4.1f} km", fontsize=9.5, va="top",
                       family="monospace")
            panel.text(0.76, y, sensors, fontsize=9.5, va="top", family="monospace")
            y -= 0.045
        if len(ranked) > self.max_rows:
            panel.text(0, y, f"+ {len(ranked) - self.max_rows} more tracks", fontsize=9,
                       color=INK["muted"], va="top")

        handles = [Line2D([], [], ls="none", marker=CLASS_STYLE[c][1], ms=8,
                          mfc=CLASS_STYLE[c][0], mec=INK["surface"], label=c)
                   for c in ("drone", "bird", "aircraft", "helicopter")]
        handles += [Line2D([], [], ls="none", marker="o", ms=5, mfc="none", mec=INK["muted"],
                           label="ground truth"),
                    Line2D([], [], ls="none", marker="o", ms=11, mfc="none", mec=CRITICAL,
                           mew=1.8, label="HIGH threat (advisory)")]
        panel.legend(handles=handles, loc="lower left", ncol=3, frameon=False, fontsize=9,
                     bbox_to_anchor=(0.0, 0.0))
        panel.text(0, 0.135, "Sensors: R radar · E daylight camera · I thermal · A acoustic · "
                   "B ADS-B", fontsize=8.5, color=INK["muted"])
        if any("PLACEHOLDER" in s.upper() for s in self.r.confusion_sources.values()):
            panel.text(0, 0.175, "Classifier confusion matrices are placeholders; "
                       "figures are illustrative.", fontsize=8.5, color=INK["secondary"])
        panel.set_xlim(0, 1)
        panel.set_ylim(0, 1)

    def save_png(self, path, k: int):
        self.draw(k)
        self.fig.savefig(path, dpi=110)

    def save_gif(self, path, every_s: float = 3.0, fps: int = 6):
        stride = max(1, int(round(every_s / self.sc.dt)))
        ks = list(range(0, len(self.r.frames), stride))
        anim = FuncAnimation(self.fig, lambda k: self.draw(k), frames=ks, blit=False)
        anim.save(path, writer=PillowWriter(fps=fps), dpi=72)

    def close(self):
        plt.close(self.fig)


def busiest_frame(result) -> int:
    """Frame index with the most tracks and threats on screen, for a representative still."""
    scores = [len(f.tracks) + 3 * sum(v.threat.level == "HIGH" for v in f.tracks)
              for f in result.frames]
    return int(np.argmax(scores))


# --- ablation charts -------------------------------------------------------------------------

ABLATION_METRICS = [
    ("drones_identified", "Drones identified", 100.0, "%", "higher is better"),
    ("drone_id_delay_s", "Median time to identify a drone", 1.0, "s", "lower is better"),
    ("loc_rmse_m", "Position error (RMSE)", 1.0, "m", "lower is better"),
    ("bird_as_drone", "Bird track-time labelled drone", 100.0, "%", "lower is better"),
]


def plot_ablation(stats: dict, scenario: str, path, highlight: str = "all", n_seeds: int = 0,
                  placeholder: bool = False):
    """stats: {config_name: {metric: (mean, std)}}; one panel per metric, one bar per config."""
    apply_style()
    configs = list(stats)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), dpi=100)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.80, bottom=0.08, wspace=0.42, hspace=0.6)
    ypos = np.arange(len(configs))[::-1]
    for ax, (key, title, scale, unit, hint) in zip(axes.flat, ABLATION_METRICS):
        means = np.array([stats[c][key][0] for c in configs], dtype=float) * scale
        stds = np.array([stats[c][key][1] for c in configs], dtype=float) * scale
        finite = np.isfinite(means)
        top = np.nanmax(means + np.nan_to_num(stds)) if finite.any() else 1.0
        top = top if top > 0 else 1.0
        colors = [GROUP["manned"] if c == highlight else INK["axis"] for c in configs]
        ax.barh(ypos[finite], means[finite], height=0.42, color=np.array(colors)[finite], zorder=2)
        for y, m, s, ok in zip(ypos, means, stds, finite):
            if not ok:
                ax.text(0, y, " n/a", va="center", fontsize=9, color=INK["muted"])
                continue
            hi = min(m + s, 100.0) if unit == "%" else m + s
            if s > 0:
                ax.plot([max(m - s, 0.0), hi], [y, y], lw=1.0, color=INK["secondary"], zorder=3)
            ax.text(max(hi, m) + 0.02 * top, y, f"{m:.0f}{unit}" if m >= 10
                    else f"{m:.1f}{unit}", va="center", fontsize=9, color=INK["secondary"])
        ax.set_yticks(ypos)
        ax.set_yticklabels(configs, fontsize=9.5, color=INK["primary"])
        ax.set_xlim(0, top * 1.25)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.grid(axis="x", color=INK["grid"], lw=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc="left", fontsize=11, pad=16)
        ax.text(0, 1.02, hint, transform=ax.transAxes, fontsize=8.5, color=INK["muted"])
    fig.suptitle(f"Sensor ablation: {scenario}", x=0.02, y=0.97, ha="left", fontsize=14,
                 weight="bold")
    sub = f"Mean over {n_seeds} seeds, whisker = 1 s.d. Highlighted bar: all sensors fused."
    if placeholder:
        sub += "  Placeholder classifier matrices: illustrative, not results."
    fig.text(0.02, 0.905, sub, fontsize=9.5, color=INK["secondary"])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fmt(value, scale=1.0, digits=1):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value * scale:.{digits}f}"
