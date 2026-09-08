"""
plot_contours.py - visual sanity check on all 31 nozzle contours
=================================================================
Plots the baseline (percent_bell=0.80, theta_n=22, theta_e=14) plus all 30
sweep_plan.csv design points, so fold-back or other geometric weirdness is
obvious by eye before any of them go into Fluent.

Writes two figures:
    contours_overlay.png - all 31 contours on one pair of axes
    contours_grid.png    - a 4x8 grid, one contour per panel, clamped rows
                           highlighted in red

Read-only: does not modify sweep_plan.csv, the meshes, or any generator.
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nozzle_contour import build_contour, load_geometry

Rt, eps = load_geometry()   # eps comes from EXPANSION_RATIO in nozzle_contour.py
SWEEP_CSV = "sweep_plan.csv"
OUT_OVERLAY = "contours_overlay.png"
OUT_GRID = "contours_grid.png"


def load_cases():
    cases = [("baseline", 0.80, 22.0, 14.0)]
    with open(SWEEP_CSV, newline="") as f:
        for row in csv.DictReader(f):
            cases.append((row["run_id"], float(row["percent_bell"]),
                          float(row["theta_n_deg"]), float(row["theta_e_deg"])))
    return cases


def main():
    cases = load_cases()
    built = []
    for run_id, pb, tn, te in cases:
        xs, rs, meta = build_contour(Rt, eps, pb, tn, te)
        mono = bool(np.all(np.diff(xs) >= -1e-9))
        built.append((run_id, pb, tn, te, xs, rs, meta, mono))

    # ---- overlay ----
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for run_id, pb, tn, te, xs, rs, meta, mono in built:
        if run_id == "baseline":
            ax.plot(xs * 1e3, rs * 1e3, "-", lw=2.4, color="k", zorder=5,
                    label="baseline")
        else:
            col = "#c0392b" if meta["clamped"] else "#7f8c8d"
            ax.plot(xs * 1e3, rs * 1e3, "-", lw=0.9, color=col, alpha=0.75)
    ax.plot([], [], "-", color="#c0392b", lw=0.9, label="clamped (Q was outside N..E)")
    ax.plot([], [], "-", color="#7f8c8d", lw=0.9, label="unclamped")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("axial x [mm]")
    ax.set_ylabel("radius r [mm]")
    ax.set_title("All 31 nozzle contours (baseline + 30 sweep points)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_OVERLAY, dpi=130)
    print(f"[plot] wrote {OUT_OVERLAY}")

    # ---- grid ----
    ncol, nrow = 8, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(20, 9), sharex=True, sharey=True)
    for ax, (run_id, pb, tn, te, xs, rs, meta, mono) in zip(axes.ravel(), built):
        col = "#c0392b" if meta["clamped"] else "#2c3e50"
        ax.plot(xs * 1e3, rs * 1e3, "-", lw=1.3, color=col)
        ax.axhline(0, color="k", lw=0.4, ls="--")
        flag = " CLAMPED" if meta["clamped"] else ""
        bad = "" if mono else "  FOLDED!"
        ax.set_title(f"{run_id}{flag}{bad}\n"
                     f"pb={pb:.3f} tn={tn:.1f} te={te:.1f}",
                     fontsize=7, color=("#c0392b" if meta["clamped"] else "k"))
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(built):]:
        ax.axis("off")
    fig.suptitle("Contour grid - red = Bezier control point was clamped", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_GRID, dpi=110)
    print(f"[plot] wrote {OUT_GRID}")

    nclamped = sum(1 for b in built if b[6]["clamped"])
    nmono = sum(1 for b in built if b[7])
    print(f"[plot] {nmono}/{len(built)} monotonic, {nclamped} clamped")


if __name__ == "__main__":
    main()
