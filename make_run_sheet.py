"""
make_run_sheet.py - one CSV to drive the 30 manual Fluent runs
===============================================================
Builds fluent_run_sheet.csv: one row per sweep point, carrying everything
needed to set up and record a run -

    * which mesh file to read
    * the bell geometry for that run (design variables + derived length)
    * the boundary / gas inputs to enter in Fluent (identical every run,
      pulled from engine_design.json so they can't drift from Stage 1)
    * the 1D theory values to sanity-check the CFD against
    * blank columns to type results into as each run finishes

Read-only with respect to sweep_plan.csv, the meshes and the generators.
"""
import csv
import json
import os

import numpy as np

from nozzle_contour import build_contour, exit_conditions, load_geometry
from make_sweep_plan import chord_angle_converged

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP_CSV = os.path.join(SCRIPT_DIR, "sweep_plan.csv")
DESIGN_JSON = os.path.join(SCRIPT_DIR, "engine_design.json")
MESH_DIR = os.path.join(SCRIPT_DIR, "meshes")
OUT_CSV = os.path.join(SCRIPT_DIR, "fluent_run_sheet.csv")

# filled in by hand as each run completes
RESULT_COLS = [
    "Cf_cfd", "thrust_N", "Isp_cfd_s", "peak_heatflux_Wm2",
    "exit_Mach_cfd", "converged", "iterations", "notes",
]


def main():
    Rt, eps = load_geometry()
    with open(DESIGN_JSON) as f:
        d = json.load(f)

    # engine_design.json's eps-dependent fields (Cf, Isp, Me_exit, Re, Ae) are
    # only valid for the eps Stage 1 computed. eps now comes from
    # EXPANSION_RATIO, so recompute them for the CURRENT eps instead.
    exit_c = exit_conditions(Rt, eps, d["gamma_exit"], d["Pc_bar"], d["Pamb_bar"])
    print(f"[runsheet] eps={eps:.4f} -> Re={exit_c['Re_m'] * 1e3:.4f} mm, "
          f"Me={exit_c['Me_exit']:.4f}, Cf_ideal={exit_c['Cf_ideal']:.5f}, "
          f"Pe={exit_c['Pe_bar']:.4f} bar")

    with open(SWEEP_CSV, newline="") as f:
        plan = list(csv.DictReader(f))

    rows = []
    missing = []
    for r in plan:
        run_id = r["run_id"]
        pb = float(r["percent_bell"])
        tn = float(r["theta_n_deg"])
        te = float(r["theta_e_deg"])

        xs, rs, meta = build_contour(Rt, eps, pb, tn, te)
        length_mm = (xs.max() - xs.min()) * 1e3

        mesh_rel = f"meshes/{run_id}.msh"
        if not os.path.exists(os.path.join(MESH_DIR, f"{run_id}.msh")):
            missing.append(mesh_rel)

        rows.append({
            "run_id": run_id,
            "mesh_file": mesh_rel,
            # --- geometry (varies per run) ---
            "percent_bell": f"{pb:.6f}",
            "theta_n_deg": f"{tn:.4f}",
            "theta_e_deg": f"{te:.4f}",
            "chord_deg": f"{chord_angle_converged(pb, Rt, eps):.4f}",
            "length_mm": f"{length_mm:.3f}",
            "Rt_mm": f"{Rt * 1e3:.4f}",
            "Re_mm": f"{meta['Re'] * 1e3:.4f}",
            "eps": f"{eps:.4f}",
            # --- Fluent inputs (identical for every run) ---
            "Pc_bar": d["Pc_bar"],
            "Tc_K": f"{d['Tc_K']:.2f}",
            "Pamb_bar": d["Pamb_bar"],
            "gamma": f"{d['gamma_chamber']:.5f}",
            "R_specific_JkgK": f"{d['R_specific_JkgK']:.3f}",
            "MW_gmol": f"{d['MW_chamber_gmol']:.4f}",
            "mdot_kgs": f"{d['mdot_kgs']:.5f}",
            # --- 1D theory reference to check CFD against ---
            # recomputed for the CURRENT eps, not read from the stale JSON
            "Cf_theory": f"{exit_c['Cf_ideal']:.5f}",
            "Pe_bar_theory": f"{exit_c['Pe_bar']:.4f}",
            "Me_exit_theory": f"{exit_c['Me_exit']:.4f}",
        })
        for c in RESULT_COLS:
            rows[-1][c] = ""

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[runsheet] wrote {OUT_CSV}")
    print(f"[runsheet] {len(rows)} runs x {len(fieldnames)} columns")
    if missing:
        print(f"[runsheet] WARNING missing mesh files: {missing}")
    else:
        print(f"[runsheet] all {len(rows)} mesh files present")
    print(f"[runsheet] fill in by hand: {', '.join(RESULT_COLS)}")


if __name__ == "__main__":
    main()
