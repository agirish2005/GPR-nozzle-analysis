"""
batch_make_meshes.py - build all meshes in sweep_plan.csv (pure Python)
========================================================================
Reads sweep_plan.csv (written by make_sweep_plan.py), calls into
nozzle_mesh.py's build_structured() / write_fluent_msh() for each of the 30
rows, and writes meshes/run_01.msh ... run_30.msh. No Fluent / PyFluent
involved - these meshes are meant to be read into the validated Fluent case
manually or via a journal file.
"""
import os

import pandas as pd

from nozzle_contour import load_geometry
from nozzle_mesh import build_structured, write_fluent_msh

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP_CSV = os.path.join(SCRIPT_DIR, "sweep_plan.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "meshes")

NI, NJ = 240, 80
FIRST_CELL_M = 3e-6


def main():
    Rt, eps = load_geometry()
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(SWEEP_CSV)
    for _, row in df.iterrows():
        run_id = row["run_id"]
        pb, tn, te = row["percent_bell"], row["theta_n_deg"], row["theta_e_deg"]

        X, R, meta = build_structured(
            Rt, eps, pb, tn, te, NI=NI, NJ=NJ, first_cell_m=FIRST_CELL_M
        )
        out_path = os.path.join(OUT_DIR, f"{run_id}.msh")
        nnodes, ncells = write_fluent_msh(out_path, X, R)
        length_mm = (X[-1, 0] - X[0, 0]) * 1e3

        print(
            f"[{run_id}] cells={ncells:6d} nodes={nnodes:6d} "
            f"length={length_mm:7.2f} mm  "
            f"(percent_bell={pb:.3f} theta_n={tn:.2f} theta_e={te:.2f})"
        )

    print(f"\n[batch] wrote {len(df)} meshes to {OUT_DIR}/")


if __name__ == "__main__":
    main()
