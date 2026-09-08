"""
compute_lengths.py - nozzle length (mm) for every design point in sweep_plan.csv
=================================================================================
Standalone, read-only report: reads sweep_plan.csv, rebuilds each wall contour
via nozzle_contour.build_contour(), and prints/saves nozzle length_mm per run
for manual entry into the results spreadsheet. No Fluent, no new dependencies.

Rt and eps are NOT columns in sweep_plan.csv - only percent_bell, theta_n_deg,
theta_e_deg are swept (nozzle_contour.py holds everything upstream of the
throat fixed). Rt/eps are loaded from engine_design.json via
nozzle_contour.load_geometry(), the same way batch_make_meshes.py loads them,
so these lengths match the actual meshes in meshes/.
"""
import csv

from nozzle_contour import build_contour, load_geometry

SWEEP_CSV = "sweep_plan.csv"
OUT_CSV = "nozzle_lengths.csv"


def main():
    Rt, eps = load_geometry()

    rows = []
    with open(SWEEP_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row["run_id"]
            pb = float(row["percent_bell"])
            tn = float(row["theta_n_deg"])
            te = float(row["theta_e_deg"])

            xs, rs, meta = build_contour(Rt, eps, pb, tn, te)
            length_mm = (xs.max() - xs.min()) * 1000.0
            rows.append((run_id, pb, tn, te, length_mm))

    header = (f"{'run_id':8s} {'percent_bell':>12s} {'theta_n_deg':>11s} "
              f"{'theta_e_deg':>11s} {'length_mm':>10s}")
    print(header)
    print("-" * len(header))
    for run_id, pb, tn, te, length_mm in rows:
        print(f"{run_id:8s} {pb:12.4f} {tn:11.3f} {te:11.3f} {length_mm:10.3f}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "percent_bell", "theta_n_deg", "theta_e_deg", "length_mm"])
        for row in rows:
            writer.writerow(row)

    print(f"\nwrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
