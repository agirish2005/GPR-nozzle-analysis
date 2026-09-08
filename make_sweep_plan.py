"""
make_sweep_plan.py - feasibility-constrained Latin Hypercube sweep of the bell
==============================================================================
Samples the three swept bell parameters from nozzle_contour.py
(percent_bell, theta_n_deg, theta_e_deg) and writes sweep_plan.csv: 30 nozzle
shapes plus empty result columns (Cf, length_mm, peak_heatflux, converged,
notes) to be filled in after each mesh is solved in the validated Fluent case.

FEASIBILITY CONSTRAINT
----------------------
The bell is a quadratic Bezier from N (throat-arc end) to E (exit) whose
control point Q is the intersection of the two tangent lines. Writing the
chord slope mc = (rE-rN)/(xE-xN) and m1 = tan(theta_n), m2 = tan(theta_e):

    Qx - xN = (xE-xN) * (mc - m2) / (m1 - m2)
    xE - Qx = (xE-xN) * (m1 - mc) / (m1 - m2)

so Q lies strictly between N and E if and only if

    theta_e  <  chord_angle  <  theta_n

If theta_n drops below the chord angle the tangent intersection overshoots E,
dx/ds goes negative near the exit and the wall folds back on itself - a
self-intersecting, physically invalid nozzle. The chord angle is set by
percent_bell (and Rt, eps), so theta_n and theta_e are NOT independent of
percent_bell; a plain box sample over the three produces infeasible corners.

This sampler therefore draws an LHS in the unit cube and maps each row's
theta_n / theta_e through that row's own feasible interval (conditional LHS).
Stratification is preserved, and every row is feasible by construction, so
nozzle_contour's clamp never has to fire.

Note: theta_n's upper bound is 30 deg, not 26. At the short-bell end
(percent_bell ~ 0.60) the chord angle is ~24.5 deg, so a 26 deg cap left an
empty feasible window. theta_n up to 30 deg is well within normal Rao TOP
bell practice.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import qmc

from nozzle_contour import RD_FACT, load_geometry

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(SCRIPT_DIR, "sweep_plan.csv")

N_RUNS = 30
SEED = 42

PCT_BELL_RANGE = (0.60, 0.90)
THETA_N_RANGE = (18.0, 30.0)   # widened from 26 -> 30, see module docstring
THETA_E_RANGE = (8.0, 16.0)
ANGLE_MARGIN = 1.5             # keep theta_n / theta_e this far off the chord
                               # angle so Q lands comfortably inside N..E


def chord_angle_deg(pb, Rt, eps, theta_n_deg):
    """Angle of the straight N->E chord [deg]. Depends weakly on theta_n
    because N itself slides along the throat downstream arc."""
    Re = Rt * np.sqrt(eps)
    Rd = RD_FACT * Rt
    L15 = (Re - Rt) / np.tan(np.radians(15.0))
    tn = np.radians(theta_n_deg)
    xN = Rd * np.sin(tn)
    rN = (Rt + Rd) - Rd * np.cos(tn)
    xE, rE = pb * L15, Re
    return float(np.degrees(np.arctan2(rE - rN, xE - xN)))


def chord_angle_converged(pb, Rt, eps, guess=22.0, iters=60):
    """Fixed-point solve of the weak theta_n dependence."""
    a = guess
    for _ in range(iters):
        a = chord_angle_deg(pb, Rt, eps, a)
    return a


def sample(Rt, eps):
    sampler = qmc.LatinHypercube(d=3, seed=SEED)
    u = sampler.random(N_RUNS)          # unit cube, stratified per dimension

    pb = PCT_BELL_RANGE[0] + u[:, 0] * (PCT_BELL_RANGE[1] - PCT_BELL_RANGE[0])

    theta_n = np.zeros(N_RUNS)
    theta_e = np.zeros(N_RUNS)
    chords = np.zeros(N_RUNS)
    for i in range(N_RUNS):
        c = chord_angle_converged(pb[i], Rt, eps)
        chords[i] = c

        # theta_n must sit ABOVE the chord angle
        n_lo = max(THETA_N_RANGE[0], c + ANGLE_MARGIN)
        n_hi = THETA_N_RANGE[1]
        if n_hi <= n_lo:
            raise ValueError(
                f"no feasible theta_n for percent_bell={pb[i]:.4f} "
                f"(chord={c:.3f} deg, needs > {n_lo:.3f}, cap {n_hi:.3f})"
            )
        theta_n[i] = n_lo + u[i, 1] * (n_hi - n_lo)

        # theta_e must sit BELOW the chord angle
        e_lo = THETA_E_RANGE[0]
        e_hi = min(THETA_E_RANGE[1], c - ANGLE_MARGIN)
        if e_hi <= e_lo:
            raise ValueError(
                f"no feasible theta_e for percent_bell={pb[i]:.4f} "
                f"(chord={c:.3f} deg, needs < {e_hi:.3f}, floor {e_lo:.3f})"
            )
        theta_e[i] = e_lo + u[i, 2] * (e_hi - e_lo)

    return pd.DataFrame({
        "percent_bell": pb,
        "theta_n_deg": theta_n,
        "theta_e_deg": theta_e,
    }), chords


def main():
    Rt, eps = load_geometry()
    df, chords = sample(Rt, eps)
    df.insert(0, "run_id", [f"run_{i + 1:02d}" for i in range(N_RUNS)])

    # feasibility must hold for every row: theta_e < chord < theta_n
    ok = (df["theta_e_deg"] < chords) & (chords < df["theta_n_deg"])
    if not ok.all():
        raise ValueError(f"infeasible rows generated: {list(df.loc[~ok, 'run_id'])}")

    for col in ("Cf", "length_mm", "peak_heatflux", "converged", "notes"):
        df[col] = ""

    df.to_csv(OUT_CSV, index=False)
    print(f"[sweep] wrote {OUT_CSV} with {N_RUNS} runs "
          f"(all feasible: theta_e < chord < theta_n)")

    show = df[["run_id", "percent_bell", "theta_n_deg", "theta_e_deg"]].copy()
    show["chord_deg"] = chords
    print(show.to_string(index=False,
                         float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
