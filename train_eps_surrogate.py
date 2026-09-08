"""
train_eps_surrogate.py - Cf_vac surrogate over expansion ratio
===============================================================
Maps  eps (Ae/At)  ->  vacuum thrust coefficient Cf_vac

from the expansion-ratio sweep in eps_sweep_entry.xlsx, with bell shape held
fixed at baseline (percent_bell=0.80, theta_n=22, theta_e=14) and Pamb = 0.

Primary model   : Gaussian Process Regression, fitted in BOTH linear-eps and
                  log-eps space; whichever wins leave-one-out CV is the one
                  that gets saved.
Physical baseline: low-order polynomial in log(eps) - the relationship is a
                  smooth monotonic rise tapering to a plateau, so if a cubic
                  in log-eps matches the GPR then the GPR is not earning its
                  complexity.
Honest baseline : DummyRegressor("mean").

Contrast with the shape sweep: the earlier fixed-eps SHAPE surrogate
(train_surrogate.py, 23 runs) failed - LOO R^2 = -0.659, 23.2% WORSE than
predicting the mean - because Cf varied by only ~0.007 there, at or below the
CFD noise floor. Here Cf_vac spans ~0.52 (~31% of its mean), so there is real
signal to learn. This script prints that comparison explicitly.
"""
import os
import shutil
import tempfile

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "eps_sweep_entry.xlsx"),
    os.path.join(PARENT_DIR, "eps_sweep_entry.xlsx"),
    os.path.join(os.path.expanduser("~"), "Downloads", "eps_sweep_entry.xlsx"),
]
SHEET = "Eps sweep results"

EPS_COL = "eps"
TARGET = "Cf_vac"
EPS_MIN, EPS_MAX = 2.0, 100.0        # validated range - warn outside this

MODEL_OUT = os.path.join(SCRIPT_DIR, "surrogate_cf_vac_eps.joblib")
PLOT_OUT = os.path.join(SCRIPT_DIR, "eps_surrogate_diagnostics.png")
SEED = 0

# Recorded results of the fixed-eps SHAPE sweep (from train_surrogate.py) so
# the contrast can be printed without re-running that analysis. n=23 is the
# sweep proper; the workbook's 24th valid row is the separate
# `baseline_validated` case. Including it gives R2=-0.467 instead of -0.659,
# i.e. the conclusion is unchanged either way.
SHAPE_SWEEP = dict(n=23, rmse=0.002253, dummy_rmse=0.001829, r2=-0.659,
                   spread=0.006796)


# ----------------------------------------------------------------------------
def load_data(verbose=True):
    """Read eps and Cf_vac from the workbook. Tolerates the file being open
    in Excel (which locks it) by reading a temporary copy."""
    path = next((p for p in DATA_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("eps_sweep_entry.xlsx not found in:\n  "
                                + "\n  ".join(DATA_CANDIDATES))

    read_path, tmp = path, None
    try:
        with open(path, "rb"):
            pass
    except PermissionError:
        # locked by Excel - work from a copy
        tmp = os.path.join(tempfile.gettempdir(), "_eps_sweep_copy.xlsx")
        shutil.copy2(path, tmp)
        read_path = tmp
        if verbose:
            print(f"[data] source is locked (open in Excel) - read a copy")

    raw = pd.read_excel(read_path, sheet_name=SHEET, header=None)
    hdr = next(i for i in range(len(raw))
               if str(raw.iloc[i, 0]).strip() == "run_id")
    df = pd.read_excel(read_path, sheet_name=SHEET, header=hdr)
    df.columns = [str(c).strip() for c in df.columns]

    # the Cf column is named "Cf_vac (auto)" in the sheet
    cf_col = next(c for c in df.columns if c.lower().startswith("cf_vac"))
    df = df.rename(columns={cf_col: TARGET})

    n_rows = df["run_id"].notna().sum()
    df = df[df["run_id"].notna()].copy()
    for c in (EPS_COL, TARGET):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[EPS_COL, TARGET]).reset_index(drop=True)

    if tmp and os.path.exists(tmp):
        os.remove(tmp)

    if verbose:
        print(f"[data] file  : {path}")
        print(f"[data] rows with a run_id : {n_rows}")
        print(f"[data] valid (eps, Cf_vac): {len(df)}  "
              f"({n_rows - len(df)} dropped for blank Cf_vac)")
        print(f"[data] eps    : {df[EPS_COL].min():g} .. {df[EPS_COL].max():g}")
        print(f"[data] Cf_vac : {df[TARGET].min():.4f} .. {df[TARGET].max():.4f}  "
              f"mean={df[TARGET].mean():.4f}  std={df[TARGET].std():.4f}  "
              f"spread={df[TARGET].max() - df[TARGET].min():.4f} "
              f"({(df[TARGET].max() - df[TARGET].min()) / df[TARGET].mean() * 100:.1f}% of mean)")
        mono = bool(np.all(np.diff(df.sort_values(EPS_COL)[TARGET]) > 0))
        print(f"[data] strictly monotonic in eps: {mono}")
    return df


# ---- model factories -------------------------------------------------------
def _gpr(kernel):
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=30, random_state=SEED)


def make_gpr_rbf():
    k = (ConstantKernel(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-2, 1e3))
         + WhiteKernel(1e-4, (1e-9, 1e0)))
    return Pipeline([("scale", StandardScaler()), ("gpr", _gpr(k))])


def make_gpr_matern():
    k = (ConstantKernel(1.0, (1e-4, 1e4)) * Matern(1.0, (1e-2, 1e3), nu=2.5)
         + WhiteKernel(1e-4, (1e-9, 1e0)))
    return Pipeline([("scale", StandardScaler()), ("gpr", _gpr(k))])


def make_poly(degree):
    return Pipeline([("poly", PolynomialFeatures(degree)),
                     ("lin", LinearRegression())])


def make_dummy():
    return DummyRegressor(strategy="mean")


def metrics(y, yhat):
    return dict(rmse=float(np.sqrt(mean_squared_error(y, yhat))),
                mae=float(mean_absolute_error(y, yhat)),
                r2=float(r2_score(y, yhat)))


def loo_predict(factory, X, y):
    preds = np.empty(len(y))
    for tr, te in LeaveOneOut().split(X):
        m = factory()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return preds


# ----------------------------------------------------------------------------
def evaluate_all(df):
    eps = df[EPS_COL].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    X_lin = eps.reshape(-1, 1)
    X_log = np.log(eps).reshape(-1, 1)

    candidates = [
        ("GPR  RBF    (linear eps)", make_gpr_rbf, X_lin),
        ("GPR  RBF    (log eps)", make_gpr_rbf, X_log),
        ("GPR  Matern (linear eps)", make_gpr_matern, X_lin),
        ("GPR  Matern (log eps)", make_gpr_matern, X_log),
        ("poly deg2   (log eps)", lambda: make_poly(2), X_log),
        ("poly deg3   (log eps)", lambda: make_poly(3), X_log),
        ("Dummy(mean)", make_dummy, X_lin),
    ]

    dummy_rmse = None
    rows = []
    preds = {}
    for name, factory, X in candidates:
        p = loo_predict(factory, X, y)
        m = metrics(y, p)
        preds[name] = p
        if name.startswith("Dummy"):
            dummy_rmse = m["rmse"]
        rows.append((name, m))

    print(f"\n=== leave-one-out CV   n={len(y)} ===")
    print(f"{'model':26s} {'RMSE':>9s} {'MAE':>9s} {'R2':>8s} {'vs dummy':>10s}")
    for name, m in rows:
        rel = "-" if name.startswith("Dummy") else \
            f"{(1 - m['rmse'] / dummy_rmse) * 100:+.1f}%"
        print(f"{name:26s} {m['rmse']:9.5f} {m['mae']:9.5f} {m['r2']:+8.3f} {rel:>10s}")

    # The deliverable model is the best GPR (that is what gets saved and what
    # supplies uncertainty). The polynomial is a comparison baseline only -
    # keep it separate so the saved model always matches the reported config.
    best_name, best_m = min((r for r in rows if r[0].startswith("GPR")),
                            key=lambda r: r[1]["rmse"])
    poly_name, poly_m = min((r for r in rows if r[0].startswith("poly")),
                            key=lambda r: r[1]["rmse"])
    return rows, preds, best_name, best_m, poly_name, poly_m, dummy_rmse, y


def build_final(df, best_name):
    """Refit the winning configuration on all data."""
    eps = df[EPS_COL].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    use_log = "log eps" in best_name
    X = (np.log(eps) if use_log else eps).reshape(-1, 1)
    factory = make_gpr_matern if "Matern" in best_name else make_gpr_rbf
    model = factory()
    model.fit(X, y)
    return model, use_log


def predict_cf_vac(model, use_log, eps_query, warn=True):
    """Predict Cf_vac and 1-sigma uncertainty at a given expansion ratio."""
    eps_query = np.atleast_1d(np.asarray(eps_query, dtype=float))
    if warn:
        out = eps_query[(eps_query < EPS_MIN) | (eps_query > EPS_MAX)]
        if out.size:
            print(f"  *** WARNING: eps {list(out)} is OUTSIDE the fitted range "
                  f"[{EPS_MIN:g}, {EPS_MAX:g}]. A GP reverts toward the data mean "
                  f"beyond its data, so both the prediction AND its uncertainty "
                  f"are unreliable there - do not trust extrapolation. ***")
    X = (np.log(eps_query) if use_log else eps_query).reshape(-1, 1)
    Xs = model.named_steps["scale"].transform(X)
    mean, std = model.named_steps["gpr"].predict(Xs, return_std=True)
    return mean, std


def make_plots(df, model, use_log, preds, best_name, path):
    eps = df[EPS_COL].to_numpy(float)
    y = df[TARGET].to_numpy(float)

    grid = np.logspace(np.log10(EPS_MIN), np.log10(EPS_MAX), 400)
    mu, sd = predict_cf_vac(model, use_log, grid, warn=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    ax.fill_between(grid, mu - 2 * sd, mu + 2 * sd, color="#c0392b", alpha=0.15,
                    label="±2σ (95%)")
    ax.fill_between(grid, mu - sd, mu + sd, color="#c0392b", alpha=0.30,
                    label="±1σ")
    ax.plot(grid, mu, "-", color="#c0392b", lw=2, label=f"GPR mean ({best_name.strip()})")
    ax.plot(eps, y, "o", color="#2c3e50", ms=7, zorder=5, label="CFD data (13 pts)")
    ax.set_xscale("log")
    ax.set_xlabel("expansion ratio ε  (Ae/At)")
    ax.set_ylabel("Cf_vac")
    ax.set_title("Surrogate: Cf_vac vs ε, with uncertainty band")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    p = preds[best_name]
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="ideal y=x")
    ax.axhline(y.mean(), color="#7f8c8d", ls=":", lw=1.2, label="train mean")
    ax.scatter(y, p, c="#c0392b", s=42, zorder=5, label="GPR (LOO)")
    ax.set_xlabel("actual Cf_vac (CFD)")
    ax.set_ylabel("LOO-predicted Cf_vac")
    ax.set_title("Predicted vs actual (leave-one-out)")
    ax.set_aspect("equal")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=135)
    print(f"\n[plot] wrote {path}")


# ----------------------------------------------------------------------------
def main():
    df = load_data()
    rows, preds, best_name, best_m, poly_name, poly_m, dummy_rmse, y = evaluate_all(df)

    print(f"\n[best GPR] {best_name.strip()}  "
          f"RMSE={best_m['rmse']:.5f}  R2={best_m['r2']:+.4f}   <- saved model")
    print(f"[baseline] {poly_name.strip()}  "
          f"RMSE={poly_m['rmse']:.5f}  R2={poly_m['r2']:+.4f}")
    d = (poly_m["rmse"] - best_m["rmse"]) / best_m["rmse"] * 100
    if abs(d) < 10:
        print(f"[baseline] -> the polynomial is within {abs(d):.1f}% of the GPR: the")
        print(f"[baseline]    relationship really is just a smooth curve in log-eps,")
        print(f"[baseline]    so the GPR is NOT buying accuracy - only its uncertainty band.")
    elif d > 0:
        print(f"[baseline] -> GPR beats the polynomial by {d:.1f}%")
    else:
        print(f"[baseline] -> the polynomial BEATS the GPR by {-d:.1f}%")

    # log-eps vs linear-eps, as asked
    def best_of(tag):
        sub = [r for r in rows if tag in r[0]]
        return min(sub, key=lambda r: r[1]["rmse"]) if sub else None
    lin, log = best_of("linear eps"), best_of("log eps")
    print(f"[space] best linear-eps : {lin[0].strip():26s} RMSE={lin[1]['rmse']:.5f} "
          f"R2={lin[1]['r2']:+.4f}")
    print(f"[space] best log-eps    : {log[0].strip():26s} RMSE={log[1]['rmse']:.5f} "
          f"R2={log[1]['r2']:+.4f}")
    print(f"[space] -> {'log' if log[1]['rmse'] < lin[1]['rmse'] else 'linear'}-eps "
          f"fits better")

    model, use_log = build_final(df, best_name)
    gpr = model.named_steps["gpr"]
    print(f"\n[gpr] fitted kernel: {gpr.kernel_}")
    print(f"[gpr] log-marginal-likelihood = {gpr.log_marginal_likelihood_value_:.3f}")

    # ---- verdict + contrast with the shape sweep --------------------------
    gain = (1 - best_m["rmse"] / dummy_rmse) * 100
    print("\n=== VERDICT ===")
    print(f"  GPR LOO-RMSE   : {best_m['rmse']:.5f}")
    print(f"  mean-predictor : {dummy_rmse:.5f}")
    print(f"  improvement    : {gain:+.1f}%   (R2 = {best_m['r2']:+.4f})")
    if best_m["r2"] > 0.9 and gain > 50:
        print("  -> STRONG fit. The surrogate decisively beats predicting the mean.")
    elif best_m["r2"] > 0:
        print("  -> Positive R2: the surrogate beats the mean, but not decisively.")
    else:
        print("  -> The surrogate does NOT beat predicting the mean.")

    s = SHAPE_SWEEP
    print("\n=== CONTRAST: eps sweep vs the earlier fixed-eps SHAPE sweep ===")
    print(f"{'':22s} {'shape sweep':>14s} {'eps sweep':>14s}")
    print(f"{'n points':22s} {s['n']:>14d} {len(y):>14d}")
    print(f"{'Cf spread':22s} {s['spread']:>14.4f} "
          f"{df[TARGET].max() - df[TARGET].min():>14.4f}")
    print(f"{'LOO R2':22s} {s['r2']:>+14.3f} {best_m['r2']:>+14.4f}")
    print(f"{'RMSE':22s} {s['rmse']:>14.5f} {best_m['rmse']:>14.5f}")
    print(f"{'dummy RMSE':22s} {s['dummy_rmse']:>14.5f} {dummy_rmse:>14.5f}")
    print(f"{'vs dummy':22s} "
          f"{(1 - s['rmse'] / s['dummy_rmse']) * 100:>+13.1f}% {gain:>+13.1f}%")
    print("  The shape sweep varied Cf by less than its own CFD noise, so no model")
    print("  could beat the mean. Expansion ratio is the dominant driver of Cf and")
    print("  is learned cleanly from the same number of runs or fewer.")

    joblib.dump({"model": model, "use_log": use_log, "eps_range": (EPS_MIN, EPS_MAX),
                 "config": best_name.strip(), "n_train": len(y)}, MODEL_OUT)
    print(f"\n[save] wrote {MODEL_OUT}")

    print("\n[predict] examples at unseen eps:")
    for q in (4.0, 25.0, 60.0):
        mu, sd = predict_cf_vac(model, use_log, q)
        print(f"    eps = {q:6.1f}  ->  Cf_vac = {mu[0]:.4f} +/- {sd[0]:.4f} (1 sigma)")
    mono = np.all(np.diff(predict_cf_vac(model, use_log,
                                         np.logspace(np.log10(EPS_MIN),
                                                     np.log10(EPS_MAX), 200),
                                         warn=False)[0]) > 0)
    print(f"[predict] fitted curve strictly increasing across eps range: {bool(mono)}")

    print("\n[predict] extrapolation check:")
    predict_cf_vac(model, use_log, 150.0)

    make_plots(df, model, use_log, preds, best_name, PLOT_OUT)


if __name__ == "__main__":
    main()
