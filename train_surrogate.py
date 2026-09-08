"""
train_surrogate.py - Cf surrogate from the manual CFD sweep
============================================================
Maps the three Rao bell shape parameters

    [percent_bell, theta_n_deg, theta_e_deg]  ->  Cf

at fixed expansion ratio, using the CFD results entered in
nozzle_sweep_entry.xlsx.

Primary model  : Gaussian Process Regression (Matern + WhiteKernel, ARD)
Cross-check    : Gradient Boosted Trees
Honest baseline: DummyRegressor("mean") - predicts the training mean

Everything is scored with leave-one-out cross-validation, which is the honest
choice at n ~ 24. The headline question this script is built to answer is NOT
"what is the R^2" but "does the surrogate beat simply predicting the mean?".
With a Cf spread of only ~0.002 it is entirely possible that it does not, and
the script is written to say so plainly rather than to flatter the model.
"""
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# The workbook has lived in a few places; take the first that exists.
DATA_CANDIDATES = [
    # in-repo copy first, so a fresh clone is reproducible
    os.path.join(SCRIPT_DIR, "nozzle_sweep_entry_constant_e.xlsx"),
    os.path.join(SCRIPT_DIR, "nozzle_sweep_entry.xlsx"),
    os.path.join(PARENT_DIR, "nozzle_sweep_entry_constant_e.xlsx"),
    os.path.join(PARENT_DIR, "nozzle_sweep_entry.xlsx"),
]
SHEET = "Sweep results"

FEATURES = ["percent_bell", "theta_n_deg", "theta_e_deg"]
TARGET = "Cf"

# --- what counts as the sweep --------------------------------------------
# The workbook holds 24 rows with a valid Cf: the 23 SWEEP RUNS plus the
# separate `baseline_validated` case. The shape sweep proper is the 23 runs,
# so the baseline is excluded by default. Set this to [] to train on all 24 -
# the conclusion is identical either way (see the note below).
#
#   n=23 (sweep only) : GPR R2 = -0.659, 23.2% worse than the mean-predictor
#   n=24 (+ baseline) : GPR R2 = -0.467, 16.1% worse than the mean-predictor
#
# Add run_ids here to drop them from training, e.g. ["run_06"]. The script
# ALSO automatically re-runs with the most extreme point removed and reports
# whether that materially changes the fit.
EXCLUDE_RUNS = ["baseline_validated"]

MODEL_OUT = os.path.join(SCRIPT_DIR, "surrogate_cf_gpr.joblib")
PLOT_OUT = os.path.join(SCRIPT_DIR, "surrogate_diagnostics.png")

SEED = 0


# ----------------------------------------------------------------------------
def load_data(verbose=True):
    """Read the sweep workbook, returning a tidy frame of valid training rows."""
    path = next((p for p in DATA_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(
            "nozzle_sweep_entry.xlsx not found in:\n  " + "\n  ".join(DATA_CANDIDATES))

    raw = pd.read_excel(path, sheet_name=SHEET, header=None)
    # the sheet has a title banner + constants block above the real header
    hdr = next(i for i in range(len(raw))
               if str(raw.iloc[i, 0]).strip() == "run_id")
    df = pd.read_excel(path, sheet_name=SHEET, header=hdr)
    df.columns = [str(c).strip() for c in df.columns]

    df = df.rename(columns={"% bell": "percent_bell",
                            "theta_n": "theta_n_deg",
                            "theta_e": "theta_e_deg"})
    keep = ["run_id"] + FEATURES + [TARGET, "eps"]
    df = df[[c for c in keep if c in df.columns]]

    n_all = df["run_id"].notna().sum()
    df = df[df["run_id"].notna() & df[TARGET].notna()].reset_index(drop=True)
    for c in FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

    if verbose:
        print(f"[data] file   : {path}")
        print(f"[data] rows with a run_id : {n_all}")
        print(f"[data] rows with valid Cf : {len(df)}  "
              f"({n_all - len(df)} dropped for blank/invalid Cf)")
        if "eps" in df:
            data_eps = sorted(df["eps"].dropna().unique())
            print(f"[data] eps     : {data_eps}")
            # the CFD data is only valid for the eps it was run at; warn loudly
            # if EXPANSION_RATIO has since been changed underneath it
            try:
                from nozzle_contour import EXPANSION_RATIO
                if any(abs(e - EXPANSION_RATIO) > 1e-6 for e in data_eps):
                    print(f"[data] *** WARNING: EXPANSION_RATIO is now "
                          f"{EXPANSION_RATIO}, but this training data was run at "
                          f"{data_eps}. The surrogate is only valid at the eps "
                          f"the CFD used - re-run the sweep before trusting it "
                          f"at the new eps. ***")
            except ImportError:
                pass
        print(f"[data] Cf       : min={df[TARGET].min():.6f}  "
              f"max={df[TARGET].max():.6f}  mean={df[TARGET].mean():.6f}  "
              f"std={df[TARGET].std():.6f}  spread={df[TARGET].max() - df[TARGET].min():.6f}")
    return df


def make_gpr():
    """Matern(ARD) + WhiteKernel, on standardized inputs and normalized y."""
    kernel = (ConstantKernel(1.0, (1e-4, 1e4))
              * Matern(length_scale=np.ones(len(FEATURES)),
                       length_scale_bounds=(1e-2, 1e3), nu=2.5)
              + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1)))
    return Pipeline([
        ("scale", StandardScaler()),
        ("gpr", GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                         n_restarts_optimizer=20,
                                         random_state=SEED)),
    ])


def make_gbt():
    return Pipeline([
        ("scale", StandardScaler()),
        ("gbt", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                          max_depth=2, random_state=SEED)),
    ])


def loo_predict(model_factory, X, y):
    """Leave-one-out predictions. Refits from scratch on each fold."""
    preds = np.empty(len(y))
    for tr, te in LeaveOneOut().split(X):
        m = model_factory()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return preds


def metrics(y, yhat):
    return dict(rmse=float(np.sqrt(mean_squared_error(y, yhat))),
                mae=float(mean_absolute_error(y, yhat)),
                r2=float(r2_score(y, yhat)))


def evaluate(df, label, verbose=True):
    """LOO-CV for GPR, GBT and the mean-predictor baseline."""
    X = df[FEATURES].to_numpy(float)
    y = df[TARGET].to_numpy(float)

    results = {}
    preds = {}
    for name, factory in [("GPR", make_gpr), ("GBT", make_gbt),
                          ("Dummy(mean)", lambda: DummyRegressor(strategy="mean"))]:
        p = loo_predict(factory, X, y)
        preds[name] = p
        results[name] = metrics(y, p)

    if verbose:
        print(f"\n=== leave-one-out CV  [{label}]  n={len(y)} ===")
        print(f"{'model':14s} {'RMSE':>10s} {'MAE':>10s} {'R2':>9s} "
              f"{'RMSE vs dummy':>15s}")
        dz = results["Dummy(mean)"]["rmse"]
        for name in ("GPR", "GBT", "Dummy(mean)"):
            r = results[name]
            rel = (r["rmse"] / dz - 1.0) * 100.0
            tag = f"{rel:+.1f}%" if name != "Dummy(mean)" else "-"
            print(f"{name:14s} {r['rmse']:10.6f} {r['mae']:10.6f} "
                  f"{r['r2']:9.3f} {tag:>15s}")
    return results, preds, y


def report_kernel(df):
    """Fit the GPR on all data and read off signal vs noise - the most direct
    evidence of whether there is any learnable structure at all."""
    X = df[FEATURES].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    m = make_gpr()
    m.fit(X, y)
    gpr = m.named_steps["gpr"]
    k = gpr.kernel_
    print(f"\n[gpr] fitted kernel: {k}")
    amp = k.k1.k1.constant_value          # signal variance (normalized y units)
    noise = k.k2.noise_level              # noise variance  (normalized y units)
    frac = noise / (amp + noise)
    print(f"[gpr] signal var = {amp:.4g}, noise var = {noise:.4g}  "
          f"-> noise is {frac * 100:.1f}% of total variance")
    print(f"[gpr] length scales (scaled units): "
          f"{np.atleast_1d(k.k1.k2.length_scale)}")
    print(f"[gpr] log-marginal-likelihood = {gpr.log_marginal_likelihood_value_:.3f}")
    return m


def predict_cf(model, percent_bell, theta_n_deg, theta_e_deg):
    """Predict Cf and the GPR 1-sigma uncertainty for a new shape."""
    x = np.array([[percent_bell, theta_n_deg, theta_e_deg]], dtype=float)
    xs = model.named_steps["scale"].transform(x)
    mean, std = model.named_steps["gpr"].predict(xs, return_std=True)
    return float(mean[0]), float(std[0])


def make_plots(y, preds, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    lo, hi = min(y.min(), preds["GPR"].min()), max(y.max(), preds["GPR"].max())
    pad = 0.08 * (hi - lo)
    for name, c, mk in [("GPR", "#c0392b", "o"), ("GBT", "#2980b9", "s")]:
        ax.scatter(y, preds[name], c=c, marker=mk, s=34, alpha=0.8, label=name)
    ax.axhline(y.mean(), color="#7f8c8d", ls=":", lw=1.2, label="train mean")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="ideal y=x")
    ax.set_xlabel("actual Cf (CFD)"); ax.set_ylabel("LOO-predicted Cf")
    ax.set_title("Predicted vs actual (leave-one-out)")
    ax.legend(fontsize=8); ax.set_aspect("equal")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)

    ax = axes[1]
    res = preds["GPR"] - y
    ax.scatter(y, res, c="#c0392b", s=34)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("actual Cf"); ax.set_ylabel("GPR residual (pred - actual)")
    ax.set_title(f"GPR residuals  (std = {res.std():.2e})")

    ax = axes[2]
    ax.hist(res, bins=10, color="#c0392b", alpha=0.75, edgecolor="k")
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("GPR residual"); ax.set_ylabel("count")
    ax.set_title("Residual distribution")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\n[plot] wrote {path}")


# ----------------------------------------------------------------------------
def main():
    df_all = load_data()

    if EXCLUDE_RUNS:
        df = df_all[~df_all["run_id"].isin(EXCLUDE_RUNS)].reset_index(drop=True)
        print(f"[data] EXCLUDE_RUNS={EXCLUDE_RUNS} -> {len(df)} rows used")
    else:
        df = df_all

    # which points are actually extreme?
    z = (df[TARGET] - df[TARGET].mean()) / df[TARGET].std()
    order = z.abs().sort_values(ascending=False)
    print("\n[outliers] most extreme rows by |z-score| of Cf:")
    for i in order.index[:4]:
        print(f"    {df.loc[i, 'run_id']:20s} Cf={df.loc[i, TARGET]:.6f}  z={z[i]:+.2f}")

    res_in, preds_in, y_in = evaluate(df, "all points")

    # sensitivity: drop the single most extreme point and re-score
    worst = order.index[0]
    worst_id = df.loc[worst, "run_id"]
    df_ex = df.drop(index=worst).reset_index(drop=True)
    res_ex, _, _ = evaluate(df_ex, f"excluding {worst_id}")

    print(f"\n[outliers] effect of dropping {worst_id}:")
    for name in ("GPR", "GBT", "Dummy(mean)"):
        print(f"    {name:14s} RMSE {res_in[name]['rmse']:.6f} -> "
              f"{res_ex[name]['rmse']:.6f}   "
              f"R2 {res_in[name]['r2']:+.3f} -> {res_ex[name]['r2']:+.3f}")

    model = report_kernel(df)

    # ---- verdict -----------------------------------------------------------
    gpr_rmse = res_in["GPR"]["rmse"]
    dum_rmse = res_in["Dummy(mean)"]["rmse"]
    gain = (1.0 - gpr_rmse / dum_rmse) * 100.0
    print("\n=== VERDICT ===")
    print(f"  GPR LOO-RMSE     : {gpr_rmse:.6f}")
    print(f"  mean-predictor   : {dum_rmse:.6f}")
    print(f"  improvement      : {gain:+.1f}%   (R2 = {res_in['GPR']['r2']:+.3f})")
    if res_in["GPR"]["r2"] <= 0 or gain <= 5.0:
        print("  -> The surrogate does NOT meaningfully beat predicting the mean.")
        print("     On this data, shape has no detectable effect on Cf beyond noise.")
    else:
        print("  -> The surrogate beats the mean-predictor.")

    joblib.dump({"model": model, "features": FEATURES,
                 "excluded": EXCLUDE_RUNS, "n_train": len(df)}, MODEL_OUT)
    print(f"\n[save] wrote {MODEL_OUT}")

    pb, tn, te = 0.80, 22.0, 14.0
    mu, sd = predict_cf(model, pb, tn, te)
    print(f"[predict] example shape percent_bell={pb}, theta_n={tn}, theta_e={te}")
    print(f"[predict]   Cf = {mu:.6f} +/- {sd:.6f} (1 sigma)")

    make_plots(y_in, preds_in, PLOT_OUT)


if __name__ == "__main__":
    main()
