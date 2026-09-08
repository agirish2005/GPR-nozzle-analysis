

# Rao Bell Nozzle — Parametric Geometry, CFD Sweep & Cf Surrogate

Parametric design and CFD study of a LOX/CH4 bell nozzle. The pipeline sizes an
engine from CEA, generates a Rao thrust-optimised (TOP) bell contour, exports a
watertight 2-D axisymmetric domain for ANSYS, builds structured Fluent meshes,
and fits Gaussian-Process surrogates to the CFD results.

This route takes an PyFluent path whereby a script is used to nuild the meshes utilsied in the CFD sweeps. However, if this does not work, I validated a different method. This method involves manually generateing your meshes for each geometry iteration. This is long-winded and brute force, but it DOES work.

---

## The one knob

**`EXPANSION_RATIO` in `nozzle_contour.py` is the single source of truth for ε.**

`load_geometry()` returns `(Rt, eps)`, taking `Rt` from `engine_design.json` and
`eps` from `EXPANSION_RATIO` — deliberately overriding the JSON. Every script
calls `load_geometry()`; nothing hardcodes geometry. Change that one number and
every downstream artefact (contour, domain, meshes, lengths, run sheet) follows.

---

## Run order

```
engine_sizing.py        # Stage 1: CEA sizing -> engine_design.json  (needs rocketcea)
nozzle_contour.py       # Stage 2: bell contour + closed 2-D domain export
make_sweep_plan.py      # LHS design points (feasibility-constrained)
batch_make_meshes.py    # 30 structured Fluent meshes -> meshes/
compute_lengths.py      # nozzle length per design point
plot_contours.py        # visual check: all contours, fold-back detection
make_run_sheet.py       # fluent_run_sheet.csv for the manual Fluent runs
export_domain.py        # DXF export of the closed domain
```

Surrogates (after the CFD runs are recorded in the spreadsheets):

```
train_eps_surrogate.py  # eps -> Cf_vac      WORKS   (R2 = +0.97)
train_surrogate.py      # shape -> Cf        FAILS   (R2 = -0.66) — kept as a result
```

---

## Importing into ANSYS

Import **`nozzle_domain_ansys.txt`** — the closed domain (wall + outlet + axis +
inlet) as 5 curve groups with bit-identical shared endpoints.

Do **not** import `nozzle_curve_ansys.txt` — that is the wall only, an open
curve, which produces *"No regions were created. Disconnected surface models are
not supported"* in Fluent Meshing.

**Set your DesignModeler/SpaceClaim session units to match `EXPORT_UNIT`
(currently `m`) before importing.** A point file carries only bare numbers with
no unit token, so a mismatch scales the model by exactly 1000× and Meshing
hangs trying to fit controls to a 100-metre nozzle.

`nozzle_domain.dxf` is an alternative: one closed `POLYLINE`, which imports as a
single closed region with no edge-joining step.

In DesignModeler: *Concept → 3D Curve* (point file) → *Concept → Surfaces From
Edges* (select all 5 edges) → one face. The 5 groups exist so each can be named
`inlet` / `outlet` / `axis` / `nozzle_wall` as a Named Selection.

---

## Results

### ε → Cf_vac — works

13 CFD points, ε ∈ [2, 100], shape fixed at baseline, vacuum (Pamb = 0).
Fitted in **log-ε** (66% lower RMSE than linear-ε).

| model | LOO RMSE | R² | vs mean-predictor |
|---|---:|---:|---:|
| **GPR Matérn (log ε)** | **0.02712** | **+0.967** | **83.1% better** |
| poly deg3 (log ε) | 0.02630 | +0.969 | 83.6% better |
| Dummy (mean) | 0.16055 | −0.174 | — |

A cubic in log-ε matches the GPR to within 3%, so the GPR is not buying
accuracy here — only its calibrated uncertainty band. Saved model:
`surrogate_cf_vac_eps.joblib`. Plot: `eps_surrogate_diagnostics.png`.

### Shape → Cf at fixed ε — does not work

**23 CFD runs** varying `percent_bell`, `theta_n`, `theta_e` at ε = 7.706.
(The workbook has 24 rows with a valid Cf: the 23 sweep runs plus the separate
`baseline_validated` case, which is excluded from the sweep by default.)

| model | LOO RMSE | R² | vs mean-predictor |
|---|---:|---:|---:|
| GPR | 0.002253 | −0.659 | 23.2% **worse** |
| GBT | 0.002403 | −0.888 | 31.4% worse |
| Dummy (mean) | 0.001829 | −0.093 | — |

Cf varied by only 0.0068 across all 23 runs — at or below the CFD convergence
noise floor — and no feature correlated with Cf (all p > 0.4). No kernel or
model configuration beat the mean-predictor. Including the baseline row (n=24)
changes nothing qualitatively: R² = −0.467, still 16.1% worse than the mean.

**This is a genuine result, not a modelling failure:** at fixed ε, bell shape
has no detectable effect on Cf in this data, while ε itself is learned cleanly
from fewer runs still.

---

## Files

| file | role |
|---|---|
| `engine_sizing.py` | Stage 1 CEA sizing → `engine_design.json` |
| `nozzle_contour.py` | contour maths, closed-domain assembly, **`EXPANSION_RATIO`** |
| `nozzle_mesh.py` | structured axisymmetric quad mesh → Fluent `.msh` |
| `make_sweep_plan.py` | feasibility-constrained Latin Hypercube sampler |
| `batch_make_meshes.py` | builds all 30 sweep meshes |
| `compute_lengths.py`, `plot_contours.py`, `make_run_sheet.py` | reporting |
| `export_domain.py` | DXF export |
| `train_eps_surrogate.py`, `train_surrogate.py` | surrogates |
| `fluent validated case/` | validated baseline case — every manual run starts here |
| `eps_sweep_entry.xlsx` | ε-sweep CFD results (13 pts) |
| `nozzle_sweep_entry_constant_e.xlsx` | shape-sweep CFD results (23 runs + baseline) |

Not tracked: `.venv/`, `meshes/` (regenerate via `batch_make_meshes.py`),
figures, trained models (`*.joblib`) and Fluent binaries (`*.h5`) — all are
generated artefacts. Run the scripts to reproduce them.

---

## Geometry notes

Two constraints are enforced in `build_contour()` because violating them
silently produces invalid geometry:

- **Bell feasibility.** The quadratic-Bézier control point must lie between N
  and E, which holds iff `theta_e < chord_angle < theta_n`. Outside that, the
  wall folds back on itself. `make_sweep_plan.py` samples inside the feasible
  region so the clamp never fires; a fold-back still raises `ValueError`.
- **Corner preservation.** The chamber→cone junction is the only true C0 corner
  (35°); all other junctions are tangent-continuous. It is written as a curve
  *group boundary* so the importer cannot spline through and round it.

## Requirements

Python 3.14 + `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`,
`joblib`, `openpyxl`. `engine_sizing.py` additionally needs `rocketcea`.
