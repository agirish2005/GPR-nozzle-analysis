"""
Stage 2 - Parametric nozzle contour (Rao thrust-optimised parabolic / "TOP" bell)
=================================================================================
Generates the full axisymmetric wall contour (chamber -> converging cone ->
throat arcs -> parabolic bell) and exports it in formats ready for ANSYS
Fluent / DesignModeler / SpaceClaim.

Design variables for the surrogate study are the THREE bell parameters:
    percent_bell   fractional length vs a 15 deg conical nozzle  (~0.6-0.9)
    theta_n        parabola initial wall angle at N   [deg]      (~18-26)
    theta_e        parabola exit wall angle at E      [deg]      (~8-16)
Everything upstream of the throat (chamber + converging section) is held fixed,
so the sweep only reshapes the diverging bell -> low-dimensional, surrogate-friendly.

Geometry input (Rt, eps, Re) is read from engine_design.json written by Stage 1,
with a manual fallback block if you want to run this standalone.

Author: (your project)   |   Units: SI (metres) internally; export unit selectable.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
# Use absolute path based on script location
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DESIGN_JSON   = os.path.join(SCRIPT_DIR, "engine_design.json")  # produced by Stage 1

# --- Bell design variables (the ones you will sweep) ---
PERCENT_BELL  = 0.722   # Rao fractional length (Ln / L_15deg_cone)
THETA_N_DEG   = 22.963    # initial parabola wall angle
THETA_E_DEG   = 15.923 # exit  parabola wall angle
EXPANSION_RATIO = 7.706  # Ae/At; change this to set the nozzle exit area

# --- Fixed upstream / discretisation geometry ---
CONTRACTION_RATIO = 8.0  # Ac/At  -> sets chamber radius Rc = Rt*sqrt(CR)
CONV_HALF_ANGLE   = 35.0 # converging cone half-angle [deg]
CHAMBER_LEN_FACT  = 1.0  # straight chamber length as multiple of Rc (cosmetic)
RU_FACT           = 1.5  # throat UPSTREAM arc radius = RU_FACT * Rt (standard 1.5)
RD_FACT           = 0.382# throat DOWNSTREAM arc radius = RD_FACT * Rt (Rao std 0.382)
QX_CLAMP_FRAC     = 0.15 # if the Bezier control point falls outside N..E, clamp it
                         # this far (as a fraction of xE-xN) inside each end

N_ARC   = 40             # points per throat arc
N_CONE  = 20             # points on converging cone
N_PARAB = 120            # points on parabola
N_CHAM  = 10             # points on straight chamber

EXPORT_UNIT = "m"        # "m" or "mm" -> MUST match your DM/SpaceClaim session units
                         # (was "mm"; a metre-based DM session read those mm numbers
                         #  as metres and inflated the model exactly 1000x)
CORNER_TURN_DEG = 5.0    # a wall slope change bigger than this is a real corner;
                         # it starts a NEW curve group in the ANSYS point file so
                         # the importer cannot spline (and round) straight through it
# NOTE ON UNITS. A DesignModeler/SpaceClaim point file carries ONLY numbers -
# there is no unit token in the format - so the importer interprets them in
# whatever unit the CAD session is currently set to. Exporting mm numbers into
# a metre-based DM session inflates the model by exactly 1000x (a 152 mm nozzle
# becomes a 152 m one, and Meshing then hangs trying to fit controls to it).
# Set EXPORT_UNIT to match your DM/SpaceClaim session units.
EDGE_SPACING_MM = 2.0    # point spacing along the 3 straight closing edges [mm]
MAX_GAP_MM      = 5.0    # validation: max allowed gap between points [mm]

OUT_CSV     = "nozzle_contour.csv"
OUT_ANSYS   = "nozzle_curve_ansys.txt"      # WALL only (open curve)
OUT_DOMAIN  = "nozzle_domain_ansys.txt"     # CLOSED domain: wall+outlet+axis+inlet
OUT_PLOT    = "nozzle_contour.png"

# Manual fallback if engine_design.json is absent (Rt only - eps always comes
# from EXPANSION_RATIO above, so the fallback path honours the knob too)
FALLBACK = dict(Rt=0.01331)   # metres


# ----------------------------------------------------------------------------
def load_geometry(verbose=True):
    """
    Single source of truth for (Rt, eps) across the whole project.

    Rt comes from engine_design.json (Stage 1). eps comes from EXPANSION_RATIO
    at the top of THIS file, deliberately overriding the JSON, so that changing
    that one number reshapes every downstream artefact. Every other script must
    call this rather than hardcoding geometry.
    """
    if os.path.exists(DESIGN_JSON):
        with open(DESIGN_JSON) as f:
            d = json.load(f)
        Rt = d["Rt_m"]
        eps_json = d.get("eps")
        src = f"Rt from {os.path.basename(DESIGN_JSON)}"
    else:
        Rt, eps_json, src = FALLBACK["Rt"], None, "Rt from FALLBACK"
    eps = float(EXPANSION_RATIO)

    if verbose:
        print(f"[geom] {src}: Rt={Rt*1e3:.3f} mm | "
              f"eps={eps:.4f} from EXPANSION_RATIO (nozzle_contour.py)")
        if eps_json is not None and abs(eps_json - eps) > 1e-9:
            print(f"[geom] NOTE eps overrides engine_design.json ({eps_json:.4f}); "
                  f"that file's eps-derived values (Re_m, Ae_m2, Cf, Isp_*, "
                  f"Me_exit) are STALE - use exit_conditions() instead.")
    return Rt, eps


# --- eps-derived quantities, so nothing has to trust the stale JSON ---------
def _area_ratio(M, g):
    return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) \
        ** ((g + 1.0) / (2.0 * (g - 1.0)))


def mach_from_area(eps, g, lo=1.0000001, hi=50.0):
    """Supersonic Mach number for a given area ratio (bisection)."""
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if _area_ratio(mid, g) > eps:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def exit_conditions(Rt, eps, gamma, Pc_bar, Pamb_bar):
    """
    Ideal 1D exit state for the CURRENT eps. Use these instead of the
    eps-dependent fields in engine_design.json, which are only valid for the
    eps Stage 1 happened to compute.
    """
    Re = Rt * np.sqrt(eps)
    Me = mach_from_area(eps, gamma)
    Pe_Pc = (1.0 + 0.5 * (gamma - 1.0) * Me * Me) ** (-gamma / (gamma - 1.0))
    g = gamma
    cf_mom = np.sqrt((2.0 * g * g / (g - 1.0))
                     * (2.0 / (g + 1.0)) ** ((g + 1.0) / (g - 1.0))
                     * (1.0 - Pe_Pc ** ((g - 1.0) / g)))
    Cf = cf_mom + (Pe_Pc - Pamb_bar / Pc_bar) * eps
    return dict(Re_m=float(Re), Ae_m2=float(np.pi * Re * Re), Me_exit=float(Me),
                Pe_bar=float(Pe_Pc * Pc_bar), Cf_ideal=float(Cf), eps=float(eps))


def build_contour(Rt, eps, percent_bell, theta_n_deg, theta_e_deg):
    """Return x, r arrays (metres) of the wall from chamber inlet to nozzle exit."""
    Re  = Rt * np.sqrt(eps)
    Rc  = Rt * np.sqrt(CONTRACTION_RATIO)
    Ru  = RU_FACT * Rt
    Rd  = RD_FACT * Rt
    b   = np.radians(CONV_HALF_ANGLE)
    tn  = np.radians(theta_n_deg)
    te  = np.radians(theta_e_deg)

    # --- Throat DOWNSTREAM arc: throat (0,Rt) -> N, sweeping 0..theta_n ---
    t = np.linspace(0.0, tn, N_ARC)
    x_dn = Rd * np.sin(t)
    r_dn = (Rt + Rd) - Rd * np.cos(t)
    xN, rN = x_dn[-1], r_dn[-1]

    # --- Parabola (quadratic Bezier) N -> E, tangents theta_n, theta_e ---
    L15 = (Re - Rt) / np.tan(np.radians(15.0))   # reference 15-deg cone length
    Ln  = percent_bell * L15                       # bell length from throat
    xE, rE = Ln, Re
    m1, m2 = np.tan(tn), np.tan(te)
    Qx = (rE - rN + m1 * xN - m2 * xE) / (m1 - m2)   # tangent-line intersection

    # The tangent-line intersection is only a usable Bezier control point while
    # it lies strictly between N and E. For shallow theta_n / short bells it
    # lands outside that span (typically Qx > xE), which drives dx/ds negative
    # near s=1 and folds the wall back on itself - a self-intersecting, non
    # single-valued contour. Clamp Q back into the interior so r(x) stays a
    # function. NOTE: this preserves theta_n exactly (Qr is still built off m1)
    # but the REALISED exit angle then differs from the requested theta_e.
    Qx_raw = Qx
    lo = xN + QX_CLAMP_FRAC * (xE - xN)
    hi = xE - QX_CLAMP_FRAC * (xE - xN)
    clamped = bool(Qx <= xN or Qx >= xE)
    if clamped:
        Qx = float(np.clip(Qx, lo, hi))
        print(f"[contour] WARNING degenerate tangent intersection: "
              f"percent_bell={percent_bell:.4f}, theta_n={theta_n_deg:.3f}, "
              f"theta_e={theta_e_deg:.3f} -> Qx {Qx_raw * 1e3:.3f} mm clamped "
              f"to {Qx * 1e3:.3f} mm (valid range "
              f"[{lo * 1e3:.3f}, {hi * 1e3:.3f}] mm); realised exit angle will "
              f"differ from requested theta_e")

    Qr = rN + m1 * (Qx - xN)
    s  = np.linspace(0.0, 1.0, N_PARAB)
    x_pb = (1 - s) ** 2 * xN + 2 * (1 - s) * s * Qx + s ** 2 * xE
    r_pb = (1 - s) ** 2 * rN + 2 * (1 - s) * s * Qr + s ** 2 * rE

    # Never let a folded bell reach the mesh generator silently.
    if not np.all(np.diff(x_pb) >= -1e-9):
        raise ValueError(f"non-monotonic contour for percent_bell={percent_bell}, "
                         f"theta_n={theta_n_deg}, theta_e={theta_e_deg}")

    # --- Throat UPSTREAM arc: chamber side (t=b) -> throat (t=0) ---
    t = np.linspace(b, 0.0, N_ARC)
    x_up = -Ru * np.sin(t)
    r_up = (Rt + Ru) - Ru * np.cos(t)
    x_arc_cham, r_arc_cham = x_up[0], r_up[0]        # chamber-side end of arc

    # --- Converging cone: chamber (Rc) -> upstream-arc end, at angle beta ---
    dr      = Rc - r_arc_cham
    x_cone0 = x_arc_cham - dr / np.tan(b)
    x_cone  = np.linspace(x_cone0, x_arc_cham, N_CONE)
    r_cone  = Rc - (x_cone - x_cone0) * np.tan(b)

    # --- Straight chamber ---
    Lc     = CHAMBER_LEN_FACT * Rc
    x_cham = np.linspace(x_cone0 - Lc, x_cone0, N_CHAM)
    r_cham = np.full_like(x_cham, Rc)

    # --- Concatenate in increasing-x order (drop duplicate joints) ---
    xs = np.concatenate([x_cham, x_cone[1:], x_up[1:], x_dn[1:], x_pb[1:]])
    rs = np.concatenate([r_cham, r_cone[1:], r_up[1:], r_dn[1:], r_pb[1:]])

    meta = dict(Rt=Rt, Re=Re, Rc=Rc, eps=eps, Ln=Ln, L15=L15,
                xN=xN, rN=rN, xE=xE, rE=rE, x_inlet=xs[0],
                Qx=Qx, Qx_raw=Qx_raw, Qr=Qr, clamped=clamped,
                theta_e_realised=float(np.degrees(np.arctan2(rE - Qr, xE - Qx))))
    return xs, rs, meta


def corner_indices(xs, rs, thresh_deg=CORNER_TURN_DEG):
    """
    Point indices where the wall turns by more than thresh_deg - i.e. genuine
    C0 corners. For this contour that is only the chamber -> converging-cone
    join; every other segment junction is tangent-continuous by construction
    (the throat arcs and the parabola are all built tangent to their neighbour).
    """
    slope = np.degrees(np.arctan2(np.diff(rs), np.diff(xs)))
    turn = np.abs(np.diff(slope))
    return [int(i) + 1 for i in np.nonzero(turn > thresh_deg)[0]]


def _interior(p, q, spacing):
    """Points strictly between p and q at ~spacing, excluding both ends."""
    length = float(np.hypot(q[0] - p[0], q[1] - p[1]))
    n = max(int(np.ceil(length / spacing)), 1)
    if n < 2:
        return []
    ts = np.linspace(0.0, 1.0, n + 1)[1:-1]
    return [(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])) for t in ts]


def unit_scale(unit):
    """Metres -> export unit. The contour maths is always in SI metres."""
    if unit == "mm":
        return 1000.0
    if unit == "m":
        return 1.0
    raise ValueError(f"EXPORT_UNIT must be 'mm' or 'm', got {unit!r}")


def build_domain_groups(xs, rs, unit=EXPORT_UNIT):
    """
    Assemble the CLOSED 2D flow domain - wall, outlet, axis, inlet - as an
    ordered list of (name, points) curve groups.

    The wall alone is an open curve, which is why DesignModeler ends up with a
    surface model enclosing nothing and Fluent then reports "No regions were
    created. Disconnected surface models are not supported". Adding the three
    closing edges makes it a watertight loop that surfaces into one face.

    The four corners are each computed ONCE here and thereafter only reused as
    the end of one group and the start of the next, so both sides of every
    junction are bit-identical - independently recomputed corners are exactly
    what leaves the hairline gaps that read as a disconnected model.
    """
    scale = unit_scale(unit)
    X, R = xs * scale, rs * scale
    # spacing is defined in mm; convert into whatever unit we are writing,
    # otherwise a metre export would space points 2 METRES apart
    spacing = EDGE_SPACING_MM * scale / 1000.0

    C_iw = (float(X[0]), float(R[0]))      # inlet  -> wall   (chamber lip)
    C_wo = (float(X[-1]), float(R[-1]))    # wall   -> outlet (exit lip)
    C_oa = (float(X[-1]), 0.0)             # outlet -> axis   (x reused exactly)
    C_ai = (float(X[0]), 0.0)              # axis   -> inlet  (x reused exactly)

    wall = [(float(x), float(r)) for x, r in zip(X, R)]

    # split the wall at its own sharp corner so the importer cannot spline
    # through it (same reason as the wall-only export above)
    ci = corner_indices(xs, rs)
    if ci:
        wall_groups = [("chamber", wall[:ci[0] + 1]), ("wall", wall[ci[0]:])]
    else:
        wall_groups = [("wall", wall)]

    groups = wall_groups + [
        ("outlet", [C_wo] + _interior(C_wo, C_oa, spacing) + [C_oa]),
        ("axis",   [C_oa] + _interior(C_oa, C_ai, spacing) + [C_ai]),
        ("inlet",  [C_ai] + _interior(C_ai, C_iw, spacing) + [C_iw]),
    ]
    corners = {
        "inlet  -> wall   (chamber lip)": C_iw,
        "wall   -> outlet (exit lip)": C_wo,
        "outlet -> axis": C_oa,
        "axis   -> inlet": C_ai,
    }
    return groups, corners


def validate_domain(groups, corners, unit=EXPORT_UNIT):
    """Closure / gap / corner-coincidence checks. Returns True on PASS."""
    pts = [p for _, g in groups for p in g]
    ok = True
    scale = unit_scale(unit)
    max_gap = MAX_GAP_MM * scale / 1000.0      # limit expressed in export units

    print(f"\n--- domain corner junctions ({unit}) ---")
    for name, (x, y) in corners.items():
        print(f"  {name:32s} x = {x:12.6f}   r = {y:12.6f}")

    print("--- closed-loop validation ---")
    # (a) closed: last point of the last group == first point of the first
    a_ok = groups[-1][1][-1] == groups[0][1][0]
    ok &= a_ok
    print(f"  (a) loop closes          : {'PASS' if a_ok else 'FAIL'}"
          f"{'   (bit-identical)' if a_ok else ''}")

    # (b) no gaps / duplicate points within each group
    seg = [float(np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]))
           for i in range(len(pts) - 1)]
    seg = [s for s in seg if s > 0.0]          # group joins repeat a point by design
    b_ok = max(seg) <= max_gap
    ok &= b_ok
    print(f"  (b) no gaps              : {'PASS' if b_ok else 'FAIL'}   "
          f"max seg = {max(seg):.6g} {unit} (limit {max_gap:.6g}), "
          f"min seg = {min(seg):.6g} {unit}")

    # (c) every junction exactly coincident
    checks = []
    for (na, ga), (nb, gb) in zip(groups, groups[1:]):
        checks.append((f"{na} end == {nb} start", ga[-1] == gb[0]))
    checks.append((f"{groups[-1][0]} end == {groups[0][0]} start",
                   groups[-1][1][-1] == groups[0][1][0]))
    c_ok = all(v for _, v in checks)
    ok &= c_ok
    print(f"  (c) junctions coincident : {'PASS' if c_ok else 'FAIL'}")
    for label, v in checks:
        print(f"        {'ok ' if v else 'BAD'}  {label}")

    ring = pts + [pts[0]]
    area = 0.5 * sum(ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
                     for i in range(len(ring) - 1))
    print(f"  enclosed area            : {abs(area):.6g} {unit}^2")

    # scale sanity: the model must import at real nozzle size, not 1000x it
    xs_ = [p[0] for p in pts]; rs_ = [p[1] for p in pts]
    diag = float(np.hypot(max(xs_) - min(xs_), max(rs_) - min(rs_)))
    diag_mm = diag * 1000.0 / scale
    print(f"  bbox diagonal            : {diag:.6g} {unit}  (= {diag_mm:.2f} mm true size)")
    print(f"  >> the file contains BARE NUMBERS in {unit}. Set your DesignModeler /")
    print(f"  >> SpaceClaim session units to {unit} BEFORE importing, or the model")
    print(f"  >> will be off by 1000x and Meshing will hang.")
    print(f"  OVERALL: {'PASS - watertight closed region' if ok else 'FAIL'}")
    return ok


def export_domain_file(groups):
    """Write the closed domain as a DesignModeler 3D-curve point file."""
    with open(OUT_DOMAIN, "w") as f:
        for gi, (_, grp) in enumerate(groups, start=1):
            for pi, (x, y) in enumerate(grp, start=1):
                f.write(f"{gi} {pi} {x:.6f} {y:.6f} 0.0\n")
    sizes = ", ".join(f"{name}={len(g)}" for name, g in groups)
    print(f"[export] wrote {OUT_DOMAIN}  ({len(groups)} groups: {sizes})")


def export(xs, rs, unit):
    scale = 1000.0 if unit == "mm" else 1.0
    X, R = xs * scale, rs * scale

    # Plain CSV (x, r)
    np.savetxt(OUT_CSV, np.column_stack([X, R]),
               delimiter=",", header=f"x[{unit}],r[{unit}]", comments="")

    # ANSYS DesignModeler / SpaceClaim 3D-curve point file:
    #   columns:  group  point  X  Y  Z   (axis = X, radius = Y, Z = 0)
    #
    # The FIRST column is the curve-group id, and the importer fits ONE spline
    # through each group. Writing every point as group 1 therefore splines
    # straight through the chamber->cone corner and rounds it off. Emit one
    # group per smooth span instead, so each real corner falls on a group
    # BOUNDARY. The corner point is repeated - last point of one group, first
    # point of the next - so the two curves share a vertex and meet sharply.
    corners = corner_indices(xs, rs)
    bounds = [0] + corners + [len(X) - 1]
    nlines = 0
    with open(OUT_ANSYS, "w") as f:
        for g, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
            for p, i in enumerate(range(a, b + 1), start=1):
                f.write(f"{g} {p} {X[i]:.6f} {R[i]:.6f} 0.0\n")
                nlines += 1

    print(f"[export] wrote {OUT_CSV} and {OUT_ANSYS}  ({len(X)} pts, unit={unit})")
    if corners:
        loc = ", ".join(f"pt {c + 1} at x={X[c]:.4f} mm" for c in corners)
        print(f"[export] {len(bounds) - 1} curve groups; sharp corner(s) kept at: {loc}")
    else:
        print(f"[export] 1 curve group (no corners above {CORNER_TURN_DEG} deg)")


def main():
    Rt, eps = load_geometry()
    xs, rs, m = build_contour(Rt, eps, PERCENT_BELL, THETA_N_DEG, THETA_E_DEG)

    print("\n--- contour summary (mm) ---")
    print(f"  throat radius Rt : {m['Rt']*1e3:8.3f}")
    print(f"  exit   radius Re : {m['Re']*1e3:8.3f}")
    print(f"  chamber radius Rc: {m['Rc']*1e3:8.3f}")
    print(f"  bell length Ln   : {m['Ln']*1e3:8.3f}  ({PERCENT_BELL*100:.0f}% of 15deg cone {m['L15']*1e3:.3f})")
    print(f"  overall length   : {(xs[-1]-xs[0])*1e3:8.3f}")

    export(xs, rs, EXPORT_UNIT)

    # Closed flow domain for direct DesignModeler import (no manual redraw)
    groups, corners = build_domain_groups(xs, rs, EXPORT_UNIT)
    validate_domain(groups, corners)
    export_domain_file(groups)

    # Quick visual check
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(xs * 1e3, rs * 1e3, "-", lw=1.6, color="#c0392b")
    ax.plot(xs * 1e3, -rs * 1e3, "-", lw=1.6, color="#c0392b")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.plot(m["xN"] * 1e3, m["rN"] * 1e3, "o", ms=4, color="#2c3e50", label="N (arc->parabola)")
    ax.plot(m["xE"] * 1e3, m["rE"] * 1e3, "s", ms=4, color="#2980b9", label="E (exit)")
    ax.set_aspect("equal")
    ax.set_xlabel("axial x [mm]"); ax.set_ylabel("radius r [mm]")
    ax.set_title(f"Rao TOP bell  |  eps={eps:.2f}, {PERCENT_BELL*100:.0f}% bell, "
                 f"theta_n={THETA_N_DEG}, theta_e={THETA_E_DEG}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_PLOT, dpi=130)
    print(f"[plot] wrote {OUT_PLOT}")


if __name__ == "__main__":
    main()
