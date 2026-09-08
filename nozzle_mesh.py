"""
nozzle_mesh.py - structured 2D axisymmetric quad mesh -> Fluent .msh
====================================================================
Takes the three bell parameters, builds the closed nozzle profile (reusing the
Stage-2 contour construction), maps a STRUCTURED quad grid onto it with strong
wall clustering (for y+ ~ 2), and writes a Fluent 2D mesh (.msh v2 ASCII) with
named boundaries: inlet, outlet, wall, axis.

No SpaceClaim, no STL, no faceting. Exact curves, exact first-cell height.

Zones written:
    interior + one fluid cell zone
    inlet   (left,  x = x_min, r: 0..Rc)
    outlet  (right, x = x_max, r: 0..Re)
    wall    (the nozzle contour, top)
    axis    (r = 0, bottom)

Grid layout (logical i x j):
    i : streamwise  (inlet -> outlet)   NI points
    j : wall-normal (axis  -> wall)      NJ points, clustered toward wall (j=NJ-1)

This is a SINGLE-BLOCK structured mesh: every streamwise station is a straight
radial line from axis (r=0) to the wall contour r_wall(x). That is exact for a
nozzle because the wall is a single-valued function r(x).
"""

import numpy as np

# ---- import the contour construction from Stage 2 --------------------------
# (build_contour returns xs, rs of the wall from inlet to exit)
from nozzle_contour import build_contour, CONTRACTION_RATIO   # reuse exactly

# ----------------------------------------------------------------------------
def wall_profile(Rt, eps, percent_bell, theta_n, theta_e, n=400):
    """Dense (x, r_wall) sampling of the wall, resampled to uniform-ish x."""
    xs, rs, meta = build_contour(Rt, eps, percent_bell, theta_n, theta_e)
    # build_contour already returns increasing-x wall points; densify by interp
    x_uniform = np.linspace(xs.min(), xs.max(), n)
    r_wall = np.interp(x_uniform, xs, rs)
    return x_uniform, r_wall, meta


def _solve_growth(L, h1, n, lo=1.0000001, hi=2.0):
    """Solve h1*(r^n - 1)/(r - 1) = L for growth ratio r (bisection)."""
    if h1 * n >= L:            # can't fit; fall back to uniform
        return None
    def f(r):
        return h1 * (r**n - 1.0) / (r - 1.0) - L
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


AXIS_EPS_M = 1e-6   # centerline node offset off r=0 (Fluent axisymmetric FPE fix)


def radial_nodes(L, h1, n):
    """
    Return n+1 node radii from ~0 (axis) to L (wall), with the SMALLEST cell
    (~h1) adjacent to the wall. Geometric growth from wall inward. The
    innermost node is offset to AXIS_EPS_M instead of sitting at exactly
    r=0, to avoid a division-by-zero in Fluent's axisymmetric solver.
    """
    r = _solve_growth(L, h1, n)
    if r is None:                          # uniform fallback
        radii = np.linspace(0.0, L, n + 1)
    else:
        # distances from the WALL inward: d_k = h1*(r^k - 1)/(r - 1), k=0..n
        k = np.arange(n + 1)
        d = h1 * (r**k - 1.0) / (r - 1.0)
        d[-1] = L                          # pin last exactly to axis
        radii = L - d                      # j=0 -> axis(0)? no: reorder below
        radii = radii[::-1]                # now index 0 = axis, last = wall
    radii[0] = AXIS_EPS_M
    return radii


def build_structured(Rt, eps, percent_bell, theta_n, theta_e,
                     NI=240, NJ=80, first_cell_m=3e-6, growth=1.15):
    """Return node arrays X[NI,NJ], R[NI,NJ] for the structured mesh."""
    x, r_wall, meta = wall_profile(Rt, eps, percent_bell, theta_n, theta_e, n=NI)

    # Per-station geometric distribution: constant PHYSICAL first-cell height
    # (~first_cell_m) at the wall everywhere, growing inward toward the axis.
    X = np.zeros((NI, NJ))
    R = np.zeros((NI, NJ))
    for i in range(NI):
        X[i, :] = x[i]
        R[i, :] = radial_nodes(r_wall[i], first_cell_m, NJ - 1)  # NJ nodes
    return X, R, meta


# ----------------------------------------------------------------------------
def cell_metrics(X, R):
    """
    Per-cell planar cross-section area (x-r plane, shoelace) and axisymmetric
    revolved volume-per-radian (two-triangle Pappus split) for every quad cell
    in a structured X[NI,NJ]/R[NI,NJ] grid. Returns (areas, volumes), each
    shaped (NI-1, NJ-1).
    """
    NI, NJ = X.shape
    areas = np.zeros((NI - 1, NJ - 1))
    volumes = np.zeros((NI - 1, NJ - 1))
    for i in range(NI - 1):
        for j in range(NJ - 1):
            x0, r0 = X[i, j], R[i, j]
            x1, r1 = X[i + 1, j], R[i + 1, j]
            x2, r2 = X[i + 1, j + 1], R[i + 1, j + 1]
            x3, r3 = X[i, j + 1], R[i, j + 1]
            areas[i, j] = 0.5 * abs(
                (x0 * r1 - x1 * r0) + (x1 * r2 - x2 * r1)
                + (x2 * r3 - x3 * r2) + (x3 * r0 - x0 * r3)
            )

            def tri_vol(xa, ra, xb, rb, xc, rc):
                a = 0.5 * abs((xb - xa) * (rc - ra) - (xc - xa) * (rb - ra))
                rbar = (ra + rb + rc) / 3.0
                return a * rbar

            volumes[i, j] = (
                tri_vol(x0, r0, x1, r1, x2, r2) + tri_vol(x0, r0, x2, r2, x3, r3)
            )
    return areas, volumes


def check_mesh_quality(X, R, label=""):
    """Print min/max cell area and revolved volume, flagging any degenerate
    (zero or negative) cell anywhere, and specifically in the axis row (j=0)."""
    tag = f" [{label}]" if label else ""
    areas, volumes = cell_metrics(X, R)
    axis_areas, axis_vols = areas[:, 0], volumes[:, 0]

    print(f"[quality{tag}] global min cell area   = {areas.min():.3e} m^2")
    print(f"[quality{tag}] axis-row min cell area  = {axis_areas.min():.3e} m^2")
    print(f"[quality{tag}] global min cell volume  = {volumes.min():.3e} m^3/rad")
    print(f"[quality{tag}] axis-row min cell volume= {axis_vols.min():.3e} m^3/rad")

    degenerate = volumes.min() <= 0.0 or areas.min() <= 0.0
    if degenerate:
        print(f"[quality{tag}] DEGENERATE CELL DETECTED (area or volume <= 0)")
    else:
        print(f"[quality{tag}] OK - no degenerate (zero/negative) cells")
    return areas, volumes


# ----------------------------------------------------------------------------
def write_fluent_msh(path, X, R):
    """
    Write a minimal Fluent 2D mesh (ASCII, format 'MSH' v2) from a structured
    NI x NJ node grid. Quad cells. Boundaries tagged inlet/outlet/wall/axis.
    """
    NI, NJ = X.shape
    # node ids: n(i,j) = i*NJ + j + 1   (1-based, Fluent is 1-based hex ids)
    def nid(i, j):
        return i * NJ + j + 1
    nnodes = NI * NJ
    ncells = (NI - 1) * (NJ - 1)

    nodes = []
    for i in range(NI):
        for j in range(NJ):
            nodes.append((X[i, j], R[i, j]))

    # cell ids: c(i,j) = i*(NJ-1) + j + 1  for i in 0..NI-2, j in 0..NJ-2
    def cid(i, j):
        return i * (NJ - 1) + j + 1

    # ---- build faces with owner/neighbour + zone tags ----
    # face types: interior, inlet(i=0), outlet(i=NI-1), axis(j=0), wall(j=NJ-1)
    interior, inlet, outlet, axis_f, wall_f = [], [], [], [], []

    # i-direction faces (constant i between nodes j..j+1): separate cells (i-1,j)|(i,j)
    for i in range(NI):
        for j in range(NJ - 1):
            n0, n1 = nid(i, j), nid(i, j + 1)
            left = cid(i - 1, j) if i - 1 >= 0 else 0
            right = cid(i, j) if i <= NI - 2 else 0
            if i == 0:
                inlet.append((n0, n1, 0, right))
            elif i == NI - 1:
                outlet.append((n0, n1, left, 0))
            else:
                interior.append((n0, n1, left, right))

    # j-direction faces (constant j between nodes i..i+1): separate (i,j-1)|(i,j)
    for j in range(NJ):
        for i in range(NI - 1):
            n0, n1 = nid(i, j), nid(i + 1, j)
            below = cid(i, j - 1) if j - 1 >= 0 else 0
            above = cid(i, j) if j <= NJ - 2 else 0
            if j == 0:
                axis_f.append((n0, n1, 0, above))
            elif j == NJ - 1:
                wall_f.append((n0, n1, below, 0))
            else:
                interior.append((n0, n1, below, above))

    # ---- zone ids ----
    Z_INTERIOR, Z_INLET, Z_OUTLET, Z_WALL, Z_AXIS = 10, 11, 12, 13, 14
    CELL_ZONE, NODE_ZONE = 2, 1

    def hex_(n):
        return format(n, "x")

    with open(path, "w") as f:
        f.write("(0 \"nozzle 2D axisymmetric structured mesh\")\n")
        f.write("(2 2)\n")  # dimension = 2

        # nodes
        f.write(f"(10 (0 1 {hex_(nnodes)} 0 2))\n")
        f.write(f"(10 ({hex_(NODE_ZONE)} 1 {hex_(nnodes)} 1 2)(\n")
        for (xx, rr) in nodes:
            f.write(f"{xx:.10e} {rr:.10e}\n")
        f.write("))\n")

        # cells - placeholder total declaration now; the real typed zone is
        # declared after the faces section (faces are what define the cells).
        f.write(f"(12 (0 1 {hex_(ncells)} 0))\n")

        # faces - one section per zone. face format: type n0 n1 cL cR (type 2 = linear)
        # Face indices are global across the whole (13 ...) face table, so each
        # zone must claim its own non-overlapping first..last slice, not restart at 1.
        face_offset = [0]
        def write_faces(zone_id, faces, bc_type):
            n = len(faces)
            first = face_offset[0] + 1
            last = face_offset[0] + n
            face_offset[0] = last
            f.write(f"(13 ({hex_(zone_id)} {hex_(first)} {hex_(last)} {bc_type} 2)(\n")
            for (a, b, cl, cr) in faces:
                f.write(f"{hex_(a)} {hex_(b)} {hex_(cl)} {hex_(cr)}\n")
            f.write("))\n")

        total_faces = len(interior)+len(inlet)+len(outlet)+len(wall_f)+len(axis_f)
        f.write(f"(13 (0 1 {hex_(total_faces)} 0))\n")
        write_faces(Z_INTERIOR, interior, 2)   # 2 = interior
        write_faces(Z_INLET,   inlet,   4)     # 4 = pressure/velocity inlet (generic)
        write_faces(Z_OUTLET,  outlet,  5)     # 5 = pressure outlet
        write_faces(Z_WALL,    wall_f,  3)     # 3 = wall
        write_faces(Z_AXIS,    axis_f,  37)    # 37 = axis (legacy v2 ascii DOES have a dedicated
                                                # axis bc-type code; using 7/symmetry here mistags
                                                # the r=0 boundary as a symmetry plane instead of an
                                                # axis, which starves the axisymmetric solver of the
                                                # 1/r pole treatment and causes floating-point
                                                # exceptions right at the centerline)

        # real typed cell zone, declared after the faces that define it
        f.write(f"(12 ({hex_(CELL_ZONE)} 1 {hex_(ncells)} 1 3))\n")

        # zone names
        f.write(f"(45 ({hex_(CELL_ZONE)} fluid fff)())\n")
        f.write(f"(45 ({hex_(Z_INLET)} pressure-inlet inlet)())\n")
        f.write(f"(45 ({hex_(Z_OUTLET)} pressure-outlet outlet)())\n")
        f.write(f"(45 ({hex_(Z_WALL)} wall wall)())\n")
        f.write(f"(45 ({hex_(Z_AXIS)} axis axis)())\n")

    return nnodes, ncells


if __name__ == "__main__":
    # baseline params for a self-test - geometry from the single source of
    # truth so this honours EXPANSION_RATIO like everything else
    from nozzle_contour import load_geometry
    Rt, eps = load_geometry()
    X, R, meta = build_structured(Rt, eps, 0.80, 22.0, 14.0,
                                  NI=240, NJ=80, first_cell_m=3e-6)
    nn, nc = write_fluent_msh("nozzle.msh", X, R)
    print(f"nodes={nn} cells={nc}  (Rt={Rt*1e3:.2f}mm eps={eps})")
    print(f"throat first-cell target = 3 um; NI x NJ = {X.shape}")