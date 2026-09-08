"""
export_domain.py - closed 2D flow domain for the Fluid Flow (Fluent) route
===========================================================================
nozzle_contour.py exports only the WALL curve. That is an open curve, so
DesignModeler / SpaceClaim end up with a surface model that encloses nothing
and Fluent meshing fails with

    "No regions were created. Disconnected surface models are not supported"

This module assembles the FULL domain boundary as one ordered, watertight
loop - wall, outlet, axis, inlet - and exports it in a form that imports as a
single closed region. It does not touch the contour shape maths: it calls
nozzle_contour.build_contour() and only adds the three closing edges.

Loop traversal in the x-r plane:

   inlet-wall corner  o---------------- wall ----------------o  wall-outlet corner
                      |                                      |
                    inlet                                  outlet
                      |                                      |
   axis-inlet corner  o------------ axis (r = 0) ------------o  outlet-axis corner

The four corners are each computed ONCE and then reused verbatim as the end of
one edge and the start of the next, so every junction is bit-identical rather
than merely "close". Independently recomputed corners are exactly what leaves
the hairline mismatches that read as a disconnected model.

Outputs
-------
nozzle_domain.dxf        RECOMMENDED. One closed POLYLINE = one closed planar
                         region. SpaceClaim / DesignModeler import this
                         directly as a closed sketch profile which surfaces
                         into a single face - no edge-joining step, and no
                         import tolerance to fight.
nozzle_domain_ansys.txt is written by nozzle_contour.py, NOT here - one
writer per file, so the two cannot disagree on units.
"""
import math
import os

import numpy as np

from nozzle_contour import (EXPORT_UNIT, build_contour, build_domain_groups,
                            load_geometry, validate_domain)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DXF = os.path.join(SCRIPT_DIR, "nozzle_domain.dxf")

# Baseline shape, passed explicitly so this script never depends on whatever
# nozzle_contour.py's module-level PERCENT_BELL / THETA_*_DEG are set to.
BASELINE = dict(percent_bell=0.80, theta_n_deg=22.0, theta_e_deg=14.0)

# NOTE: point spacing, gap limits and export units are all owned by
# nozzle_contour.py (EDGE_SPACING_MM / MAX_GAP_MM / EXPORT_UNIT). Nothing is
# redefined here on purpose - one owner per setting.


# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def write_dxf(path, loop):
    """
    Minimal R12 ASCII DXF holding ONE closed POLYLINE. The closed flag (group
    code 70 = 1) is what makes the importer treat it as a single closed region,
    so the first vertex is NOT repeated at the end - that would add a
    zero-length segment.
    """
    pts = loop[:-1]                      # drop the explicit closing duplicate
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    out = []

    def g(code, val):
        out.append(str(code))
        out.append(str(val))

    g(0, "SECTION"); g(2, "HEADER")
    g(9, "$ACADVER"); g(1, "AC1009")
    g(9, "$EXTMIN"); g(10, f"{min(xs):.6f}"); g(20, f"{min(ys):.6f}"); g(30, "0.0")
    g(9, "$EXTMAX"); g(10, f"{max(xs):.6f}"); g(20, f"{max(ys):.6f}"); g(30, "0.0")
    g(0, "ENDSEC")

    g(0, "SECTION"); g(2, "ENTITIES")
    g(0, "POLYLINE"); g(8, "NOZZLE"); g(66, 1); g(70, 1)   # 70 = 1 -> CLOSED
    for x, y in pts:
        g(0, "VERTEX"); g(8, "NOZZLE")
        g(10, f"{x:.6f}"); g(20, f"{y:.6f}"); g(30, "0.0")
    g(0, "SEQEND"); g(8, "NOZZLE")
    g(0, "ENDSEC"); g(0, "EOF")

    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return len(pts)



# ----------------------------------------------------------------------------
def main():
    """
    DXF-only exporter. The closed loop itself is built by
    nozzle_contour.build_domain_groups() - the single implementation - so the
    two exporters cannot drift apart on units or geometry. nozzle_contour.py
    owns nozzle_domain_ansys.txt; this script must NOT write that file.
    """
    Rt, eps = load_geometry()
    xs, rs, meta = build_contour(Rt, eps, BASELINE["percent_bell"],
                                 BASELINE["theta_n_deg"], BASELINE["theta_e_deg"])
    print(f"[domain] baseline percent_bell={BASELINE['percent_bell']}, "
          f"theta_n={BASELINE['theta_n_deg']}, theta_e={BASELINE['theta_e_deg']}")
    print(f"[domain] Rt={Rt * 1e3:.4f} mm, eps={eps:.4f}, "
          f"Re={meta['Re'] * 1e3:.4f} mm, Rc={meta['Rc'] * 1e3:.4f} mm")

    groups, corners = build_domain_groups(xs, rs, EXPORT_UNIT)
    ok = validate_domain(groups, corners, EXPORT_UNIT)

    # Flatten the groups into one ring. Consecutive groups deliberately REPEAT
    # their shared junction point, so drop those repeats here - a duplicated
    # vertex would become a zero-length DXF segment.
    loop = []
    for _, g in groups:
        for p in g:
            if not loop or p != loop[-1]:
                loop.append(p)
    if loop[-1] == loop[0]:                   # ring already meets at the start
        loop.pop()
    loop.append(loop[0])                      # explicit closing duplicate
    npts = write_dxf(OUT_DXF, loop)
    print(f"\n[domain] wrote {OUT_DXF}")
    print(f"[domain]   1 closed POLYLINE, {npts} vertices, units = {EXPORT_UNIT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
