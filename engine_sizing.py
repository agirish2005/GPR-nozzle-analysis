"""
Stage 1 - Engine sizing with NASA CEA (via rocketcea) for LOX/CH4
=================================================================
Sweeps O/F, picks the mixture ratio, sizes throat/exit from ideal rocket
theory, and writes engine_design.json for Stage 2 (nozzle_contour.py) and
the Fluent gas-property inputs.

INSTALL (do this on your WSL/Ubuntu box):
    sudo apt install gfortran           # CEA is FORTRAN; needs a compiler
    pip install rocketcea

WHAT YOU GET OUT:
    - Isp / chamber-temperature vs O/F sweep (plot + printed table)
    - chosen design point (MR, eps) at a target exit pressure
    - throat/exit areas & radii for the requested thrust
    - gas properties (gamma, R, T0) to feed the Fluent density-based solver
    - engine_design.json handoff file

Units: everything SI (bar, K, m/s) via rocketcea's units-aware CEA_Obj.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from rocketcea.cea_obj_w_units import CEA_Obj

# ----------------------------------------------------------------------------
# DESIGN INPUTS  (edit these)
# ----------------------------------------------------------------------------
OX, FUEL   = "LOX", "CH4"     # methalox (Raptor-class). Swap FUEL="RP1" for kerolox.
F_TARGET   = 5000.0           # design thrust [N]
PC_BAR     = 50.0             # chamber pressure [bar]
PE_BAR     = 1.013            # design exit pressure [bar]  (1.013 = sea-level optimum)
PAMB_BAR   = 1.013            # ambient pressure at design altitude [bar]
MR_RANGE   = (2.4, 4.2)       # O/F sweep range
MR_N       = 37               # sweep resolution
MR_OVERRIDE = None            # set a float to force MR; None = pick max delivered Isp

G0         = 9.80665
OUT_JSON   = "engine_design.json"
OUT_PLOT   = "sizing_sweep.png"


# ----------------------------------------------------------------------------
def make_cea():
    return CEA_Obj(
        oxName=OX, fuelName=FUEL,
        isp_units="sec",
        cstar_units="m/s",
        pressure_units="bar",
        temperature_units="K",
        sonic_velocity_units="m/s",
        density_units="kg/m^3",
        specific_heat_units="J/kg-K",
    )


def area_ratio(M, g):
    return (1.0 / M) * ((2.0 / (g + 1.0)) *
                        (1.0 + 0.5 * (g - 1.0) * M * M)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def mach_from_area(eps, g):
    """Supersonic root of the area-Mach relation by bisection."""
    lo, hi = 1.0 + 1e-6, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if area_ratio(mid, g) > eps:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def ambient_isp(C, Pc, MR, eps, Pamb):
    """Delivered Isp at ambient pressure, with a graceful fallback."""
    try:
        isp_amb, _mode = C.estimate_Ambient_Isp(Pc=Pc, MR=MR, eps=eps, Pamb=Pamb)
        return float(isp_amb)
    except Exception:
        # fallback: Isp_amb = Isp_vac - (Pe*Ae)/(mdot*g0); approximate via c*,Cf
        isp_vac = C.get_Isp(Pc=Pc, MR=MR, eps=eps)
        cstar   = C.get_Cstar(Pc=Pc, MR=MR)
        # vacuum Cf from Isp_vac, subtract pressure term (Pamb-0)*eps/Pc
        cf_vac  = isp_vac * G0 / cstar
        cf_amb  = cf_vac - (Pamb / Pc) * eps
        return cf_amb * cstar / G0


def main():
    C = make_cea()
    PcOvPe = PC_BAR / PE_BAR

    # --- O/F sweep at the design exit-pressure condition -------------------
    MRs = np.linspace(*MR_RANGE, MR_N)
    isp_amb, isp_vac, Tc, eps_arr = [], [], [], []
    for MR in MRs:
        eps = C.get_eps_at_PcOvPe(Pc=PC_BAR, MR=MR, PcOvPe=PcOvPe)
        eps_arr.append(eps)
        isp_vac.append(C.get_Isp(Pc=PC_BAR, MR=MR, eps=eps))
        isp_amb.append(ambient_isp(C, PC_BAR, MR, eps, PAMB_BAR))
        Tc.append(C.get_Tcomb(Pc=PC_BAR, MR=MR))
    isp_amb = np.array(isp_amb); isp_vac = np.array(isp_vac)
    Tc = np.array(Tc); eps_arr = np.array(eps_arr)

    MR = MR_OVERRIDE if MR_OVERRIDE else float(MRs[int(np.argmax(isp_amb))])
    print(f"[sweep] max delivered Isp at O/F = {MRs[int(np.argmax(isp_amb))]:.3f} "
          f"(Isp_amb={isp_amb.max():.1f} s).  Real methalox engines often run "
          f"~5-8% richer for wall temperature.")
    print(f"[design] using O/F = {MR:.3f}")

    # --- Design-point properties -------------------------------------------
    eps      = float(C.get_eps_at_PcOvPe(Pc=PC_BAR, MR=MR, PcOvPe=PcOvPe))
    cstar    = float(C.get_Cstar(Pc=PC_BAR, MR=MR))
    Tc_d, Tt_d, Te_d = [float(x) for x in C.get_Temperatures(Pc=PC_BAR, MR=MR, eps=eps)]
    mw_c, g_c = [float(x) for x in C.get_Chamber_MolWt_gamma(Pc=PC_BAR, MR=MR, eps=eps)]
    try:
        mw_e, g_e = [float(x) for x in C.get_exit_MolWt_gamma(Pc=PC_BAR, MR=MR, eps=eps)]
    except Exception:
        mw_e, g_e = mw_c, g_c
    isp_amb_d = ambient_isp(C, PC_BAR, MR, eps, PAMB_BAR)
    isp_vac_d = float(C.get_Isp(Pc=PC_BAR, MR=MR, eps=eps))

    R_spec = 8314.462 / mw_c                       # J/kg-K  (chamber gas)
    Pc_pa  = PC_BAR * 1e5

    # --- Size from ideal rocket theory -------------------------------------
    mdot = F_TARGET / (isp_amb_d * G0)             # kg/s
    At   = mdot * cstar / Pc_pa                     # m^2   (c* = Pc*At/mdot)
    Rt   = np.sqrt(At / np.pi)
    Ae   = eps * At
    Re   = np.sqrt(Ae / np.pi)
    Me   = mach_from_area(eps, g_e)
    Cf   = isp_amb_d * G0 / cstar

    # --- Report -------------------------------------------------------------
    print("\n================  DESIGN POINT  ================")
    print(f"  propellants        : {OX}/{FUEL}")
    print(f"  O/F (MR)           : {MR:.3f}")
    print(f"  Pc / Pe / Pamb     : {PC_BAR:.1f} / {PE_BAR:.3f} / {PAMB_BAR:.3f} bar")
    print(f"  expansion ratio eps: {eps:.3f}")
    print(f"  c*                 : {cstar:.1f} m/s")
    print(f"  Isp (vac / amb)    : {isp_vac_d:.1f} / {isp_amb_d:.1f} s")
    print(f"  Cf (ambient)       : {Cf:.4f}")
    print(f"  Tc / Tthroat / Te  : {Tc_d:.0f} / {Tt_d:.0f} / {Te_d:.0f} K")
    print(f"  gamma (cham / exit): {g_c:.4f} / {g_e:.4f}")
    print(f"  MW (chamber)       : {mw_c:.3f} g/mol   ->  R = {R_spec:.1f} J/kg-K")
    print(f"  mdot               : {mdot:.4f} kg/s")
    print(f"  throat  At / Rt    : {At*1e6:.2f} mm^2 / {Rt*1e3:.3f} mm")
    print(f"  exit    Ae / Re    : {Ae*1e6:.2f} mm^2 / {Re*1e3:.3f} mm")
    print(f"  exit Mach (ideal)  : {Me:.3f}")
    print("================================================\n")

    # --- Handoff JSON (Stage 2 reads Rt_m, eps; Fluent reads gas props) ----
    design = dict(
        propellants=f"{OX}/{FUEL}", MR=MR, Pc_bar=PC_BAR, Pe_bar=PE_BAR, Pamb_bar=PAMB_BAR,
        eps=eps, cstar_ms=cstar, Isp_vac_s=isp_vac_d, Isp_amb_s=isp_amb_d, Cf=Cf,
        Tc_K=Tc_d, Tthroat_K=Tt_d, Texit_K=Te_d,
        gamma_chamber=g_c, gamma_exit=g_e, MW_chamber_gmol=mw_c, R_specific_JkgK=R_spec,
        mdot_kgs=mdot, At_m2=At, Rt_m=Rt, Ae_m2=Ae, Re_m=Re, Me_exit=Me,
    )
    with open(OUT_JSON, "w") as f:
        json.dump(design, f, indent=2)
    print(f"[handoff] wrote {OUT_JSON}")

    # --- Sweep plot ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.plot(MRs, isp_amb, color="#c0392b", label="Isp (ambient)")
    ax1.plot(MRs, isp_vac, color="#e67e22", ls="--", label="Isp (vacuum)")
    ax1.axvline(MR, color="#2c3e50", ls=":", lw=1)
    ax1.set_xlabel("O/F  (mixture ratio)"); ax1.set_ylabel("Isp [s]")
    ax2 = ax1.twinx()
    ax2.plot(MRs, Tc, color="#2980b9", label="chamber T")
    ax2.set_ylabel("chamber temperature [K]", color="#2980b9")
    ax1.legend(loc="lower center", fontsize=8)
    ax1.set_title(f"{OX}/{FUEL} sizing sweep @ Pc={PC_BAR} bar, Pe={PE_BAR} bar")
    fig.tight_layout(); fig.savefig(OUT_PLOT, dpi=130)
    print(f"[plot] wrote {OUT_PLOT}")


if __name__ == "__main__":
    main()
