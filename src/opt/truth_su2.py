"""Black-box SU2 evaluator on cst22 (Phase D, the expensive oracle = headline truth).

`f(cst22) -> (CD, CL, CMz, t_c, feasible)` via the VALIDATED decode->gmsh->SU2 pipeline
(cf su2_recipe.py, CST gap <1%), in FIXED_CL mode (SU2 adjusts alpha to hold TARGET_CL).
Fixed-CL params replicated from the proven RAE case (turb_SA_RAE2822.cfg).

File-based architecture (non-intrusive, cf KTH BO_GP):
  LOCAL  : prep_local(cst22) -> airfoil.dat + t_c (decode + tc_fc)         [no local gmsh]
  CLUSTER: gmshairfoil2d (meshes) + SU2_CFD (solves) -> history.csv         [the runai job]
  LOCAL  : parse_forces(history.csv) -> CD, CL, CMz, converged ; feasible(...) applies constraints

The 2 LOCKED cases (same as the SU2 shape_opt refs, ported to RANS/cst22):
  naca0012 : RANS-SA M0.15/Re6e6, fixed-CL 0.5,   min CD, t/c>=0.12                (subsonic)
  rae2822  : RANS-SA M0.729/Re6.5e6, fixed-CL 0.724, min CD, |CMz|<0.093 & t/c>=0.12 (transonic)
Design box: x0 +/- 0.03 per cst22 coeff.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

from src.opt.cst_geom import decode, N_CST
from src.opt.evaluators import tc_fc

REF = Path(__file__).resolve().parent / "references"

CASES = {
    "naca0012": dict(
        mach=0.15, Re=6.0e6, target_cl=0.5, alpha0=4.0,
        mesh=dict(size=0.005, first_layer="1e-6", nb_layers=80, ratio=1.18, farfield=50),
        cfl=10.0, cfl_adapt=False, iter=8000,
        cmz_max=None, x0="naca0012_cst22.npy"),  # tc_min = t_c(x0), see tc_min()
    "rae2822": dict(
        mach=0.729, Re=6.5e6, target_cl=0.724, alpha0=2.31,
        mesh=dict(size=0.005, first_layer="1e-6", nb_layers=80, ratio=1.18, farfield=100),
        cfl=4.0, cfl_adapt=True, iter=15000, iter_fast=6000,   # transonic fixed-CL: the angle alpha
        cmz_max=0.093, x0="rae2822_cst22.npy"),  # converges slowly -> 3000 insufficient (CL not held), 6000 OK (2.5x < HF)
    # C3 = transonic NACA0012 (SU2 TUTORIAL case ADODG-1). Exact tutorial values: M0.8, AOA1.25 FIXED,
    # min DRAG, t/c>0.12. Two versions: euler (identical to tutorial, ceiling) + rans (zoo-compatible).
    "naca0012t_euler": dict(
        solver="EULER", fixed_cl=False, mach=0.8, alpha0=1.25, Re=None, target_cl=None,
        mesh=dict(size=0.005, first_layer="1e-6", nb_layers=80, ratio=1.18, farfield=100),
        cfl=4.0, cfl_adapt=True, iter=8000,
        cmz_max=None, x0="naca0012_cst22.npy"),
    "naca0012t_rans": dict(
        solver="RANS", fixed_cl=False, mach=0.8, alpha0=1.25, Re=3.0e6, target_cl=None,
        mesh=dict(size=0.005, first_layer="1e-6", nb_layers=80, ratio=1.18, farfield=100),
        cfl=4.0, cfl_adapt=True, iter=15000, iter_fast=6000,   # M0.8 strong shock -> converges slowly, 6000 fast (cf rae)
        cmz_max=None, x0="naca0012_cst22.npy"),
    # 2 ADDITIONAL well-posed regimes (fixed-CL -> stable multi-seed, attached RANS -> fast valid).
    # Same NACA0012 geometry -> isolates the REGIME (conditions), not the shape. Exercise the experts
    # subsonic_compressible (M0.5 subcritical, no shock) and low_re (Re3e5).
    "naca0012_msub": dict(          # subcritical compressible subsonic (between NACA M0.15 and RAE transonic)
        mach=0.50, Re=6.0e6, target_cl=0.5, alpha0=3.0,
        mesh=dict(size=0.005, first_layer="1e-6", nb_layers=80, ratio=1.18, farfield=50),
        cfl=8.0, cfl_adapt=True, iter=10000,
        cmz_max=None, x0="naca0012_cst22.npy"),
    "naca0012_msub_ld": dict(       # M0.5 subcritical compressible, MAX L/D FIXED-alpha (dashboard case 2, intermediate regime)
        solver="RANS", fixed_cl=False, mach=0.50, Re=6.0e6, alpha0=3.0, target_cl=None,
        mesh=dict(size=0.005, first_layer="1e-6", nb_layers=80, ratio=1.18, farfield=50),
        cfl=8.0, cfl_adapt=True, iter=10000, iter_fast=4000,
        cmz_max=None, x0="naca0012_cst22.npy"),
    "naca0012_nearstall": dict(     # high-CL near separation (CL~2/3 of CL_max ; SA delays separation)
        mach=0.15, Re=6.0e6, target_cl=1.0, alpha0=9.0,
        mesh=dict(size=0.005, first_layer="1e-6", nb_layers=80, ratio=1.18, farfield=50),
        cfl=6.0, cfl_adapt=True, iter=12000, iter_fast=5000,   # near-stall settles slowly -> cap 5000
        cmz_max=None, x0="naca0012_cst22.npy"),
    # DEMO NACA -> RAE: NACA0012 baseline BUT RAE2822 transonic conditions -> the optimum should
    # be supercritical, RAE-like. WIDE box (OPT_BOX_K=1.6) to contain RAE (at 1.59σ). cmz_max=None
    # (free exploration towards supercritical). Validates "right conditions + box -> right shape emerges".
    "naca2rae": dict(               # CL=0.5 (not 0.724): NACA0012 copes better at M0.729 -> feasible anchor
        mach=0.729, Re=6.5e6, target_cl=0.5, alpha0=1.5,
        mesh=dict(size=0.005, first_layer="1e-6", nb_layers=80, ratio=1.18, farfield=100),
        cfl=4.0, cfl_adapt=True, iter=15000, iter_fast=6000,
        cmz_max=None, x0="naca0012_cst22.npy"),
    # STEP-3: MAX L/D fixed-ALPHA (continuation of the simple use case, but cst22 + RANS). Same conditions
    # as naca0012 (M0.15/Re6e6, validated numerics, ref 99.8 cts) BUT alpha FIXED at 5deg + L/D objective.
    # Symmetric NACA0012 start -> camber raises CL/(L/D) (non-flat); at high CL we enter
    # the near_stall regime -> NF becomes unreliable -> the near_stall expert should be useful there.
    "naca0012_ld": dict(
        solver="RANS", fixed_cl=False, mach=0.15, Re=6.0e6, alpha0=5.0, target_cl=None,
        mesh=dict(size=0.005, first_layer="1e-6", nb_layers=80, ratio=1.18, farfield=50),
        cfl=10.0, cfl_adapt=False, iter=8000, iter_fast=4000,
        cmz_max=None, x0="naca0012_cst22.npy"),
    "naca0012_m03": dict(           # NEW (09-04): M0.3 MILDLY compressible + alpha0=4 (alpha study: α4 attached with MARGIN,
        solver="RANS", fixed_cl=False, mach=0.30, Re=6.0e6, alpha0=4.0, target_cl=None,  # α8 diverges; margin for the camber added by the optimizer). Tests zoo global+xfoil.
        mesh=dict(size=0.005, first_layer="1e-6", nb_layers=80, ratio=1.18, farfield=50),
        cfl=10.0, cfl_adapt=False, iter=8000, iter_fast=4000,
        cmz_max=None, x0="naca0012_cst22.npy"),
    # 10-case GENERALIZATION (09-11). Extra regimes/objectives to show that prior/blend
    # generalizes beyond the 4 anchors. Varied geometry (cambered naca4412) + extended Re (3e5 -> 1.2e7).
    "naca0012_lowre_ld": dict(      # LOW-Re Re3e5 (sailplane/UAV, laminar bubble): SU2-SA less reliable, XFoil strong -> tests the low_re expert
        solver="RANS", fixed_cl=False, mach=0.10, Re=3.0e5, alpha0=5.0, target_cl=None,
        mesh=dict(size=0.005, first_layer="3e-6", nb_layers=80, ratio=1.18, farfield=50),
        cfl=6.0, cfl_adapt=True, iter=12000, iter_fast=5000,   # low-Re settles slowly
        cmz_max=None, x0="naca0012_cst22.npy"),
    "naca0012_nearstall_ld": dict(  # NEAR-STALL max L/D α9 (high lift/separation): global dominates (B1), NF/xfoil unreliable
        solver="RANS", fixed_cl=False, mach=0.15, Re=6.0e6, alpha0=9.0, target_cl=None,
        mesh=dict(size=0.005, first_layer="1e-6", nb_layers=80, ratio=1.18, farfield=50),
        cfl=6.0, cfl_adapt=True, iter=12000, iter_fast=5000,   # near-stall settles slowly
        cmz_max=None, x0="naca0012_cst22.npy"),
    "naca4412_ld": dict(            # DIFFERENT GEOMETRY (cambered NACA4412) + HIGH-Re 1.2e7 + M0.3: shows not single-airfoil, not single-Re
        solver="RANS", fixed_cl=False, mach=0.30, Re=1.2e7, alpha0=3.0, target_cl=None,
        mesh=dict(size=0.005, first_layer="7e-7", nb_layers=90, ratio=1.16, farfield=50),
        cfl=10.0, cfl_adapt=False, iter=8000, iter_fast=4000,
        cmz_max=None, x0="naca4412_cst22.npy"),
    # STEP-3 SHOWCASE: transonic ADODG Case 2 — M0.734/CL0.824, STRONG upper-surface SHOCK, constrained
    # min-CD (fixed-CL lift + cmz moment + thickness/area). Real descent (~40-80 counts) AND
    # NeuralFoil BLIND (incompressible -> misses the shock) => maximal curse + switch to the transonic expert.
    "rae2822_adodg": dict(
        mach=0.734, Re=6.5e6, target_cl=0.824, alpha0=2.9,
        mesh=dict(size=0.005, first_layer="1e-6", nb_layers=80, ratio=1.18, farfield=100),
        cfl=4.0, cfl_adapt=True, iter=15000, iter_fast=6000,
        cmz_max=0.092, x0="rae2822_cst22.npy"),
    # ROBUST transonic SHOWCASE: RAE2822 M0.734 FIXED alpha (2.3°) -> max L/D. Keeps the SHOCK (NF blind
    # = curse) + the descent (reducing wave drag raises L/D), BUT zero fixed-CL fragility.
    "rae2822_ld": dict(
        solver="RANS", fixed_cl=False, mach=0.734, Re=6.5e6, alpha0=2.3, target_cl=None,
        mesh=dict(size=0.005, first_layer="1e-6", nb_layers=80, ratio=1.18, farfield=100),
        cfl=4.0, cfl_adapt=True, iter=15000, iter_fast=6000,
        cmz_max=None, x0="rae2822_cst22.npy"),
    "rae2822_ld_od": dict(          # RAE2822 OFF-DESIGN (alpha 3.0 > design 2.3) -> RAE NOT optimal -> room to improve (transonic case 3)
        solver="RANS", fixed_cl=False, mach=0.734, Re=6.5e6, alpha0=3.0, target_cl=None,
        mesh=dict(size=0.005, first_layer="1e-6", nb_layers=80, ratio=1.18, farfield=100),
        cfl=4.0, cfl_adapt=True, iter=15000, iter_fast=6000,
        cmz_max=None, x0="rae2822_cst22.npy"),
    "rae2822_ld_od4": dict(         # RAE2822 FURTHER OFF-DESIGN (alpha 4.0) -> RAE clearly sub-optimal -> more margin (case 3 big gain)
        solver="RANS", fixed_cl=False, mach=0.734, Re=6.5e6, alpha0=4.0, target_cl=None,
        mesh=dict(size=0.005, first_layer="1e-6", nb_layers=80, ratio=1.18, farfield=100),
        cfl=4.0, cfl_adapt=True, iter=15000, iter_fast=6000,
        cmz_max=None, x0="rae2822_cst22.npy"),
    # NACA0012 -> RAE transonic conditions (M0.734, FIXED alpha 2.3°), max L/D. We KNOW the optimum ≈ RAE
    # (L/D~51.4, since rae2822_ld = 0 descent). NACA0012 start = poor in transonic (strong shock) -> BIG
    # descent towards a KNOWN target. Run with OPT_BOX_K=1.6 (contains RAE at ~1.6σ from NACA0012).
    "naca2rae_ld": dict(
        solver="RANS", fixed_cl=False, mach=0.734, Re=6.5e6, alpha0=2.3, target_cl=None,
        mesh=dict(size=0.005, first_layer="1e-6", nb_layers=80, ratio=1.18, farfield=100),
        cfl=4.0, cfl_adapt=True, iter=15000, iter_fast=6000,
        cmz_max=None, x0="naca0012_cst22.npy"),
}
# default solver/fixed_cl for the legacy cases (C1/C2 = RANS + fixed-CL)
for _c in CASES.values():
    _c.setdefault("solver", "RANS"); _c.setdefault("fixed_cl", True)

# ── PER-COEFFICIENT design box ("de facto" standard: data hypercube) ──────────────
# Bounds = x0 +/- BOX_K * sigma_i, where sigma_i = std of coeff i in dataset_v3.
# Better than uniform ±0.06: each coeff explores in proportion to its real variation
# (large thickness/shock coeff = wide; small LE coeff = narrow). Lit: bounds = data hypercube.
# If the box generates OOD -> NOT a problem: the router/IO-CUE must detect that the surrogate is
# poor there and switch to RANS (that is the test of the tool). Safeguard = GEOMETRIC validity only.
SIGMA = np.load(REF / "cst_sigma.npy")          # (22,) per-coeff std over dataset_v3
BOX_K = float(os.environ.get("OPT_BOX_K", 0.5)) # k=0.5 -> median ±0.155 (wider than ±0.06)
BOX = BOX_K * SIGMA                              # vector of half-widths (22,)


def box(case):
    """(lb, ub) cst22 for a case: x0 +/- BOX_K*sigma (per coefficient)."""
    x0 = load_x0(case)
    return x0 - BOX, x0 + BOX


def geom_valid(cst22):
    """Valid geometry = no self-intersection (non-negative thickness) + reasonable t/c.
    Rejects crossed airfoils (spaghetti) that the wide box can generate. The trailing edge is
    naturally thin (th->0) -> we check th>=0 on [0.02, 0.90] with a small numerical tolerance."""
    from src.utils.cst import class_function, bernstein_basis
    c = np.asarray(cst22, float); A_u, A_l = c[:N_CST], c[N_CST:2 * N_CST]
    xq = np.linspace(0.02, 0.90, 100); C = class_function(xq); B = bernstein_basis(xq, N_CST - 1)
    th = C * (B @ A_u) - C * (B @ A_l)           # thickness(x) excluding the thin trailing edge
    return bool(th.min() > -1e-4 and 0.03 < th.max() < 0.40)   # >=0 (no crossing) + sane t/c

# Convergence = CD-Cauchy (the CD coeff has settled), NOT the density rms.
# In transonic flow the rms plateaus (shock limit cycle) while the CD has settled (checked:
# design rms=-5.84 -> CD p2p=0.13 cts over 1000 iters). The rms only serves as anti-divergence.
RMS_FLOOR = -4.0    # above = numerical blow-up -> rejected
CD_CAUCHY = 1e-4    # CD settled if peak-to-peak (last 1000 iters) < 1 count

# RANS-fast = intermediate fidelity (~5 min). ITER cap (CD settles well before 8000):
# validated on 6 regimes (subsonic/transonic/low-Re/near-stall/thick/compressible) -> ~±0.5 ct
# at 3000 iters, RANK preserved (the bias is near-systematic). CD read as a WINDOWED MEAN (ripple).
ITER_FAST = 3000
FAST_WINDOW = 500   # CD = mean of the last 500 iters (smooths the transonic/near-stall limit cycle)
ITER_COARSE = 3000  # COARSE-RANS: the RANS CD converges slowly -> enough iters (calib d0: 1200 fails, CD not converged). Speedup via a coarse MESH (OPT_COARSE_MESH), not the tolerance alone.


def load_x0(case: str) -> np.ndarray:
    return np.load(REF / CASES[case]["x0"])


# ── SU2 .cfg generation (RANS-SA + fixed-CL + per-case numerics) ───────────
def make_cfg(case: str, fast: bool = False, fidelity: str = None) -> str:
    c = CASES[case]
    fid = fidelity or ("fast" if fast else "hf")                    # fidelities: coarse (cheap OOD anchoring) | medium (moderate CD-Cauchy, near-reliable) | fast (fast fine, validated ≈hf) | hf (truth)
    niter = {"coarse": c.get("iter_coarse", ITER_COARSE), "medium": c.get("iter_fast", ITER_FAST),
             "fast": c.get("iter_fast", ITER_FAST), "hf": c["iter"]}[fid]
    # CONV_FIELD=(LIFT,DRAG): the Cauchy criterion applies TO CD/CL (not the residual). Loose coarse 1E-4 breaks at the aggressive optimum (CD not converged at the cap). medium 5E-5 = better-converged CD, ~2× coarse.
    cauchy = {"coarse": "1E-4", "medium": "5E-5"}.get(fid, "1E-6")  # medium: between coarse (1E-4) and fine (1E-6)
    startit = "1500" if fid == "coarse" else "1000"                 # coarse: Cauchy CHECK after the transient (CD plateaus early then drops)
    # environment overrides for the convergence study; default = unchanged behaviour
    cfl_v = float(os.environ.get("OPT_CFL", c["cfl"]))
    _ad = os.environ.get("OPT_CFL_ADAPT", "")
    adapt_v = (_ad == "1") if _ad else c["cfl_adapt"]
    cfl_block = ("CFL_ADAPT= YES\nCFL_ADAPT_PARAM= ( 0.1, 2.0, 4.0, 1000.0 )"
                 if adapt_v else "CFL_ADAPT= NO")
    euler = c["solver"] == "EULER"
    # solver block: Euler (inviscid, slip wall, no Re/turb) vs RANS-SA (viscous)
    if euler:
        solver_block = ("SOLVER= EULER\nMATH_PROBLEM= DIRECT\nRESTART_SOL= NO\n"
                        f"MACH_NUMBER= {c['mach']}\nAOA= {c['alpha0']}\n"
                        "FREESTREAM_TEMPERATURE= 288.15")
        wall = "MARKER_EULER= ( airfoil )"                 # slip wall (inviscid)
    else:
        # SA_OPTIONS: model variants (BCM = Bas-Cakmakcioglu-Menter transition).
        # Not emitted by default, so behaviour unchanged.
        _sa = os.environ.get("OPT_SA_OPTIONS", "")
        sa_opt = f"SA_OPTIONS= {_sa}\n" if _sa else ""
        solver_block = (f"SOLVER= RANS\nKIND_TURB_MODEL= SA\n{sa_opt}MATH_PROBLEM= DIRECT\nRESTART_SOL= NO\n"
                        f"MACH_NUMBER= {c['mach']}\nAOA= {c['alpha0']}\n"
                        "FREESTREAM_TEMPERATURE= 288.15\n"
                        f"REYNOLDS_NUMBER= {c['Re']}\nREYNOLDS_LENGTH= 1.0\n"
                        "FREESTREAM_TURBULENCEINTENSITY= 0.001\nFREESTREAM_TURB2LAMVISCRATIO= 100.0")
        wall = "MARKER_HEATFLUX= ( airfoil, 0.0 )"
    # CL block: fixed-CL (SU2 adjusts alpha) vs fixed alpha (tutorial)
    cl_block = (f"FIXED_CL_MODE= YES\nTARGET_CL= {c['target_cl']}\nDCL_DALPHA= 0.2\n"
                "UPDATE_AOA_ITER_LIMIT= 100\nITER_DCL_DALPHA= 500\nEVAL_DOF_DCX= NO"
                if c["fixed_cl"] else "FIXED_CL_MODE= NO")
    turb_num = "" if euler else ("CONV_NUM_METHOD_TURB= SCALAR_UPWIND\nMUSCL_TURB= NO\n"
                                 "SLOPE_LIMITER_TURB= VENKATAKRISHNAN\nTIME_DISCRE_TURB= EULER_IMPLICIT\n")
    return f"""{solver_block}
{cl_block}
REF_ORIGIN_MOMENT_X= 0.25
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH= 1.0
REF_AREA= 1.0
MESH_FILENAME= mesh.su2
MESH_FORMAT= SU2
{wall}
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( airfoil )
MARKER_MONITORING= ( airfoil )
NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CFL_NUMBER= {cfl_v}
{cfl_block}
MGLEVEL= 3
MGCYCLE= W_CYCLE
MG_PRE_SMOOTH= ( 4, 4, 4, 4 )
MG_POST_SMOOTH= ( 4, 4, 4, 4 )
MG_CORRECTION_SMOOTH= ( 1, 1, 1, 1 )
MG_DAMP_RESTRICTION= 0.5
MG_DAMP_PROLONGATION= 0.5
CONV_NUM_METHOD_FLOW= JST
JST_SENSOR_COEFF= ( 0.5, 0.02 )
TIME_DISCRE_FLOW= EULER_IMPLICIT
{turb_num}LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= LU_SGS
LINEAR_SOLVER_ERROR= 1E-1
LINEAR_SOLVER_ITER= 3
ITER= {os.environ.get('OPT_ITER', niter)}
CONV_FIELD= (LIFT, DRAG)
CONV_STARTITER= {startit}
CONV_CAUCHY_ELEMS= 100
CONV_CAUCHY_EPS= {cauchy}
TABULAR_FORMAT= CSV
CONV_FILENAME= history
SURFACE_FILENAME= surface_flow
OUTPUT_FILES= ( RESTART, SURFACE_CSV )
OUTPUT_WRT_FREQ= 500
SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY, LIFT, DRAG, MOMENT_Z)
HISTORY_OUTPUT= ITER, RMS_RES, AERO_COEFF
"""


def mesh_cmd(case: str, outdir: str, coarse: bool = False) -> list:
    m = dict(CASES[case]["mesh"])
    # E2 REGIME-ADAPTIVE: CONTINUOUS coarseness factor cf ∈ [0,1]. cf=1 aggressive (attached, cheap) …
    # cf=0 fine. Mild (cf~0.3-0.5) under separation where aggressive fails (near-stall calib ±6453). Env OPT_COARSE_LEVEL.
    cf = 0.0
    if coarse is True:
        cf = float(os.environ.get("OPT_COARSE_LEVEL", "1.0"))   # default=1.0 -> aggressive (exact back-compat)
    elif coarse:                                                 # float ∈ (0,1] = explicit level
        cf = float(coarse)
    cf = min(max(cf, 0.0), 1.0)
    if cf > 0.0:                                                 # coarsen surface+BL+farfield ; KEEPS first_layer (y+~1 SA valid)
        m = dict(m, size=round(m["size"] * (1.0 + 1.5 * cf), 5),
                 nb_layers=max(int(round(m["nb_layers"] * (1.0 - 0.5 * cf))), 35),
                 farfield=max(int(round(m["farfield"] * (1.0 - 0.5 * cf))), 25),
                 ratio=round(1.18 + 0.07 * cf, 3))
    gmsh = os.environ.get("GMSH_BIN", "/opt/conda/envs/mesh/bin/gmshairfoil2d")
    return [gmsh, "--airfoil_path", f"{outdir}/airfoil.dat",
            "--format", "su2", "--output", outdir, "--structured",
            "--airfoil_mesh_size", str(m["size"]), "--first_layer", str(m["first_layer"]),
            "--nb_layers", str(m["nb_layers"]), "--ratio", str(m["ratio"]),
            "--arg_struc", f"{m['farfield']}x{m['farfield']}"]


# ── local prep: cst22 -> airfoil.dat + t_c (no local gmsh) ───────────
def prep_local(cst22: np.ndarray, outdir: str, n: int = 160) -> float:
    d = Path(outdir); d.mkdir(parents=True, exist_ok=True)
    xy = decode(cst22, n=n)
    with open(d / "airfoil.dat", "w") as f:
        f.write("opt\n"); [f.write(f"{x:.6f} {y:.6f}\n") for x, y in xy]
    return tc_fc(cst22)[0]   # t_c from cst22 (consistent with the written geometry)


# ── parse SU2 forces from history.csv ──────────────────────────────────
def parse_forces(rundir: str, fast: bool = False) -> dict:
    p = Path(rundir) / "history.csv"
    log = Path(rundir) / "su2.log"                        # a killed SU2 (disk full, preempted pod) leaves a truncated
    done = log.exists() and "Exit Success" in log.read_text(errors="ignore")[-5000:]   # history: never a result
    if not p.exists() or not done:
        return {"CD": None, "CL": None, "CMz": None, "rms": None, "physical": False, "cd_p2p": float("inf")}
    df = pd.read_csv(p)
    df.columns = [c.strip().strip('"').strip() for c in df.columns]
    w = FAST_WINDOW if fast else 1                        # fast: CD/CL/CMz = windowed mean (smooths the ripple)
    def col(*names):
        for n in names:
            for k in df.columns:
                if k.lower() == n.lower():
                    return float(np.mean(df[k].to_numpy(float)[-w:]))
        return None
    cd, cl, cmz = col("CD"), col("CL"), col("CMz")
    rms = col("rms[Rho]", "rms_Rho")
    physical = (cd is not None and np.isfinite(cd) and 0.0 < cd < 0.05
                and cl is not None and np.isfinite(cl))   # finite + CD within a physical range
    # CD-Cauchy: has the CD coeff stopped moving? (robust convergence, transonic included)
    cd_ser = next((df[k].to_numpy(float) for k in df.columns if k.lower() == "cd"), None)
    if cd_ser is not None and len(cd_ser) >= 50:
        cd_p2p = float(np.ptp(cd_ser[-min(1000, len(cd_ser)):]))
    else:
        cd_p2p = float("inf")                             # not enough history -> cannot judge -> rejected
    return {"CD": cd, "CL": cl, "CMz": cmz, "rms": rms, "cd_p2p": cd_p2p, "physical": physical}


def tc_min(case: str) -> float:
    """Thickness constraint = t_c of the decoded baseline ('no thinner than x0')."""
    return tc_fc(load_x0(case))[0]


def feasible(res: dict, t_c: float, case: str, fast: bool = False) -> bool:
    """converged + CL reaches the target + GEOMETRIC constraint (t/c). CMz is NO LONGER constrained
    (MLP<->RANS fairness: the MLPs do not predict CM) -> it is only REPORTED (res['CMz']).
    Bad-design safeguards: non-physical / NaN / CL missed / too thin -> False.
    fast=True: convergence via the ITER cap (validated by rank), not the strict CD-Cauchy."""
    c = CASES[case]
    if not res.get("physical", False):                    # finite + CD within a physical range
        return False
    if res["rms"] is None or res["rms"] > RMS_FLOOR:      # anti-divergence (numerical blow-up)
        return False
    if not fast and res.get("cd_p2p", float("inf")) > CD_CAUCHY:   # HF: CD-Cauchy; fast: ITER cap is enough
        return False
    if c["fixed_cl"] and (res["CL"] is None or abs(res["CL"] - c["target_cl"]) > 0.02):
        return False                                      # did fixed-CL hold the target? (if fixed-CL mode)
    if t_c < tc_min(case) - 1e-6:                         # no thinner than the baseline (GEO constraint)
        return False
    return True                                           # CMz not constrained (fairness), only reported


# ── sanity: generate the cfg + prep the baseline, check locally ─────────────
if __name__ == "__main__":
    for case in ("naca0012", "rae2822"):
        c = CASES[case]
        x0 = load_x0(case)
        t_c = prep_local(x0, f"/tmp/su2eval_{case}")
        cfg = make_cfg(case)
        assert "FIXED_CL_MODE= YES" in cfg and f"TARGET_CL= {c['target_cl']}" in cfg
        print(f"{case}: t_c(x0)={t_c*100:.2f}% (=tc_min baseline) | "
              f"M={c['mach']} Re={c['Re']:.1e} CLtarget={c['target_cl']} "
              f"farfield={c['mesh']['farfield']}c cfl={c['cfl']}(adapt={c['cfl_adapt']}) "
              f"cmz_max={c['cmz_max']} | cfg OK, airfoil.dat written")
