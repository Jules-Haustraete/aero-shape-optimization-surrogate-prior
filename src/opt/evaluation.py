"""Evaluate a batch of designs: SU2 RANS (the truth) or a zoo member standing in for it.

`eval_batch(cst_list, case, workdir, tag, fidelity)` is the only entry point the optimizer uses.
fidelity = "fast" | "hf" | "coarse" | "medium" runs SU2 (truth_su2 builds the mesh and the config),
any member of src/zoo.py gives a cheap local proxy of the truth (to test the pipeline without SU2).
Fixed-lift cases invert alpha by bisection, as SU2 does internally.
"""
import os
from pathlib import Path
import numpy as np

from src import zoo
from src.opt.evaluators import tc_fc
from src.opt import truth_su2 as T

SURROGATES = set(zoo.MEMBERS)


def _surrogate_clcd(model, cst22, mach, alpha, Re):
    return zoo.predict(model, cst22, mach, alpha, Re)


def _alpha_for_cl(model, cst22, case):
    """Fixed-CL: INVERTS alpha (bisection) to hold target_cl, as SU2 does internally."""
    from scipy.optimize import brentq
    c = T.CASES[case]; mach = c["mach"]; Re = c.get("Re") or 1e6; tgt = c["target_cl"]
    try:
        return brentq(lambda a: _surrogate_clcd(model, cst22, mach, a, Re)[0] - tgt, -5.0, 12.0, maxiter=50)
    except Exception:
        return c["alpha0"]


# L1 (ladder): PESSIMISTIC objective = CD + lambda*sigma_hat (IO-CUE). lambda=0 -> L0 (raw curse).
# Feasibility stays on the RAW CD (physical design); only the OBJECTIVE is penalized -> the optimizer
# avoids regions where the surrogate is uncertain. Control rung: tests whether per-point UQ beats the curse.
PESSIMISM = float(os.environ.get("GPBO_PESSIMISM", 0.0))


# ── GP with DIMENSION-SCALED lengthscale PRIOR (Hvarfner et al., ICML 2024) ─
# The key fix for high dimension: without it, the GP assumes small lengthscales ->
# does not model globally in 22D -> EI proposes anything -> BO does not progress.
# LogNormal(loc=sqrt2 + ln(D)*0.5, scale=sqrt3) makes the GP smoother as D grows
# (distances ~sqrt(D)) -> "vanilla BO performs great in high dimensions". D=22 -> loc~2.96.
def _dim_scaled_covar(dim):
    """RBF-ARD kernel + lengthscale prior LogNormal(sqrt2+ln(D)*0.5, sqrt3), bound 0.025.
    Built in to recent botorch; otherwise we rebuild it (same values, Hvarfner 2024)."""
    try:
        from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
        return get_covar_module_with_dim_scaled_prior(ard_num_dims=dim)
    except ImportError:
        import math
        from gpytorch.kernels import RBFKernel, ScaleKernel
        from gpytorch.priors import LogNormalPrior, GammaPrior
        from gpytorch.constraints import GreaterThan
        ls_prior = LogNormalPrior(loc=math.sqrt(2) + math.log(dim) * 0.5, scale=math.sqrt(3))
        k = RBFKernel(ard_num_dims=dim, lengthscale_prior=ls_prior,
                      lengthscale_constraint=GreaterThan(2.5e-2))
        return ScaleKernel(k, outputscale_prior=GammaPrior(2.0, 0.15))


# ── SURROGATE eval (cheap, local): cst22 -> (CL,CD) via MLP/NF/TUM, fixed-CL by alpha inversion ─
def eval_surrogate_batch(cst_list, case, model):
    c = T.CASES[case]; mach = c["mach"]; Re = c.get("Re") or 1e6
    out = []
    for cst in cst_list:
        tc = tc_fc(cst)[0]
        if not T.geom_valid(cst):
            out.append({"CD": None, "CL": None, "CMz": None, "rms": None,
                        "physical": False, "cd_p2p": 0.0, "tc": tc}); continue
        a = _alpha_for_cl(model, cst, case) if c["fixed_cl"] else c["alpha0"]
        cl, cd = _surrogate_clcd(model, cst, mach, a, Re)
        rec = {"CD": cd, "CL": cl, "CMz": None, "rms": -99.0,           # dummy rms (no RANS)
               "physical": (0.0 < cd < 0.05), "cd_p2p": 0.0, "tc": tc, "alpha": a}
        if PESSIMISM > 0.0:                                             # L1: pessimistic objective CD+λσ̂
            from src.opt.iocue import sigma_hat
            rec["obj_cd"] = cd + PESSIMISM * sigma_hat(cst, mach, a, Re, model)
        out.append(rec)
    return out


# ── eval_batch: ROUTES to surrogate (cheap) OR SU2 (expensive). SAME optimizer for all. ────────────
def eval_batch(cst_list, case, workdir, tag, fidelity):
    if fidelity in SURROGATES:
        return eval_surrogate_batch(cst_list, case, fidelity)
    coarse = fidelity == "coarse"                                   # COARSE-RANS: anchoring low-fi (Cauchy 1e-4 + iter cap; coarse mesh if OPT_COARSE_MESH=1)
    medium = fidelity == "medium"                                   # MEDIUM: CD-Cauchy 5E-5 + iter_fast cap = almost reliable everywhere except stall, ~2× coarse
    fast = fidelity in ("fast", "rans_fast")
    win = fast or coarse or medium                                  # windowed parse (averaged CD, cap-based)
    coarse_mesh = coarse and os.environ.get("OPT_COARSE_MESH", "0") == "1"   # phase 2 (default off = phase 1 tol-only)
    import subprocess
    GMSH = os.environ.get("GMSH_BIN", "/opt/conda/envs/mesh/bin/gmshairfoil2d")
    SU2 = os.environ.get("SU2_BIN", "/opt/su2/bin/SU2_CFD")
    MAXPAR = int(os.environ.get("OPT_MAXPAR", 8))
    dirs = []
    for i, c in enumerate(cst_list):
        d = Path(workdir) / f"{tag}_{i}"; d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*"):
            try: f.unlink()
            except IsADirectoryError: pass
        tc = T.prep_local(c, str(d))
        (d / "run.cfg").write_text(T.make_cfg(case, fidelity=fidelity) if (coarse or medium) else T.make_cfg(case, fast=fast))
        dirs.append((d, tc))

    def run_wave(cmds):
        procs = []
        for cmd, cwd, logn in cmds:
            lf = open(Path(cwd) / logn, "w")
            procs.append((subprocess.Popen(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT), lf))
            if len(procs) >= MAXPAR:
                for p, f in procs: p.wait(); f.close()
                procs = []
        for p, f in procs: p.wait(); f.close()

    run_wave([(T.mesh_cmd(case, str(d), coarse=coarse_mesh), str(d), "gmsh.log") for d, _ in dirs])
    for d, _ in dirs:
        cand = [p for p in d.glob("*.su2") if p.name != "mesh.su2"]
        if cand: cand[0].rename(d / "mesh.su2")
    run_wave([([SU2, "run.cfg"], str(d), "su2.log") for d, _ in dirs if (d / "mesh.su2").exists()])
    out = []
    for d, tc in dirs:
        if not (d / "mesh.su2").exists():
            out.append({"CD": None, "CL": None, "CMz": None, "rms": None,
                        "physical": False, "cd_p2p": float("inf"), "tc": tc})
            continue
        res = T.parse_forces(str(d), fast=win); res["tc"] = tc
        out.append(res)
    return out
