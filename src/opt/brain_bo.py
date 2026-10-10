"""THE BRAIN — multi-model router + RANS verification (step-3, cst22, real zoo, real IO-CUE).
Implements exactly fig_router_brain.png. REUSABLE objective (propagated from CL/CD primitives).

Loop, for each proposed design x (pool-LogEI on the objective GP):
  1. CHEAP eval of the whole zoo -> (CLᵢ, CDᵢ).
  2. IO-CUE.propagate per model -> (Jᵢ corrected for bias δ̂, σ̂ᵢ propagated to the objective).
  3. SELECT i* = argmin σ̂ᵢ  (the most RELIABLE, never the most optimistic).
  4. GATE: σ*<τ -> TRUST (cost 0); σ*≥τ ∧ EI≥τ_EI -> VERIFY RANS (cost 1); else SKIP (trust).
  5. if VERIFY: update GP/best + record residuals of the 2 primitives of ALL models -> enriches IO-CUE.
  6. at the end: flush_enriched -> <name>_resid_v_online.npz (extended coverage for the next optimization).

Launch: python main.py optimize (presets of the thesis arms) or python -m src.opt.brain_bo.
Env: BRAIN_CASE, BRAIN_FID (fast|hf = RANS; or a zoo member = local proxy truth), BRAIN_OBJ
(ld|cd), BRAIN_ZOO, BRAIN_TAU, BRAIN_TAU_EI, BRAIN_NINIT, BRAIN_NITER, BRAIN_POOL, BRAIN_SEED, IOCUE_SUFFIX.
"""
import os
import numpy as np
import torch
from pathlib import Path
from scipy.stats import norm
from scipy.spatial.distance import cdist
from torch.quasirandom import SobolEngine

from src.opt import evaluation as G
from src.opt import truth_su2 as T
from src.opt import iocue as IO
from src.opt.evaluators import tc_fc
from src.opt.cst_geom import decode
from src import zoo
from src.opt.turbo_state import TurboState, update_state           # TuRBO TR (grafted onto the router)

ROOTP = Path(__file__).resolve().parents[2]
CASE = os.environ.get("BRAIN_CASE", "naca0012_ld")
FID = os.environ.get("BRAIN_FID", "global")            # 'fast'/'hf' = RANS; else local proxy
OBJ = os.environ.get("BRAIN_OBJ", "ld")                # ld (max L/D) | cd (min CD)
COMPOSITE = os.environ.get("BRAIN_COMPOSITE", "0") == "1"  # composite BO (Astudillo&Frazier 2019): 2 GPs (CL,CD) instead of scalar L/D -> the acquisition exploits the L/D=CL/CD structure. GP on VERIFIED points only.
PRIOR_MEAN = os.environ.get("BRAIN_PRIOR_MEAN", "0") == "1"
PRIOR_SELECT = os.environ.get("BRAIN_PRIOR_SELECT", "")   # ONLINE selection of the prior among the zoo by RANK agreement with the RANS anchors (Feurer RGPE): "" (off: route/FUSION) | argmin (hard switch) | rgpe (weighted bootstrap + dilution). Zero extra RANS; sees the optimum region.
RGPE_S = int(os.environ.get("BRAIN_RGPE_S", 100))
PSEL_NONE = os.environ.get("BRAIN_PSEL_NONE", "1") == "1"
PSEL_WARM = int(os.environ.get("BRAIN_PSEL_WARM", 0))          # warm-up: while n_anchors < PSEL_WARM, prior = fixed PSEL_WARM_ZOO (no selection); then online selection. 0=off
PSEL_WARM_ZOO = os.environ.get("BRAIN_PSEL_WARM_ZOO", "neuralfoil")
PSEL_FREEZE = int(os.environ.get("BRAIN_PSEL_FREEZE", 0))      # FREEZE: once n_anchors >= N, the selector weights are no longer recomputed. Rank agreement is only readable on a SPREAD population; the optimizer destroys it by concentrating (measured: exact tau at 25 anchors, dead beyond 50). Also freezes the residual-GP target, which otherwise changes retroactively at each re-weighting. 0=off
PINIT_RULE = os.environ.get("BRAIN_PINIT_RULE", "zscore")   # init pre-screen: zscore (log lscale + ρ + −sres) | smooth (argmax box smoothness; if all rough < PINIT_LS_MIN -> PINIT_FALLBACK)

PINIT_SRES_MAX = float(os.environ.get("BRAIN_PINIT_SRES_MAX", 2.0))  # guard: candidate rejected if std(residual) > k×std(y) at init anchors (amplitude that drowns the signal)
PINIT_REVISE = int(os.environ.get("BRAIN_PINIT_REVISE", 12))   # revisits the decision once N designs are verified: the tie-break relies on a residual, whose variance drops with anchors (measured: stability 38-58 % at 6 anchors -> 92-100 % at 12-16)
PINIT_STAB_MIN = float(os.environ.get("BRAIN_PINIT_STAB_MIN", 0.6))    # the choice must survive anchor resampling: below this rate, the decision is a coin toss -> fallback
PINIT_STAB_B = int(os.environ.get("BRAIN_PINIT_STAB_B", 100))
PINIT_SPREAD_MIN = float(os.environ.get("BRAIN_PINIT_SPREAD_MIN", 0.25))  # guard: a QUASI-CONSTANT field is infinitely "smooth" with residual 1.0 -> it would win the selection while saying nothing. It must vary by at least this factor × std(y).
PINIT_LS_BAND = float(os.environ.get("BRAIN_PINIT_LS_BAND", 0.8))   # smoothness band: candidates at ≥80 % of the best ℓ are tie-broken by the residual, not by 0.02 of ℓ
PINIT_RAW = os.environ.get("BRAIN_PINIT_RAW", "1") == "1"            # includes RAW candidates (δ̂ off for THAT member): sometimes the only smooth field (subcomp CD) or conversely the only informative one (nf at M0.8 = δ̂)
PINIT_LS_MIN = float(os.environ.get("BRAIN_PINIT_LS_MIN", 0.3))
PINIT_FALLBACK = os.environ.get("BRAIN_PINIT_FALLBACK", "global")   # 1 = "none" candidate (GP-LOO target, RGPE shrink); 0 = base-models only (weights forced on the zoo)  # prior-mean in STANDARD BO (no TRMM): GP fit on residual (truth − f_low surrogate), acquisition on (f_low + residual). Isolates the prior from the trust-region.
KOH = os.environ.get("BRAIN_KOH", "auto")              # #3 coarse->fine correction (affine KOH per regime, CD=ρ·cd+β, CL=cl+offset). auto=ON if FID=coarse | on | off. Valid for attached/msub ONLY (calib). DO NOT enable for transonic/stall (nonlinear bias).
_KOH_ON = (KOH == "on") or (KOH == "auto" and FID == "coarse")
_KOH_C = None
if _KOH_ON:
    import json as _json
    try:
        _kd = _json.load(open(ROOTP / "models/coarse_koh.json"))[CASE]   # ρ,β = fine≈ρ·coarse+β (calib coarse vs fine, see models/coarse_koh.json)
        _KOH_C = (_kd["CL"]["rho"], _kd["CL"]["beta"], _kd["CD"]["rho"], _kd["CD"]["beta"])
    except Exception:
        _KOH_ON = False                                                  # no calib for this CASE (regime not covered) -> raw coarse
# ZOO v4: regime experts (plan4_v4) + source models (source_v4) + cheap oracles. TUM excluded (slow diffusion).
ZOO = os.environ.get("BRAIN_ZOO", "global,unifoil,ramos,turbulent_attached,subsonic_compressible,"
                     "transonic,near_stall,post_stall,neuralfoil,airfoilann").split(",")
TAU = float(os.environ.get("BRAIN_TAU", 3.0))          # error TOLERANCE accepted to BELIEVE (objective units; CQR => 90% error)
TAU_PCT = float(os.environ.get("BRAIN_TAU_PCT", 0.0))  # if >0: tolerance in % of |base perf| (scale-free = engineering decision)
TAU_EI = float(os.environ.get("BRAIN_TAU_EI", 0.0))    # EI gate (0 = no gate)
GATE = os.environ.get("BRAIN_GATE", "tau")             # + CONTROLS (09-17): random (rate matched to the shadow σ̂ gate) | oracle (decides on the TRUE error, hidden RANS not counted)
AUDIT = os.environ.get("BRAIN_AUDIT", "0") == "1"       # CONTROL: hidden RANS NOT counted on every believed point -> val_true in the traj (phantom rate of the gate decisions)
_SHADOW = [0, 0]                                        # random: [n decisions, n trust] of the shadow σ̂ gate -> matched p_trust             # gate POLICY: tau (σ̂<τ%·base, default) | margin | regret | adapt | ctd | hybrid | escalate (3-TIER: trust surrogate < cheap physical medium < fine truth; escalates cost only if needed)
ESC_MED = os.environ.get("BRAIN_ESC_MED", "medium")    # escalate: fidelity of the INTERMEDIATE tier (medium=reliable cheap CD-Cauchy). Low tier=surrogate (trust), high=hf (fine).
MARGIN_K = float(os.environ.get("BRAIN_MARGIN_K", 1.0))       # margin: trust IFF σ̂ < K·|belief−best| (K=1 → believes if the uncertainty is smaller than the gain at stake; far from optimum large gap→trust, near→verify since σ̂ dwarfs the gap = anti-curse)
MARGIN_FLOOR = float(os.environ.get("BRAIN_MARGIN_FLOOR", 0.02))  # margin: margin floor (frac of |best|) to avoid τ→0 exactly at the plateau (else verifies everything)
FUSION = os.environ.get("BRAIN_FUSION", "argmin")      # zoo fusion: argmin (default, predicted σ̂) | invvar (LCWF inverse-variance) | skill (MODEL-SELECTION by REAL ERROR MEASURED at neighbouring RANS anchors — not the blind σ̂; the true truth-based model-selection)
SKILL_K = int(os.environ.get("BRAIN_SKILL_K", 5))      # skill: nb of neighbouring RANS anchors to measure each model's local skill
_SKILL_HIST = []                                        # history [(x_unit, {model: real |err_cd|})] filled at verifications
CTD_HORIZON = float(os.environ.get("BRAIN_CTD_HORIZON", 0.0))  # ctd: iteration horizon to pace the budget (0=auto 2·BUDGET)
ADAPT_SLOPE = float(os.environ.get("BRAIN_ADAPT_SLOPE", 0.9))  # C: τ_t = τ·(1 − slope·frac_budget) → τ decreases over the run (tight verification of finalists)
ADAPT_FLOOR = float(os.environ.get("BRAIN_ADAPT_FLOOR", 0.1))  # C: floor of τ_t (fraction of τ) so it does not drop to 0
ADAPT_POW = float(os.environ.get("BRAIN_ADAPT_POW", 1.0))     # decay shape: 1=LINEAR; >1=CONVEX (τ stays high, drops LATE = better matches the curse, which is at the end)
REGRET_MARGIN = float(os.environ.get("BRAIN_REGRET_MARGIN", 0.0))  # B-frugal: margin (frac of |best|) the optimistic belief must EXCEED to verify (0=default over-verifies; 0.05=requires +5% = RANS cost)
LAZY = os.environ.get("BRAIN_LAZY", "0") == "1"        # ① LAZY VERIFICATION: trusts EVERYTHING in the loop (0 inline RANS); only the final-verify spends RANS ("trust to search, verify to deliver" policy)
RACE = os.environ.get("BRAIN_RACE", "0") == "1"        # ② FINAL-VERIFY as a RACE: verifies the podium by decreasing upper bound (belief+σ̂), STOP as soon as the proven beats the upper bound of ALL the rest (best-arm ID)
CONFORMAL_LAMBDA = float(os.environ.get("BRAIN_CONFORMAL_LAMBDA", 1.0))  # ③ σ̂ recalibration for the calibrated RACE upper bound: bound = belief + 1.645·λ·σ̂. λ=0.64 → EXACT 90% CI (−36% vs raw σ̂, coverage kept) → podium decided faster = less RANS. 1.0=raw
N_INIT = int(os.environ.get("BRAIN_NINIT", 6))
INIT_PRIOR = int(os.environ.get("BRAIN_INIT_PRIOR", 0))   # k>0: k of the N_INIT anchors = top-k of the debiased PRIOR (argmax belief over 3000 feasible Sobol, diversified) instead of pure Sobol -> prior-guided init
INIT_RANS = int(os.environ.get("BRAIN_INIT_RANS", 999))  # nb of RANS ANCHORS at init (base included); the rest of N_INIT = surrogate beliefs (honest σ̂² weight, 0 RANS). Default 999 = all-RANS (current behaviour)
N_ITER = int(os.environ.get("BRAIN_NITER", 300))       # hard cap; the real stop = budget/patience
POOL = int(os.environ.get("BRAIN_POOL", 2000))
SEED = int(os.environ.get("BRAIN_SEED", 42))
BUDGET = int(os.environ.get("BRAIN_BUDGET", 100))      # RANS cap (verifications) — the budget = TOTAL RANS (base+init+loop+final-verify), HARD
PATIENCE = int(os.environ.get("BRAIN_PATIENCE", 12))   # stop if verified best plateaus for N verifications (convergence)
FINAL_RESERVE = int(os.environ.get("BRAIN_FINAL_RESERVE", 0))  # RANS reserved for the FINAL-VERIFY: the loop stops at BUDGET−RESERVE, the final fills up to BUDGET. Router: =FINALK (banks the trusted designs); baseline τ0: 0 (all inline). Guarantees total=BUDGET on BOTH sides.
FINAL_FID = os.environ.get("BRAIN_FINAL_FID", "")            # "hf": FINAL-VERIFY in FINE even if loop is coarse (search cheap, DELIVER fine-verified). Empty=same fidelity as the loop. The DELIVERED best = best_fine (real fine units, NOT coarse-KOH).
FLUSH = os.environ.get("BRAIN_FLUSH", "1") == "1"      # 0 for the sweep (avoids _v_online collision)
ENRICH_EVERY = int(os.environ.get("BRAIN_ENRICH_EVERY", 0))  # #4 ONLINE ENRICHMENT (live): refit zoo δ̂/σ̂ every N RANS verifications (0=off). Integrates anchors near the optimum -> local self-correction (the only anti-curse survivor). ≠ FLUSH (which only persists for the next run).
USE_TR = os.environ.get("BRAIN_TR", "0") == "1"        # 1 = adaptive trust-region (instead of fixed box)
TR_FAILTOL = int(os.environ.get("BRAIN_TR_FAILTOL", 4))  # verifications without gain before shrinking the TR (verifications = sparse)
TRMM = os.environ.get("BRAIN_TRMM", "0") == "1"          # TRMM POLICY (March&Willcox 2011): RANS-verifies the surrogate optimum INSIDE the TR (1/iter = ANCHORING) + resize by ρ=actual/predicted (CONVERGENCE, anti-curse). m(x)=GP mean (=f_low+δ̂ corrected). Docs docs/ml/trmm_march_algorithm.md
TRMM_G0 = float(os.environ.get("BRAIN_TRMM_G0", 0.5))    # TR radius contraction if ρ<η (surrogate over-promised -> tighten = anti-curse)
TRMM_G1 = float(os.environ.get("BRAIN_TRMM_G1", 2.0))    # TR radius expansion if ρ≥η (surrogate reliable -> widen)
TRMM_ETA = float(os.environ.get("BRAIN_TRMM_ETA", 0.2))  # ρ threshold to expand (March: 0.2)
TRMM_DELTA0 = float(os.environ.get("BRAIN_TRMM_DELTA0", 0.4))  # initial TR radius (unit space)
TRMM_DMAX = float(os.environ.get("BRAIN_TRMM_DMAX", 1.0))      # max TR radius
TRMM_EPS = float(os.environ.get("BRAIN_TRMM_EPS", 0.01))       # convergence: floor radius -> STOP (HF optimum found)
TRMM_PRIOR = os.environ.get("BRAIN_TRMM_PRIOR", "0") == "1"     # TRMM v2: m_k(x)=f_low(x)+e_k(x) FAITHFUL to March (f_low=surrogate belief route(), e_k=GP on residuals truth−f_low) instead of GP-on-truth-only (v1 wastes the surrogate). FINALLY uses the zoo for the march.
TRMM_RESTART = os.environ.get("BRAIN_TRMM_RESTART", "0") == "1" # TRMM v2: at convergence (radius<EPS), RE-EXPAND + restart (multi-start, like TuRBO) instead of hard STOP -> works around the stop-brake, uses the remaining budget, escapes the local optimum
INIT_FRAC = float(os.environ.get("BRAIN_INIT_FRAC", 1.0))  # <1: init near x0 (WIDE box for TR reach + feasible init)
STOP_EI = float(os.environ.get("BRAIN_STOP_EI", 0.0))      # EI floor (0=off): stop if max EI < threshold (Nguyen 2017 / cost-aware)
STOP_QUIET = int(os.environ.get("BRAIN_STOP_QUIET", 0))    # quiet gate (0=off): stop if K iters WITHOUT verification (surrogate reliable where the optimizer goes)
AGP_HARD = os.environ.get("BRAIN_AGP", "0") == "1"         # 1 = HARD Augmented-GP (Candelieri): belief INCONSISTENT with the GP → quasi-excluded (anti-phantom-poison)
AGP_BETA = float(os.environ.get("BRAIN_AGP_BETA", 2.0))    # consistency tolerance: |belief−μ_GP| ≤ β·√(σ_GP²+σ̂²) else drop
MIN_TC = float(os.environ.get("BRAIN_MIN_TC", 0.0))        # manufacturability: min thickness t/c (0=off). Blocks a too-thin optimum
MAX_FC = float(os.environ.get("BRAIN_MAX_FC", 1.0))        # max camber |f/c| = GEOMETRIC CM proxy (1=off). Surrogates do not predict CM
# PHYSICAL / manufacturability constraints (objective rules from the papers, NOT data-dependent). Ref Kolla 2021, Martins.
MAX_CURV = int(os.environ.get("BRAIN_MAX_CURV", 2))        # DEFAULT 2: manufacturability ALWAYS ON (anti-waviness; smooth~1-2; bases pass). Ref Kolla 2021. Set 0 to disable.
TE_MIN = float(os.environ.get("BRAIN_TE_MIN", 0.0))       # min trailing-edge thickness (0=off; real airfoils often TE~0)
LE_MIN = float(os.environ.get("BRAIN_LE_MIN", 0.02))      # DEFAULT 0.02: manufacturability ALWAYS ON (LE radius, avoids too sharp; bases pass). Ref Abdul-Kaiyoom & Martins 2024
ACQ = os.environ.get("BRAIN_ACQ", "ei")                    # ei | kg | ucb | pims | es: acquisition. kg=VOI/cost; pims=Prob. of exceeding the max of 1 sample path (Takeno 2023); es=MES max-value entropy (Wang&Jegelka 2017, reduces uncertainty on the optimum). pims/es τ-free, no κ
KG_EPS = float(os.environ.get("BRAIN_KG_EPS", 0.01))       # KG threshold (std units): verify if KG≥ε else trust. REPLACES τ+τ_EI
KG_NZ = int(os.environ.get("BRAIN_KG_NZ", 128))            # MC samples for the KG
KG_POOL = int(os.environ.get("BRAIN_KG_POOL", 256))        # feasible candidates evaluated by the KG (cost M²·NZ)
UCB_KAPPA = float(os.environ.get("BRAIN_UCB_KAPPA", 2.0))  # UCB = μ + κ·σ: low κ=exploit, high κ=explore (ACQ=ucb)
ES_NS = int(os.environ.get("BRAIN_ES_NS", 12))             # ACQ=es (MES, max-value entropy search): nb of sampled max-values m*_s (info-theoretic, reduces uncertainty on the optimum)
NOISE_FLOOR = float(os.environ.get("BRAIN_NOISE", 0.25))  # RANS obs noise variance (quasi-deterministic)
TRUST_WEIGHT = float(os.environ.get("BRAIN_TRUST_WEIGHT", 1.0))  # multiplies σ̂² of the TRUSTED surrogate point in the GP: 1=default, >1=believes LESS (down-weight), <1=believes MORE (towards poison), 1e6=quasi-excluded (GP sees only RANS)
TRUST_EQUAL = os.environ.get("BRAIN_TRUST_EQUAL", "0") == "1"     # 1 = EQUAL WEIGHT: trusted surrogate point enters the GP with the FLOOR noise (like a RANS) -> densifies fully, tests speed-up VS poison
TW_ADAPT = float(os.environ.get("BRAIN_TW_ADAPT", 0.0))          # ADAPTIVE anchoring: TRUST_WEIGHT·(1+TW_ADAPT·frac) -> more down-weight LATE (anchors less near the optimum where the poison tail bites). 0=constant
WD = Path(os.environ.get("OPT_WORKDIR", f"/tmp/brain_{CASE}_{FID}")); WD.mkdir(parents=True, exist_ok=True)

C = T.CASES[CASE]; MACH = C["mach"]; RE = C.get("Re") or 1e6; ALPHA = C["alpha0"]
if os.environ.get("BRAIN_ZOO_AUTO", "0") == "1":                 # AUTO-CURATION: via the IO-CUE (the reference), keeps ONLY the models reliable in the case's regime; drops the out-of-regime experts (poison). "Only these models compete."
    _iod = ROOTP / os.environ.get("IOCUE_DIR", "models/iocue_v4"); _kept = []
    for _m in ZOO:
        _p = _iod / f"{_m}_resid.npz"
        if not _p.exists(): _kept.append(_m); continue           # no IO-CUE -> keep (conservative)
        _z = dict(np.load(_p)); _X = _z["X"]; _r = _z["r_cd"]
        _sel = (np.abs(_X[:, 0] - MACH) < 0.1) & (np.abs(_X[:, 2] - ALPHA) < 3.0)   # probe subset of the REGIME (mach col0, alpha col2)
        if _sel.sum() < 40: _kept.append(_m); continue           # too little data -> keep
        _err = float(np.abs(_r[_sel]).mean() * 1e4)              # mean |r_cd| vs RANS in this regime (counts)
        if _err < 300: _kept.append(_m)                          # < 300c = usable; else out-of-regime expert = poison
        else: print(f"[brain] ZOO-AUTO drops {_m} (|r_cd|={_err:.0f}c out of regime)", flush=True)
    ZOO = _kept
    print(f"[brain] ZOO-AUTO curated {CASE} (M{MACH} α{ALPHA}): {ZOO}", flush=True)
LB, UB = T.box(CASE); DIM = len(LB)
to_cst = lambda u: LB + np.asarray(u) * (UB - LB)
to_unit = lambda x: (np.asarray(x) - LB) / (UB - LB)

# ── POD DIM REDUCTION (BRAIN_POD_K>0): optimize in a kD latent instead of CST-22 ──────────
# GP goes from 22D (n≪D, flat) to kD (dense) + search RESTRICTED to the airfoil manifold (less OOD).
# The bridge stays to_cst/to_unit: everything downstream (feasible, RANS, surrogate, GP) is unchanged.
POD_K = int(os.environ.get("BRAIN_POD_K", 0))
if POD_K > 0:
    _x0c = T.load_x0(CASE).astype(float)                          # POD centre = x0 of the case
    _pod = np.load(ROOTP / "models/pod_cst.npz"); _V = _pod["V"][:, :POD_K]   # (22,k) principal modes
    _Zb = np.abs(_V).T @ ((UB - LB) / 2.0)                        # latent bounds = reach of the box projected on each mode (L1)
    DIM = POD_K
    to_cst = lambda u: np.clip(_x0c + _V @ ((2.0 * np.asarray(u) - 1.0) * _Zb), LB, UB)   # u∈[0,1]^k -> z∈±Zb -> cst clipped to box
    to_unit = lambda x: np.clip((_V.T @ (np.asarray(x) - _x0c) / _Zb + 1.0) / 2.0, 0.0, 1.0)  # cst -> z -> u (exact round-trip for points from to_cst)
    print(f"[brain] POD-{POD_K}D active: latent search (captured var {(_pod['S'][:POD_K]**2).sum()/(_pod['S']**2).sum():.3f}), CST box kept for clip/feasible", flush=True)


def alpha_for(m, cst):
    """fixed-CL -> each model INVERTS its alpha (zoo_predict) to hold target_cl; else fixed alpha0."""
    if not C["fixed_cl"]:
        return ALPHA
    from scipy.optimize import brentq
    try:
        return brentq(lambda a: zoo_predict(m, cst, MACH, a, RE)[0] - C["target_cl"], -5.0, 12.0, maxiter=40)
    except Exception:
        return C["alpha0"]


# ── WHITE-BOX global: black-vs-white ablation (levers if we control global) ──────────
#   BRAIN_WB=off (default): nothing changes, global stays black-box (frozen + IO-CUE).
#   BRAIN_WB=on: global belief = LIVE model (allows fine-tune), post-hoc δ̂ BYPASSED (white repairs, does not patch).
#   BRAIN_WB_FT=K  (axis C): fine-tune global on the RANS anchors every K verified.
#   BRAIN_WB_UQ=mcdrop (axis B): global σ̂ = epistemic MC-dropout instead of IO-CUE.
#   BRAIN_WB_PROPOSE=grad (axis A): candidate = gradient ascent on global (autograd) instead of the acquisition.
WB = os.environ.get("BRAIN_WB", "off") != "off"
WB_FT = int(os.environ.get("BRAIN_WB_FT", 0))                 # 0=no retraining
WB_UQ = os.environ.get("BRAIN_WB_UQ", "")                     # ""=IO-CUE; "mcdrop"=MC-dropout; "ensemble"=deep ensemble
WB_ENS = os.environ.get("BRAIN_WB_ENS", "")                   # members dir (e.g. models/plan4_ens) -> deep ensemble
WB_PROPOSE = os.environ.get("BRAIN_WB_PROPOSE", "")           # ""=acquisition; "grad"=gradient
WB_GLR = float(os.environ.get("BRAIN_WB_GLR", 0.05))         # gradient step = frac of the box
WB_GSTEPS = int(os.environ.get("BRAIN_WB_GSTEPS", 15)); WB_GSTARTS = int(os.environ.get("BRAIN_WB_GSTARTS", 5))
_WB = None
if WB:
    raise SystemExit("BRAIN_WB: the white-box ablation of the thesis is not part of this release")
    from src.opt.wb_global import WBGlobal
    _WB = WBGlobal(ROOTP, MACH, RE, ens_dir=WB_ENS)
    _WB_ANCH = []                                             # (feats27, [CL, CD]) verified RANS anchors for the fine-tune
    print(f"[brain] WHITE-BOX global active: FT={WB_FT} UQ={WB_UQ or 'iocue'} PROPOSE={WB_PROPOSE or 'acq'} members={len(_WB.members)}", flush=True)


def feats27(cst, alpha=ALPHA):
    t_c, f_c = tc_fc(cst)
    return np.concatenate([[MACH, np.log10(RE), alpha, t_c, f_c], np.asarray(cst, float)])


def _curv_sign_changes(co, n=80):
    """#curvature sign changes (y'') on the 2 surfaces = waviness (smooth airfoils~1-2). Ref Kolla 2021."""
    x, y = co[:, 0], co[:, 1]; le = int(np.argmin(x)); out = 0
    for surf in [slice(0, le + 1), slice(le, None)]:
        xs, ys = x[surf], y[surf]; o = np.argsort(xs); xs, ys = xs[o], ys[o]
        xu = np.linspace(max(xs.min(), 0.005), min(xs.max(), 0.995), n)
        yu = np.interp(xu, xs, ys); ypp = np.gradient(np.gradient(yu, xu), xu)
        s = np.sign(ypp[np.abs(ypp) > 0.5])
        out += int(np.sum(s[1:] * s[:-1] < 0))
    return out


def _te_thickness(co):
    """trailing-edge thickness = |y_upper − y_lower| at the TE (x≈1)."""
    return abs(float(co[0, 1] - co[-1, 1]))


def _le_thickness(co, x0=0.02):
    """LE radius proxy: thickness (upper−lower) near the leading edge (small x); too sharp LE → small."""
    x, y = co[:, 0], co[:, 1]; le = int(np.argmin(x))
    xu, yu = x[:le + 1], y[:le + 1]; xl, yl = x[le:], y[le:]
    ou, ol = np.argsort(xu), np.argsort(xl)
    return float(np.interp(x0, xu[ou], yu[ou]) - np.interp(x0, xl[ol], yl[ol]))


def feasible(c):
    """PHYSICAL / manufacturability constraints (OBJECTIVE rules from the papers, NOT data-dependent):
    valid geometry + t/c + camber + smoothness (curvature, Kolla) + TE + LE. 0 RANS."""
    if not T.geom_valid(c):
        return False
    tc, fc = tc_fc(c)
    if not (tc >= MIN_TC and abs(fc) <= MAX_FC):
        return False
    if MAX_CURV > 0 or TE_MIN > 0 or LE_MIN > 0:                            # checks on the decoded geometry (last = expensive)
        try:
            co = decode(np.asarray(c, float))
            if MAX_CURV > 0 and _curv_sign_changes(co) > MAX_CURV:
                return False
            if TE_MIN > 0 and _te_thickness(co) < TE_MIN:
                return False
            if LE_MIN > 0 and _le_thickness(co) < LE_MIN:
                return False
        except Exception:
            return False                                                    # degenerate airfoil (decode fails) -> infeasible
    return True


def zoo_predict(m, cst, mach, alpha, Re):
    """(CL, CD) of zoo member `m` (src/zoo.py) at the point (mach, alpha, Re)."""
    return zoo.predict(m, cst, mach, alpha, Re)


def to_max(J, s):
    """(corrected objective value, σ) -> (value to MAXIMIZE, σ in that scale)."""
    return (-J * 1e4, s * 1e4) if OBJ == "cd" else (J, s)     # cd: minimize -> maximize -CD (counts)


def truth_val(cst, tag, fid=None):
    """RANS truth (or proxy) -> (value to maximize, CL, CD). fid override (final-verify in hf even if loop is coarse)."""
    _fid = fid or FID
    r = G.eval_batch([np.asarray(cst)], CASE, WD, tag, _fid)[0]
    cl, cd = r.get("CL"), r.get("CD")
    if _KOH_C is not None and _fid == "coarse" and cl is not None and cd is not None:   # #3 KOH: coarse-RANS -> fine estimate (ONLY if the eval was coarse; hf=already truth)
        _rcl, _bcl, _rcd, _bcd = _KOH_C
        cl = _rcl * cl + _bcl; cd = _rcd * cd + _bcd
    if cd is None or cl is None or not (0 < cd < 0.05):
        return None, None, None
    J = (cl / max(cd, 1e-4)) if OBJ == "ld" else cd
    v, _ = to_max(J, 0.0)
    return v, cl, cd


def route(cst):
    """The brain at 1 point: returns model=argmin σ̂, belief (max-scale), sigma, evals={m:(cl,cd,alpha)}."""
    evals = {}; permodel = {}; best = None
    for m in ZOO:
        a = alpha_for(m, cst)
        cl, cd = zoo_predict(m, cst, MACH, a, RE)
        evals[m] = (cl, cd, a)
        if not (np.isfinite(cl) and np.isfinite(cd)):                    # XFoil DIVERGED -> (nan,nan): DO NOT call IO.propagate (GBR crashes on NaN in X). Huge σ̂ = never trusted (argmin/invvar ignore it).
            permodel[m] = (0.0, 1e9); continue
        J, s = IO.propagate(feats27(cst, a)[None], [[cl, cd]], m, OBJ)   # corrects bias + propagates σ
        v, sig = to_max(J[0], s[0])
        if WB and m == "global":                                         # ── WHITE-BOX: belief = LIVE model (fine-tuned), post-hoc δ̂ BYPASSED (repair != patch) ──
            clw, cdw = _WB.predict(cst, a); evals[m] = (clw, cdw, a)
            Jw = (clw / max(cdw, 1e-4)) if OBJ == "ld" else cdw
            v, _ = to_max(Jw, 0.0)                                        # raw belief of the white model (no IO-CUE bias correction)
            if WB_UQ in ("mcdrop", "ensemble"):                           # axis B: native epistemic σ̂ (MC-dropout OR deep ensemble) instead of IO-CUE
                _, _, scl, scd = _WB.mc_sigma(cst, a, 24)
                s_ld = (scl / max(cdw, 1e-4) + clw * scd / max(cdw, 1e-4) ** 2) if OBJ == "ld" else scd * 1e4  # propagates σ to the L/D scale (or CD counts)
                sig = abs(float(s_ld))
        if not (np.isfinite(v) and np.isfinite(sig)):                    # model unavailable (e.g. XFoil DIVERGED -> nan): huge σ̂ = ignored by argmin AND invvar (weight 1/σ̂²≈0)
            v, sig = 0.0, 1e9
        permodel[m] = (float(v), float(sig))                             # per-model log (interactive dashboard)
        if best is None or sig < best[1]:
            best = (m, sig, v)
    if FUSION == "mean":                                                 # UNIFORM MIXTURE of the whole zoo: the consensus direction, no per-member decision. Offline: tau +0.43/+0.67 in box vs +0.45/+0.70 for the best member, and seed-independent (RGPE: 3 seeds, 3 winners). An aberrant magnitude weighs 1/N instead of being concentrated.
        ok = [m for m in ZOO if permodel[m][1] < 1e8]                    # available members (σ̂=1e9 = unavailable, e.g. diverged XFoil)
        if ok:
            bel = float(np.mean([permodel[m][0] for m in ok]))
            sig_f = float(np.sqrt(np.mean([permodel[m][1] ** 2 for m in ok])))   # RMS σ̂ of the mixture (not the inverse-variance formula: no member is favoured)
            return dict(model="mean", belief=bel, sigma=sig_f, evals=evals, permodel=permodel)
    if FUSION == "invvar":                                               # LCWF-family (March §VI): INVERSE-VARIANCE weighted fusion of the WHOLE zoo (vs argmin=1 model)
        vs = np.array([permodel[m][0] for m in ZOO], float)
        sg = np.array([max(permodel[m][1], 1e-6) for m in ZOO], float)
        w = (1.0 / sg ** 2); w = w / w.sum()
        bel = float((w * vs).sum()); sig_f = float(np.sqrt(1.0 / (1.0 / sg ** 2).sum()))  # σ̂_fus < min(σ̂) (Eq 26)
        return dict(model="invvar", belief=bel, sigma=sig_f, evals=evals, permodel=permodel)
    if FUSION == "skill" and len(_SKILL_HIST) >= SKILL_K:                 # MODEL-SELECTION by MEASURED skill: the most accurate model at neighbouring RANS anchors
        xu = to_unit(cst)
        d = np.array([np.linalg.norm(xu - h[0]) for h in _SKILL_HIST])
        idx = np.argsort(d)[:SKILL_K]                                    # K nearest anchors
        merr = {m: float(np.mean([_SKILL_HIST[i][1].get(m, 1e9) for i in idx])) for m in ZOO}
        bm = min(ZOO, key=lambda m: merr[m])                            # model with the smallest local REAL error
        return dict(model=f"skill:{bm}", belief=permodel[bm][0], sigma=permodel[bm][1], evals=evals, permodel=permodel)
    return dict(model=best[0], belief=best[2], sigma=best[1], evals=evals, permodel=permodel)


def cand_value(r, cand):
    """Value ('to maximize' scale) of a prior CANDIDATE at a point, from a route() result.
    cand = "<member>" -> DEBIASED belief (IO-CUE δ̂ applied); cand = "<member>|raw" -> RAW prediction of the member (δ̂ off for it alone)."""
    m, _, mode = cand.partition("|")
    if mode != "raw":
        return float(r["permodel"][m][0])
    cl, cd, _a = r["evals"][m]
    if not (np.isfinite(cl) and np.isfinite(cd)) or cd <= 0:
        return float("nan")
    J = (cl / max(cd, 1e-4)) if OBJ == "ld" else cd
    v, _ = to_max(J, 0.0)
    return float(v)


def cand_field(cst, cand):
    """Candidate value at ONE point, evaluating ONLY its member (route() would evaluate the whole zoo: 12x more expensive per point)."""
    m, _, mode = cand.partition("|")
    a = alpha_for(m, cst)
    cl, cd = zoo_predict(m, cst, MACH, a, RE)
    if not (np.isfinite(cl) and np.isfinite(cd)) or cd <= 0:
        return float("nan")
    if mode == "raw":
        J = (cl / max(cd, 1e-4)) if OBJ == "ld" else cd
        v, _ = to_max(J, 0.0)
        return float(v)
    J, sg = IO.propagate(feats27(cst, a)[None], [[cl, cd]], m, OBJ)
    v, _ = to_max(J[0], sg[0])
    return float(v)


def prior_candidates():
    """List of prior candidates: each zoo member, debiased and (optionally) raw, + NONE (no prior)."""
    c = list(ZOO) + ([f"{m}|raw" for m in ZOO] if PINIT_RAW else [])
    return c


def prescreen_prior_scores2(Xu_init, Y_init, n_box=300):
    """EXTENDED PRE-SCREEN (0 RANS) over the candidates (member × δ̂ on/off): field smoothness over the BOX,
    Spearman and residual amplitude at the init anchors, field spread over the box.
    Returns {cand: (lscale, rho, sres, spread)}; non-finite candidates discarded."""
    from scipy.stats import spearmanr
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
    Ub = SobolEngine(DIM, scramble=True, seed=SEED + 777).draw(4 * n_box).numpy()
    Ub = np.array([u for u in Ub if feasible(to_cst(u))][:n_box])
    Rb = [route(to_cst(u)) for u in Ub]                        # a single route() call per point: yields both permodel AND evals
    Ra = [route(to_cst(u)) for u in Xu_init]
    y = np.asarray(Y_init, float); sy = float(np.std(y)) + 1e-9; out = {}; FA = {}
    for c in prior_candidates():
        fb = np.array([cand_value(r, c) for r in Rb]); fa = np.array([cand_value(r, c) for r in Ra])
        if not (np.isfinite(fb).all() and np.isfinite(fa).all()) or float(np.std(fb)) < 1e-9:
            continue                                           # member unavailable (diverged XFoil, out-of-regime expert) -> not a candidate
        k = Ck(1.0, (1e-2, 1e3)) * RBF(np.ones(DIM), (1e-2, 1e2)) + WhiteKernel(1e-2, (1e-6, 1e1))
        try:
            g = GaussianProcessRegressor(kernel=k, normalize_y=True, random_state=0).fit(Ub, fb)
            ls = float(np.exp(np.mean(np.log(np.exp(g.kernel_.theta)[1:1 + DIM]))))
        except Exception:
            ls = 1e-2
        rho = float(spearmanr(fa, y)[0]) if len(y) > 3 else 0.0
        out[c] = (ls, rho if np.isfinite(rho) else float("nan"), float(np.std(y - fa)) / sy, float(np.std(fb)) / sy)
        FA[c] = fa
    prescreen_prior_scores2._fields = (FA, y, {c: out[c][0] for c in out}, {c: out[c][3] for c in out})   # for the stability test
    return out


def pick_prior_candidate(scores):
    """Decision rule: among candidates with a reasonable residual amplitude, the SMOOTHEST.
    No smooth candidate -> None (run WITHOUT a prior rather than with a bad one)."""
    ok = {c: v for c, v in scores.items()
          if v[2] <= PINIT_SRES_MAX and v[3] >= PINIT_SPREAD_MIN and np.isfinite(v[1]) and np.isfinite(v[0])}
    if not ok:
        return None, "no informative candidate (residual too large or near-constant field)"
    ls_max = max(v[0] for v in ok.values())
    if ls_max < PINIT_LS_MIN:
        return None, f"no smooth field (best ℓ={ls_max:.2f} < {PINIT_LS_MIN})"
    band = {c: v for c, v in ok.items() if v[0] >= PINIT_LS_BAND * ls_max}   # smoothness is a THRESHOLD, not a ranking: at comparable smoothness, tie-break on the residual
    best = min(band, key=lambda c: band[c][2])
    if len(band) > 1:                                          # ── STABILITY: the tie-break rests on the residual of 6 anchors. Is it reproducible? ──
        fr = _choice_stability(best)
        if fr is not None and fr < PINIT_STAB_MIN:             # measured: when the choice oscillates, perf drops and scatters (n4412 ±1.8, raeod ±5.4) -> the default is better
            fb_ = PINIT_FALLBACK if PINIT_FALLBACK in band else None
            return fb_, f"unstable choice ({fr:.0%} < {PINIT_STAB_MIN:.0%} of the resamples) -> {'fallback ' + PINIT_FALLBACK if fb_ else 'no prior'}"
        return best, f"smoothness (ℓ≥{PINIT_LS_BAND:.0%} of max), min residual among {len(band)}, stable {fr:.0%}" if fr is not None else f"smoothness then min residual among {len(band)}"
    return best, f"smoothness (ℓ≥{PINIT_LS_BAND:.0%} of max) then min residual among {len(band)}"


def _choice_stability(best):
    """Frequency at which `best` stays chosen when the anchors are resampled (bootstrap). None if unavailable."""
    F = getattr(prescreen_prior_scores2, "_fields", None)
    if F is None: return None
    from scipy.stats import spearmanr
    FA, y, LS, SP = F
    n = len(y)
    if n < 5: return None
    rng = np.random.default_rng(0); k = max(4, int(round(0.8 * n))); hit = 0
    for _ in range(PINIT_STAB_B):
        sel = rng.choice(n, size=k, replace=False); ys = y[sel]; sy = float(np.std(ys)) + 1e-9
        sc = {}
        for c in FA:
            fa = FA[c][sel]; rho = spearmanr(fa, ys)[0]
            sc[c] = (LS[c], rho if np.isfinite(rho) else float("nan"), float(np.std(ys - fa)) / sy, SP[c])
        ok = {c: v for c, v in sc.items() if v[2] <= PINIT_SRES_MAX and v[3] >= PINIT_SPREAD_MIN and np.isfinite(v[1]) and np.isfinite(v[0])}
        if not ok: continue
        lm = max(v[0] for v in ok.values())
        if lm < PINIT_LS_MIN: continue
        bd = {c: v for c, v in ok.items() if v[0] >= PINIT_LS_BAND * lm}
        if min(bd, key=lambda c: bd[c][2]) == best: hit += 1
    return hit / PINIT_STAB_B


def prescreen_prior_scores(Xu_init, Y_init, n_box=300):
    """PRE-SCREEN AT INIT (0 extra RANS): for each zoo member, (a) smoothness of its predicted field over the BOX
    (correlation length of an RBF-ARD GP fitted on n_box feasible Sobol), (b) Spearman surrogate↔RANS and
    (c) std(residual)/std(y) on the (spread) INIT anchors. Score = sum of z-scores (log lscale, ρ, −std_res).
    Measures the property that makes a good prior (smooth + well oriented far from the optimum), where late anchors cannot see it."""
    from scipy.stats import spearmanr
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
    Ub = SobolEngine(DIM, scramble=True, seed=SEED + 777).draw(4 * n_box).numpy()
    Ub = np.array([u for u in Ub if feasible(to_cst(u))][:n_box])
    PMb = [route(to_cst(u))["permodel"] for u in Ub]
    PMa = [route(to_cst(u))["permodel"] for u in Xu_init]
    y = np.asarray(Y_init, float); sy = float(np.std(y)) + 1e-9; sc = {}; raw = {}
    for m in ZOO:
        fb = np.array([r[m][0] for r in PMb]); fa = np.array([r[m][0] for r in PMa])
        k = Ck(1.0, (1e-2, 1e3)) * RBF(np.ones(DIM), (1e-2, 1e2)) + WhiteKernel(1e-2, (1e-6, 1e1))
        try:
            g = GaussianProcessRegressor(kernel=k, normalize_y=True, random_state=0).fit(Ub, fb)
            ls = float(np.exp(np.mean(np.log(np.exp(g.kernel_.theta)[1:1 + DIM]))))
        except Exception:
            ls = 1e-2
        rho = float(spearmanr(fa, y)[0]) if len(y) > 3 else 0.0
        sres = float(np.std(y - fa)) / sy
        raw[m] = (np.log(max(ls, 1e-3)), rho if np.isfinite(rho) else 0.0, -sres)
    Z = np.array([raw[m] for m in ZOO], float); Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    for i, m in enumerate(ZOO): sc[m] = float(Z[i].sum())
    return sc, raw


def fit_gp_hetero(Xa, Ya, Yva):
    """GP with PER-POINT OBS NOISE (clean MFBO): verified RANS -> floor noise; trusted surrogate ->
    noise = σ̂² (IO-CUE) → the GP DOWN-WEIGHTS uncertain points (no poison). Returns (gp, ym, ys)
    to standardize the EI correctly (the GP standardizes Y internally)."""
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    X = torch.tensor(np.array(Xa), dtype=torch.double)
    Y = torch.tensor(np.array(Ya), dtype=torch.double)
    Yv = torch.tensor(np.array(Yva), dtype=torch.double)
    ym = Y.mean(); ys = Y.std() + 1e-9
    Yn = ((Y - ym) / ys).unsqueeze(-1)
    Yvn = (Yv / ys**2).clamp_min(1e-8).unsqueeze(-1)               # standardized noise
    covar = G._dim_scaled_covar(X.shape[-1])                        # same HD prior as the refs
    gp = SingleTaskGP(X, Yn, train_Yvar=Yvn, covar_module=covar)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp, float(ym), float(ys)


def mc_kg(gp, U, lam, nz=128, seed=0, off=None):
    """Discrete Monte-Carlo KG (Frazier): for each candidate j, expected gain of the MAX of the posterior mean
    if j is VERIFIED with RANS (noise λ). μ,Σ standardized (the GP is on standardized Y). Returns (kg (len U), mu)."""
    with torch.no_grad():
        post = gp.posterior(torch.tensor(np.asarray(U), dtype=torch.double))
        mu = post.mean.squeeze(-1).double().numpy()
        cov = post.mvn.covariance_matrix.double().numpy()
    if off is not None:                                             # prior-mean: the target is f_low + residual; off = (f_low + ym)/ys in residual-GP units
        mu = mu + np.asarray(off, float)
    var = np.clip(np.diag(cov), 1e-12, None)
    sigt = cov / np.sqrt(var + lam)[None, :]                        # σ̃[i,j] = cov[i,j] / sqrt(var_j + λ)
    m0 = float(mu.max()); Z = np.random.default_rng(seed).standard_normal(nz)
    kg = np.array([(mu[:, None] + np.outer(sigt[:, j], Z)).max(0).mean() - m0 for j in range(len(mu))])
    return np.clip(kg, 0.0, None), mu


def main():
    print(f"[brain] case={CASE} obj={OBJ} fid={FID} zoo={ZOO} τ={TAU} τ_EI={TAU_EI} dim={DIM}", flush=True)
    sob = SobolEngine(DIM, scramble=True, seed=SEED)          # TOLERANT init: redraw until N_INIT feasible
    class _Traj(list):                                         # CHECKPOINT: persists the trajectory at EVERY verified RANS -> resume after preemption (the parquet used to be written only at the end)
        def append(self, row):
            list.append(self, row)
            if row.get("verified"):
                try:
                    import pandas as _pd
                    _tmp = WD / f".traj_brain_{OBJ}_tau{TAU}.parquet.tmp"
                    _pd.DataFrame(list(self)).to_parquet(_tmp); os.replace(_tmp, WD / f"traj_brain_{OBJ}_tau{TAU}.parquet")   # atomic write
                except Exception:
                    pass                                           # a failed checkpoint must NEVER stop the optimization
    Xu, Y, Yv = [], [], []; best_true = -np.inf; best_cst = None; traj = _Traj(); n_truth = 0; beliefs = []
    Xuv, YCL, YCD = [], [], []                                 # composite: (unit, CL, CD) at RANS-VERIFIED points only
    def _add_comp(u_pt, cl, cd):                               # composite helper: records verified CL/CD
        if COMPOSITE and cl is not None and cd is not None and np.isfinite(cl) and np.isfinite(cd):
            Xuv.append(np.asarray(u_pt, float)); YCL.append(float(cl)); YCD.append(float(cd))
    base_val = None; tau_eff = TAU                            # tau_eff = effective tolerance (absolute OR %·|base|)
    x0 = T.load_x0(CASE).astype(float)
    _self_ck = WD / f"traj_brain_{OBJ}_tau{TAU}.parquet"       # AUTO-RESUME after preemption: if a checkpoint of THIS workdir exists, restart from its verified RANS
    RESUME = os.environ.get("BRAIN_RESUME_TRAJ", "") or (str(_self_ck) if _self_ck.exists() else ""); resumed = False; bv = None
    if RESUME and Path(RESUME).exists():                      # RESUME: reuses the already VERIFIED designs (RANS)
        import pandas as _pd
        dfr = _pd.read_parquet(RESUME); vr = dfr[dfr.verified & (dfr.val > -1e8)]
        for _, rr in vr.iterrows():
            c = np.asarray(rr.cst22, float); vv = float(rr.val)
            Xu.append(to_unit(c)); Y.append(vv); Yv.append(NOISE_FLOOR)
            if vv > best_true: best_true, best_cst = vv, c
            traj.append(dict(it=-1, n_truth=len(Xu), verified=True, model="RESUME", sigma=0.0,
                             val=vv, best=best_true, cst22=c))   # logs the resumed history (full curve)
        n_truth = len(Xu)
        br = dfr[dfr.model == "BASE"]; base_val = float(br.val.iloc[0]) if len(br) else best_true
        tau_eff = (TAU_PCT / 100.0 * abs(base_val)) if TAU_PCT > 0 else TAU
        resumed = len(Xu) >= 3
        print(f"[brain] RESUME {len(Xu)} verified from {RESUME} | best={best_true:.2f} n_truth={n_truth}", flush=True)
    if not resumed:
        bv, bcl, bcd = truth_val(x0, "base"); n_truth += 1
    if bv is not None:
        base_val = bv; Xu.append(to_unit(x0)); Y.append(bv); Yv.append(NOISE_FLOOR); best_true, best_cst = bv, x0
        _add_comp(to_unit(x0), bcl, bcd)
        tau_eff = (TAU_PCT / 100.0 * abs(bv)) if TAU_PCT > 0 else TAU   # % tolerance -> absolute (scale-free)
        for m in ZOO:
            a = alpha_for(m, x0); em = zoo_predict(m, x0, MACH, a, RE)
            IO.record_residual(m, feats27(x0, a), em[0], em[1], bcl, bcd)
        traj.append(dict(it=-1, n_truth=n_truth, verified=True, model="BASE", sigma=0.0, val=bv, best=bv, cst22=x0))
        print(f"[brain] base {CASE}: {('CD=%.1f cts' % (bcd*1e4)) if OBJ=='cd' else ('L/D=%.2f' % (bcl/max(bcd,1e-4)))}", flush=True)
    attempts = 0
    _pq = []                                                  # queue of anchors proposed by the PRIOR (guided init)
    if INIT_PRIOR > 0 and not resumed:
        _Up = SobolEngine(DIM, scramble=True, seed=SEED + 555).draw(3000).numpy(); _cand = []
        for _u in _Up:
            _c = to_cst(_u)
            if not feasible(_c): continue
            _r = route(_c)
            if _r["belief"] is not None and np.isfinite(_r["belief"]): _cand.append((float(_r["belief"]), _u))
        _cand.sort(key=lambda t: -t[0])
        for _b, _u in _cand:                                  # diversified top-k (unit distance >= 0.15 between them)
            if all(np.linalg.norm(_u - _v) >= 0.15 for _v in _pq): _pq.append(_u)
            if len(_pq) >= INIT_PRIOR: break
        print(f"[brain] INIT_PRIOR={INIT_PRIOR}: {len(_pq)} anchors proposed by the prior (max belief {_cand[0][0]:.2f} over {len(_cand)} feasible)", flush=True)
    while len(Xu) < N_INIT and attempts < N_INIT * 12:
        u = _pq.pop(0) if _pq else sob.draw(1).numpy()[0]; attempts += 1
        if INIT_FRAC < 1.0 and not (INIT_PRIOR > 0 and attempts <= INIT_PRIOR):
            u = np.clip(0.5 + (u - 0.5) * INIT_FRAC, 0.0, 1.0)   # init near x0 (unit 0.5 = box centre)
        c = to_cst(u)
        if not feasible(c): continue                          # free GEOMETRIC + manufacturability rejection (no RANS)
        if n_truth < INIT_RANS:                               # ── RANS ANCHOR (calibrates the fidelity correlation) ──
            v, cl, cd = truth_val(c, f"init{n_truth}"); n_truth += 1
            if v is None: continue                            # RANS infeasible -> redraw
            Xu.append(u); Y.append(v); Yv.append(NOISE_FLOOR) # RANS = floor noise
            _add_comp(u, cl, cd)
            for m in ZOO:                                     # enriches from init (verified residuals)
                a = alpha_for(m, c); em = zoo_predict(m, c, MACH, a, RE)
                IO.record_residual(m, feats27(c, a), em[0], em[1], cl, cd)
            if v > best_true: best_true, best_cst = v, c
            traj.append(dict(it=-1, n_truth=n_truth, verified=True, model="RANS", sigma=0.0,
                             val=v, best=best_true, cst22=np.asarray(c, float)))
        else:                                                 # ── surrogate BELIEF (HYBRID init: 0 RANS, HONEST σ̂² weight) ──
            r = route(c)                                       # belief = obj − δ̂ (debiased); σ̂ = conformal uncertainty
            if r["belief"] is None or not np.isfinite(r["belief"]): continue
            Xu.append(u); Y.append(r["belief"])
            Yv.append(TRUST_WEIGHT * max(r["sigma"], 1e-3) ** 2)   # σ̂² = honest RESIDUAL variance (not floor); best_true NOT updated (unproven)
            traj.append(dict(it=-1, n_truth=n_truth, verified=False, model=r["model"], sigma=r["sigma"],
                             reason="init-trust", val=r["belief"], best=best_true, cst22=np.asarray(c, float)))
    print(f"[brain] init {len(Xu)} feasible in {n_truth} RANS ({attempts} draws) | best={best_true:.2f}", flush=True)
    if len(Xu) < 3:                                            # GUARD: GP unfittable -> clean abort (no crash)
        print(f"[brain] ABORT: <3 feasible init (box too hard) -> reduce OPT_BOX_K", flush=True); return

    it = 0; no_improve = 0; quiet = 0                          # STOP = RANS budget OR plateau OR EI floor OR quiet gate
    tr = None; trmm_delta = TRMM_DELTA0; _sig_hist = []        # TRMM: TR radius; _sig_hist: σ̂ seen (for CTD pacing)
    if USE_TR and not TRMM:                                    # TuRBO: adaptive LOCAL box (centre = verified best)
        tr = TurboState(dim=DIM, batch_size=1, best_value=best_true); tr.failure_tolerance = TR_FAILTOL
    while n_truth < BUDGET - FINAL_RESERVE and no_improve < PATIENCE and it < N_ITER:
        gp, ym, ys = fit_gp_hetero(Xu, Y, Yv)
        _pm_flowU = None                                       # prior-mean BO (no TRMM): GP re-fit on the residual (truth − f_low)
        _pm_w = None                                           # selector weights, reused at the candidates (line ~765)
        if PRIOR_MEAN and not TRMM:
            if PRIOR_SELECT in ("argmin", "rgpe"):             # same selector as the TRMM branch, but WITHOUT trust region
                from src.opt.prior_select import select_prior, combine, gp_loo_mean
                _ra = [route(to_cst(u))["permodel"] for u in Xu]
                _pm_Ba = {m: np.array([r[m][0] for r in _ra]) for m in ZOO}
                _tgt = None
                try:                                           # target of the "no prior" candidate = GP LOO on the truth
                    with torch.no_grad():
                        _Xt = torch.tensor(np.array(Xu), dtype=torch.double)
                        _K = gp.covar_module(_Xt).evaluate().double().numpy() + np.diag(np.asarray(Yv, float) / (ys ** 2) + 1e-6)
                    _tgt = gp_loo_mean(_K, (np.asarray(Y, float) - ym) / ys) * ys + ym
                except Exception:
                    _tgt = None
                _pm_w, _pm_info = select_prior(_pm_Ba, np.asarray(Y, float), mode=PRIOR_SELECT,
                                               target_pred=(_tgt if PSEL_NONE else None), S=RGPE_S, seed=SEED + it)
                _flow_a = combine(_pm_w, _pm_Ba)
                if it % 10 == 0:
                    print(f"[brain] PRIOR-SELECT({PRIOR_SELECT}, no TRMM) it{it} n_anchors={len(Xu)} weights={ {m: round(w, 2) for m, w in _pm_w.items() if w > 0} } losses={ {m: v for m, v in _pm_info.get('losses', {}).items()} }", flush=True)
            else:
                _flow_a = np.array([route(to_cst(u))["belief"] for u in Xu])
            gp, ym, ys = fit_gp_hetero(Xu, list(np.asarray(Y) - _flow_a), Yv)
        gp_cl = gp_cd = None                                   # composite: 2 separate GPs on verified CL, CD
        if COMPOSITE and len(Xuv) >= 3:
            _yvc = [NOISE_FLOOR] * len(Xuv)
            gp_cl, _, _ = fit_gp_hetero(Xuv, YCL, _yvc)
            gp_cd, _, _ = fit_gp_hetero(Xuv, YCD, _yvc)
        if TRMM:                                               # TRMM: box centred on the proven BEST, radius = trmm_delta (resize by ρ)
            xc = to_unit(best_cst)
            lb_tr = np.clip(xc - trmm_delta, 0.0, 1.0); ub_tr = np.clip(xc + trmm_delta, 0.0, 1.0)
            U = lb_tr + SobolEngine(DIM, scramble=True, seed=SEED + 2000 + it).draw(POOL).numpy() * (ub_tr - lb_tr)
        elif USE_TR:                                           # candidates INSIDE the TR (weights = GP lengthscales, ARD)
            xc = to_unit(best_cst)
            _b = getattr(gp.covar_module, "base_kernel", gp.covar_module)
            _ls = _b.lengthscale.detach().double().numpy().ravel()
            w = _ls / _ls.mean(); w = w / (np.prod(w) ** (1.0 / DIM))
            lb_tr = np.clip(xc - w * tr.length / 2, 0.0, 1.0); ub_tr = np.clip(xc + w * tr.length / 2, 0.0, 1.0)
            U = lb_tr + SobolEngine(DIM, scramble=True, seed=SEED + 1000 + it).draw(POOL).numpy() * (ub_tr - lb_tr)
        else:
            U = SobolEngine(DIM, scramble=True, seed=SEED + 1000 + it).draw(POOL).numpy()
        gv = np.array([feasible(to_cst(u)) for u in U])       # free geometric + manufacturability mask (t/c, camber)
        mask = gv & (cdist(U, np.array(Xu)).min(1) > 1e-4)
        if not mask.any():
            it += 1; continue
        if TRMM:                                               # ══ TRMM POLICY (March&Willcox): 1 RANS/iter at the TR optimum + ρ-resize ══
            if TRMM_PRIOR:                                     # v2 FAITHFUL to March: m_k(x) = f_low(x) + e_k(x)
                _psel_w = None
                if PRIOR_SELECT == "init":                        # ── PRE-SCREEN AT INIT: prior chosen ONCE on box-smoothness + init-correlation (0 RANS) ──
                    from src.opt.prior_select import combine
                    if not hasattr(main, "_pinit_w"):
                        _sc, _raw = prescreen_prior_scores(Xu, Y)
                        if PINIT_RULE == "smooth":                    # smoothness rule: the prior = the smoothest field over the box; all rough -> RANS generalist
                            _ls = {m: float(np.exp(_raw[m][0])) for m in ZOO}; _bm = max(_ls, key=_ls.get)
                            if _ls[_bm] < PINIT_LS_MIN: _bm = PINIT_FALLBACK if PINIT_FALLBACK in ZOO else _bm
                            _sc = _ls
                        else:
                            _bm = max(_sc, key=_sc.get)
                        main._pinit_w = {m: (1.0 if m == _bm else 0.0) for m in ZOO}
                        print(f"[brain] PRIOR-INIT chosen={_bm} scores={ {m: round(v, 2) for m, v in _sc.items()} } raw(log_ls,ρ,-sres)={ {m: tuple(round(x, 2) for x in v) for m, v in _raw.items()} }", flush=True)
                    _psel_w = main._pinit_w
                    _ra = [route(to_cst(u))["permodel"] for u in Xu]; _Ba = {m: np.array([r[m][0] for r in _ra]) for m in ZOO}
                    flow_a = combine(_psel_w, _Ba)
                    _rU = [route(to_cst(u))["permodel"] for u in U]; flow_U = combine(_psel_w, {m: np.array([r[m][0] for r in _rU]) for m in ZOO})
                    _rc = route(best_cst)["permodel"]; _fc = float(combine(_psel_w, {m: np.array([_rc[m][0]]) for m in ZOO})[0])
                elif PRIOR_SELECT == "init2":                      # ── EXTENDED PRE-SCREEN: candidates = member × (δ̂ on/off), amplitude/spread guards, no-prior fallback, REVISION at PINIT_REVISE anchors ──
                    _nrev = int(getattr(main, "_pinit2_n", -1))
                    if not hasattr(main, "_pinit2") or (PINIT_REVISE and _nrev < PINIT_REVISE <= len(Xu)):
                        _first = not hasattr(main, "_pinit2")
                        _sc2 = prescreen_prior_scores2(Xu, Y)
                        _cand, _why = pick_prior_candidate(_sc2)
                        _old = getattr(main, "_pinit2", None)
                        main._pinit2 = _cand; main._pinit2_n = len(Xu)
                        _tab = " | ".join(f"{c}: ℓ{v[0]:.2f} ρ{v[1]:+.2f} r{v[2]:.2f} a{v[3]:.2f}" for c, v in sorted(_sc2.items(), key=lambda t: -t[1][0])[:6])
                        _tag = "PRIOR-INIT2" if _first else f"PRIOR-INIT2 REVISION({len(Xu)} anchors, before={_old or 'none'})"
                        print(f"[brain] {_tag} chosen={_cand or 'NONE (no prior)'} ({_why}) | top: {_tab}", flush=True)
                    _cand = main._pinit2
                    if _cand is None:                              # no trustworthy prior -> f_low ≡ 0 = pure TRMM (the GP carries everything)
                        flow_a = np.zeros(len(Xu)); flow_U = np.zeros(len(U)); _fc = 0.0
                    else:
                        flow_a = np.array([cand_field(to_cst(u), _cand) for u in Xu])
                        flow_U = np.array([cand_field(to_cst(u), _cand) for u in U])
                        _fc = float(cand_field(best_cst, _cand))
                elif PRIOR_SELECT in ("argmin", "rgpe"):         # ── ONLINE PRIOR SELECTION (RGPE-like): which candidate best ranks the RANS ANCHORS? ──
                    from src.opt.prior_select import select_prior, combine, gp_loo_mean, NONE as _NONE
                    _ra = [route(to_cst(u))["permodel"] for u in Xu]                 # PER-model beliefs at the anchors
                    _Ba = {m: np.array([r[m][0] for r in _ra]) for m in ZOO}
                    _tgt = None
                    try:                                                            # target = GP-on-truth (fit at loop start), closed-form LOO on its anchors
                        with torch.no_grad():
                            _Xt = torch.tensor(np.array(Xu), dtype=torch.double)
                            _K = gp.covar_module(_Xt).evaluate().double().numpy() + np.diag(np.asarray(Yv, float) / (ys ** 2) + 1e-6)
                        _tgt = gp_loo_mean(_K, (np.asarray(Y, float) - ym) / ys) * ys + ym
                    except Exception:
                        _tgt = None
                    if PSEL_FREEZE > 0 and hasattr(main, "_psel_frozen"):   # FREEZE active: reuse the frozen weights, no more recomputation
                        _psel_w, _psel_info = main._psel_frozen, dict(reason=f"freeze@{PSEL_FREEZE}", losses={})
                    elif PSEL_WARM > 0 and len(Xu) < PSEL_WARM:           # WARM-UP: fixed default prior until PSEL_WARM anchors (decision too noisy before)
                        _psel_w = {m: (1.0 if m == PSEL_WARM_ZOO else 0.0) for m in ZOO}; _psel_info = dict(reason=f"warm<{PSEL_WARM}", losses={})
                    else:
                        _psel_w, _psel_info = select_prior(_Ba, np.asarray(Y, float), mode=PRIOR_SELECT, target_pred=(_tgt if PSEL_NONE else None), S=RGPE_S, seed=SEED + it)
                        if PSEL_FREEZE > 0 and len(Xu) >= PSEL_FREEZE:     # last estimate made on a still-spread population -> freeze it
                            main._psel_frozen = _psel_w
                            print(f"[brain] PRIOR-SELECT FROZEN at {len(Xu)} anchors, frozen weights = { {m: round(w, 2) for m, w in _psel_w.items() if w > 0.01} }", flush=True)
                    flow_a = combine(_psel_w, _Ba)
                    _rU = [route(to_cst(u))["permodel"] for u in U]
                    flow_U = combine(_psel_w, {m: np.array([r[m][0] for r in _rU]) for m in ZOO})
                    _rc = route(best_cst)["permodel"]; _fc = float(combine(_psel_w, {m: np.array([_rc[m][0]]) for m in ZOO})[0])
                    if it % 10 == 0:
                        print(f"[brain] PRIOR-SELECT({PRIOR_SELECT}) it{it} n_anchors={len(Xu)} weights={ {m: round(w, 2) for m, w in _psel_w.items() if w > 0} } losses={ {m: v for m, v in _psel_info.get('losses', {}).items()} }", flush=True)
                else:
                    flow_a = np.array([route(to_cst(u))["belief"] for u in Xu])        # f_low = surrogate belief at the ANCHORS
                    flow_U = np.array([route(to_cst(u))["belief"] for u in U])         # f_low at the TR candidates
                    _fc = float(route(best_cst)["belief"])
                gp_e, em_e, es_e = fit_gp_hetero(Xu, list(np.asarray(Y) - flow_a), Yv)  # e_k = GP on the discrepancy (truth − f_low)
                with torch.no_grad():
                    ek = gp_e.posterior(torch.tensor(U, dtype=torch.double)).mean.squeeze(-1).numpy() * es_e + em_e
                    ek_c = float(gp_e.posterior(torch.tensor(to_unit(best_cst)[None], dtype=torch.double)).mean.item()) * es_e + em_e
                mgp = flow_U + ek                              # m_k = f_low + e_k (CORRECTED surrogate, the zoo finally guides the march)
                m_ctr = _fc + ek_c
            else:                                              # v1: GP mean on the truth only (surrogate ignored)
                with torch.no_grad():
                    mgp = gp.posterior(torch.tensor(U, dtype=torch.double)).mean.squeeze(-1).numpy() * ys + ym
                    m_ctr = float(gp.posterior(torch.tensor(to_unit(best_cst)[None], dtype=torch.double)).mean.item()) * ys + ym
            k = int(np.argmax(np.where(mask, mgp, -1e18)))     # step1: surrogate optimum INSIDE the TR
            c = to_cst(U[k]); m_cand = float(mgp[k])
            v, cl, cd = truth_val(c, f"trmm{it}"); n_truth += 1  # step2: HF (RANS) VERIFICATION at the TR optimum = THE ANCHORING
            if v is None:
                trmm_delta *= TRMM_G0; it += 1; continue        # failed eval -> tighten, retry
            pred = max(m_cand - m_ctr, 1e-6)                    # PREDICTED improvement (surrogate; floor avoids /0)
            rho = float(np.clip((v - best_true) / pred, -100.0, 100.0))  # step3: ρ = actual / predicted (clamped, robust)
            for _m in ZOO:                                      # enriches the IO-CUE (verified residual)
                _a = alpha_for(_m, c); _em = zoo_predict(_m, c, MACH, _a, RE); IO.record_residual(_m, feats27(c, _a), _em[0], _em[1], cl, cd)
            Xu.append(to_unit(c)); Y.append(v); Yv.append(NOISE_FLOOR)   # RANS anchor in the GP
            _add_comp(to_unit(c), cl, cd)
            improved = v > best_true
            if improved:                                        # step4: accept if real improvement
                best_true, best_cst = v, c; no_improve = 0
            else:
                no_improve += 1
            trmm_delta = min(trmm_delta * TRMM_G1, TRMM_DMAX) if rho >= TRMM_ETA else trmm_delta * TRMM_G0  # step5: resize by ρ
            _distt = float(np.min(np.linalg.norm(np.array(Xu[:-1]) - to_unit(c), axis=1))) if len(Xu) > 1 else float("nan")
            _pm = "TRMM" if not (TRMM_PRIOR and PRIOR_SELECT) else ("TRMM:" + "+".join(f"{m}{w:.2f}" for m, w in sorted((_psel_w or {}).items(), key=lambda kv: -kv[1]) if w > 0.05))
            traj.append(dict(it=it, n_truth=n_truth, verified=True, model=_pm, sigma=float(pred),
                             belief=float(m_cand), dist_anchor=_distt,          # belief=surrogate m_k(x); sigma here=PREDICTED improvement (≠σ̂); dist_anchor=OOD proxy
                             ei=float(rho), val=float(v), best=best_true, cst22=np.asarray(c, float),
                             reason="trmm", cost=1, mach=MACH, Re=RE, alpha=ALPHA))
            beliefs.append((float(v), np.asarray(c, float), True, 0.0))
            if it % 10 == 0:
                print(f"[brain] TRMM it{it} cost={n_truth} best={best_true:.2f} | ρ={rho:+.2f} Δ={trmm_delta:.3f} {'ACCEPT' if improved else 'reject'}", flush=True)
            if trmm_delta < TRMM_EPS:                           # step7: convergence (TR collapsed)
                if TRMM_RESTART and n_truth < BUDGET - FINAL_RESERVE - 1:   # v2: RE-EXPAND + restart (multi-start) -> works around the brake, uses the remaining budget, escapes the local optimum
                    trmm_delta = TRMM_DELTA0
                    print(f"[brain] TRMM RESTART (it{it}, cost={n_truth}, best={best_true:.2f}): re-expand radius->{TRMM_DELTA0}", flush=True); it += 1; continue
                print(f"[brain] TRMM CONVERGED: Δ={trmm_delta:.4f} < {TRMM_EPS} (it{it}, cost={n_truth}, best={best_true:.2f})", flush=True); it += 1; break
            it += 1; continue                                  # skips the standard acquisition+gate
        if PRIOR_MEAN and not TRMM:                            # prior at the candidates, same weights as at the anchors, for ALL acquisitions
            if _pm_w is not None:
                from src.opt.prior_select import combine as _cmb
                _rU = [route(to_cst(uu))["permodel"] for uu in U]
                _pm_flowU = _cmb(_pm_w, {m: np.array([r[m][0] for r in _rU]) for m in ZOO})
            else:
                _pm_flowU = np.array([route(to_cst(uu))["belief"] for uu in U])
        if ACQ == "random":                                    # ── BRUTE-FORCE: no surrogate (neither GP nor zoo). Random draw of a feasible candidate, RANS-verified. Hardest baseline. ──
            idxf = np.where(mask)[0]
            k = int(np.random.default_rng(SEED + 7000 + it).choice(idxf)); acq_x = 0.0
        elif ACQ == "kg":                                      # ── Knowledge-Gradient/cost acquisition (VOI, NO τ to tune) ──
            idx = np.where(mask)[0]
            if len(idx) > KG_POOL:
                idx = np.random.default_rng(SEED + it).choice(idx, KG_POOL, replace=False)
            kg, _ = mc_kg(gp, U[idx], NOISE_FLOOR / (ys ** 2), KG_NZ, seed=it,
                          off=(None if _pm_flowU is None else (_pm_flowU[idx] + ym) / ys))
            jj = int(np.argmax(kg)); k = int(idx[jj]); acq_x = float(kg[jj])
        else:                                                  # ── EI or UCB (standardized posterior μ, σ) ──
            with torch.no_grad():
                post = gp.posterior(torch.tensor(U, dtype=torch.double))
                mu = post.mean.squeeze(-1).numpy(); sd = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).numpy()
            if PRIOR_MEAN and not TRMM:                           # prior-mean: μ = f_low + residual-GP, in RAW units (the residual GP is standardized: ×ys + ym)
                mu = _pm_flowU + (mu * ys + ym); sd = sd * ys
            if COMPOSITE and gp_cl is not None and OBJ == "ld":   # composite: μ/σ of L/D=CL/CD via 2 GPs (delta method) instead of the scalar L/D GP
                with torch.no_grad():
                    _pcl = gp_cl.posterior(torch.tensor(U, dtype=torch.double)); _pcd = gp_cd.posterior(torch.tensor(U, dtype=torch.double))
                    _mcl = _pcl.mean.squeeze(-1).numpy(); _scl = _pcl.variance.clamp_min(1e-12).sqrt().squeeze(-1).numpy()
                    _mcd = _pcd.mean.squeeze(-1).numpy(); _scd = _pcd.variance.clamp_min(1e-12).sqrt().squeeze(-1).numpy()
                _cdc = np.clip(_mcd, 1e-4, None)
                mu = _mcl / _cdc                                # mean L/D
                sd = np.sqrt((_scl / _cdc) ** 2 + (_mcl * _scd / _cdc ** 2) ** 2)   # delta propagation CL/CD
            if ACQ == "ucb":                                   # UCB = μ + κ·σ (κ = explore/exploit)
                acq = np.where(mask, mu + UCB_KAPPA * sd, -1e18)
            elif ACQ == "pims":                                # PIMS (Takeno 2023): P(f>m*) where m*=max of ONE GP sample path -> τ-free, no κ
                idxf = np.where(mask)[0]
                sub = idxf if len(idxf) <= 256 else np.random.default_rng(SEED + it).choice(idxf, 256, replace=False)
                with torch.no_grad():                          # 1 JOINT sample path (cholesky) on the subset -> its max = m*
                    torch.manual_seed(SEED + 1000 + it)        # DETERMINISTIC rsample per (SEED,it): otherwise the global torch RNG desyncs the PIMS path between full/router BEFORE any trust -> unpaired 1-seed comparison. With this, at equal it both draw the SAME path -> only the gate (trust) diverges.
                    if COMPOSITE and gp_cl is not None and OBJ == "ld":   # composite: m* = max of a JOINT CL/CD sample -> L/D
                        _sc = gp_cl.posterior(torch.tensor(U[sub], dtype=torch.double)).rsample(torch.Size([1])).squeeze().numpy()
                        _sd2 = gp_cd.posterior(torch.tensor(U[sub], dtype=torch.double)).rsample(torch.Size([1])).squeeze().numpy()
                        gsamp = _sc / np.clip(_sd2, 1e-4, None)
                    else:
                        gsamp = gp.posterior(torch.tensor(U[sub], dtype=torch.double)).rsample(torch.Size([1])).squeeze().numpy()
                        if PRIOR_MEAN and not TRMM and _pm_flowU is not None:   # prior-mean: path = f_low + sampled residual
                            gsamp = _pm_flowU[sub] + (gsamp * ys + ym)   # raw units, like μ
                mstar = float(np.max(gsamp))
                acq = np.where(mask, norm.cdf((mu - mstar) / np.maximum(sd, 1e-9)), -1.0)
            elif ACQ == "es":                                  # ES = MES (Wang&Jegelka 2017, max-value entropy search): info gain on the optimum; m*_s = max of ES_NS paths
                idxf = np.where(mask)[0]
                sub = idxf if len(idxf) <= 256 else np.random.default_rng(SEED + it).choice(idxf, 256, replace=False)
                with torch.no_grad():                          # ES_NS joint paths -> ES_NS plausible max-values m*_s (standardized units = mu/sd)
                    torch.manual_seed(SEED + 1000 + it)        # same as PIMS: deterministic rsample per (SEED,it) for a clean paired comparison
                    paths = gp.posterior(torch.tensor(U[sub], dtype=torch.double)).rsample(torch.Size([ES_NS])).squeeze(-1).numpy()
                if PRIOR_MEAN and not TRMM and _pm_flowU is not None:   # prior-mean: paths = f_low + residual, raw units like μ
                    paths = _pm_flowU[sub][None, :] + (paths * ys + ym)
                mstars = paths.max(1)                           # (ES_NS,) max-values
                g = np.clip((mstars[None, :] - mu[:, None]) / np.maximum(sd[:, None], 1e-9), -8, 8)
                Phi = np.clip(norm.cdf(g), 1e-12, 1.0); phi = norm.pdf(g)
                mes = (g * phi / (2 * Phi) - np.log(Phi)).mean(1)   # analytic info gain averaged over the m*_s
                acq = np.where(mask, mes, -1.0)
            else:                                              # EI (STANDARDIZED units = the GP)
                best_std = best_true if (PRIOR_MEAN and not TRMM) else (best_true - ym) / ys   # prior-mean: μ, σ in raw units
                zc = (mu - best_std) / np.maximum(sd, 1e-9)
                acq = np.where(mask, (mu - best_std) * norm.cdf(zc) + sd * norm.pdf(zc), -1.0)
            k = int(np.argmax(acq)); acq_x = float(acq[k])
        if WB_PROPOSE == "grad":                               # ── WHITE-BOX axis A: candidate = gradient ascent on global (autograd), projected on the manifold via to_unit ──
            _starts = np.argsort(-acq)[:WB_GSTARTS] if 'acq' in dir() else np.where(mask)[0][:WB_GSTARTS]
            _bu, _bv = None, -1e18
            for _si in _starts:
                _u = U[_si].copy()
                for _ in range(WB_GSTEPS):
                    _cc = to_cst(_u)
                    _val, _g = _WB.ld_grad(_cc, alpha_for("global", _cc), OBJ)
                    _step = WB_GLR * (UB - LB) * _g / (np.linalg.norm(_g) + 1e-9)   # step relative to the box, gradient direction
                    _u = to_unit(np.clip(_cc + _step, LB, UB))                       # projects onto the manifold (POD or box)
                _cc = to_cst(_u)
                if not feasible(_cc):
                    continue
                _val, _ = _WB.ld_grad(_cc, alpha_for("global", _cc), OBJ)
                if _val > _bv:
                    _bv, _bu = _val, _u
            if _bu is not None:
                U[k] = _bu; acq_x = float(TAU_EI + 1e-6)        # forces the 'promising' gate (the gradient already selected the optimistic one)
        c = to_cst(U[k])
        with torch.no_grad():                                  # GP posterior at the chosen point (AGP + log), de-standardized
            _pc = gp.posterior(torch.tensor(U[k:k + 1], dtype=torch.double))
            gp_mu_c = float(_pc.mean.squeeze().item()) * ys + ym
            gp_sd_c = float(_pc.variance.clamp_min(1e-12).sqrt().squeeze().item()) * ys

        r = route(c)                                           # ── THE BRAIN ──
        if GATE in ("adapt", "hybrid"):                        # C: τ shrinks with the budget (frac^POW: POW=1 linear, POW>1 convex = stays high, drops late)
            _frac = (n_truth / max(BUDGET, 1)) ** ADAPT_POW
            tau_now = max(tau_eff * (1.0 - ADAPT_SLOPE * _frac), tau_eff * ADAPT_FLOOR)
        else:
            tau_now = tau_eff
        _vfid = FID                                            # verification fidelity (default=FID; escalate picks medium/hf per point)
        _pre = None; _shadow = bool(r["sigma"] < tau_now)      # _pre = hidden truth (oracle/audit); _shadow = what the standard σ̂ gate would decide
        if GATE == "oracle":
            _pre = truth_val(c, f"or{it}", fid=_vfid)          # hidden RANS, NOT counted: the oracle decides on the real error
        if LAZY:                                               # ① LAZY: never verify inline -> all trusted, RANS is spent only at final-verify
            do_verify = False; reason = "lazy-trust"
        elif GATE == "escalate":                               # 3-TIER: trust (surrogate) if confident; else verify in MEDIUM (far, cheap) or FINE (near the optimum, tight decision)
            if r["sigma"] < tau_eff:
                do_verify = False; reason = "trust"            # cheap #1: surrogate (0 cost)
            else:
                _gap = abs(r["belief"] - best_true) if np.isfinite(best_true) else abs(r["belief"])
                _vfid = "hf" if r["sigma"] >= _gap else ESC_MED   # σ̂ dwarfs the margin (near optimum) → fine; else cheap MEDIUM #2
                do_verify = True; reason = f"verify-{_vfid}"
        elif GATE == "margin":                                 # MARGIN: trust IFF σ̂ < K·LOCAL-margin (|belief−best|) instead of fixed τ%·|base|
            _gap = abs(r["belief"] - best_true) if np.isfinite(best_true) else abs(r["belief"])  # what THIS decision puts at stake
            _floor = MARGIN_FLOOR * abs(best_true) if best_true not in (0, -np.inf) else 0.0
            tau_mrg = MARGIN_K * max(_gap, _floor)             # ADAPTIVE tolerance: large far from the optimum, collapses near it (gap→0)
            do_verify = r["sigma"] >= tau_mrg; reason = "verify" if do_verify else "trust"
        elif GATE == "hybrid":                                 # HYBRID: adapt (clock=economy far away) + MARGIN override (tight local decision=verify near the optimum). do_verify = σ̂ ≥ MIN(τ_adapt, τ_margin) = the stricter gate wins.
            _gap = abs(r["belief"] - best_true) if np.isfinite(best_true) else abs(r["belief"])
            _floor = MARGIN_FLOOR * abs(best_true) if best_true not in (0, -np.inf) else 0.0
            tau_mrg = MARGIN_K * max(_gap, _floor)
            do_verify = (r["sigma"] >= tau_now) or (r["sigma"] >= tau_mrg)  # verifies if either one triggers (economy far via adapt, safe near via margin)
            reason = "verify" if do_verify else "trust"
        elif GATE == "ctd":                                    # CTD (Calibrate-Then-Delegate): τ PACED to spend the budget verifying the most UNCERTAIN; risk bounded by conformal σ̂
            _sig_hist.append(r["sigma"])
            _hor = CTD_HORIZON if CTD_HORIZON > 0 else 2.0 * BUDGET
            _pace = float(np.clip(BUDGET / max(_hor, 1.0), 0.05, 0.95))     # target fraction to verify (budget guarantee)
            _tau_ctd = float(np.quantile(_sig_hist, 1.0 - _pace)) if len(_sig_hist) >= 8 else tau_now
            do_verify = r["sigma"] >= _tau_ctd; reason = "verify" if do_verify else "ctd-trust"
        elif ACQ == "kg":                                      # KG: verifies IFF the RANS is worth its cost (KG ≥ ε); else believes
            do_verify = acq_x >= KG_EPS; reason = "verify" if do_verify else "trust"
        elif GATE == "regret":                                 # B: VOI regret band (NO τ) — verifies IFF the OPTIMISTIC belief (belief+σ̂) beats the best by AT LEAST a margin (=RANS cost; EIC Hu 2022)
            _mrg = REGRET_MARGIN * abs(best_true) if best_true not in (0, -np.inf) else 0.0
            do_verify = (r["belief"] + r["sigma"]) > best_true + _mrg; reason = "verify" if do_verify else "trust"
        elif GATE == "random":                                 # UNINFORMED CONTROL: random trust, at the SAME rate as the σ̂ gate (tracked in shadow)
            _SHADOW[0] += 1; _SHADOW[1] += int(_shadow)
            _p = (_SHADOW[1] + 1.0) / (_SHADOW[0] + 2.0)
            do_verify = bool(np.random.default_rng(SEED + 9000 + it).random() >= _p); reason = "verify" if do_verify else "trust"
        elif GATE == "oracle":                                 # IDEAL CONTROL: trust iff |belief − truth| < τ (upper bound of what a gate can save)
            _vt = _pre[0]
            do_verify = (_vt is None) or (abs(float(r["belief"]) - float(_vt)) >= tau_now); reason = "verify" if do_verify else "trust"
        elif ACQ == "ucb":                                     # UCB: τ gate (the acq only changes the point SELECTION)
            do_verify = r["sigma"] >= tau_now; reason = "verify" if do_verify else "trust"
        elif r["sigma"] < tau_now:                             # TRUST: uncertainty < tolerance
            do_verify = False; reason = "trust"
        elif acq_x >= TAU_EI:                                  # VERIFY: promising ∧ uncertain
            do_verify = True; reason = "verify"
        else:                                                  # SKIP: uncertain but low EI -> believe
            do_verify = False; reason = "skip"
        if do_verify:
            v, cl, cd = (_pre if _pre is not None else truth_val(c, f"bo{it}", fid=_vfid)); n_truth += 1   # escalate: _vfid = medium OR hf per point (else FID); oracle: reuses the hidden truth
            verified = v is not None
            if verified:
                _skerr = {}
                for m in ZOO:                                  # enriches IO-CUE (2 primitives, all models)
                    cle, cde, a = r["evals"][m]
                    IO.record_residual(m, feats27(c, a), cle, cde, cl, cd)
                    _skerr[m] = abs(float(cde) - float(cd))    # skill: REAL CD error of this model at this RANS anchor
                _SKILL_HIST.append((to_unit(c), _skerr))       # history for FUSION=skill (model-selection by measured truth)
                if WB:                                         # ── WHITE-BOX axis C: the model learns the RANS truth (repairs, does not patch) ──
                    _WB_ANCH.append((feats27(c, r["evals"]["global"][2]), [float(cl), float(cd)]))
                    if WB_FT > 0 and n_truth % WB_FT == 0:
                        _fl = _WB.finetune([a_[0] for a_ in _WB_ANCH], [a_[1] for a_ in _WB_ANCH])
                        print(f"[white-box] global fine-tuned on {len(_WB_ANCH)} RANS anchors (loss={_fl}) @ cost={n_truth}", flush=True)
        else:
            v, verified = r["belief"], False
            if AUDIT and _pre is None: _pre = truth_val(c, f"au{it}", fid=_vfid)   # AUDIT: hidden RANS NOT counted on the believed point
        if v is None: it += 1; continue
        Xu.append(to_unit(c)); Y.append(v)
        _twf = TRUST_WEIGHT * (1.0 + TW_ADAPT * (n_truth / max(BUDGET, 1)))   # adaptive anchoring: more down-weight late (high frac)
        yv = NOISE_FLOOR if (verified or TRUST_EQUAL) else _twf * max(r["sigma"], 1e-3) ** 2   # trusted = w·σ̂² -> down-weighted; TRUST_EQUAL -> floor (RANS weight)
        if AGP_HARD and not verified:                                       # HARD AGP (Candelieri): belief ↔ GP consistency
            if abs(v - gp_mu_c) > AGP_BETA * np.sqrt(gp_sd_c ** 2 + yv):    # belief CONTRADICTS the accumulated GP
                yv = 1e6; reason = "trust-drop"                            # -> huge noise = quasi out of the GP (anti-phantom-poison)
        Yv.append(yv)
        improved = verified and v > best_true
        if improved: best_true, best_cst = v, c
        if verified: no_improve = 0 if improved else no_improve + 1   # patience = verifications WITHOUT gain
        if ENRICH_EVERY > 0 and verified and n_truth % ENRICH_EVERY == 0:   # #4 ONLINE enrichment (live): integrates the buffered RANS anchors -> refit zoo δ̂/σ̂ (local self-correction near the optimum)
            IO.online_update_all(ZOO)
        if USE_TR and verified:                                       # TR adapts on VERIFICATIONS (not beliefs)
            tr = update_state(tr, torch.tensor([float(v)], dtype=torch.double))
            if tr.restart_triggered:                                  # TR collapsed -> re-expand around the best
                tr = TurboState(dim=DIM, batch_size=1, best_value=best_true); tr.failure_tolerance = TR_FAILTOL
        _dist = float(np.min(np.linalg.norm(np.array(Xu[:-1]) - to_unit(c), axis=1))) if len(Xu) > 1 else float("nan")  # geometric OOD PROXY: distance of the candidate to the anchors (in-dist if small; "never-OOD" evidence)
        _row = dict(it=it, n_truth=n_truth, verified=verified, model=r["model"], sigma=r["sigma"],
                    belief=float(r["belief"]), dist_anchor=_dist,                # belief=surrogate belief (→ realized error |belief−val| at verifications = σ̂ coverage); dist_anchor=OOD proxy
                    ei=float(acq_x), val=float(v), best=best_true, cst22=np.asarray(c, float),
                    reason=reason, cost=(1 if verified else 0), mach=MACH, Re=RE, alpha=ALPHA,
                    val_true=(float(_pre[0]) if (_pre is not None and _pre[0] is not None) else float("nan")),   # hidden truth (oracle/audit), nan otherwise
                    shadow_trust=_shadow)                                                                            # decision the σ̂ gate would have taken
        for _m in ZOO:                                          # pred + σ̂ of EACH model (interactive dashboard)
            _row[f"pred_{_m}"] = r["permodel"][_m][0]; _row[f"sig_{_m}"] = r["permodel"][_m][1]
        traj.append(_row)
        beliefs.append((float(v), np.asarray(c, float), verified, float(r["sigma"])))   # (belief, cst, verified, σ̂) for the FINAL-VERIFY / racing
        if it % 10 == 0:
            print(f"[brain] it{it} cost={n_truth} best={best_true:.2f} | "
                  f"{'VERIF' if verified else 'trust'} {r['model']} σ*={r['sigma']:.2f} noimp={no_improve}"
                  f"{f' TRlen={tr.length:.2f}' if USE_TR else ''}", flush=True)
        quiet = 0 if verified else quiet + 1                        # iterations WITHOUT verification (quiet gate)
        if (STOP_EI > 0 and acq_x < STOP_EI) or (STOP_QUIET > 0 and quiet >= STOP_QUIET):
            print(f"[brain] EARLY STOP: {'EI<floor' if (STOP_EI>0 and acq_x<STOP_EI) else 'quiet-gate'} (it{it}, cost={n_truth})", flush=True)
            it += 1; break
        it += 1
    stop = "budget" if n_truth >= BUDGET else ("plateau" if no_improve >= PATIENCE else "itmax")
    print(f"[brain] STOP={stop} | cost={n_truth} best={best_true:.2f}", flush=True)

    # FINAL-VERIFY ("trust to explore, VERIFY to deliver"): verifies the top-K most
    # PROMISING designs (belief) not yet verified -> banks what the router found by trusting + catches
    # a possible phantom. Without it, best=verified-only misses the good design found via trust.
    K = int(os.environ.get("BRAIN_FINALK", 5))
    _ffid = FINAL_FID or None                                  # fine-final: coarse-KOH loop -> DELIVERY in real fine
    _fine_final = _ffid is not None and _ffid != FID
    best_fine = -np.inf; best_fine_cst = None                  # best in FINE units (≠ best_true=coarse-KOH loop)
    unv = [(bv, bc, sg) for bv, bc, ver, sg in beliefs if not ver]
    ub = lambda b, s: b + 1.645 * CONFORMAL_LAMBDA * s         # ③ CALIBRATED upper bound (90% CI recalibrated by λ)
    cands = sorted(unv, key=lambda x: -(ub(x[0], x[2]) if RACE else x[0]))
    if _fine_final and best_cst is not None:
        cands = [(best_true, np.asarray(best_cst, float), 0.0)] + cands   # fine-final: ALSO verifies the loop best (coarse) in fine
    seen = []; nfin = 0
    for j, (bv, bc, sg) in enumerate(cands):
        if nfin >= K or n_truth >= BUDGET: break                # HARD CAP: total RANS ≤ BUDGET (final-verify included)
        if any(np.linalg.norm(bc - s) < 1e-3 for s in seen):
            continue
        if RACE and not _fine_final:                           # RACE disabled in fine-final (coarse loop units ≠ fine, invalid comparison)
            rest_ub = max((ub(c[0], c[2]) for c in cands[j:]), default=-1e18)
            if best_true >= rest_ub:
                print(f"[brain] calibrated RACE-STOP (λ={CONFORMAL_LAMBDA}): best {best_true:.2f} >= upper-bound90 of the rest {rest_ub:.2f} (verifs={nfin}, cost={n_truth})", flush=True)
                break
        seen.append(bc); vv, _, _ = truth_val(bc, f"final{nfin}", fid=_ffid); n_truth += 1; nfin += 1
        if _fine_final:                                        # DELIVERED best = max of FINE verifications (honest)
            if vv is not None and vv > best_fine:
                best_fine, best_fine_cst = vv, bc
            _bshow = best_fine
        else:
            if vv is not None and vv > best_true:
                best_true, best_cst = vv, bc
            _bshow = best_true
        traj.append(dict(it=900 + nfin, n_truth=n_truth, verified=True, model="FINAL",
                         sigma=0.0, val=float(vv) if vv is not None else -1e9, best=_bshow, cst22=bc))
    if _fine_final and best_fine_cst is not None:              # delivery = real fine
        print(f"[brain] FINE-FINAL {nfin} hf verifs -> DELIVERED best_fine = {best_fine:.2f} (coarse-KOH loop claimed {best_true:.2f}) (total cost={n_truth})", flush=True)
        best_true, best_cst = best_fine, best_fine_cst
    elif nfin:
        print(f"[brain] FINAL-VERIFY{'(race)' if RACE else ''} {nfin} -> best={best_true:.2f} (total cost={n_truth})", flush=True)

    if FLUSH:
        for m in ZOO: IO.flush_enriched(m)                     # persists the extended IO-CUE (off for the sweep)
    if best_cst is not None:
        np.save(WD / f"best_cst22_{OBJ}_tau{TAU}.npy", np.asarray(best_cst, float))   # for the airfoil PNG
    import pandas as pd
    out = WD / f"traj_brain_{OBJ}_tau{TAU}.parquet"; pd.DataFrame(traj).to_parquet(out)
    from collections import Counter
    prefer = Counter(t["model"] for t in traj if t["it"] >= 0)                       # argmin σ̂ (the switch)
    trusted = Counter(t["model"] for t in traj if t["it"] >= 0 and not t["verified"])
    print(f"[brain] DONE cost={n_truth} best={best_true:.2f} | preferred(argmin σ)={dict(prefer)} "
          f"| trusted={dict(trusted)} -> {out}", flush=True)
    gain = (best_true - base_val) if base_val is not None else float("nan")
    n_tr = sum(1 for t in traj if t["it"] >= 0 and not t["verified"])
    n_ve = sum(1 for t in traj if t["it"] >= 0 and t["verified"]); tot = n_tr + n_ve
    opt_nt = next((t["n_truth"] for t in traj if t["best"] >= best_true - 1e-9), n_truth)
    unit = "CD counts saved" if OBJ == "cd" else "L/D gained"
    print(f"[brain] GAIN vs base = {gain:+.1f} {unit} | RANS→optimum={opt_nt} | total RANS={n_truth} "
          f"| trusted designs (RANS avoided) = {n_tr}/{tot} = {100*n_tr/max(1,tot):.0f}%", flush=True)


if __name__ == "__main__":
    main()
