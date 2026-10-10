"""IO-CUE — Input-Output-Conditioned Uncertainty Estimation (arxiv 2506.00918), PRIMITIVES version.

Models the surrogate error on the PHYSICAL QUANTITIES CL and CD (not on an objective) -> REUSABLE
for any optimization (min CD, max L/D, CD@fixed CL...). The router then PROPAGATES to the objective.

Per surrogate `name` and per primitive `output ∈ {cl, cd}`:
  δ̂(x)  = SIGNED conditional bias (median GBR, robust to off-regime noise).  correction: out_c = out - δ̂
  σ̂(x)  = uncertainty, IOCUE_SIGMA method:
             gauss    : mean GBR on |residual|            (Gaussian baseline — under-covers the tail)
             quantile : quantile-Q GBR on |residual|      (conservative)
             cqr      : quantile-Q + SPLIT-CONFORMAL offset (probe=fit, test=calib) -> guaranteed coverage
  I/O-conditioned: features = [27D inputs, CL_hat, CD_hat]. At fixed alpha, CL_hat encodes the regime
  -> regime-adaptive δ̂/σ̂ WITHOUT a regime classifier.

RCP-PORTABLE: residuals are stored as .npz (pure numpy, stable 1.x<->2.x); GBR re-fit at load time
(never a sklearn pickle -> immune to the cross-version numpy._core crash). Online enrichment:
record_residual() buffers at each RANS check, flush_enriched() -> <name>_resid_v_online.npz.

Residual training: training/train_iocue.py and training/fit_iocue_su2probe.py (python main.py train-iocue).
"""
import os, sys, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IOCUE_DIR = ROOT / os.environ.get("IOCUE_DIR", "models/iocue_v4"); IOCUE_DIR.mkdir(parents=True, exist_ok=True)
FEATS = ["mach", "logRe", "alpha", "t_c", "f_c"] + [f"cst_{i}" for i in range(22)]   # 27D = MLP contract
OUT_FEATS = ["cl_hat", "cd_hat"]                        # +2: surrogate output (I/O conditioning)
SUFFIX = os.environ.get("IOCUE_SUFFIX", "")             # "" base | "_v_online" enriched (previous optimization)
Q = float(os.environ.get("IOCUE_Q", 0.9))              # target coverage level of σ̂
SIGMA_METHOD = os.environ.get("IOCUE_SIGMA", "cqr")    # gauss | quantile | cqr
DEBIAS = os.environ.get("IOCUE_SIGMA_DEBIAS", "0") == "1"  # σ̂ on |r−δ̂| (POST-correction residual) instead of raw |r|
NODEBIAS = os.environ.get("IOCUE_NODEBIAS", "0") == "1"    # 1 = DISABLES δ̂ (belief = RAW pred). δ̂ calibrated on dataset_v3 (mixed solvers) HURTS vs OUR SU2 (measured: 4.2→4.4)
_BUFFER = {}                                            # {name: [feats_list, rcl_list, rcd_list]} online


def _fit_gbr(X, y, q=None):
    """GBR features->target. q=None: L2 mean. q=0.5: median (robust δ̂). q=0.9: quantile (σ̂)."""
    from sklearn.ensemble import GradientBoostingRegressor
    kw = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
    if q is not None: kw.update(loss="quantile", alpha=float(q))
    m = GradientBoostingRegressor(**kw); m.fit(X, y); return m


# ── loaded/cached models (re-fit at load = portable) ─────────────────────────
_LOADED = {}
def _load(name):
    if name not in _LOADED:
        p = IOCUE_DIR / f"{name}_resid{SUFFIX}.npz"
        _LOADED[name] = dict(np.load(p)) if p.exists() else None
    return _LOADED[name]


_DELTA, _SIGMA = {}, {}
def _delta_get(name, output):                                          # median δ̂ per primitive
    key = (name, output, SUFFIX)
    if key not in _DELTA:
        d = _load(name)
        _DELTA[key] = None if d is None else _fit_gbr(d["X"][~d["cal"]], d[f"r_{output}"][~d["cal"]], q=0.5)
    return _DELTA[key]


def _sigma_get(name, output):                                          # σ̂ (gauss|quantile|cqr) per primitive
    key = (name, output, SUFFIX, SIGMA_METHOD, Q, DEBIAS)
    if key not in _SIGMA:
        d = _load(name)
        if d is None:
            _SIGMA[key] = None
        else:
            pr, cal = ~d["cal"], d["cal"]
            if DEBIAS:                                              # post-correction residual: |r − δ̂(x)|
                dm = _delta_get(name, output); ar = np.abs(d[f"r_{output}"] - dm.predict(d["X"]))
            else:
                ar = np.abs(d[f"r_{output}"])
            if SIGMA_METHOD == "gauss":
                m = _fit_gbr(d["X"][pr], ar[pr], q=None); off = 0.0
            else:
                m = _fit_gbr(d["X"][pr], ar[pr], q=Q)
                off = 0.0 if SIGMA_METHOD == "quantile" else \
                    float(np.quantile(ar[cal] - m.predict(d["X"][cal]),
                                      min(math.ceil((cal.sum() + 1) * Q) / cal.sum(), 1.0)))
            _SIGMA[key] = (m, off)
    return _SIGMA[key]


# ── batch (routeur) : feats27 (N,27) + out2 (N,2)=[CL_hat,CD_hat] -> primitive ─────────────────
def _full(feats27, out2):
    return np.hstack([np.asarray(feats27, float), np.asarray(out2, float)])


SGATE_CD = float(os.environ.get("IOCUE_SGATE_CD", "0"))   # >0: δ̂_cd applied ONLY if σ̂_cd < threshold (counts). Otherwise δ̂=0 (erratic/shock residual = unreliable median). 0=off
SGATE_CL = float(os.environ.get("IOCUE_SGATE_CL", "0"))   # same for cl (raw CL units)


def delta_hat_out(feats27, out2, name, output):
    if NODEBIAS:                                          # δ̂=0: belief = raw pred (δ̂ miscalibrated vs our SU2)
        return np.zeros(len(feats27))
    m = _delta_get(name, output)
    if m is None:
        return np.zeros(len(feats27))
    dh = m.predict(_full(feats27, out2))
    gate = SGATE_CD if output == "cd" else SGATE_CL
    if gate > 0:                                          # σ̂-GATE: applies δ̂ only where σ̂ is small (smooth, reliable bias); erratic (shock) -> raw pred
        sh = sigma_hat_out(feats27, out2, name, output)  # σ̂ (raw primitive units)
        thr = gate / 1e4 if output == "cd" else gate     # cd: threshold in counts -> raw
        dh = np.where(sh < thr, dh, 0.0)
    return dh


def sigma_hat_out(feats27, out2, name, output):
    mo = _sigma_get(name, output)
    if mo is None: return np.zeros(len(feats27))
    m, off = mo
    return np.maximum(0.0, m.predict(_full(feats27, out2)) + off)


def propagate(feats27, out2, name, objective="cd"):
    """Builds the objective + its σ from the primitives (bias-corrected + propagation). REUSABLE.
    out2=[CL_hat,CD_hat]. Returns (corrected_objective_value, σ_objective).  objective ∈ {cd, ld}."""
    cl, cd = np.asarray(out2, float)[:, 0], np.asarray(out2, float)[:, 1]
    cl_c = cl - delta_hat_out(feats27, out2, name, "cl")
    cd_c = np.maximum(cd - delta_hat_out(feats27, out2, name, "cd"), 1e-4)
    s_cl = sigma_hat_out(feats27, out2, name, "cl")
    s_cd = sigma_hat_out(feats27, out2, name, "cd")
    if objective == "ld":                                              # L/D = CL/CD, delta-method
        ld = cl_c / cd_c
        s = np.abs(ld) * np.sqrt((s_cl / np.abs(cl_c).clip(1e-3)) ** 2 + (s_cd / cd_c) ** 2)
        return ld, s
    return cd_c, s_cd                                                  # min CD : direct


# ── single-point (gpbo L1 compat + convenience) ──────────────────────────────────
def _out_for(name, cst22, mach, alpha, Re):
    from src import zoo
    return zoo.predict(name, cst22, mach, alpha, Re)


def _feats1(cst22, mach, alpha, Re):
    from src.opt.evaluators import tc_fc
    t_c, f_c = tc_fc(cst22)
    return np.concatenate([[mach, np.log10(Re), alpha, t_c, f_c], np.asarray(cst22, float)])[None]


def sigma_hat(cst22, mach, alpha, Re, name, output="cd", q=None):
    """σ̂ (in primitive units) at the point. output ∈ {cl, cd}. gpbo L1 compat (default CD)."""
    cl_hat, cd_hat = _out_for(name, cst22, mach, alpha, Re)
    return float(sigma_hat_out(_feats1(cst22, mach, alpha, Re), [[cl_hat, cd_hat]], name, output)[0])


def delta_hat(cst22, mach, alpha, Re, name, output="cd"):
    cl_hat, cd_hat = _out_for(name, cst22, mach, alpha, Re)
    return float(delta_hat_out(_feats1(cst22, mach, alpha, Re), [[cl_hat, cd_hat]], name, output)[0])


# ── ONLINE enrichment (both primitives) ───────────────────────────────────
def record_residual(name, feats27, cl_hat, cd_hat, cl_rans, cd_rans):
    """At each RANS check: append (I/O features, SIGNED CL & CD residuals) to the buffer of `name`."""
    x = np.concatenate([np.asarray(feats27, float).ravel(), [float(cl_hat), float(cd_hat)]])
    b = _BUFFER.setdefault(name, [[], [], []])
    b[0].append(x); b[1].append(float(cl_hat) - float(cl_rans)); b[2].append(float(cd_hat) - float(cd_rans))


def flush_enriched(name, version="v_online"):
    """offline + online -> <name>_resid_<version>.npz (versioned + provenance). New points
    are marked cal=False (used for the fit, not for conformal calibration)."""
    p0 = IOCUE_DIR / f"{name}_resid{SUFFIX}.npz"; Xs, rc, rd, cl, prov = [], [], [], [], []  # SUFFIX-aware: accumulates ACROSS runs (run k reads _v_online of run k-1) instead of restarting from the base. SUFFIX="" = base (unchanged).
    if p0.exists():
        z = np.load(p0); Xs.append(z["X"]); rc.append(z["r_cl"]); rd.append(z["r_cd"])
        cl.append(z["cal"]); prov.append(np.zeros(len(z["r_cd"]), np.int16))
    b = _BUFFER.get(name)
    if b and b[0]:
        Xs.append(np.array(b[0], float)); rc.append(np.array(b[1], float)); rd.append(np.array(b[2], float))
        cl.append(np.zeros(len(b[1]), bool)); prov.append(np.ones(len(b[1]), np.int16))
    if not Xs: return None
    out = IOCUE_DIR / f"{name}_resid_{version}.npz"
    np.savez(out, X=np.vstack(Xs).astype(np.float32), r_cl=np.concatenate(rc).astype(np.float32),
             r_cd=np.concatenate(rd).astype(np.float32), cal=np.concatenate(cl),
             provenance=np.concatenate(prov))
    n_on = len(b[1]) if (b and b[0]) else 0
    print(f"[iocue] flush {name}: +{n_on} online -> {out}")
    return out


# ── ONLINE RE-TRAINING (in memory, no restart) ──────────────────────────
def online_update(name):
    """Merges the RANS-check buffer (record_residual) INTO the loaded residual set of `name`
    + invalidates the δ̂/σ̂ caches -> refit on offline+online at the NEXT sigma/delta call.
    Online points enter the FIT (cal=False): the σ̂ GBR sees fresh OOD residuals
    -> larger σ̂ at the boundary (addresses tail under-coverage). The conformal offset stays
    anchored on the held-out calib (conservative bound). Cost ~ms-s (GBR, small n) << 1 RANS.
    Returns the number of merged points (0 if buffer empty)."""
    b = _BUFFER.get(name)
    if not b or not b[0]:
        return 0
    d = _load(name)
    if d is None:
        return 0
    Xon = np.asarray(b[0], np.float32)
    rcl = np.asarray(b[1], np.float32)
    rcd = np.asarray(b[2], np.float32)
    _LOADED[name] = dict(                                       # incremental accumulation (merge into the _load cache)
        X=np.vstack([d["X"], Xon]),
        r_cl=np.concatenate([d["r_cl"], rcl]),
        r_cd=np.concatenate([d["r_cd"], rcd]),
        cal=np.concatenate([d["cal"], np.zeros(len(rcl), bool)]),
    )
    for cache in (_DELTA, _SIGMA):                              # invalidates this model's fitted GBRs -> refit at next access
        for k in [k for k in cache if k[0] == name]:
            del cache[k]
    _BUFFER[name] = [[], [], []]                                # buffer consumed (points now in _LOADED)
    return len(rcl)


def online_update_all(names=None):
    """Re-trains the WHOLE zoo: merges each model's buffered residuals. The GBR refit is
    LAZY (at the next sigma/delta) -> here we only merge + invalidate (almost free);
    the real GBR cost is paid at the 1st subsequent call. Returns {name: n_merged_pts}."""
    ns = names if names is not None else list(_BUFFER.keys())
    out = {}
    for nm in ns:
        n = online_update(nm)
        if n:
            out[nm] = n
    return out
