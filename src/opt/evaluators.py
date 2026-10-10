"""Geometry helpers and the external oracles of the zoo, on the shared design vector cst22.

  - tc_fc            : thickness and camber of a cst22 shape (same Bernstein basis as decode)
  - eval_nf          : NeuralFoil on the decoded geometry
  - eval_xfoil       : XFoil through a subprocess, NaN when absent or not converged
  - eval_airfoilann  : AirfoilANN (Khan et al.), pure numpy forward pass
  - eval_mlp         : alias of src.zoo.predict for the MLP members
"""
import numpy as np
from pathlib import Path
from functools import lru_cache

from src.opt.cst_geom import decode, N_CST
from src.utils.cst import class_function, bernstein_basis


def tc_fc(cst22: np.ndarray) -> tuple[float, float]:
    """t_c, f_c from cst22 (reconstruction WITHOUT TE, same grid as fit_cst)."""
    c = np.asarray(cst22, float)
    A_u, A_l = c[:N_CST], c[N_CST:2 * N_CST]
    xq = np.linspace(0.01, 0.99, 200)
    C = class_function(xq); B = bernstein_basis(xq, N_CST - 1)
    yu = C * (B @ A_u); yl = C * (B @ A_l)
    th = yu - yl; cm = 0.5 * (yu + yl)
    return float(th.max()), float(cm[np.argmax(np.abs(cm))])


def eval_mlp(cst22: np.ndarray, mach: float, alpha: float, Re: float,
             name: str = "global") -> tuple[float, float]:
    """cst22 + conditions -> (CL, CD) via the MLP member <name> of the zoo (src/zoo.py)."""
    from src import zoo
    return zoo.predict(name, cst22, mach, alpha, Re)


def eval_nf(cst22: np.ndarray, alpha: float, Re: float,
            model_size: str = "large") -> tuple[float, float]:
    """cst22 -> (CL, CD) via NeuralFoil on the decoded geometry (same shape as SU2)."""
    import neuralfoil as nf
    coords = decode(cst22, n=160)
    r = nf.get_aero_from_coordinates(coords, alpha, Re, model_size=model_size)
    return float(np.asarray(r["CL"]).ravel()[0]), float(np.asarray(r["CD"]).ravel()[0])


def eval_xfoil(cst22: np.ndarray, alpha: float, Re: float, mach: float = 0.0) -> tuple[float, float]:
    """cst22 -> (CL, CD) via XFoil (REAL physics: viscous-inviscid panel + convergence flag).
    DIRECT subprocess (aerosandbox crashes); RELATIVE names (Fortran XFoil truncates long paths).
    Diverges -> (nan, nan) = honest flag (the IO-CUE xfoil_resid has a huge σ̂ there). Binary = env XFOIL_BIN.

    Two pitfalls measured on the RCP container, both silent:
      * without PLOP/G F, XFoil computes then dies on "Cannot open display...aborting";
      * even with graphics off, PACC never writes the data line to the polar file
        (only the header), while CL and CD are correct on standard output.
        So we read standard output and no longer touch the file.
    """
    import os, re, subprocess, tempfile
    import shutil
    XBIN = os.environ.get("XFOIL_BIN") or shutil.which("xfoil") or ""
    if not XBIN or not Path(XBIN).exists():
        return float("nan"), float("nan")                                  # no XFoil (e.g. RCP container without XFoil)
    try:
        coords = decode(cst22, n=120)
        with tempfile.TemporaryDirectory() as d:
            with open(Path(d) / "af.dat", "w") as f:
                f.write("af\n")
                for x, y in coords:
                    f.write(f"{x:.6f} {y:.6f}\n")
            cmd = (f"PLOP\nG F\n\nLOAD af.dat\nPANE\nOPER\nVISC {float(Re):.0f}\n"
                   f"MACH {float(min(mach, 0.6)):.3f}\nITER 200\nALFA {float(alpha):.3f}\n\nQUIT\n")
            out = subprocess.run([XBIN], input=cmd, text=True, capture_output=True,
                                 timeout=25, cwd=d).stdout
        # XFoil prints "Convergence failed" on intermediate passes even when the
        # final point converges: only the LAST state counts, not the presence of the word.
        m_cl = [m for m in re.finditer(r"CL\s*=\s*(-?\d+\.\d+)", out)]
        m_cd = [m for m in re.finditer(r"CD\s*=\s*(-?\d+\.\d+)", out)]
        if not m_cl or not m_cd:
            return float("nan"), float("nan")
        last_fail = out.rfind("Convergence failed")
        if last_fail > m_cl[-1].start():                                   # the failure comes after the last solution
            return float("nan"), float("nan")                              # not converged = honest flag
        return float(m_cl[-1].group(1)), float(m_cd[-1].group(1))
    except Exception:
        return float("nan"), float("nan")


# ── AirfoilANN (Khan et al., JavaFoil-trained Keras MLP): pure NUMPY forward ───
# Weights dumped from MODEL_512.256.128.3_D_all_10.h5 (23->512->256->128->3, relu x3+linear)
# + StandardScaler(23). Runs natively in base (no TF/keras). Features = 10 cosine
# x-stations (yU,yL) from cst22 + [Re, mach, alpha]. DOMAIN: NACA 4/5-digit, Re<=5e5,
# M<=0.3, |a|<=10 -> OUT-OF-DOMAIN on our cases (strong extrapolation = expected curse).
_AANN_NPZ = Path(__file__).resolve().parents[2] / "models/airfoilann/aann_weights.npz"   # + MIT LICENSE
_th_aann = np.linspace(-np.pi / 2, np.pi / 2, 12)
_XST_AANN = ((np.sin(_th_aann) + 1) / 2)[1:-1]                                   # (10,)
from scipy.special import comb as _comb
_B10_AANN = np.stack([_comb(10, i) * _XST_AANN ** i * (1 - _XST_AANN) ** (10 - i) for i in range(11)], 1)  # (10,11)
_C_AANN = _XST_AANN ** 0.5 * (1 - _XST_AANN) ** 1.0                              # class function


@lru_cache(maxsize=1)
def _aann():
    z = np.load(_AANN_NPZ, allow_pickle=True)
    n = int(z["n_layers"][0])
    Ws = [z[f"W{i}"].astype(np.float64) for i in range(n)]
    bs = [z[f"b{i}"].astype(np.float64) for i in range(n)]
    acts = [str(a) for a in z["acts"]]
    return dict(Ws=Ws, bs=bs, acts=acts,
                mean=z["scaler_mean"].astype(np.float64), scale=z["scaler_scale"].astype(np.float64))


def eval_airfoilann(cst22: np.ndarray, mach: float, alpha: float, Re: float) -> tuple[float, float]:
    """cst22 + conditions -> (CL, CD) via AirfoilANN (numpy forward). Out-of-domain on our cases."""
    c = np.asarray(cst22, float)
    Au, Al = c[:11], c[11:22]
    yU = _C_AANN * (_B10_AANN @ Au); yL = _C_AANN * (_B10_AANN @ Al)             # (10,),(10,)
    x = np.concatenate([yU, yL, [float(Re), float(mach), float(alpha)]])          # (23,)
    net = _aann()
    x = (x - net["mean"]) / net["scale"]
    for W, b, a in zip(net["Ws"], net["bs"], net["acts"]):
        x = x @ W + b
        if a == "relu": x = np.maximum(x, 0.0)
    return float(x[0]), float(x[1])   # CL, CD (ignore CM)
