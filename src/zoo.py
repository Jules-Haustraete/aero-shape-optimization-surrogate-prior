"""The zoo: every cheap aerodynamic estimator of the thesis behind one call (Chapter 2).

    from src import zoo
    cl, cd = zoo.predict("global", cst22, mach=0.3, alpha=4.0, Re=6e6)

Members
  regime experts   MLPs trained on one flow regime of the harmonised RANS dataset   models/plan4_v4
  source models    MLPs trained on one source dataset, hence one solver each        models/source_v4
  neuralfoil       NeuralFoil (Sharpe), incompressible: mach is ignored
  airfoilann       AirfoilANN (Khan et al.), JavaFoil-trained, out of its domain on most cases
  xfoil            XFoil panel code; needs XFOIL_BIN or xfoil on PATH, NaN when absent or not converged

Every MLP has the same architecture and the same 27 inputs
[mach, log10 Re, alpha, t/c, f/c, 22 CST coefficients] and predicts (CL, CD).
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.opt.evaluators import tc_fc, eval_nf, eval_xfoil, eval_airfoilann

ROOT = Path(__file__).resolve().parents[1]
REGIME_EXPERTS = ["global", "turbulent_attached", "subsonic_compressible", "transonic",
                  "low_re", "near_stall", "post_stall"]
SOURCE_MODELS = {"unifoil": ["unifoil_ft", "unifoil_nlf"],               # member -> dataset sources
                 "ramos": ["ramos2023a_turb_model", "ramos2023b"],
                 "agarwal": ["agarwal2024"]}
ORACLES = ["neuralfoil", "airfoilann", "xfoil"]
MLP_MEMBERS = REGIME_EXPERTS + list(SOURCE_MODELS)
MEMBERS = MLP_MEMBERS + ORACLES

CST = [f"cst_{i}" for i in range(22)]
FEATS = ["mach", "logRe", "alpha", "t_c", "f_c"] + CST
TARGS = ["CL_rans", "CD_rans"]


class MLP(nn.Module):
    """27 -> 256 -> 128 -> 64 -> 2, ReLU, dropout 0.05 (inactive at inference)."""

    def __init__(self, d_in=len(FEATS), d_out=len(TARGS), hidden=(256, 128, 64), p=0.05):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]
            d = h
        layers += [nn.Linear(d, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def mlp_path(name):
    folder = "source_v4" if name in SOURCE_MODELS else "plan4_v4"
    return ROOT / "models" / folder / f"{name}.pt"


@lru_cache(maxsize=None)
def load_mlp(name):
    """(network, input mean, input std, output mean, output std, log-CD target?) of an MLP member."""
    d = torch.load(mlp_path(name), map_location="cpu", weights_only=False)
    net = MLP(len(d["feats"]), len(d["targs"]))
    net.load_state_dict(d["state_dict"])
    net.eval()
    return (net, np.array(d["xm"]), np.array(d["xs"]), np.array(d["ym"]), np.array(d["ys"]),
            bool(d.get("logcd", False)))


def features27(cst22, mach, alpha, Re):
    t_c, f_c = tc_fc(cst22)
    return np.concatenate([[mach, np.log10(Re), alpha, t_c, f_c], np.asarray(cst22, float)])


def predict_features(name, F):
    """MLP member on rows of 27 features, shape (N, 27) -> (N, 2) = [CL, CD]."""
    net, xm, xs, ym, ys, logcd = load_mlp(name)
    x = np.ascontiguousarray((np.atleast_2d(F) - xm) / xs, dtype=np.float32)
    with torch.no_grad():
        out = net(torch.tensor(x)).numpy()
    pred = out * ys + ym
    if logcd:
        pred[:, 1] = np.exp(pred[:, 1])
    return pred


def predict(name, cst22, mach, alpha, Re):
    """(CL, CD) of member `name` for one airfoil (22 CST coefficients) at (mach, alpha in degrees, Re)."""
    if name == "neuralfoil":
        return eval_nf(cst22, alpha, Re)
    if name == "xfoil":
        return eval_xfoil(cst22, alpha, Re, mach)
    if name == "airfoilann":
        return eval_airfoilann(cst22, mach, alpha, Re)
    if name not in MLP_MEMBERS:
        raise KeyError(f"unknown zoo member {name!r}, expected one of {MEMBERS}")
    cl, cd = predict_features(name, features27(cst22, mach, alpha, Re))[0]
    return float(cl), float(cd)
