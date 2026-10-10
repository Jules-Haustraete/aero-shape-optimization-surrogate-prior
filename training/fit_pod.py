"""POD of the CST coefficients of real airfoils (Section 2.3): the latent space of the optimizer.

The airfoils are those of the probe set (7468); one CST vector per airfoil, centred, SVD.
Eight modes keep 99 % of the variance. brain_bo searches z in [-Zb, Zb]^k around the case's
starting airfoil, x = x0 + V[:, :k] z, clipped to the CST box (BRAIN_POD_K=8).

Input   data/common_probe.parquet
Output  models/pod_cst.npz   (V: modes as columns, S: singular values, mu: mean, n: airfoils)
"""
import numpy as np
import pandas as pd

from src.zoo import ROOT, CST


def main():
    pr = pd.read_parquet(ROOT / "data/common_probe.parquet")
    X = pr.groupby("airfoil_name")[CST].first().to_numpy(float)
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    out = ROOT / "models/pod_cst.npz"
    np.savez(out, V=Vt.T, S=S, mu=mu, n=len(X))
    var = np.cumsum(S ** 2) / (S ** 2).sum()
    print(f"POD of {len(X)} airfoils: variance kept by 4/8/12 modes = "
          f"{var[3]:.3f} / {var[7]:.3f} / {var[11]:.3f} -> {out}")
