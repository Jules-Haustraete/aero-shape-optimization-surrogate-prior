"""IO-CUE residuals of every zoo member on the probe set (Section 2.5).

For each member: features = the 27 inputs + the member's own (CL, CD) prediction (input-output
conditioning, 29 dimensions), signed residuals against the RANS reference, and the fit / calib mask
by airfoil. Only the residuals are stored: the gradient-boosting heads (median bias delta-hat,
conformal quantile sigma-hat) are refit from them when src/opt/iocue.py loads a member, so no
sklearn pickle ever crosses versions.

Input   data/common_probe.parquet, data/predictions_all.parquet
Output  models/iocue_v4/<member>_resid.npz, results/iocue_v4_report.csv
"""
import json
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as GBR

from src.zoo import ROOT, FEATS

OUT = ROOT / "models/iocue_v4"
Q = 0.9


def cqr_coverage(X, absr, cal):
    """Quantile-Q gradient boosting on the fit airfoils + one-sided conformal offset on the calib ones
    -> empirical coverage on calib (a check, the heads used at run time are refit by iocue.py)."""
    m = GBR(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.9, loss="quantile", alpha=Q)
    m.fit(X[~cal], absr[~cal])
    sc = absr[cal] - m.predict(X[cal])
    off = float(np.quantile(sc, min(math.ceil((cal.sum() + 1) * Q) / cal.sum(), 1.0)))
    return float((absr[cal] <= np.maximum(0, m.predict(X[cal]) + off)).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pr = pd.read_parquet(ROOT / "data/common_probe.parquet").assign(logRe=lambda d: np.log10(d.Re))
    pred = pd.read_parquet(ROOT / "data/predictions_all.parquet")
    df = pr.merge(pred, on="row_id")
    X27 = df[FEATS].to_numpy(float); cal = (df.iocue_split == "calib").to_numpy()
    rep = []
    for nm in sorted({c[:-3] for c in pred.columns if c.endswith("_cl")}):
        cl_h, cd_h = df[f"{nm}_cl"].to_numpy(float), df[f"{nm}_cd"].to_numpy(float)
        ok = np.isfinite(cl_h) & np.isfinite(cd_h)
        X = np.hstack([X27, cl_h[:, None], cd_h[:, None]])[ok]
        r_cl = (cl_h - df.CL_rans.to_numpy(float))[ok]; r_cd = (cd_h - df.CD_rans.to_numpy(float))[ok]
        c = cal[ok]
        np.savez(OUT / f"{nm}_resid.npz", X=X.astype(np.float32), r_cl=r_cl.astype(np.float32),
                 r_cd=r_cd.astype(np.float32), cal=c)
        cov = cqr_coverage(X, np.abs(r_cd), c) if c.sum() > 20 else float("nan")
        rep.append({"member": nm, "n": int(ok.sum()), "bias_CD_counts": round(float(np.median(r_cd)) * 1e4, 1),
                    "mae_CD_counts": round(float(np.abs(r_cd).mean()) * 1e4, 1),
                    "mae_CL": round(float(np.abs(r_cl).mean()), 3), "cqr_coverage_CD": round(cov, 3)})
        print(json.dumps(rep[-1]), flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    pd.DataFrame(rep).to_csv(ROOT / "results/iocue_v4_report.csv", index=False)
