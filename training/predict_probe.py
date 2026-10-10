"""Run every zoo member on exactly the probe set, so that their residuals are comparable.

Input   data/common_probe.parquet, the trained members (models/)
Output  data/predictions_all.parquet   (row_id, <member>_cl, <member>_cd)
XFoil is skipped unless asked for explicitly: about half a second per point.
"""
import time

import numpy as np
import pandas as pd

from src import zoo
from src.zoo import ROOT, FEATS, CST


def main(members=None):
    pr = pd.read_parquet(ROOT / "data/common_probe.parquet")
    F = pr.assign(logRe=np.log10(pr.Re))[FEATS].to_numpy(float)
    cst = pr[CST].to_numpy(float); mach, alpha, Re = pr.mach.to_numpy(), pr.alpha.to_numpy(), pr.Re.to_numpy()
    res = {"row_id": pr.row_id.to_numpy()}
    for nm in (members or [m for m in zoo.MEMBERS if m != "xfoil"]):
        t = time.time()
        if nm in zoo.MLP_MEMBERS:                                       # batched forward pass
            out = zoo.predict_features(nm, F)
            res[f"{nm}_cl"], res[f"{nm}_cd"] = out[:, 0], out[:, 1]
        else:                                                           # oracles, point by point
            cl = np.full(len(pr), np.nan); cd = np.full(len(pr), np.nan)
            for i in range(len(pr)):
                try:
                    cl[i], cd[i] = zoo.predict(nm, cst[i], float(mach[i]), float(alpha[i]), float(Re[i]))
                except Exception:
                    pass
            res[f"{nm}_cl"], res[f"{nm}_cd"] = cl, cd
        ok = np.isfinite(res[f"{nm}_cl"]).sum()
        print(f"{nm}: {ok}/{len(pr)} predictions in {time.time() - t:.1f}s", flush=True)
    out = ROOT / "data/predictions_all.parquet"
    pd.DataFrame(res).to_parquet(out, index=False)
    print(f"-> {out}", flush=True)
