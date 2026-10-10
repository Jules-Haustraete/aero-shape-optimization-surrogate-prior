"""Refit the IO-CUE residuals of the prior members against OUR SU2 instead of the mixed-solver dataset.

Same recipe as train_iocue.py, on the SU2 probe (data/su2_probe.parquet: probe-set airfoils solved
with the SU2 setup of the optimizer). The calibration third is drawn BY AIRFOIL, never by row.
These heads are the ones of the final arm.

Input   data/su2_probe.parquet, the trained members
Output  models/iocue_v4/<member>_resid<IOCUE_OUT_SUFFIX>.npz   (default _su2probefix, never overwrites
        the thesis heads _su2probe; selected with `main.py optimize prior ... --heads fixed`)

Two sets are shipped: _su2probe, the heads of the thesis runs, which evaluated the MLP members at
the flight condition of the optimization case; and _su2probefix, written by this script, where
every member is evaluated at the probe row's own Mach and Reynolds numbers through src/zoo.py.
"""
import os

import numpy as np
import pandas as pd

from src import zoo
from src.opt.evaluators import tc_fc

MEMBERS = ["turbulent_attached", "global", "unifoil", "subsonic_compressible", "neuralfoil", "ramos", "airfoilann"]
KEEP = {"strict": ["strict"], "standard": ["strict", "standard"], "large": ["strict", "standard", "large"]}


def main(qual="standard"):
    out_dir = zoo.ROOT / "models/iocue_v4"
    suffix = os.environ.get("IOCUE_OUT_SUFFIX", "_su2probefix")
    D = pd.read_parquet(zoo.ROOT / "data/su2_probe.parquet")
    D = D[D.qual.isin(KEEP[qual])].reset_index(drop=True)
    aids = np.array(sorted(D.aid.unique()))
    rng = np.random.default_rng(0); rng.shuffle(aids)
    cal_row = D.aid.isin(set(aids[:max(1, len(aids) // 3)])).to_numpy()
    print(f"SU2 probe '{qual}': {len(D)} rows, {len(aids)} airfoils, calibration {int(cal_row.sum())} rows")

    X22 = np.stack([np.asarray(c, float)[:22] for c in D.cst22])
    ma, re, al = D.mach.to_numpy(float), D.Re.to_numpy(float), D.alpha.to_numpy(float)
    CLt, CDt = D.CL.to_numpy(float), D.CD.to_numpy(float)
    TF = np.array([tc_fc(x) for x in X22])
    F27 = np.column_stack([ma, np.log10(re), al, TF[:, 0], TF[:, 1], X22])

    for m in MEMBERS:
        pcl = np.full(len(D), np.nan); pcd = np.full(len(D), np.nan)
        for i in range(len(D)):
            try:
                cl, cd = zoo.predict(m, X22[i], ma[i], al[i], re[i])
                if np.isfinite(cl) and np.isfinite(cd) and cd > 0 and abs(cl) < 5:
                    pcl[i], pcd[i] = cl, cd
            except Exception:
                pass
        ok = np.isfinite(pcl) & np.isfinite(pcd)
        cal = cal_row[ok]
        if ok.sum() < 200 or cal.sum() < 60 or (~cal).sum() < 120:
            print(f"  {m:24s} skipped: {ok.sum()} usable rows"); continue
        X = np.column_stack([F27[ok], pcl[ok], pcd[ok]]).astype(np.float32)
        r_cl = (pcl[ok] - CLt[ok]).astype(np.float32); r_cd = (pcd[ok] - CDt[ok]).astype(np.float32)
        p = out_dir / f"{m}_resid{suffix}.npz"
        np.savez(p, X=X, r_cl=r_cl, r_cd=r_cd, cal=cal)
        print(f"  {m:24s} n={ok.sum():4d}  bias CL {np.median(r_cl[~cal]):+.3f}  "
              f"bias CD {np.median(r_cd[~cal]) * 1e4:+6.1f} counts -> {p.name}")
