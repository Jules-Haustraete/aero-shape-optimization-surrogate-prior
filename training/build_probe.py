"""Build the probe set (Section 2.2): the common points every zoo member is measured on.

A stratified sample of the VALIDATION airfoils of the dataset (never used to fit any MLP), at most
6000 points per regime, tagged fit / calib BY AIRFOIL so that the conformal calibration of the IO-CUE
heads never sees an airfoil used to fit them. Its airfoils also define the POD latent (fit_pod.py).

Input   data/dataset_v4_clean.parquet
Output  data/common_probe.parquet
"""
import numpy as np
import pandas as pd

from src.zoo import ROOT, CST

CAP = 6000                                                    # max points per regime


def main():
    df = pd.read_parquet(ROOT / "data/dataset_v4_clean.parquet")
    val = df[df.split == "val"].copy()
    rng = np.random.default_rng(42)
    parts = [g if len(g) <= CAP else g.sample(CAP, random_state=42) for _, g in val.groupby("regime")]
    probe = pd.concat(parts, ignore_index=True)

    afs = probe.airfoil_name.unique(); rng.shuffle(afs)
    fit_af = set(afs[:int(0.70 * len(afs))])
    probe["iocue_split"] = np.where(probe.airfoil_name.isin(fit_af), "fit", "calib")

    keep = ["row_id", "airfoil_name", "source", "regime", "iocue_split", "mach", "Re", "alpha",
            "t_c", "f_c", "CL_rans", "CD_rans"] + CST
    out = ROOT / "data/common_probe.parquet"
    probe[keep].to_parquet(out, index=False)
    print(f"probe set: {len(probe)} points, {probe.airfoil_name.nunique()} airfoils "
          f"(fit {int((probe.iocue_split == 'fit').sum())} / calib {int((probe.iocue_split == 'calib').sum())}) -> {out}")
    print(probe.groupby("regime").size().to_string())
