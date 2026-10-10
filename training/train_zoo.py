"""Train the MLP members of the zoo (Section 2.1).

Regime experts learn one flow regime of the harmonised RANS dataset (`global` learns all of it),
source models learn one source dataset, hence one solver each. The split is by AIRFOIL: early
stopping watches airfoils never seen in training, and the probe set of the IO-CUE heads is drawn
from those same validation airfoils, never from the training ones.

Input   data/dataset_v4_clean.parquet  (split, regime, source, mach, Re, alpha, t_c, f_c,
                                        cst_0..cst_21, CL_rans, CD_rans)
Output  models/plan4_v4/<member>.pt, models/source_v4/<member>.pt
        LOGCD=1 trains on log CD and writes <member>_logcd.pt
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src import zoo
from src.zoo import FEATS, TARGS, CST

ROOT = zoo.ROOT
SRC = ROOT / "data/dataset_v4_clean.parquet"
SEED = 42
LOGCD = os.environ.get("LOGCD", "0") == "1"


def _t(a):
    return torch.tensor(np.ascontiguousarray(a, dtype=np.float32))


def train(name, df):
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if name in zoo.SOURCE_MODELS:
        rows = df.source.isin(zoo.SOURCE_MODELS[name])
    elif name == "global":
        rows = np.ones(len(df), bool)
    else:
        rows = df.regime == name

    def subset(split):
        return df[rows & (df.split == split)].dropna(subset=["mach", "Re", "alpha", "t_c", "f_c"] + CST + TARGS)

    dtr, dva = subset("train"), subset("val")
    assert len(dtr) > 300, f"too few train rows for {name}: {len(dtr)}"

    def target(d):
        Y = d[TARGS].to_numpy().copy()
        if LOGCD:
            Y[:, 1] = np.log(np.clip(Y[:, 1], 1e-5, None))
        return Y

    X, Y = dtr[FEATS].to_numpy(), target(dtr)
    xm, xs = X.mean(0), X.std(0) + 1e-8
    ym, ys = Y.mean(0), Y.std(0) + 1e-8
    Xtr, Ytr = _t((X - xm) / xs).to(dev), _t((Y - ym) / ys).to(dev)
    Xva, Yva = _t((dva[FEATS].to_numpy() - xm) / xs).to(dev), _t((target(dva) - ym) / ys).to(dev)

    model = zoo.MLP(len(FEATS), len(TARGS)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    lossf = nn.MSELoss(); bs = 8192; nb = (len(Xtr) + bs - 1) // bs
    best, best_state, bad, t0, ep = 1e9, None, 0, time.time(), 0
    for ep in range(80):
        model.train(); order = torch.randperm(len(Xtr), device=dev)
        for b in range(nb):
            sl = order[b * bs:(b + 1) * bs]
            opt.zero_grad(); loss = lossf(model(Xtr[sl]), Ytr[sl]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = lossf(model(Xva), Yva).item()
        sched.step(vloss)
        if vloss < best - 1e-5:
            best, best_state, bad = vloss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 12:                                   # early stopping on unseen airfoils
                break
    model.load_state_dict(best_state); model.eval()

    with torch.no_grad():
        pv = model(Xva).cpu().numpy() * ys + ym
        pt = model(Xtr).cpu().numpy() * ys + ym
    if LOGCD:
        pv[:, 1] = np.exp(pv[:, 1]); pt[:, 1] = np.exp(pt[:, 1])
    yv, yt = dva[TARGS].to_numpy(), dtr[TARGS].to_numpy()
    r2v = 1 - ((yv - pv) ** 2).sum(0) / ((yv - yv.mean(0)) ** 2).sum(0)
    r2t = 1 - ((yt - pt) ** 2).sum(0) / ((yt - yt.mean(0)) ** 2).sum(0)
    tag = name + ("_logcd" if LOGCD else "")
    diag = {"name": tag, "logcd": LOGCD, "n_train": len(dtr), "n_val": len(dva), "epochs": ep + 1,
            "sec": round(time.time() - t0, 1),
            "val_R2": {t: round(float(r), 4) for t, r in zip(TARGS, r2v)},
            "train_R2": {t: round(float(r), 4) for t, r in zip(TARGS, r2t)}}
    out = zoo.mlp_path(name).with_name(f"{tag}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "feats": FEATS, "targs": TARGS, "xm": xm.tolist(), "xs": xs.tolist(),
                "ym": ym.tolist(), "ys": ys.tolist(), "diag": diag, "logcd": LOGCD}, out)
    print(json.dumps(diag), flush=True)
    return diag


def main(members=None):
    df = pd.read_parquet(SRC)
    if "split" not in df.columns:                           # airfoil split shipped separately
        df = df.merge(pd.read_parquet(ROOT / "data/splits_v4_airfoil.parquet")[["row_id", "split"]],
                      on="row_id", how="left")
    df["logRe"] = np.log10(df["Re"])
    for name in (members or zoo.MLP_MEMBERS):
        train(name, df)
