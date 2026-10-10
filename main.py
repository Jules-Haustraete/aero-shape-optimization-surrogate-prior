"""Single entry point of the repository.

    python main.py optimize <arm> <case> [--seed 42]     run one arm of the thesis on one case
    python main.py predict <member> --airfoil naca0012 --mach 0.3 --alpha 4 --re 6e6

    python main.py fit-pod                               the training chain, in order (Chapter 2):
    python main.py train-zoo [member ...]                data/dataset_v4_clean.parquet -> models/
    python main.py build-probe
    python main.py predict-probe [member ...]
    python main.py train-iocue
    python main.py fit-iocue-su2

The optimizer reads its configuration from BRAIN_* environment variables when it is imported,
so `optimize` sets them first and only then runs src/opt/brain_bo.py.
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Table 3.1 of the thesis: number -> (case of src/opt/truth_su2.CASES, objective)
CASES = {1: ("naca0012_ld", "ld"),            # max L/D, M 0.15
         2: ("naca0012_msub_ld", "ld"),       # max L/D, M 0.50
         3: ("naca0012_m03", "ld"),           # max L/D, M 0.30
         4: ("rae2822_ld_od", "ld"),          # RAE 2822 off design, M 0.734
         5: ("naca0012_lowre_ld", "ld"),      # low Re, 3e5
         6: ("naca0012_nearstall_ld", "ld"),  # near stall, alpha 9
         7: ("naca4412_ld", "ld"),            # NACA 4412, Re 1.2e7
         8: ("naca0012", "cd"),               # min CD at CL* = 0.5
         9: ("naca0012t_rans", "cd")}         # min CD, M 0.80

ZOO7 = "turbulent_attached,global,unifoil,subsonic_compressible,neuralfoil,ramos,airfoilann"
# The zoo of the thesis has seven members; the gate runs of Chapter 4 used this wider pool
ZOO13 = ("global,turbulent_attached,subsonic_compressible,transonic,near_stall,post_stall,low_re,"
         "unifoil,ramos,agarwal,neuralfoil,airfoilann,xfoil")
VERIFY_ALL = dict(BRAIN_TAU="0", BRAIN_TAU_PCT="0", BRAIN_TAU_EI="0")

# The four arms, settings copied from the cluster launch scripts of the thesis runs
ARMS = {
    "baseline":  dict(VERIFY_ALL, BRAIN_ACQ="ei", BRAIN_POD_K="0", BRAIN_NITER="600", BRAIN_ZOO=ZOO7),
    "reference": dict(VERIFY_ALL, BRAIN_ACQ="pims", BRAIN_POD_K="8", BRAIN_NITER="600", BRAIN_ZOO=ZOO7),
    "gate":      dict(BRAIN_TAU_PCT="20", BRAIN_TAU_EI="0", BRAIN_GATE="tau", BRAIN_ACQ="pims",
                      BRAIN_POD_K="8", BRAIN_NITER="400", BRAIN_ZOO=ZOO13),
    "prior":     dict(BRAIN_TAU_PCT="0", BRAIN_ACQ="pims", BRAIN_POD_K="8", BRAIN_NITER="600",
                      BRAIN_TRMM="1", BRAIN_TRMM_PRIOR="1", BRAIN_TRMM_RESTART="1", BRAIN_PRIOR_SELECT="rgpe",
                      IOCUE_SUFFIX="_su2probe", BRAIN_ZOO=ZOO7),
}
# Protocol shared by every arm: fast SU2 fidelity, 100 RANS, 6 initial designs, box of 0.5 sigma
COMMON = dict(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", IOCUE_DIR="models/iocue_v4",
              BRAIN_PATIENCE="999", BRAIN_NINIT="6", BRAIN_INIT_RANS="999", OPT_BOX_K="0.5", BRAIN_FLUSH="0",
              BRAIN_LAZY="0", BRAIN_RACE="0", BRAIN_FINAL_RESERVE="0", BRAIN_FUSION="argmin", BRAIN_ZOO_AUTO="0")


def optimize(a):
    case, obj = CASES[int(a.case)] if a.case.isdigit() else (a.case, a.obj)
    wd = str((ROOT / (a.workdir or f"results/{a.arm}_{case}_s{a.seed}")).resolve())   # absolute: SU2 and the mesher run in sub-directories
    os.environ.update(COMMON)
    os.environ.update(ARMS[a.arm])
    if a.arm == "prior" and a.heads == "fixed":
        os.environ["IOCUE_SUFFIX"] = "_su2probefix"
    os.environ.update(BRAIN_FID=a.fid, BRAIN_BUDGET=str(a.budget), BRAIN_CASE=case, BRAIN_OBJ=obj,
                      BRAIN_SEED=str(a.seed), OPT_WORKDIR=wd)
    print(f"[main] {a.arm} on {case} ({obj}), seed {a.seed}, truth={a.fid}, budget {a.budget} -> {wd}", flush=True)
    runpy.run_module("src.opt.brain_bo", run_name="__main__")


def predict(a):
    import numpy as np
    from src import zoo
    p = Path(a.airfoil)
    cst = np.load(p if p.suffix == ".npy" else ROOT / f"src/opt/references/{a.airfoil}_cst22.npy")
    cl, cd = zoo.predict(a.member, cst, a.mach, a.alpha, a.re)
    print(f"{a.member}: CL = {cl:.4f}  CD = {cd * 1e4:.1f} counts  L/D = {cl / cd:.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("optimize", help="run one arm of the thesis on one case")
    o.add_argument("arm", choices=list(ARMS))
    o.add_argument("case", help="1-9 (Table 3.1) or a key of src/opt/truth_su2.CASES")
    o.add_argument("--seed", type=int, default=42, help="the thesis uses 42, 123, 456")
    o.add_argument("--fid", default="fast", help="fast | hf (SU2), or a zoo member as a cheap stand-in for the truth")
    o.add_argument("--budget", type=int, default=100, help="RANS evaluations")
    o.add_argument("--obj", default="ld", choices=["ld", "cd"], help="objective, when case is a CASES key")
    o.add_argument("--workdir", default=None)
    o.add_argument("--heads", default="thesis", choices=["thesis", "fixed"],
                   help="prior arm: IO-CUE heads of the thesis runs, or the ones refit by "
                        "training/fit_iocue_su2probe.py (each probe row at its own flight condition)")

    p = sub.add_parser("predict", help="one zoo member on one airfoil")
    p.add_argument("member")
    p.add_argument("--airfoil", default="naca0012", help="naca0012 | naca4412 | rae2822 | path to a cst22 .npy")
    p.add_argument("--mach", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=4.0)
    p.add_argument("--re", type=float, default=6e6)

    sub.add_parser("fit-pod", help="POD latent of the CST coefficients")
    t = sub.add_parser("train-zoo", help="train the MLP members"); t.add_argument("members", nargs="*")
    sub.add_parser("build-probe", help="probe set, stratified by regime, split by airfoil")
    q = sub.add_parser("predict-probe", help="every member on the probe set"); q.add_argument("members", nargs="*")
    sub.add_parser("train-iocue", help="IO-CUE residuals against the dataset RANS")
    sub.add_parser("fit-iocue-su2", help="IO-CUE residuals of the prior members against our SU2")

    a = ap.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    if a.cmd == "optimize":
        return optimize(a)
    if a.cmd == "predict":
        return predict(a)
    import importlib
    mod = importlib.import_module("training." + {"fit-pod": "fit_pod", "train-zoo": "train_zoo",
                                                 "build-probe": "build_probe", "predict-probe": "predict_probe",
                                                 "train-iocue": "train_iocue", "fit-iocue-su2": "fit_iocue_su2probe"}[a.cmd])
    return mod.main(a.members or None) if a.cmd in ("train-zoo", "predict-probe") else mod.main()


if __name__ == "__main__":
    main()
