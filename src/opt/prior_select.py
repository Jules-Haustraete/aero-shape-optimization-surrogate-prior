"""ONLINE selection of the surrogate prior from a zoo, by RANK agreement with the RANS anchors already verified.
Inspired by RGPE (Feurer et al. 2018, "Practical Transfer Learning for BO"):
  - loss of a candidate = number of DISCORDANT PAIRS between its ranking of the anchors and the true ranking (scale-free)
  - 'rgpe': bootstrap (S draws of the anchors with replacement) -> w_m = frequency with which m is the argmin; the TARGET
             (GP on the truth, closed-form LOO predictions) is a candidate; DILUTION: a base model whose median loss exceeds
             the 95th percentile of the target loss gets 0.
  - 'argmin': hard switch = candidate with minimal loss over all anchors.
Key difference with the offline selectors (σ̂, probe accuracy): here the scores come from the RANS of THIS run, hence from
the optimum region (where the in-dist probe is blind). Zero extra RANS.
"""
import numpy as np

NONE = "none"   # 'no prior' candidate (f_low = 0 -> GP on the truth alone)


def discordant_loss(pred, y):
    """Number of pairs (j<k) where the predicted order contradicts the true order. Ties ignored."""
    pred = np.asarray(pred, float); y = np.asarray(y, float); n = len(y)
    if n < 2: return 0.0
    dp = np.sign(pred[:, None] - pred[None, :]); dy = np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(n, 1)
    return float(np.sum((dp[iu] * dy[iu]) < 0))


def gp_loo_mean(K, y):
    """Closed-form LOO mean of a zero-mean GP (Sundararajan&Keerthi 2001):
    mu_loo_i = y_i - [K^-1 y]_i / [K^-1]_ii.  K = covariance matrix of the ANCHORS (noise included on the diagonal)."""
    K = np.asarray(K, float); y = np.asarray(y, float)
    Ki = np.linalg.pinv(K + 1e-9 * np.eye(len(y)))
    return y - (Ki @ y) / np.clip(np.diag(Ki), 1e-12, None)


def select_prior(beliefs, y, mode="rgpe", target_pred=None, S=100, seed=0, dilution_pct=95):
    """beliefs : {model: array(n_anchors)} beliefs of each candidate prior at the anchors.
       y       : array(n_anchors) RANS truth (scale to MAXIMIZE).
       target_pred : array(n) LOO predictions of the target (GP without prior). None -> 'none' candidate scored as a constant (max loss/2).
       Returns (weights dict {model: w}, info dict). weights sum to 1, NONE included."""
    models = list(beliefs.keys()); y = np.asarray(y, float); n = len(y)
    cands = {m: np.asarray(beliefs[m], float) for m in models}
    if target_pred is not None: cands[NONE] = np.asarray(target_pred, float)
    names = list(cands.keys())
    if n < 4:                                                    # too few anchors: uniform over the zoo (no none)
        w = {m: 1.0 / len(models) for m in models}; w[NONE] = 0.0
        return w, dict(reason="n<4", losses={})
    full = {m: discordant_loss(cands[m], y) for m in names}
    if mode == "argmin":
        best = min(names, key=lambda m: (full[m], names.index(m)))
        w = {m: (1.0 if m == best else 0.0) for m in names}
        if NONE not in w: w[NONE] = 0.0
        return w, dict(reason="argmin", losses=full, chosen=best)
    # ── RGPE bootstrap ──
    rng = np.random.default_rng(seed); counts = {m: 0.0 for m in names}; boot_losses = {m: [] for m in names}
    for _ in range(S):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(idx)) < 2: continue
        L = {m: discordant_loss(cands[m][idx], y[idx]) for m in names}
        for m in names: boot_losses[m].append(L[m])
        mn = min(L.values()); ties = [m for m in names if L[m] == mn]
        for m in ties: counts[m] += 1.0 / len(ties)
    tot = sum(counts.values()) or 1.0
    w = {m: counts[m] / tot for m in names}
    # dilution (Feurer): base model whose median loss > 95th pct of the target loss -> 0
    if NONE in cands and boot_losses[NONE]:
        thr = np.percentile(boot_losses[NONE], dilution_pct)
        for m in models:
            if boot_losses[m] and np.median(boot_losses[m]) > thr: w[m] = 0.0
        s = sum(w.values())
        w = {m: v / s for m, v in w.items()} if s > 0 else {m: (1.0 if m == NONE else 0.0) for m in names}
    if NONE not in w: w[NONE] = 0.0
    return w, dict(reason="rgpe", losses=full, boot_median={m: float(np.median(v)) if v else None for m, v in boot_losses.items()})


def combine(weights, beliefs_at, none_value=0.0):
    """f_low(x) = Σ_m w_m · belief_m(x) (+ w_none · 0). beliefs_at : {model: array} at the requested points."""
    out = None
    for m, w in weights.items():
        if m == NONE or w == 0.0: continue
        b = np.asarray(beliefs_at[m], float) * w
        out = b if out is None else out + b
    if out is None:
        any_m = next(iter(beliefs_at)); out = np.zeros_like(np.asarray(beliefs_at[any_m], float)) + none_value
    return out
