"""Airfoil geometry descriptors from CST coefficients or coordinates (plan4 §12 toolkit).

Basis-independent shape descriptors used by the geometry-risk / OOD layer. Works from raw
coordinates (any source: our CST, Kulfan, .dat) so references in different bases are comparable.
"""
import numpy as np
from scipy.special import comb

DESCRIPTORS = ["t_c", "camber", "le_thick", "xcamber_loc", "reflex", "thick_25", "thick_75", "tmax_loc", "waviness"]
_NS = 25
_GRID = np.linspace(0.02, 0.98, _NS)


def recon_coords(cst22, n=100):
    """Reconstruct airfoil coordinates from 22 CST coefficients (11 upper, 11 lower; n_cst=11)."""
    c = np.asarray(cst22, float); A_u, A_l = c[:11], c[11:22]
    x = (1 - np.cos(np.linspace(0, np.pi, n))) / 2
    xs = np.clip(x, 1e-10, 1 - 1e-10)
    C = xs ** 0.5 * (1 - xs) ** 1.0
    B = np.stack([comb(10, i) * x ** i * (1 - x) ** (10 - i) for i in range(11)], 1)
    yu = C * (B @ A_u); yl = C * (B @ A_l)
    upper = np.stack([x[::-1], yu[::-1]], 1); lower = np.stack([x[1:], yl[1:]], 1)
    return np.vstack([upper, lower])


def descriptors(coords, with_shape=False):
    """Scalar shape descriptors from coordinates. If with_shape, also return the 2*NS profile vector.
    Returns a dict (and optionally the shape vector), or None if the airfoil is degenerate."""
    coords = np.asarray(coords, float); x, y = coords[:, 0], coords[:, 1]
    le = int(np.argmin(x))
    xu, yu = x[:le + 1][::-1], y[:le + 1][::-1]; xl, yl = x[le:], y[le:]
    if len(np.unique(xu)) < 4 or len(np.unique(xl)) < 4:
        return (None, None) if with_shape else None
    yui = np.interp(_GRID, xu, yu); yli = np.interp(_GRID, xl, yl)
    th = yui - yli; cm = 0.5 * (yui + yli)
    at = lambda g: float(th[np.argmin(np.abs(_GRID - g))])
    sc = dict(t_c=float(th.max()), tmax_loc=float(_GRID[th.argmax()]),
              camber=float(np.abs(cm).max()), xcamber_loc=float(_GRID[np.abs(cm).argmax()]),
              le_thick=at(0.03), reflex=float(cm[np.argmin(np.abs(_GRID - 0.90))]),
              thick_25=at(0.25), thick_75=at(0.75), waviness=float(np.sum(np.abs(np.diff(yui, 2)))))
    return (sc, np.concatenate([th, cm])) if with_shape else sc


def descriptors_from_cst(cst22, with_shape=False):
    """Convenience: CST coefficients -> shape descriptors."""
    return descriptors(recon_coords(cst22), with_shape=with_shape)


def descriptor_matrix(cst_array):
    """(n, 22) CST -> (n, len(DESCRIPTORS)) descriptor matrix; rows that fail are NaN."""
    out = np.full((len(cst_array), len(DESCRIPTORS)), np.nan)
    for i, c in enumerate(cst_array):
        d = descriptors_from_cst(c)
        if d: out[i] = [d[k] for k in DESCRIPTORS]
    return out
