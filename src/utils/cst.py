"""
CST (Kulfan) Airfoil Parameterization — Phase A
================================================
Fits a 18-parameter CST representation to any airfoil .dat file:
  - 8 upper surface Bernstein coefficients  (A_upper[0..7])
  - 8 lower surface Bernstein coefficients  (A_lower[0..7])
  - 1 trailing-edge thickness upper         (dz_te_upper)
  - 1 trailing-edge thickness lower         (dz_te_lower)

Plus 2 scalar geometric descriptors computed via AeroSandbox:
  - t/c  : maximum thickness ratio (%)
  - f/c  : maximum camber ratio (%)

Total feature vector for MLP: 18 CST + t/c + f/c = 20 geometric features.

Reference: Kulfan, B. (2008). "Universal Parametric Geometry Representation Method."
           Journal of Aircraft, 45(1), 142–158.

Usage:
    from src.utils.cst import fit_cst, load_and_fit

    params = fit_cst(upper_coords, lower_coords, n_cst=8)
    # → dict with keys: A_upper, A_lower, dz_te_upper, dz_te_lower, t_c, f_c,
    #                   reconstruction_error_upper, reconstruction_error_lower

    result = load_and_fit("data/airfoils/naca0012.dat", n_cst=8)
"""

import numpy as np
from pathlib import Path
from scipy.special import comb


# ── Bernstein basis ──────────────────────────────────────────────────────────

def bernstein_basis(x: np.ndarray, n: int) -> np.ndarray:
    """
    Compute the (n+1) Bernstein basis polynomials of degree n at points x.

    Returns matrix B of shape (len(x), n+1) where:
        B[:, i] = C(n,i) * x^i * (1-x)^(n-i)
    """
    x = np.asarray(x, dtype=float).ravel()
    B = np.zeros((len(x), n + 1))
    for i in range(n + 1):
        B[:, i] = comb(n, i, exact=True) * (x ** i) * ((1 - x) ** (n - i))
    return B


def class_function(x: np.ndarray, N1: float = 0.5, N2: float = 1.0) -> np.ndarray:
    """
    CST class function C(x) = x^N1 * (1-x)^N2.
    Defaults: N1=0.5 (blunt leading edge), N2=1.0 (sharp trailing edge).
    """
    x = np.asarray(x, dtype=float).ravel()
    # Avoid numerical issues at endpoints
    x_safe = np.clip(x, 1e-10, 1 - 1e-10)
    return (x_safe ** N1) * ((1 - x_safe) ** N2)


# ── Core fitting ─────────────────────────────────────────────────────────────

def fit_surface(
    x: np.ndarray,
    y: np.ndarray,
    n_cst: int = 8,
) -> tuple[np.ndarray, float, float]:
    """
    Fit CST coefficients + TE thickness to one surface (upper or lower).

    The CST model for one surface:
        y(x) = C(x) · Σ Aᵢ · Bᵢ,ₙ(x)  +  x · dz_te

    This is linear in [A₀..Aₙ, dz_te] → solved by least squares.

    Args:
        x      : x-coordinates ∈ [0, 1], shape (M,)
        y      : y-coordinates, shape (M,)
        n_cst  : number of Bernstein coefficients per surface (default 8)

    Returns:
        coeffs     : array of shape (n_cst + 1,) = [A₀..A_{n-1}, dz_te]
        rmse       : root mean squared reconstruction error
        max_err    : max absolute reconstruction error
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    # Remove leading/trailing edge exact points to avoid division by zero
    mask = (x > 1e-6) & (x < 1 - 1e-6)
    x_fit, y_fit = x[mask], y[mask]

    C  = class_function(x_fit)                          # shape (M,)
    B  = bernstein_basis(x_fit, n_cst - 1)              # shape (M, n_cst)

    # Design matrix: C(x)*B columns + x column for TE thickness
    phi = C[:, None] * B                                # (M, n_cst)
    phi = np.hstack([phi, x_fit[:, None]])              # (M, n_cst+1)

    # Least squares solve
    coeffs, _, _, _ = np.linalg.lstsq(phi, y_fit, rcond=None)

    # Reconstruction error
    y_pred = phi @ coeffs
    residuals = y_fit - y_pred
    rmse    = float(np.sqrt(np.mean(residuals ** 2)))
    max_err = float(np.max(np.abs(residuals)))

    return coeffs, rmse, max_err


def reconstruct_surface(
    x: np.ndarray,
    coeffs: np.ndarray,
    n_cst: int = 8,
) -> np.ndarray:
    """
    Reconstruct y-coordinates from CST coefficients.

    Args:
        x      : x-coordinates ∈ [0, 1]
        coeffs : array [A₀..A_{n-1}, dz_te] of length n_cst+1
        n_cst  : number of Bernstein coefficients

    Returns:
        y_reconstructed : shape (len(x),)
    """
    x = np.asarray(x, dtype=float).ravel()
    C = class_function(x)
    B = bernstein_basis(x, n_cst - 1)
    A    = coeffs[:n_cst]
    dz   = coeffs[n_cst]
    return C * (B @ A) + x * dz


# ── Surface splitting ─────────────────────────────────────────────────────────

def split_surfaces(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split airfoil coordinates into upper and lower surfaces.

    Handles both Selig format (TE→LE→TE) and split format.
    Returns (upper, lower) each as array of shape (N, 2) sorted by x ascending.
    """
    coords = np.asarray(coords, dtype=float)

    # Find leading edge (minimum x)
    le_idx = np.argmin(coords[:, 0])

    # Split at leading edge
    part1 = coords[:le_idx + 1]   # from start to LE
    part2 = coords[le_idx:]        # from LE to end

    # Determine which is upper vs lower by sign of y at mid-chord
    def mean_y_at_mid(pts):
        mid_mask = (pts[:, 0] > 0.3) & (pts[:, 0] < 0.7)
        if mid_mask.sum() == 0:
            return np.mean(pts[:, 1])
        return np.mean(pts[mid_mask, 1])

    if mean_y_at_mid(part1) >= mean_y_at_mid(part2):
        upper_raw, lower_raw = part1, part2
    else:
        upper_raw, lower_raw = part2, part1

    # Sort by x ascending, clip to [0,1]
    def clean(pts):
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] <= 1)]
        return pts[np.argsort(pts[:, 0])]

    return clean(upper_raw), clean(lower_raw)


# ── Main fitting function ─────────────────────────────────────────────────────

def fit_cst(
    coords: np.ndarray,
    n_cst: int = 8,
) -> dict:
    """
    Fit 18-parameter CST representation to airfoil coordinates.

    Args:
        coords : airfoil coordinates, shape (N, 2), any standard format
        n_cst  : Bernstein polynomial order per surface (default 8 → 18 params)

    Returns dict with:
        A_upper             : ndarray (n_cst,) — upper Bernstein coefficients
        A_lower             : ndarray (n_cst,) — lower Bernstein coefficients
        dz_te_upper         : float — upper TE thickness
        dz_te_lower         : float — lower TE thickness
        t_c                 : float — max thickness / chord (fraction)
        f_c                 : float — max camber / chord (fraction)
        rmse_upper          : float — upper surface reconstruction RMSE
        rmse_lower          : float — lower surface reconstruction RMSE
        max_err_upper       : float — upper surface max absolute error
        max_err_lower       : float — lower surface max absolute error
        feature_vector      : ndarray (20,) — [A_upper(8), A_lower(8),
                                               dz_te_upper, dz_te_lower,
                                               t_c, f_c]
    """
    upper, lower = split_surfaces(coords)

    # Fit each surface
    coeffs_u, rmse_u, max_u = fit_surface(upper[:, 0], upper[:, 1], n_cst)
    coeffs_l, rmse_l, max_l = fit_surface(lower[:, 0], lower[:, 1], n_cst)

    A_upper = coeffs_u[:n_cst]
    A_lower = coeffs_l[:n_cst]
    dz_te_u = float(coeffs_u[n_cst])
    dz_te_l = float(coeffs_l[n_cst])

    # Compute t/c and f/c from coordinates
    x_query = np.linspace(0.01, 0.99, 200)
    y_u = reconstruct_surface(x_query, coeffs_u, n_cst)
    y_l = reconstruct_surface(x_query, coeffs_l, n_cst)
    thickness = y_u - y_l
    camber    = (y_u + y_l) / 2.0
    t_c = float(np.max(thickness))
    f_c = float(camber[np.argmax(np.abs(camber))])

    feature_vector = np.concatenate([
        A_upper, A_lower,
        [dz_te_u, dz_te_l, t_c, f_c]
    ])

    return {
        "A_upper":       A_upper,
        "A_lower":       A_lower,
        "dz_te_upper":   dz_te_u,
        "dz_te_lower":   dz_te_l,
        "t_c":           t_c,
        "f_c":           f_c,
        "rmse_upper":    rmse_u,
        "rmse_lower":    rmse_l,
        "max_err_upper": max_u,
        "max_err_lower": max_l,
        "feature_vector": feature_vector,  # shape (20,)
    }


# ── File-level entry point ────────────────────────────────────────────────────

def load_and_fit(dat_path: str, n_cst: int = 8) -> dict | None:
    """
    Load a .dat airfoil file and return its CST fit.

    Args:
        dat_path : path to .dat file (Selig format, first line = name)
        n_cst    : Bernstein order per surface (default 8)

    Returns:
        dict from fit_cst(), with added key "airfoil_name", or None on failure.
    """
    path = Path(dat_path)
    try:
        coords = np.loadtxt(path, skiprows=1)
        if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 20:
            return None
        result = fit_cst(coords, n_cst=n_cst)
        result["airfoil_name"] = path.stem.lower()
        result["dat_path"]     = str(path)
        return result
    except Exception as e:
        return None


# ── Batch fitting ─────────────────────────────────────────────────────────────

def fit_all(
    dat_paths: list[str],
    n_cst: int = 8,
    verbose: bool = True,
) -> "pd.DataFrame":
    """
    Fit CST to a list of .dat files. Returns a DataFrame with one row per profile.

    Columns: airfoil_name, A_upper_0..7, A_lower_0..7,
             dz_te_upper, dz_te_lower, t_c, f_c,
             rmse_upper, rmse_lower, max_err_upper, max_err_lower
    """
    import pandas as pd
    from tqdm import tqdm

    records = []
    failed  = 0
    iterator = tqdm(dat_paths, desc="CST fitting") if verbose else dat_paths

    for path in iterator:
        result = load_and_fit(path, n_cst=n_cst)
        if result is None:
            failed += 1
            continue
        row = {"airfoil_name": result["airfoil_name"], "dat_path": result["dat_path"]}
        for i in range(n_cst):
            row[f"A_upper_{i}"] = float(result["A_upper"][i])
            row[f"A_lower_{i}"] = float(result["A_lower"][i])
        row["dz_te_upper"]   = result["dz_te_upper"]
        row["dz_te_lower"]   = result["dz_te_lower"]
        row["t_c"]           = result["t_c"]
        row["f_c"]           = result["f_c"]
        row["rmse_upper"]    = result["rmse_upper"]
        row["rmse_lower"]    = result["rmse_lower"]
        row["max_err_upper"] = result["max_err_upper"]
        row["max_err_lower"] = result["max_err_lower"]
        records.append(row)

    if verbose:
        print(f"Fitted: {len(records)}  |  Failed: {failed}")
    return pd.DataFrame(records)
