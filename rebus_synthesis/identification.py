"""REBUS identification and robust safety-bound synthesis.

This module implements the constrained regression / robust optimization pipeline for
estimating one-sided safety constants used in the REBUS supervisor.

Dependencies
------------
- numpy
- cvxpy

The implementation follows a bootstrap-scenario approach for uncertainty handling:
1. Fit constrained local models and scalar gate/cost models.
2. Bootstrap contiguous blocks or episodes.
3. Use bootstrap scenarios in robust semidefinite programs to compute
   one-sided lower/upper confidence bounds.

Main entry points
-----------------
- estimate_homeostatic_center
- identify_rebus_bounds
- synthesize_supervisor_gains
- make_synthetic_rebus_data
- run_demo
- synthetic_smoke_test

Data conventions
----------------
All time-varying arrays are expected to be aligned along the last axis. For example:
    X_t    : shape (n_x, T)
    U_t    : shape (n_u, T)
    X_tp1  : shape (n_x, T)
Scalar series can be passed as 1-D arrays of length T.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cvxpy as cp
except Exception:  # pragma: no cover
    cp = None

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]
OPT_OK = {"optimal", "optimal_inaccurate"}

def _require_cvxpy() -> None:
    if cp is None:
        raise ImportError(
            "cvxpy is required for this function. Install it in the target environment before use."
        )



@dataclass(frozen=True)
class RebusBounds:
    """One-sided robust bounds returned by the identification pipeline."""

    a_lb: float
    P: np.ndarray
    lambda_lb: float
    mu_lb: float
    kappa_ub: float
    c_alpha_ub: float
    c_nu_ub: float
    b0_ub: float
    b1_ub: float
    b2_ub: float
    b3_ub: float


@dataclass(frozen=True)
class SupervisorGains:
    """Lyapunov-composite gain synthesis output."""

    p: float
    q: float
    delta_safe: float
    Gamma: float


@dataclass(frozen=True)
class SyntheticTruth:
    """Ground-truth parameters for the built-in synthetic REBUS scaffold."""

    A: np.ndarray
    B: np.ndarray
    d: np.ndarray
    phi_alpha: float
    eta_omega: float
    eta_chi: float
    phi_e: float
    psi_e: float
    direct_kappa: float
    c0: float
    c_alpha: float
    c_nu: float


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------


def _as_float_array(arr: ArrayLike, ndim: Optional[int] = None) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"Expected array with ndim={ndim}, got {out.ndim}.")
    return out



def _as_1d(arr: ArrayLike) -> np.ndarray:
    out = np.asarray(arr, dtype=float).reshape(-1)
    return out



def _as_2d_time(arr: ArrayLike) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    if out.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D time-series array, got ndim={out.ndim}.")
    return out



def _validate_same_time_length(*arrays: np.ndarray) -> int:
    lengths = [arr.shape[-1] for arr in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"Mismatched time dimensions: {lengths}")
    return lengths[0]


def _validate_paired_lengths(
    reference_len: int,
    reference_name: str,
    paired: Mapping[str, ArrayLike],
) -> None:
    """Raise ValueError if any paired 1-D series doesn't match reference_len.

    Used by bootstrap_pipeline to surface array-alignment errors as clear
    contract violations instead of mid-loop np.take IndexErrors when bootstrap
    indices drawn from `reference_name` are reused on shorter paired series.
    """
    bad = {
        name: len(_as_1d(arr))
        for name, arr in paired.items()
        if len(_as_1d(arr)) != reference_len
    }
    if bad:
        details = ", ".join(f"{n}(T={ln})" for n, ln in sorted(bad.items()))
        raise ValueError(
            f"Series sharing bootstrap indices with "
            f"{reference_name}(T={reference_len}) have mismatched lengths: "
            f"{details}"
        )



def _safe_quantile(values: Sequence[float], q: float) -> float:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        raise ValueError("Cannot compute quantile of an empty sequence.")
    return float(np.quantile(vals, q))


# ---------------------------------------------------------------------------
# Centering / preprocessing
# ---------------------------------------------------------------------------


def estimate_homeostatic_center(
    Y_closed: ArrayLike,
    method: str = "median",
    huber_M: float = 1.0,
    solver: str = "SCS",
    verbose: bool = False,
) -> np.ndarray:
    """Estimate the homeostatic center from closed-mode observations.

    Parameters
    ----------
    Y_closed:
        Array of shape (n_y, T_closed).
    method:
        Either "median" (coordinatewise median) or "huber".
    huber_M:
        Huber threshold for the "huber" estimator.
    solver:
        CVXPY solver name for the Huber estimator.
    verbose:
        Passed through to CVXPY.

    Returns
    -------
    np.ndarray of shape (n_y, 1)
    """
    Y = _as_2d_time(Y_closed)
    if Y.shape[1] == 0:
        raise ValueError("Y_closed must contain at least one sample.")

    if method.lower() == "median":
        return np.median(Y, axis=1, keepdims=True)

    if method.lower() != "huber":
        raise ValueError("method must be 'median' or 'huber'.")

    # Only the Huber branch needs cvxpy — defer the import check so callers
    # explicitly requesting method="median" work in slim installs.
    _require_cvxpy()
    n_y, _ = Y.shape
    y = cp.Variable((n_y, 1))
    resid = Y - y @ np.ones((1, Y.shape[1]))
    prob = cp.Problem(cp.Minimize(cp.sum(cp.huber(resid, M=huber_M))))
    prob.solve(solver=solver, verbose=verbose)
    if prob.status not in OPT_OK or y.value is None:
        raise RuntimeError(f"Failed to estimate homeostatic center; status={prob.status}")
    return np.asarray(y.value, dtype=float)



def center_state(Y: ArrayLike, y_star: ArrayLike) -> np.ndarray:
    """Center state observations around the homeostatic operating point."""
    Y2 = _as_2d_time(Y)
    y0 = np.asarray(y_star, dtype=float)
    if y0.ndim == 1:
        y0 = y0.reshape(-1, 1)
    elif y0.ndim == 2 and y0.shape[0] == 1 and y0.shape[1] == Y2.shape[0]:
        y0 = y0.reshape(-1, 1)
    elif y0.ndim != 2:
        raise ValueError("y_star must be a vector or column matrix.")
    if y0.shape[1] != 1:
        raise ValueError("y_star must have shape (n_y, 1) or be broadcastable to that form.")
    if y0.shape[0] != Y2.shape[0]:
        raise ValueError("State dimension mismatch between Y and y_star.")
    return Y2 - y0


# ---------------------------------------------------------------------------
# Convex loss helpers
# ---------------------------------------------------------------------------


def huber_sum(resid: Union[np.ndarray, "cp.Expression"], M: float) -> "cp.Expression":
    """Sum of elementwise Huber penalties."""
    _require_cvxpy()
    return cp.sum(cp.huber(resid, M=M))



def pinball_sum(resid: Union[np.ndarray, "cp.Expression"], q: float) -> "cp.Expression":
    """Sum of pinball (quantile) losses at quantile q in [0,1]."""
    _require_cvxpy()
    if not (0.0 < q < 1.0):
        raise ValueError("Quantile q must lie strictly between 0 and 1.")
    return cp.sum(q * cp.pos(resid) + (1.0 - q) * cp.pos(-resid))


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def moving_block_indices(T: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Generate bootstrap indices by resampling contiguous blocks with replacement."""
    if T <= 0:
        raise ValueError("T must be positive.")
    if block_len <= 0:
        raise ValueError("block_len must be positive.")
    if block_len > T:
        block_len = T
    n_blocks = int(np.ceil(T / block_len))
    starts = rng.integers(0, T - block_len + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:T]
    return idx.astype(int)



def bootstrap_take(arr: ArrayLike, idx: np.ndarray) -> np.ndarray:
    """Take bootstrap samples along the last axis."""
    a = np.asarray(arr)
    return np.take(a, idx, axis=-1)



def box_vertices(alpha_bar: float, e_bar: float, omega_bar: float) -> List[np.ndarray]:
    """Vertices of the input uncertainty box [0,alpha_bar] x [0,e_bar] x [0,omega_bar]."""
    verts: List[np.ndarray] = []
    for a in (0.0, float(alpha_bar)):
        for e in (0.0, float(e_bar)):
            for w in (0.0, float(omega_bar)):
                verts.append(np.array([[a], [e], [w]], dtype=float))
    return verts


# ---------------------------------------------------------------------------
# Identification problems P1-P4
# ---------------------------------------------------------------------------


def fit_local_plant(
    X_t: ArrayLike,
    U_t: ArrayLike,
    X_tp1: ArrayLike,
    tau_x: float = 1.0,
    rho_theta: float = 1e-4,
    eps_A: float = 1e-2,
    C_theta: Optional[np.ndarray] = None,
    h_theta: Optional[np.ndarray] = None,
    solver: str = "SCS",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Fit the local linear plant x_{t+1} = A x_t + B u_t + d + w_t.

    Solves a constrained robust ridge problem with Huber loss and a spectral-norm
    contraction surrogate on A.
    """
    _require_cvxpy()
    X_t2 = _as_2d_time(X_t)
    U_t2 = _as_2d_time(U_t)
    X_tp12 = _as_2d_time(X_tp1)
    T = _validate_same_time_length(X_t2, U_t2, X_tp12)

    n_x = X_t2.shape[0]
    n_u = U_t2.shape[0]
    Z = np.vstack([X_t2, U_t2, np.ones((1, T))])

    Theta = cp.Variable((n_x, n_x + n_u + 1))
    A = Theta[:, :n_x]
    B = Theta[:, n_x : n_x + n_u]
    d = Theta[:, -1]

    resid = X_tp12 - Theta @ Z

    constraints: List[cp.Constraint] = [cp.norm(A, 2) <= 1.0 - eps_A]

    if C_theta is not None or h_theta is not None:
        if C_theta is None or h_theta is None:
            raise ValueError("Both C_theta and h_theta must be provided together.")
        C_theta = np.asarray(C_theta, dtype=float)
        h_theta = np.asarray(h_theta, dtype=float).reshape(-1)
        theta_vec = cp.reshape(Theta, (n_x * (n_x + n_u + 1),), order="F")
        constraints.append(C_theta @ theta_vec <= h_theta)

    objective = cp.Minimize(huber_sum(resid, tau_x) + rho_theta * cp.sum_squares(Theta))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    result: Dict[str, Any] = {"status": prob.status, "obj": prob.value}
    if prob.status in OPT_OK and Theta.value is not None:
        theta_val = np.asarray(Theta.value, dtype=float)
        result.update(
            {
                "Theta": theta_val,
                "A": theta_val[:, :n_x],
                "B": theta_val[:, n_x : n_x + n_u],
                "d": theta_val[:, -1].reshape(-1, 1),
            }
        )
    else:
        result.update({"Theta": None, "A": None, "B": None, "d": None})
    return result



def fit_alpha_gate(
    alpha_t: ArrayLike,
    alpha_tp1: ArrayLike,
    omega_t: ArrayLike,
    chi_t: ArrayLike,
    tau_alpha: float = 1.0,
    solver: str = "SCS",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Fit the openness gate alpha_{t+1} = phi alpha_t + eta_w omega_t - eta_c chi_t + d."""
    _require_cvxpy()
    a_t = _as_1d(alpha_t)
    a_tp1 = _as_1d(alpha_tp1)
    w_t = _as_1d(omega_t)
    c_t = _as_1d(chi_t)
    if not (len(a_t) == len(a_tp1) == len(w_t) == len(c_t)):
        raise ValueError("All alpha-gate series must have the same length.")

    phi_alpha = cp.Variable()
    eta_omega = cp.Variable(nonneg=True)
    eta_chi = cp.Variable(nonneg=True)
    d_alpha = cp.Variable()

    resid = a_tp1 - (phi_alpha * a_t + eta_omega * w_t - eta_chi * c_t + d_alpha)
    constraints = [phi_alpha >= 0.0, phi_alpha <= 1.0]

    prob = cp.Problem(cp.Minimize(huber_sum(resid, tau_alpha)), constraints)
    prob.solve(solver=solver, verbose=verbose)

    result: Dict[str, Any] = {"status": prob.status}
    if prob.status in OPT_OK and phi_alpha.value is not None:
        phi_val = float(phi_alpha.value)
        result.update(
            {
                "phi_alpha": phi_val,
                "eta_omega": float(eta_omega.value),
                "eta_chi": float(eta_chi.value),
                "d_alpha": float(d_alpha.value),
                "lambda_hat": 1.0 - phi_val,
            }
        )
    else:
        result.update(
            {"phi_alpha": None, "eta_omega": None, "eta_chi": None, "d_alpha": None, "lambda_hat": None}
        )
    return result



def fit_excess_gate(
    e_t: ArrayLike,
    e_tp1: ArrayLike,
    alpha_t: ArrayLike,
    tau_e: float = 1.0,
    solver: str = "SCS",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Fit topology-excess dynamics e_{t+1} = phi_e e_t + psi_e alpha_t + noise."""
    _require_cvxpy()
    e_t1 = _as_1d(e_t)
    e_tp11 = _as_1d(e_tp1)
    a_t = _as_1d(alpha_t)
    if not (len(e_t1) == len(e_tp11) == len(a_t)):
        raise ValueError("All excess-dynamics series must have the same length.")

    phi_e = cp.Variable()
    psi_e = cp.Variable(nonneg=True)
    resid = e_tp11 - (phi_e * e_t1 + psi_e * a_t)

    constraints = [phi_e >= 0.0, phi_e <= 1.0]
    prob = cp.Problem(cp.Minimize(huber_sum(resid, tau_e)), constraints)
    prob.solve(solver=solver, verbose=verbose)

    result: Dict[str, Any] = {"status": prob.status}
    if prob.status in OPT_OK and phi_e.value is not None:
        phi_val = float(phi_e.value)
        psi_val = float(psi_e.value)
        result.update({"phi_e": phi_val, "psi_e": psi_val, "mu_hat": 1.0 - phi_val, "kappa_hat": psi_val})
    else:
        result.update({"phi_e": None, "psi_e": None, "mu_hat": None, "kappa_hat": None})
    return result



def fit_kappa_quantile(
    nu_t: ArrayLike,
    alpha_t: ArrayLike,
    q: float = 0.95,
    reg: float = 1e-8,
    solver: str = "SCS",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Fit a conservative upper-envelope quantile model nu_t ≈ kappa * alpha_t."""
    _require_cvxpy()
    nu = _as_1d(nu_t)
    alpha = _as_1d(alpha_t)
    if len(nu) != len(alpha):
        raise ValueError("nu_t and alpha_t must have the same length.")

    kappa = cp.Variable(nonneg=True)
    resid = nu - kappa * alpha
    prob = cp.Problem(cp.Minimize(pinball_sum(resid, q) + reg * cp.square(kappa)))
    prob.solve(solver=solver, verbose=verbose)

    result: Dict[str, Any] = {"status": prob.status}
    if prob.status in OPT_OK and kappa.value is not None:
        result["kappa"] = float(kappa.value)
    else:
        result["kappa"] = None
    return result



def fit_budget_quantile(
    strain_delta_t: ArrayLike,
    alpha_t: ArrayLike,
    nu_t: ArrayLike,
    q: float = 0.99,
    reg: float = 1e-8,
    solver: str = "SCS",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Fit a conservative quantile model for strain increments.

    Solves:
        strain_delta_t ≈ c0 + c_alpha * alpha_t + c_nu * nu_t
    with nonnegative coefficients.
    """
    _require_cvxpy()
    s = _as_1d(strain_delta_t)
    alpha = _as_1d(alpha_t)
    nu = _as_1d(nu_t)
    if not (len(s) == len(alpha) == len(nu)):
        raise ValueError("All budget-regression series must have the same length.")

    c0 = cp.Variable(nonneg=True)
    c_alpha = cp.Variable(nonneg=True)
    c_nu = cp.Variable(nonneg=True)

    pred = c0 + c_alpha * alpha + c_nu * nu
    resid = s - pred

    prob = cp.Problem(
        cp.Minimize(pinball_sum(resid, q) + reg * (cp.square(c_alpha) + cp.square(c_nu)))
    )
    prob.solve(solver=solver, verbose=verbose)

    result: Dict[str, Any] = {"status": prob.status}
    if prob.status in OPT_OK and c_alpha.value is not None and c_nu.value is not None:
        result.update({"c0": float(c0.value), "c_alpha": float(c_alpha.value), "c_nu": float(c_nu.value)})
    else:
        result.update({"c0": None, "c_alpha": None, "c_nu": None})
    return result


# ---------------------------------------------------------------------------
# Bootstrap stage
# ---------------------------------------------------------------------------


def _check_required_keys(data: Mapping[str, Any], keys: Sequence[str]) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(f"Missing required data keys: {missing}")



def bootstrap_pipeline(
    data: Mapping[str, Any],
    B: int = 200,
    block_len: int = 32,
    seed: int = 0,
    solver: str = "SCS",
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Run moving-block bootstrap and refit all model components on each resample."""
    _check_required_keys(
        data,
        [
            "X_t",
            "U_t",
            "X_tp1",
            "alpha_t",
            "alpha_tp1",
            "omega_t",
            "chi_t",
            "e_t",
            "e_tp1",
            "alpha_t_for_e",
            "nu_t",
            "strain_delta_t",
            "alpha_t_for_b",
            "nu_t_for_b",
        ],
    )

    rng = np.random.default_rng(seed)
    samples: List[Dict[str, Any]] = []

    X_t = _as_2d_time(data["X_t"])
    U_t = _as_2d_time(data["U_t"])
    X_tp1 = _as_2d_time(data["X_tp1"])

    T_H = _validate_same_time_length(X_t, U_t, X_tp1)
    T_a = len(_as_1d(data["alpha_t"]))
    T_e = len(_as_1d(data["e_t"]))
    T_b = len(_as_1d(data["strain_delta_t"]))

    # idx_a / idx_e / idx_b are drawn from T_a / T_e / T_b and reused on every
    # paired series fed into fit_*_gate below. A shorter paired series would
    # otherwise hit np.take IndexError mid-loop inside bootstrap_take, surfacing
    # as an opaque, intermittent crash. Validate every paired length up front
    # so misalignment becomes a clear input-contract error.
    _validate_paired_lengths(
        T_a,
        "alpha_t",
        {
            "alpha_tp1": data["alpha_tp1"],
            "omega_t": data["omega_t"],
            "chi_t": data["chi_t"],
        },
    )
    _validate_paired_lengths(
        T_e,
        "e_t",
        {
            "e_tp1": data["e_tp1"],
            "alpha_t_for_e": data["alpha_t_for_e"],
            "nu_t": data["nu_t"],
        },
    )
    _validate_paired_lengths(
        T_b,
        "strain_delta_t",
        {
            "alpha_t_for_b": data["alpha_t_for_b"],
            "nu_t_for_b": data["nu_t_for_b"],
        },
    )

    for _ in range(B):
        idx_H = moving_block_indices(T_H, block_len, rng)
        idx_a = moving_block_indices(T_a, block_len, rng)
        idx_e = moving_block_indices(T_e, block_len, rng)
        idx_b = moving_block_indices(T_b, block_len, rng)

        local_fit = fit_local_plant(
            bootstrap_take(X_t, idx_H),
            bootstrap_take(U_t, idx_H),
            bootstrap_take(X_tp1, idx_H),
            solver=solver,
            verbose=verbose,
        )
        alpha_fit = fit_alpha_gate(
            bootstrap_take(data["alpha_t"], idx_a),
            bootstrap_take(data["alpha_tp1"], idx_a),
            bootstrap_take(data["omega_t"], idx_a),
            bootstrap_take(data["chi_t"], idx_a),
            solver=solver,
            verbose=verbose,
        )
        excess_fit = fit_excess_gate(
            bootstrap_take(data["e_t"], idx_e),
            bootstrap_take(data["e_tp1"], idx_e),
            bootstrap_take(data["alpha_t_for_e"], idx_e),
            solver=solver,
            verbose=verbose,
        )
        kappa_fit = fit_kappa_quantile(
            bootstrap_take(data["nu_t"], idx_e),
            bootstrap_take(data["alpha_t_for_e"], idx_e),
            solver=solver,
            verbose=verbose,
        )
        budget_fit = fit_budget_quantile(
            bootstrap_take(data["strain_delta_t"], idx_b),
            bootstrap_take(data["alpha_t_for_b"], idx_b),
            bootstrap_take(data["nu_t_for_b"], idx_b),
            solver=solver,
            verbose=verbose,
        )

        fits = [local_fit, alpha_fit, excess_fit, kappa_fit, budget_fit]
        if all(f["status"] in OPT_OK for f in fits):
            samples.append(
                {
                    "A": np.asarray(local_fit["A"], dtype=float),
                    "B": np.asarray(local_fit["B"], dtype=float),
                    "d": np.asarray(local_fit["d"], dtype=float),
                    "phi_alpha": float(alpha_fit["phi_alpha"]),
                    "phi_e": float(excess_fit["phi_e"]),
                    "psi_e": float(excess_fit["psi_e"]),
                    "kappa": float(kappa_fit["kappa"]),
                    "c_alpha": float(budget_fit["c_alpha"]),
                    "c_nu": float(budget_fit["c_nu"]),
                }
            )

    if not samples:
        raise RuntimeError("No bootstrap fits solved successfully.")
    return samples



def one_sided_bounds(
    samples: Sequence[Mapping[str, Any]],
    delta_alpha: float = 0.05,
    delta_e: float = 0.05,
    delta_b: float = 0.05,
) -> Dict[str, float]:
    """Compute one-sided scalar confidence bounds from bootstrap samples.

    For kappa, the function uses the maximum of the excess-dynamics envelope
    quantile and the direct birth-mass quantile when both are available.
    """
    phi_alpha = [float(s["phi_alpha"]) for s in samples]
    phi_e = [float(s["phi_e"]) for s in samples]
    psi_e = [float(s["psi_e"]) for s in samples]
    c_alpha = [float(s["c_alpha"]) for s in samples]
    c_nu = [float(s["c_nu"]) for s in samples]
    direct_kappa = [float(s["kappa"]) for s in samples if "kappa" in s and s["kappa"] is not None]

    phi_alpha_ub = _safe_quantile(phi_alpha, 1.0 - delta_alpha)
    phi_e_ub = _safe_quantile(phi_e, 1.0 - delta_e)
    psi_e_ub = _safe_quantile(psi_e, 1.0 - delta_e)
    c_alpha_ub = _safe_quantile(c_alpha, 1.0 - delta_b)
    c_nu_ub = _safe_quantile(c_nu, 1.0 - delta_b)
    if direct_kappa:
        direct_kappa_ub = _safe_quantile(direct_kappa, 1.0 - delta_e)
        kappa_ub = max(psi_e_ub, direct_kappa_ub)
    else:
        direct_kappa_ub = np.nan
        kappa_ub = psi_e_ub

    return {
        "phi_alpha_ub": phi_alpha_ub,
        "phi_e_ub": phi_e_ub,
        "psi_e_ub": psi_e_ub,
        "direct_kappa_ub": direct_kappa_ub,
        "lambda_lb": 1.0 - phi_alpha_ub,
        "mu_lb": 1.0 - phi_e_ub,
        "kappa_ub": kappa_ub,
        "c_alpha_ub": c_alpha_ub,
        "c_nu_ub": c_nu_ub,
    }


# ---------------------------------------------------------------------------
# Robust optimization stage R1-R2
# ---------------------------------------------------------------------------


def solve_P_feasibility(
    A_scenarios: Sequence[np.ndarray],
    a_candidate: float,
    pmin: float = 1e-4,
    solver: str = "SCS",
    verbose: bool = False,
) -> Tuple[bool, Optional[np.ndarray], str]:
    """Check feasibility of the robust contraction certificate for a fixed a_candidate."""
    _require_cvxpy()
    if not A_scenarios:
        raise ValueError("A_scenarios must contain at least one matrix.")
    n_x = A_scenarios[0].shape[0]
    if any(A.shape != (n_x, n_x) for A in A_scenarios):
        raise ValueError("All A scenarios must have identical square shapes.")

    I = np.eye(n_x)
    P = cp.Variable((n_x, n_x), PSD=True)

    constraints: List[cp.Constraint] = [P >> pmin * I, cp.trace(P) == 1.0]
    for A in A_scenarios:
        A_const = np.asarray(A, dtype=float)
        constraints.append(A_const.T @ P @ A_const - (1.0 - a_candidate) * P << 0)

    prob = cp.Problem(cp.Minimize(0.0), constraints)
    prob.solve(solver=solver, verbose=verbose, warm_start=True)

    feasible = prob.status in OPT_OK and P.value is not None
    P_val = None
    if feasible and P.value is not None:
        P_val = 0.5 * (np.asarray(P.value, dtype=float) + np.asarray(P.value, dtype=float).T)
    return feasible, P_val, prob.status



def bisection_a_lower_bound(
    A_scenarios: Sequence[np.ndarray],
    a_lo: float = 0.0,
    a_hi: float = 0.999,
    tol: float = 1e-3,
    max_iter: int = 40,
    pmin: float = 1e-4,
    solver: str = "SCS",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Bisection search for the largest robustly feasible contraction rate lower bound."""
    # Bisection only ever tests strict midpoints 0.5*(a_lo+a_hi), never a_lo
    # itself. For a marginally-stable scenario whose feasible set is exactly
    # {a_lo} (e.g., A=I where a=0 is feasible but every a>0 is infeasible),
    # best_P would otherwise stay None and the function would raise despite
    # a_lo being a valid lower bound. Seed best_P with a_lo's certificate
    # when feasible; if a_lo itself is infeasible, no a>=a_lo can be either,
    # but preserve the existing late-RuntimeError behavior for compatibility.
    feasible_lo, P_lo, _ = solve_P_feasibility(
        A_scenarios=A_scenarios,
        a_candidate=a_lo,
        pmin=pmin,
        solver=solver,
        verbose=verbose,
    )
    best_P: Optional[np.ndarray] = P_lo if feasible_lo else None

    for _ in range(max_iter):
        a_mid = 0.5 * (a_lo + a_hi)
        feasible, P_val, _ = solve_P_feasibility(
            A_scenarios=A_scenarios,
            a_candidate=a_mid,
            pmin=pmin,
            solver=solver,
            verbose=verbose,
        )
        if feasible:
            a_lo = a_mid
            best_P = P_val
        else:
            a_hi = a_mid
        if (a_hi - a_lo) <= tol:
            break

    if best_P is None:
        raise RuntimeError("No feasible contraction certificate found for any tested a.")
    return {"a_lb": float(a_lo), "P": best_P}



def robust_b_upper_bounds(
    P: ArrayLike,
    a_lb: float,
    theta_scenarios: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    barV: float,
    alpha_bar: float,
    e_bar: float,
    omega_bar: float,
    weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    solver: str = "SCS",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Robust upper bounds on b0,b1,b2,b3 using the S-procedure LMI at box vertices."""
    _require_cvxpy()
    Pm = _as_float_array(P)
    if Pm.ndim != 2 or Pm.shape[0] != Pm.shape[1]:
        raise ValueError("P must be a square matrix.")
    n_x = Pm.shape[0]
    if len(weights) != 4:
        raise ValueError("weights must have length 4.")
    if not theta_scenarios:
        raise ValueError("theta_scenarios must contain at least one scenario.")

    vertices = box_vertices(alpha_bar, e_bar, omega_bar)
    b = cp.Variable(4, nonneg=True)  # [b0, b1, b2, b3]

    constraints: List[cp.Constraint] = []
    for A, B, d in theta_scenarios:
        A = _as_float_array(A)
        B = _as_float_array(B)
        d = _as_2d_time(d)
        if d.shape[1] != 1:
            raise ValueError("Scenario d must have shape (n_x, 1).")
        if A.shape != (n_x, n_x):
            raise ValueError("Each scenario A must match the dimensions of P.")
        if B.shape[0] != n_x or B.shape[1] != 3:
            raise ValueError("Each scenario B must have shape (n_x, 3).")

        M = A.T @ Pm @ A - (1.0 - a_lb) * Pm
        for u in vertices:
            alpha_v = float(u[0, 0])
            e_v = float(u[1, 0])
            omega_v = float(u[2, 0])
            c = B @ u + d
            q = A.T @ Pm @ c
            cPc = float((c.T @ Pm @ c).item())

            tau = cp.Variable(nonneg=True)
            r = cPc - b[0] - b[1] * alpha_v - b[2] * e_v - b[3] * omega_v

            LMI = cp.bmat(
                [
                    [M + tau * Pm, q],
                    [q.T, cp.reshape(r - tau * barV, (1, 1), order="F")],
                ]
            )
            constraints.append(LMI << 0)

    objective = cp.Minimize(np.asarray(weights, dtype=float) @ b)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose, warm_start=True)

    result: Dict[str, Any] = {"status": prob.status}
    if prob.status in OPT_OK and b.value is not None:
        b_val = np.asarray(b.value, dtype=float).reshape(-1)
        result.update(
            {
                "b0_ub": float(b_val[0]),
                "b1_ub": float(b_val[1]),
                "b2_ub": float(b_val[2]),
                "b3_ub": float(b_val[3]),
            }
        )
    else:
        result.update({"b0_ub": None, "b1_ub": None, "b2_ub": None, "b3_ub": None})
    return result


# ---------------------------------------------------------------------------
# End-to-end orchestration
# ---------------------------------------------------------------------------


def identify_rebus_bounds(
    data: Mapping[str, Any],
    B: int = 200,
    block_len: int = 32,
    deltas: Tuple[float, float, float] = (0.05, 0.05, 0.05),
    alpha_bar: float = 1.0,
    e_bar: float = 1.0,
    omega_bar: float = 1.0,
    barV: float = 1.0,
    solver: str = "SCS",
    verbose: bool = False,
) -> RebusBounds:
    """Run the complete REBUS identification and robust-bound synthesis pipeline.

    Required data keys
    ------------------
    X_t, U_t, X_tp1
    alpha_t, alpha_tp1, omega_t, chi_t
    e_t, e_tp1, alpha_t_for_e
    nu_t
    strain_delta_t, alpha_t_for_b, nu_t_for_b
    """
    _check_required_keys(
        data,
        [
            "X_t",
            "U_t",
            "X_tp1",
            "alpha_t",
            "alpha_tp1",
            "omega_t",
            "chi_t",
            "e_t",
            "e_tp1",
            "alpha_t_for_e",
            "nu_t",
            "strain_delta_t",
            "alpha_t_for_b",
            "nu_t_for_b",
        ],
    )

    nominal_local = fit_local_plant(data["X_t"], data["U_t"], data["X_tp1"], solver=solver, verbose=verbose)
    nominal_alpha = fit_alpha_gate(
        data["alpha_t"], data["alpha_tp1"], data["omega_t"], data["chi_t"], solver=solver, verbose=verbose
    )
    nominal_excess = fit_excess_gate(
        data["e_t"], data["e_tp1"], data["alpha_t_for_e"], solver=solver, verbose=verbose
    )
    nominal_budget = fit_budget_quantile(
        data["strain_delta_t"], data["alpha_t_for_b"], data["nu_t_for_b"], solver=solver, verbose=verbose
    )

    nominal_fits = [nominal_local, nominal_alpha, nominal_excess, nominal_budget]
    bad_statuses = [f["status"] for f in nominal_fits if f["status"] not in OPT_OK]
    if bad_statuses:
        raise RuntimeError(f"One or more nominal fits failed: {bad_statuses}")

    samples = bootstrap_pipeline(data, B=B, block_len=block_len, solver=solver, verbose=verbose)
    scalar_bounds = one_sided_bounds(samples, delta_alpha=deltas[0], delta_e=deltas[1], delta_b=deltas[2])

    A_scenarios = [np.asarray(nominal_local["A"], dtype=float)] + [np.asarray(s["A"], dtype=float) for s in samples]
    a_cert = bisection_a_lower_bound(A_scenarios, solver=solver, verbose=verbose)

    theta_scenarios: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = [
        (
            np.asarray(nominal_local["A"], dtype=float),
            np.asarray(nominal_local["B"], dtype=float),
            np.asarray(nominal_local["d"], dtype=float),
        )
    ]
    theta_scenarios.extend(
        [
            (np.asarray(s["A"], dtype=float), np.asarray(s["B"], dtype=float), np.asarray(s["d"], dtype=float))
            for s in samples
        ]
    )

    certified_a_lb = float(a_cert["a_lb"])
    b_bounds = {"status": "not_attempted"}
    b_a_lb = certified_a_lb
    for margin in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.25, 0.0):
        b_a_lb = certified_a_lb * margin
        b_bounds = robust_b_upper_bounds(
            P=a_cert["P"],
            a_lb=b_a_lb,
            theta_scenarios=theta_scenarios,
            barV=barV,
            alpha_bar=alpha_bar,
            e_bar=e_bar,
            omega_bar=omega_bar,
            solver=solver,
            verbose=verbose,
        )
        if b_bounds["status"] in OPT_OK:
            break
    if b_bounds["status"] not in OPT_OK:
        raise RuntimeError(f"Robust b-upper-bound solve failed: {b_bounds['status']}")

    return RebusBounds(
        a_lb=float(b_a_lb),
        P=np.asarray(a_cert["P"], dtype=float),
        lambda_lb=float(scalar_bounds["lambda_lb"]),
        mu_lb=float(scalar_bounds["mu_lb"]),
        kappa_ub=float(scalar_bounds["kappa_ub"]),
        c_alpha_ub=float(scalar_bounds["c_alpha_ub"]),
        c_nu_ub=float(scalar_bounds["c_nu_ub"]),
        b0_ub=float(b_bounds["b0_ub"]),
        b1_ub=float(b_bounds["b1_ub"]),
        b2_ub=float(b_bounds["b2_ub"]),
        b3_ub=float(b_bounds["b3_ub"]),
    )



def synthesize_supervisor_gains(
    bounds: Union[RebusBounds, Mapping[str, Any]],
    eta: float,
    eps_p: float = 1e-3,
    eps_q: float = 1e-3,
) -> SupervisorGains:
    """Synthesize composite-Lyapunov supervisor gains from one-sided bounds."""
    if isinstance(bounds, RebusBounds):
        a_lb = bounds.a_lb
        lambda_lb = bounds.lambda_lb
        mu_lb = bounds.mu_lb
        kappa_ub = bounds.kappa_ub
        b1_ub = bounds.b1_ub
        b2_ub = bounds.b2_ub
        b3_ub = bounds.b3_ub
    else:
        a_lb = float(bounds["a_lb"])
        lambda_lb = float(bounds["lambda_lb"])
        mu_lb = float(bounds["mu_lb"])
        kappa_ub = float(bounds["kappa_ub"])
        b1_ub = float(bounds["b1_ub"])
        b2_ub = float(bounds["b2_ub"])
        b3_ub = float(bounds["b3_ub"])

    if lambda_lb <= 0 or mu_lb <= 0:
        raise ValueError("lambda_lb and mu_lb must be strictly positive for gain synthesis.")

    q = (b2_ub + eps_q) / mu_lb
    p = (b1_ub + q * kappa_ub + eps_p) / lambda_lb
    delta_safe = min(a_lb, eps_q / q, eps_p / p)
    Gamma = b3_ub + p * eta
    return SupervisorGains(p=float(p), q=float(q), delta_safe=float(delta_safe), Gamma=float(Gamma))


# ---------------------------------------------------------------------------
# Built-in synthetic scaffold and standalone demo
# ---------------------------------------------------------------------------


def make_synthetic_rebus_data(
    T: int = 96,
    nx: int = 3,
    seed: int = 7,
) -> Tuple[Dict[str, Any], SyntheticTruth]:
    """Generate a small stable dataset compatible with identify_rebus_bounds."""
    if T < 16:
        raise ValueError("T must be at least 16 for a meaningful bootstrap scaffold.")
    if nx < 1:
        raise ValueError("nx must be positive.")

    rng = np.random.default_rng(seed)

    Q, _ = np.linalg.qr(rng.normal(size=(nx, nx)))
    eigs = np.linspace(0.55, 0.78, nx)
    A = Q @ np.diag(eigs) @ Q.T

    B_base = np.array(
        [
            [0.12, 0.05, 0.10],
            [0.04, 0.10, 0.06],
            [0.08, 0.03, 0.09],
        ],
        dtype=float,
    )
    if nx <= 3:
        B = B_base[:nx, :]
    else:
        B = np.vstack([B_base, rng.normal(scale=0.03, size=(nx - 3, 3))])
    d = rng.normal(scale=0.01, size=(nx, 1))

    phi_alpha = 0.78
    eta_omega = 0.35
    eta_chi = 0.18
    phi_e = 0.72
    psi_e = 0.20
    direct_kappa = 0.18
    c0 = 0.03
    c_alpha = 0.35
    c_nu = 0.50

    x = np.zeros((nx, T + 1), dtype=float)
    alpha = np.zeros(T + 1, dtype=float)
    e = np.zeros(T + 1, dtype=float)
    omega = np.zeros(T, dtype=float)
    chi = np.zeros(T, dtype=float)
    nu = np.zeros(T, dtype=float)
    strain_delta = np.zeros(T, dtype=float)

    alpha[0] = 0.08
    e[0] = 0.04
    surprise = 0.0

    for t in range(T):
        pulse = rng.uniform(0.6, 1.2) if rng.random() < 0.10 else 0.0
        surprise = 0.65 * surprise + rng.normal(scale=0.18) + pulse
        omega[t] = float(np.clip(max(0.0, surprise), 0.0, 1.0))

        chi[t] = float(0.12 + 0.08 * e[t] + 0.02 * np.linalg.norm(x[:, t]))

        alpha[t + 1] = float(
            np.clip(
                phi_alpha * alpha[t] + eta_omega * omega[t] - eta_chi * chi[t] + rng.normal(scale=0.015),
                0.0,
                1.0,
            )
        )
        nu[t] = float(np.clip(direct_kappa * alpha[t] + rng.normal(scale=0.01), 0.0, 1.0))
        e[t + 1] = float(np.clip(phi_e * e[t] + psi_e * alpha[t] + rng.normal(scale=0.01), 0.0, 1.0))

        u_t = np.array([[alpha[t]], [e[t]], [omega[t]]], dtype=float)
        x[:, [t + 1]] = A @ x[:, [t]] + B @ u_t + d + rng.normal(scale=0.02, size=(nx, 1))

        strain_delta[t] = float(max(0.0, c0 + c_alpha * alpha[t] + c_nu * nu[t] + rng.normal(scale=0.01)))

    data: Dict[str, Any] = {
        "X_t": x[:, :-1],
        "U_t": np.vstack([alpha[:-1], e[:-1], omega]),
        "X_tp1": x[:, 1:],
        "alpha_t": alpha[:-1],
        "alpha_tp1": alpha[1:],
        "omega_t": omega,
        "chi_t": chi,
        "e_t": e[:-1],
        "e_tp1": e[1:],
        "alpha_t_for_e": alpha[:-1],
        "nu_t": nu,
        "strain_delta_t": strain_delta,
        "alpha_t_for_b": alpha[:-1],
        "nu_t_for_b": nu,
        "alpha_bar": 1.0,
        "e_bar": 1.0,
        "omega_bar": 1.0,
        "barV": float(1.25 * np.max(np.sum(x[:, :-1] ** 2, axis=0)) + 1e-6),
    }

    truth = SyntheticTruth(
        A=A,
        B=B,
        d=d,
        phi_alpha=phi_alpha,
        eta_omega=eta_omega,
        eta_chi=eta_chi,
        phi_e=phi_e,
        psi_e=psi_e,
        direct_kappa=direct_kappa,
        c0=c0,
        c_alpha=c_alpha,
        c_nu=c_nu,
    )
    return data, truth


def _bounds_to_summary(bounds: RebusBounds) -> Dict[str, float]:
    return {
        "a_lb": float(bounds.a_lb),
        "lambda_lb": float(bounds.lambda_lb),
        "mu_lb": float(bounds.mu_lb),
        "kappa_ub": float(bounds.kappa_ub),
        "c_alpha_ub": float(bounds.c_alpha_ub),
        "c_nu_ub": float(bounds.c_nu_ub),
        "b0_ub": float(bounds.b0_ub),
        "b1_ub": float(bounds.b1_ub),
        "b2_ub": float(bounds.b2_ub),
        "b3_ub": float(bounds.b3_ub),
        "P_trace": float(np.trace(bounds.P)),
        "P_min_eig": float(np.min(np.linalg.eigvalsh(bounds.P))),
    }


def _truth_to_summary(truth: SyntheticTruth) -> Dict[str, float]:
    return {
        "phi_alpha": float(truth.phi_alpha),
        "phi_e": float(truth.phi_e),
        "direct_kappa": float(truth.direct_kappa),
        "c_alpha": float(truth.c_alpha),
        "c_nu": float(truth.c_nu),
    }


def run_demo(
    T: int = 96,
    nx: int = 3,
    seed: int = 7,
    B: int = 20,
    block_len: int = 16,
    eta: float = 0.25,
    solver: str = "SCS",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a small end-to-end synthetic demo using only this module."""
    _require_cvxpy()
    data, truth = make_synthetic_rebus_data(T=T, nx=nx, seed=seed)
    bounds = identify_rebus_bounds(
        data,
        B=B,
        block_len=block_len,
        alpha_bar=float(data["alpha_bar"]),
        e_bar=float(data["e_bar"]),
        omega_bar=float(data["omega_bar"]),
        barV=float(data["barV"]),
        solver=solver,
        verbose=verbose,
    )
    gains = synthesize_supervisor_gains(bounds, eta=eta)
    return {
        "bounds": bounds,
        "gains": gains,
        "truth": truth,
        "data": data,
        "summary": {
            "bounds": _bounds_to_summary(bounds),
            "gains": {
                "p": float(gains.p),
                "q": float(gains.q),
                "delta_safe": float(gains.delta_safe),
                "Gamma": float(gains.Gamma),
            },
            "truth": _truth_to_summary(truth),
        },
    }


def synthetic_smoke_test(
    T: int = 40,
    nx: int = 2,
    seed: int = 5,
    B: int = 6,
    block_len: int = 8,
    eta: float = 0.25,
    solver: str = "SCS",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a small self-contained smoke test for this module."""
    data, truth = make_synthetic_rebus_data(T=T, nx=nx, seed=seed)
    required = {
        "X_t", "U_t", "X_tp1", "alpha_t", "alpha_tp1", "omega_t", "chi_t",
        "e_t", "e_tp1", "alpha_t_for_e", "nu_t", "strain_delta_t",
        "alpha_t_for_b", "nu_t_for_b", "alpha_bar", "e_bar", "omega_bar", "barV",
    }
    checks: Dict[str, Any] = {
        "required_keys_present": required.issubset(data.keys()),
        "X_t_shape": tuple(data["X_t"].shape),
        "U_t_shape": tuple(data["U_t"].shape),
        "X_tp1_shape": tuple(data["X_tp1"].shape),
        "barV_positive": float(data["barV"]) > 0.0,
        "truth_finite": bool(np.all(np.isfinite(truth.A))),
    }
    if cp is None:
        return {
            "status": "skipped_no_cvxpy",
            "checks": checks,
            "data": data,
            "truth": truth,
        }

    result = run_demo(T=T, nx=nx, seed=seed, B=B, block_len=block_len, eta=eta, solver=solver, verbose=verbose)
    bounds = result["bounds"]
    gains = result["gains"]
    checks.update({
        "a_lb_nonnegative": float(bounds.a_lb) >= 0.0,
        "lambda_lb_positive": float(bounds.lambda_lb) > 0.0,
        "mu_lb_positive": float(bounds.mu_lb) > 0.0,
        "kappa_ub_nonnegative": float(bounds.kappa_ub) >= 0.0,
        "p_positive": float(gains.p) > 0.0,
        "q_positive": float(gains.q) > 0.0,
    })
    return {
        "status": "passed",
        "checks": checks,
        "bounds": bounds,
        "gains": gains,
        "truth": truth,
        "data": data,
        "summary": result["summary"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the built-in synthetic demo."""
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(description="Run a synthetic REBUS identification demo.")
    parser.add_argument("--T", type=int, default=96, help="Number of one-step transitions.")
    parser.add_argument("--nx", type=int, default=3, help="State dimension.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--B", type=int, default=20, help="Bootstrap resamples.")
    parser.add_argument("--block-len", type=int, default=16, help="Moving-block bootstrap length.")
    parser.add_argument("--eta", type=float, default=0.25, help="Supervisor gain-synthesis eta.")
    parser.add_argument("--solver", type=str, default="SCS", help="CVXPY solver name.")
    parser.add_argument("--verbose", action="store_true", help="Enable solver verbosity.")
    parser.add_argument("--smoke-test", action="store_true", help="Run the built-in smoke test instead of the full demo.")
    args = parser.parse_args(argv)

    if args.smoke_test:
        result = synthetic_smoke_test(
            T=args.T,
            nx=args.nx,
            seed=args.seed,
            B=args.B,
            block_len=args.block_len,
            eta=args.eta,
            solver=args.solver,
            verbose=args.verbose,
        )
        pprint(result)
        return 0

    result = run_demo(
        T=args.T,
        nx=args.nx,
        seed=args.seed,
        B=args.B,
        block_len=args.block_len,
        eta=args.eta,
        solver=args.solver,
        verbose=args.verbose,
    )
    print("Synthetic REBUS identification demo")
    print("----------------------------------")
    pprint(result["summary"])
    return 0


# ---------------------------------------------------------------------------
# Optional example scaffold
# ---------------------------------------------------------------------------


def example_usage() -> None:
    """Print a minimal usage sketch for the module."""
    print(
        """
Example workflow
----------------
1) Build closed-mode feature matrix Y_closed and estimate y_star:
       y_star = estimate_homeostatic_center(Y_closed, method='median')

2) Build centered state trajectories X_t, X_tp1 and inputs U_t = [alpha_t; e_t; omega_t].

3) Prepare the data dictionary:
       data = {
           'X_t': X_t,
           'U_t': U_t,
           'X_tp1': X_tp1,
           'alpha_t': alpha_t,
           'alpha_tp1': alpha_tp1,
           'omega_t': omega_t,
           'chi_t': chi_t,
           'e_t': e_t,
           'e_tp1': e_tp1,
           'alpha_t_for_e': alpha_t_for_e,
           'nu_t': nu_t,
           'strain_delta_t': strain_delta_t,
           'alpha_t_for_b': alpha_t_for_b,
           'nu_t_for_b': nu_t_for_b,
       }

4) Identify robust bounds:
       bounds = identify_rebus_bounds(
           data,
           B=300,
           block_len=32,
           deltas=(0.05, 0.05, 0.05),
           alpha_bar=alpha_bar,
           e_bar=e_bar,
           omega_bar=omega_bar,
           barV=barV,
       )

5) Synthesize supervisor gains:
       gains = synthesize_supervisor_gains(bounds, eta=eta)

6) Run the built-in synthetic demo:
       result = run_demo(T=96, nx=3, B=20, block_len=16, eta=0.25)
       print(result["summary"])

7) Run the built-in smoke test:
       result = synthetic_smoke_test(T=40, nx=2, B=6, block_len=8, eta=0.25)
       print(result["checks"])
        """.strip()
    )

__all__ = [
    "RebusBounds",
    "SupervisorGains",
    "SyntheticTruth",
    "estimate_homeostatic_center",
    "center_state",
    "huber_sum",
    "pinball_sum",
    "moving_block_indices",
    "bootstrap_take",
    "box_vertices",
    "fit_local_plant",
    "fit_alpha_gate",
    "fit_excess_gate",
    "fit_kappa_quantile",
    "fit_budget_quantile",
    "bootstrap_pipeline",
    "one_sided_bounds",
    "solve_P_feasibility",
    "bisection_a_lower_bound",
    "robust_b_upper_bounds",
    "identify_rebus_bounds",
    "synthesize_supervisor_gains",
    "make_synthetic_rebus_data",
    "run_demo",
    "synthetic_smoke_test",
    "example_usage",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
