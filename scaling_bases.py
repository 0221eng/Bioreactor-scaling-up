# scaling_bases.py

import numpy as np
from typing import Sequence, Tuple
from bioreactor_core import ParameterVector, BasisFunction

print("scaling_bases imported.")


def build_power_log_vector_bases(
    V_list: Sequence[float],
    P_list: Sequence[ParameterVector],
):
    """
    Build vector-valued power and log bases for a d-dimensional parameter vector.
    Works for d = 3 (Cmax, mu, lam) or d = 4 (C0, Cmax, mu, lam), etc.
    """
    Vs   = np.asarray(V_list, float).reshape(-1)
    Pmat = np.vstack([p.values for p in P_list])   # (n, d)
    n, d = Pmat.shape
    logV = np.log(Vs)

    a_pow     = np.zeros(d)
    b_pow     = np.zeros(d)
    alpha_log = np.zeros(d)
    beta_log  = np.zeros(d)

    for k in range(d):
        # Power law: log(P_k) = log(a_k) + b_k log(V)
        y_pow = np.log(Pmat[:, k])
        A_pow = np.column_stack([np.ones(n), logV])
        coeff_pow, *_ = np.linalg.lstsq(A_pow, y_pow, rcond=None)
        log_a, b = coeff_pow
        a_pow[k] = np.exp(log_a)
        b_pow[k] = b

        # Log-linear: P_k = alpha_k + beta_k log(V)
        y_log = Pmat[:, k]
        A_log = np.column_stack([np.ones(n), logV])
        coeff_log, *_ = np.linalg.lstsq(A_log, y_log, rcond=None)
        alpha, beta  = coeff_log
        alpha_log[k] = alpha
        beta_log[k]  = beta

    def f_power(V: float) -> ParameterVector:
        V = float(V)
        vec = a_pow * (V**b_pow)
        vec = np.maximum(vec, 1e-6)
        return ParameterVector(vec)

    def f_log(V: float) -> ParameterVector:
        V = float(V)
        vec = alpha_log + beta_log*np.log(V)
        vec = np.maximum(vec, 1e-6)
        return ParameterVector(vec)

    return f_power, f_log


    def f_power(V: float) -> ParameterVector:
        V = float(V)
        vec = a_pow * (V**b_pow)
        vec = np.maximum(vec, 1e-6)
        return ParameterVector(vec)

    def f_log(V: float) -> ParameterVector:
        V = float(V)
        vec = alpha_log + beta_log*np.log(V)
        vec = np.maximum(vec, 1e-6)
        return ParameterVector(vec)

    return f_power, f_log


def build_piecewise_log_linear_basis_3D(
    V_list: Sequence[float],
    P_list: Sequence[ParameterVector],
) -> BasisFunction:
    """
    Piecewise-log-linear basis for (Cmax, mu, lam).
    """
    Vs   = np.asarray(V_list, float)
    Pmat = np.vstack([p.values for p in P_list])   # (3, 3)

    idx   = np.argsort(Vs)
    Vs    = Vs[idx]
    Pmat  = Pmat[idx]
    logV  = np.log(Vs)

    logP = np.log(Pmat + 1e-12)

    m1 = (logP[1] - logP[0]) / (logV[1] - logV[0])
    m2 = (logP[2] - logP[1]) / (logV[2] - logV[1])

    b1 = logP[0] - m1 * logV[0]
    b2 = logP[1] - m2 * logV[1]

    V_break = Vs[1]
    logV_break = logV[1]

    def f_pwl(V: float) -> ParameterVector:
        LV = np.log(float(V))
        if LV <= logV_break:
            vec_log = m1 * LV + b1
        else:
            vec_log = m2 * LV + b2
        vec = np.exp(vec_log)
        vec = np.maximum(vec, 1e-6)
        return ParameterVector(vec)

    return f_pwl


def make_cfd_surrogate_4D_calibrated(
        V_list: Sequence[float],
        P4_list: Sequence[ParameterVector],
        V_ref: float = 10.0,
) -> BasisFunction:
    """
    Calibrated CFD-inspired surrogate for a 4D parameter vector:
        (C0, Cmax, mu, lam).

    - C0: kept constant = C0_ref (no explicit scaling).
    - Cmax, mu, lam: scaled as before using log-laws.
    """
    Vs = np.asarray(V_list, float).reshape(-1)
    Pmat = np.vstack([p.values for p in P4_list])  # (n, 4)

    C0_vals = Pmat[:, 0]
    Cmax_vals = Pmat[:, 1]
    mu_vals = Pmat[:, 2]
    lam_vals = Pmat[:, 3]

    # Reference point at V_ref
    idx_ref = np.argmin(np.abs(Vs - V_ref))
    C0_ref = float(C0_vals[idx_ref])
    Cmax_ref = float(Cmax_vals[idx_ref])
    mu_ref = float(mu_vals[idx_ref])
    lam_ref = float(lam_vals[idx_ref])

    log_scale = np.log(Vs / V_ref + 1e-12)

    # ---- mu scaling: log(mu/mu_ref) = alpha * log(V/V_ref)
    log_mu_ratio = np.log(mu_vals / mu_ref)
    A_mu = log_scale.reshape(-1, 1)
    alpha_vec, *_ = np.linalg.lstsq(A_mu, log_mu_ratio, rcond=None)
    alpha = float(alpha_vec[0])

    # ---- lam scaling: log(lam/lam_ref) = -beta * log(V/V_ref)
    log_lam_ratio = np.log(lam_vals / lam_ref)
    A_lam = -log_scale.reshape(-1, 1)
    beta_vec, *_ = np.linalg.lstsq(A_lam, log_lam_ratio, rcond=None)
    beta = float(beta_vec[0])

    # ---- Cmax scaling: (Cmax_ref - Cmax)/Cmax_ref ≈ gamma ln(V/V_ref)
    y_c = (Cmax_ref - Cmax_vals) / Cmax_ref
    A_c = log_scale.reshape(-1, 1)
    gamma_vec, *_ = np.linalg.lstsq(A_c, y_c, rcond=None)
    gamma = float(gamma_vec[0])

    print("Calibrated CFD exponents (4D):",
          f"alpha(mu)={alpha:.3f}",
          f"beta(lam)={beta:.3f}",
          f"gamma(Cmax)={gamma:.3f}")

    def f_cfd(V: float) -> ParameterVector:
        V = float(V)
        scale = max(V / V_ref, 1e-9)

        C0 = C0_ref
        Cmax = Cmax_ref * (1.0 - gamma * np.log(scale))
        mu = mu_ref * (scale ** alpha)
        lam = lam_ref * (scale ** (-beta))

        vec = np.array([
            max(C0, 1e-6),
            max(Cmax, 1e-6),
            max(mu, 1e-6),
            max(lam, 0.0),
        ], float)
        return ParameterVector(vec)

    return f_cfd

