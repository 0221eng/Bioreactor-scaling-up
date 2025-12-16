# hybrid_model.py

import numpy as np
from dataclasses import dataclass
from typing import List, Sequence, Optional
from scipy.optimize import minimize

from bioreactor_core import ParameterVector, BasisFunction

print("hybrid_model imported.")


@dataclass
class HybridModel:
    basis_functions: List[BasisFunction]
    weights: np.ndarray

    def __post_init__(self):
        w = np.clip(self.weights, 0.0, None)
        s = w.sum()
        if s == 0:
            raise ValueError("All weights are zero.")
        self.weights = w / s

    def predict(self, V: float) -> ParameterVector:
        d = self.basis_functions[0](V).d
        acc = np.zeros(d, float)
        for wj, fj in zip(self.weights, self.basis_functions):
            acc += wj * fj(V).values
        return ParameterVector(acc)


def fit_convex_weights_multifidelity(
    volumes: Sequence[float],
    observed_params: Sequence[ParameterVector],
    basis_functions: List[BasisFunction],
    cfd_index: int,
    lambda_cfd: float = 0.5,
    reg_lambda: float = 0.0,
    w0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Multi-fidelity convex fit:

      min_w  0.5 ||F w - y||^2
           + 0.5*reg_lambda ||w - w0||^2
           + 0.5*lambda_cfd * w[cfd_index]^2

      s.t.   w_j >= 0, sum_j w_j = 1
    """
    Vs = np.asarray(volumes, float).reshape(-1)
    Ps = np.vstack([p.values for p in observed_params])  # (n, d)
    n, d = Ps.shape
    m = len(basis_functions)

    F = np.zeros((n*d, m))
    y = Ps.reshape(-1)

    for i, V in enumerate(Vs):
        for j, fj in enumerate(basis_functions):
            F[i*d:(i+1)*d, j] = fj(V).values

    if w0 is None:
        w0 = np.ones(m, float) / m
    else:
        w0 = np.asarray(w0, float).reshape(-1)

    def objective(w):
        resid = F @ w - y
        val = 0.5*np.dot(resid, resid)
        if reg_lambda > 0:
            diff = w - w0
            val += 0.5*reg_lambda*np.dot(diff, diff)
        if lambda_cfd > 0:
            val += 0.5*lambda_cfd*(w[cfd_index]**2)
        return float(val)

    cons   = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)]*m

    out = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if not out.success:
        raise RuntimeError("Multi-fidelity optimisation failed: " + out.message)

    w = np.clip(out.x, 0.0, None)
    return w / w.sum()
