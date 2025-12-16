# bioreactor_core.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence, Callable, Tuple
from scipy.optimize import least_squares, minimize_scalar

print("bioreactor_core imported.")

# ---------- Core parameter type ----------

@dataclass
class ParameterVector:
    """Generic parameter vector in R^d."""
    values: np.ndarray

    def __post_init__(self):
        self.values = np.asarray(self.values, float).reshape(-1)

    @property
    def d(self) -> int:
        return self.values.size


BasisFunction = Callable[[float], ParameterVector]


# ---------- Baranyi–Roberts model ----------

def baranyi_roberts_model(t, C0, Cmax, mu, lam):
    """
    Baranyi–Roberts model in log space, then exponentiated back to C(t).
    """
    t = np.asarray(t, float)

    C0   = max(C0,   1e-6)
    Cmax = max(Cmax, 1e-6)
    mu   = max(mu,   1e-6)
    lam  = max(lam,  0.0)

    y0   = np.log(C0)
    ymax = np.log(Cmax)

    A = t + (1/mu) * np.log(1 + np.exp(-mu*t) * (np.exp(-mu*lam) - 1))
    exp_muA = np.exp(mu*A)
    y = y0 + mu*A - np.log(1 + (exp_muA - 1) / np.exp(ymax - y0))
    return np.exp(y)


# ---------- GrowthFit container ----------

@dataclass
class GrowthFit:
    model_name: str
    volume: float
    params: dict
    t: np.ndarray
    C_obs: np.ndarray
    C_fit: np.ndarray
    residuals: np.ndarray

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2)))


# ---------- Baranyi parameter fitting ----------

def _baranyi_bounds(t: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list]:
    C = np.asarray(C, float)
    Cmax_data = float(np.max(C))
    lb = np.array([1e-3, 0.5*Cmax_data, 1e-4, 0.0])
    ub = np.array([0.9*Cmax_data, np.inf, 1.0, 40.0])
    names = ["C0", "Cmax", "mu", "lam"]
    return lb, ub, names


def fit_baranyi_robust(t: Sequence[float], C: Sequence[float], volume: float) -> GrowthFit:
    t = np.asarray(t, float)
    C = np.asarray(C, float)

    lb, ub, names = _baranyi_bounds(t, C)
    Cmax_data = float(np.max(C))
    Cmin_data = float(np.min(C))

    lam_target = 10.0
    lam_width  = 10.0

    t_min, t_max = t.min(), t.max()
    t_mid = 0.5*(t_min + t_max)
    mid_idx = np.argmin(np.abs(t - t_mid))
    if 0 < mid_idx < len(t)-1:
        slope = (C[mid_idx+1] - C[mid_idx-1])/(t[mid_idx+1]-t[mid_idx-1])
    else:
        slope = (C[-1] - C[0])/max(t_max - t_min, 1e-6)
    mu0 = max(slope/Cmax_data, 1e-4)

    C0_list   = [max(1e-3, 0.5*Cmin_data), max(1e-3, Cmin_data)]
    Cmax_list = [Cmax_data, 1.05*Cmax_data]
    mu_list   = [0.5*mu0, mu0, 1.5*mu0]
    lam_list  = [0.0, 5.0, 10.0, 20.0]

    def residuals_with_penalty(p):
        C0, Cmax, mu, lam = p
        C_model = baranyi_roberts_model(t, C0, Cmax, mu, lam)
        res_data = C_model - C
        res_lam  = (lam - lam_target)/lam_width
        return np.concatenate([res_data, [res_lam]])

    best_cost = np.inf
    best_p    = None
    best_Cfit = None

    for C0g in C0_list:
        for Cmaxg in Cmax_list:
            for mug in mu_list:
                for lamg in lam_list:
                    p0 = np.array([C0g, Cmaxg, mug, lamg], float)
                    p0 = np.clip(p0, lb, ub)
                    try:
                        out = least_squares(
                            residuals_with_penalty,
                            p0,
                            bounds=(lb, ub),
                            loss="soft_l1",
                            max_nfev=400,
                        )
                    except Exception:
                        continue
                    if not out.success:
                        continue
                    Cfit = baranyi_roberts_model(t, *out.x)
                    cost = np.mean((Cfit - C)**2)
                    if cost < best_cost:
                        best_cost = cost
                        best_p    = out.x
                        best_Cfit = Cfit

    if best_p is None:
        raise RuntimeError("Robust Baranyi fit failed.")

    params = {name: float(val) for name, val in zip(names, best_p)}
    return GrowthFit("baranyi", float(volume), params, t, C, best_Cfit, best_Cfit - C)


# ---------- Data loader ----------

def load_bioreactor_data(path: str = "bioreactor_scales.csv"):
    """
    Load bioreactor data from a long-format CSV with columns:
        scale, V, t, C

    Returns:
        df   – full DataFrame
        data – dict with keys "10L", "100L", "4m3", "100m3"
    """
    df = pd.read_csv(path)

    print("Loaded file:", path)
    print("Head:")
    print(df.head())

    def get_scale(vol: float):
        sub = df[df["V"] == vol].copy()
        sub = sub.dropna(subset=["t", "C"]).sort_values("t")
        return {
            "t": sub["t"].to_numpy(float),
            "C": sub["C"].to_numpy(float),
            "V": float(vol),
        }

    data = {
        "10L":    get_scale(10.0),
        "100L":   get_scale(100.0),
        "4m3":    get_scale(4000.0),
        "100m3":  get_scale(100000.0),
    }

    print("Data loaded OK.")
    return df, data


# ---------- Parameter utilities ----------

def extract_params_3d(fit: GrowthFit) -> ParameterVector:
    """Extract only (Cmax, mu, lam) as a 3D vector."""
    return ParameterVector(np.array([
        fit.params["Cmax"],
        fit.params["mu"],
        fit.params["lam"],
    ], dtype=float))


def choose_C0_for_large_scale(
    fit_10: GrowthFit,
    fit_100: GrowthFit,
    fit_4m3: GrowthFit,
    mode: str = "4m3",
) -> float:
    """
    Strategy for C0 at 100 m³.
    """
    if mode == "4m3":
        return float(fit_4m3.params["C0"])
    elif mode == "100L":
        return float(fit_100.params["C0"])
    elif mode == "mean":
        return float(
            (fit_10.params["C0"] + fit_100.params["C0"] + fit_4m3.params["C0"]) / 3.0
        )
    else:
        raise ValueError(f"Unknown C0 mode: {mode}")


def fit_C0_given_others(t, C_obs, Cmax, mu, lam, C0_init=3.0):
    """
    Fit C0 only, holding (Cmax, mu, lam) fixed.
    """
    t = np.asarray(t, float)
    C_obs = np.asarray(C_obs, float)

    def obj(C0):
        C_pred = baranyi_roberts_model(t, C0, Cmax, mu, lam)
        return np.mean((C_pred - C_obs)**2)

    res = minimize_scalar(obj, bounds=(1e-3, 0.9*Cmax), method="bounded")
    if not res.success:
        return C0_init
    return float(res.x)
