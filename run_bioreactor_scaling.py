import numpy as np
import pandas as pd
from bioreactor_core import (
    load_bioreactor_data,
    fit_baranyi_robust,
    baranyi_roberts_model,
    ParameterVector,
)
from scaling_bases import (
    build_power_log_vector_bases,
    build_piecewise_log_linear_basis_3D,
    make_cfd_surrogate_4D_calibrated,
)
from hybrid_model import HybridModel, fit_convex_weights_multifidelity
from ml_baselines import run_ml_baselines
from plotting import (
    plot_model_fits,
    plot_parameter_trends,
    plot_rmse_bars,
    plot_convex_weights,
    plot_residuals_all_scales,
    plot_bases_vs_hybrid_100m3,
    # NEW: single hybrid 4D plot
    plot_time_vs_conc_hybrid4D,
)

if __name__ == "__main__":
    # 1. Load data
    df_raw, data = load_bioreactor_data("bioreactor_scales.csv")

    # 2. Fit Baranyi at each scale
    fit_10    = fit_baranyi_robust(data["10L"]["t"],   data["10L"]["C"],   data["10L"]["V"])
    fit_100   = fit_baranyi_robust(data["100L"]["t"],  data["100L"]["C"],  data["100L"]["V"])
    fit_4m3   = fit_baranyi_robust(data["4m3"]["t"],   data["4m3"]["C"],   data["4m3"]["V"])
    fit_100m3 = fit_baranyi_robust(data["100m3"]["t"], data["100m3"]["C"], data["100m3"]["V"])

    print("Fitted Baranyi parameters (all four scales):")
    for label, fit in [("10 L", fit_10), ("100 L", fit_100),
                       ("4 m³", fit_4m3), ("100 m³", fit_100m3)]:
        print(f"  {label}: {fit.params}   (RMSE = {fit.rmse:.3f} g/L)")

    # 3. Extract 3D vectors
    # 3. Build 4D parameter vectors (C0, Cmax, mu, lam)
    P4_10 = ParameterVector(np.array([
        fit_10.params["C0"],
        fit_10.params["Cmax"],
        fit_10.params["mu"],
        fit_10.params["lam"],
    ], float))

    P4_100 = ParameterVector(np.array([
        fit_100.params["C0"],
        fit_100.params["Cmax"],
        fit_100.params["mu"],
        fit_100.params["lam"],
    ], float))

    P4_4m3 = ParameterVector(np.array([
        fit_4m3.params["C0"],
        fit_4m3.params["Cmax"],
        fit_4m3.params["mu"],
        fit_4m3.params["lam"],
    ], float))

    # For comparison only (not used in training)
    P4_100m3 = ParameterVector(np.array([
        fit_100m3.params["C0"],
        fit_100m3.params["Cmax"],
        fit_100m3.params["mu"],
        fit_100m3.params["lam"],
    ], float))

    V_10   = data["10L"]["V"]
    V_100  = data["100L"]["V"]
    V_4    = data["4m3"]["V"]
    V_100k = data["100m3"]["V"]

    V_calib  = [V_10, V_100, V_4]
    P4_calib = [P4_10, P4_100, P4_4m3]

    # 4. Bases
    f_power4, f_log4 = build_power_log_vector_bases(V_calib, P4_calib)
    f_pwl4           = build_piecewise_log_linear_basis_3D(V_calib, P4_calib)
    f_cfd4           = make_cfd_surrogate_4D_calibrated(V_calib, P4_calib, V_ref=V_10)

    basis_fns_4D = [f_power4, f_log4, f_pwl4, f_cfd4]
    cfd_index    = 3

    # 5. Multi-fidelity convex weights
    lambda_cfd = 0.05
    w0 = np.array([0.30, 0.30, 0.30, 0.10])
    reg_lambda = 0.10

    w_mf = fit_convex_weights_multifidelity(
        V_calib,
        P4_calib,
        basis_fns_4D,
        cfd_index=cfd_index,
        lambda_cfd=lambda_cfd,
        reg_lambda=reg_lambda,
        w0=w0,
    )
    labels = ["Power", "Log", "PWL", "CFD"]
    for name, w in zip(labels, w_mf):
        print(f"{name}: {w:.6f} ({w * 100:.2f}%)")

    # 6. Hybrid 4D model and prediction at 100 m³
    hybrid4D = HybridModel(basis_fns_4D, w_mf)
    P4_100m3_pred = hybrid4D.predict(V_100k)  # (C0, Cmax, mu, lam)

    print("Full predicted parameters at 100 m³ (hybrid 4D):")
    for name, val in zip(["C0", "Cmax", "mu", "lam"], P4_100m3_pred.values):
        print(f"  {name} = {val:.4f}")
    # 7. RMSE at 100 m³
    t_eval = data["100m3"]["t"]
    C_exp  = data["100m3"]["C"]

    # Evaluate hybrid 4D curve at 100 m³
    C_pred = baranyi_roberts_model(
        t_eval,
        C0=P4_100m3_pred.values[0],
        Cmax=P4_100m3_pred.values[1],
        mu=P4_100m3_pred.values[2],
        lam=P4_100m3_pred.values[3],
    )

    rmse_hybrid_100m3 = float(np.sqrt(np.mean((C_pred - C_exp) ** 2)))
    print(f"RMSE at 100 m³ (hybrid 4D): {rmse_hybrid_100m3:.3f} g/L")

    # 9. Parameter comparison table (optional, still in main)
    params_direct_100m3 = fit_100m3.params
    params_hybrid_4D = {
        "C0": float(P4_100m3_pred.values[0]),
        "Cmax": float(P4_100m3_pred.values[1]),
        "mu": float(P4_100m3_pred.values[2]),
        "lam": float(P4_100m3_pred.values[3]),
    }

    df_params = pd.DataFrame({
        "Direct Fit 100 m³": params_direct_100m3,
        "Hybrid 4D": params_hybrid_4D,
    })

    print("\nParameter comparison at 100 m³:")
    print(df_params)

    fits_dict = {
        "10 L": fit_10,
        "100 L": fit_100,
        "4 m³": fit_4m3,
        "100 m³": fit_100m3,
    }
    # FIGURE 5 — Residuals vs time (modular function call)
    plot_residuals_all_scales(fits_dict)

    # 10. ML baselines
    V_train = np.array([V_10, V_100, V_4]).reshape(-1, 1)
    P_train = np.array([
        [fit_10.params["Cmax"],  fit_10.params["mu"],  fit_10.params["lam"]],
        [fit_100.params["Cmax"], fit_100.params["mu"], fit_100.params["lam"]],
        [fit_4m3.params["Cmax"], fit_4m3.params["mu"], fit_4m3.params["lam"]],
    ])
    P_true_100m3 = np.array([
        fit_100m3.params["Cmax"],
        fit_100m3.params["mu"],
        fit_100m3.params["lam"],
    ])
    V_test = np.array([[V_100k]])

    rmse_dict = run_ml_baselines(V_train, P_train, V_test, P_true_100m3)

    # 11. Figures
    t10, C10     = data["10L"]["t"],   data["10L"]["C"]
    t100, C100   = data["100L"]["t"],  data["100L"]["C"]
    t4, C4       = data["4m3"]["t"],   data["4m3"]["C"]
    t100m3, C100m3 = data["100m3"]["t"], data["100m3"]["C"]

    fits_dict = {
        "10 L":  fit_10,
        "100 L": fit_100,
        "4 m³":  fit_4m3,
        "100 m³": fit_100m3,
    }
    series_dict = {
        "10 L":  (t10, C10),
        "100 L": (t100, C100),
        "4 m³":  (t4, C4),
        "100 m³": (t100m3, C100m3),
    }

    plot_model_fits(series_dict, fits_dict)

    volumes = np.array([V_10, V_100, V_4, V_100k])
    Cmax_vals = [fit_10.params["Cmax"],  fit_100.params["Cmax"],
                 fit_4m3.params["Cmax"], fit_100m3.params["Cmax"]]
    mu_vals   = [fit_10.params["mu"],    fit_100.params["mu"],
                 fit_4m3.params["mu"],   fit_100m3.params["mu"]]
    lam_vals  = [fit_10.params["lam"],   fit_100.params["lam"],
                 fit_4m3.params["lam"],  fit_100m3.params["lam"]]

    plot_parameter_trends(volumes, Cmax_vals, mu_vals, lam_vals)
    plot_rmse_bars(rmse_dict, rmse_hybrid_100m3, fit_100m3.rmse)
    plot_convex_weights(w_mf)
    plot_time_vs_conc_hybrid4D(t_eval, C_exp, P4_100m3_pred, rmse_hybrid_100m3)
    # ============================================
    # FIGURE 6 — 100 m³: individual bases vs Hybrid vs experimental
    # ============================================

    plot_bases_vs_hybrid_100m3(
        t_eval=t_eval,
        C_exp=C_exp,
        V_100k=V_100k,
        basis_fns_4D=basis_fns_4D,
        P4_100m3_pred=P4_100m3_pred,
    )




