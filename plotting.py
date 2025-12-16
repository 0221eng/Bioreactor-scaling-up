# plotting.py

import numpy as np
import matplotlib.pyplot as plt

from bioreactor_core import baranyi_roberts_model, GrowthFit

print("plotting imported.")


def plot_model_fits(series_dict, fits_dict):
    plt.figure(figsize=(10, 8))
    for i, (label, (t, C)) in enumerate(series_dict.items()):
        p = fits_dict[label].params
        tfit = np.linspace(min(t), max(t), 300)
        Cfit = baranyi_roberts_model(tfit, p["C0"], p["Cmax"], p["mu"], p["lam"])
        plt.subplot(2, 2, i + 1)
        plt.scatter(t, C, s=20, label="Data")
        plt.plot(tfit, Cfit, 'r', label="BR fit")
        plt.title(label)
        plt.xlabel("Time (h)")
        plt.ylabel("Biomass (g/L)")
        plt.legend()
    plt.tight_layout()
    plt.show()


def plot_parameter_trends(volumes, Cmax_vals, mu_vals, lam_vals):
    plt.figure(figsize=(10, 6))
    plt.plot(volumes, Cmax_vals, '-o', label="Cmax")
    plt.plot(volumes, mu_vals,   '-o', label="μ")
    plt.plot(volumes, lam_vals,  '-o', label="λ")
    plt.xscale("log")
    plt.xlabel("Volume (L)")
    plt.ylabel("Parameter value")
    plt.legend()
    plt.title("Scale-dependent evolution of kinetic parameters")
    plt.show()


def plot_rmse_bars(rmse_dict, rmse_hybrid_4D, rmse_direct_100m3):
    """
    Bar plot of RMSE for ML baselines vs hybrid 4D vs direct 100 m³ fit.

    rmse_dict should have keys: "GPR", "KRR", "RF", "MLP".
    """
    labels = ["GPR", "KRR", "RF", "MLP", "Hybrid 4D", "Direct 100 m³"]
    rmse_vals = [
        rmse_dict["GPR"],
        rmse_dict["KRR"],
        rmse_dict["RF"],
        rmse_dict["MLP"],
        rmse_hybrid_4D,
        rmse_direct_100m3,
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, rmse_vals)
    plt.yscale("log")
    plt.ylabel("RMSE (g/L)")
    plt.title("Hybrid 4D mechanistic scaling vs ML baselines")
    plt.tight_layout()
    plt.show()



def plot_convex_weights(w_mf):
    labels_w = ["Power", "Log", "PWL", "CFD"]
    sizes = np.array(w_mf)

    fig, ax = plt.subplots(figsize=(10, 6))

    wedges, _ = ax.pie(
        sizes,
        wedgeprops=dict(width=0.35, edgecolor="white"),
        startangle=90
    )

    label_pos = {
        "Power": (-1.6,  1.0),
        "Log":   (-1.6, -1.0),
        "PWL":   ( 1.6, -1.0),
        "CFD":   ( 1.6,  1.0),
    }

    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8)
    labels = ["Power", "Log", "PWL", "CFD"]

    for wedge, label, value in zip(wedges, labels, sizes):
        theta = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
        x = 0.7 * np.cos(theta)
        y = 0.7 * np.sin(theta)

        xt, yt = label_pos[label]

        ax.annotate(
            f"{label}: {value*100:.2f}%",
            xy=(x, y),
            xytext=(xt, yt),
            ha="center",
            va="center",
            arrowprops=dict(arrowstyle="-"),
            bbox=bbox_props,
            fontsize=11,
        )

    ax.set_title("Hybrid model convex weights", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_time_vs_conc_hybrid(t_eval, C_exp, P_100m3_pred, rmse_hybrid_100m3):
    t_dense = np.linspace(t_eval.min(), t_eval.max(), 300)

    C_fixed_curve = baranyi_roberts_model(
        t_dense,
        P_100m3_pred.values[0],
        P_100m3_pred.values[1],
        P_100m3_pred.values[2],
        P_100m3_pred.values[3],
    )

    plt.figure(figsize=(7, 5))
    plt.scatter(t_eval, C_exp, s=35, label="Experimental", color="black")
    plt.plot(t_dense, C_fixed_curve, linewidth=2.5,
             label="Hybrid (fixed C₀)", color="tab:blue")

    plt.xlabel("Time (h)")
    plt.ylabel("Biomass concentration C (g/L)")
    plt.title(f"100 m³ — Hybrid Prediction (Fixed C₀)\nRMSE = {rmse_hybrid_100m3:.3f} g/L")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_time_vs_conc_optC0(t_eval, C_exp, C0_h_opt, P3_100m3_pred, rmse_optC0):
    t_dense = np.linspace(t_eval.min(), t_eval.max(), 300)

    C_opt_curve = baranyi_roberts_model(
        t_dense,
        C0_h_opt,
        P3_100m3_pred.values[0],
        P3_100m3_pred.values[1],
        P3_100m3_pred.values[2],
    )

    plt.figure(figsize=(7, 5))
    plt.scatter(t_eval, C_exp, s=35, label="Experimental", color="black")
    plt.plot(t_dense, C_opt_curve, linewidth=2.5,
             label="Hybrid (C₀ optimised)", color="tab:green")

    plt.xlabel("Time (h)")
    plt.ylabel("Biomass concentration C (g/L)")
    plt.title(f"100 m³ — Hybrid Prediction (C₀ Optimised)\nRMSE = {rmse_optC0:.3f} g/L")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_time_vs_conc_hybrid4D(t_eval, C_exp, P4_100m3_pred, rmse_hybrid):
    import numpy as np
    import matplotlib.pyplot as plt
    from bioreactor_core import baranyi_roberts_model

    t_dense = np.linspace(t_eval.min(), t_eval.max(), 300)

    C_curve = baranyi_roberts_model(
        t_dense,
        P4_100m3_pred.values[0],
        P4_100m3_pred.values[1],
        P4_100m3_pred.values[2],
        P4_100m3_pred.values[3],
    )

    plt.figure(figsize=(7, 5))
    plt.scatter(t_eval, C_exp, s=35, color="black", label="Experimental")
    plt.plot(t_dense, C_curve, linewidth=2.5, label="Hybrid 4D", color="tab:blue")
    plt.xlabel("Time (h)")
    plt.ylabel("Biomass concentration C (g/L)")
    plt.title(f"100 m³ — Hybrid 4D Prediction\nRMSE = {rmse_hybrid:.3f} g/L")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals_all_scales(fits_dict, figsize=(10, 8)):
    """
    Plot residuals (C_fit - C_exp) vs time for all available scales.

    Parameters
    ----------
    fits_dict : dict
        Dictionary mapping labels → GrowthFit objects, e.g.,
        {
            "10 L": fit_10,
            "100 L": fit_100,
            "4 m³": fit_4m3,
            "100 m³": fit_100m3
        }

    figsize : tuple
        Figure size

    Returns
    -------
    None
        Displays the plot.
    """

    n = len(fits_dict)
    rows = 2
    cols = 2

    plt.figure(figsize=figsize)

    for i, (label, fit) in enumerate(fits_dict.items()):
        plt.subplot(rows, cols, i + 1)
        plt.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")

        plt.scatter(fit.t, fit.residuals, s=25, color="tab:blue")

        plt.title(f"{label} — Residuals")
        plt.xlabel("Time (h)")
        plt.ylabel("Residual (C_fit - C_exp) [g/L]")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_bases_vs_hybrid_100m3(t_eval, C_exp, V_100k, basis_fns_4D, P4_100m3_pred):
    import numpy as np
    import matplotlib.pyplot as plt
    from bioreactor_core import baranyi_roberts_model

    t_dense = np.linspace(t_eval.min(), t_eval.max(), 300)

    labels = ["Power basis", "Log basis", "PWL basis", "CFD basis"]
    curves = []

    for f in basis_fns_4D:
        p = f(V_100k).values  # [C0, Cmax, mu, lam]
        C_curve = baranyi_roberts_model(t_dense, p[0], p[1], p[2], p[3])
        curves.append(C_curve)

    # hybrid 4D curve
    p_h = P4_100m3_pred.values
    C_hyb = baranyi_roberts_model(t_dense, p_h[0], p_h[1], p_h[2], p_h[3])

    plt.figure(figsize=(8, 6))
    plt.scatter(t_eval, C_exp, s=35, color="black", label="Experimental")

    for lab, C_curve in zip(labels, curves):
        plt.plot(t_dense, C_curve, linewidth=1.8, label=lab)

    plt.plot(t_dense, C_hyb, linewidth=2.4, label="Hybrid (convex)", color="tab:red")

    plt.xlabel("Time (h)")
    plt.ylabel("Biomass concentration C (g/L)")
    plt.title("100 m³ — Individual bases vs Hybrid vs Experimental")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_leave_one_scale_out(volumes, errors):
    """
    Plot RMSE vs Volume for leave-one-scale-out cross-validation.
    """
    plt.figure(figsize=(7,5))
    plt.plot(volumes, errors, "-o", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Held-out Volume (L)")
    plt.ylabel("RMSE (g/L)")
    plt.title("Leave-One-Scale-Out Hybrid Model Performance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

