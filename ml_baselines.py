# ml_baselines.py

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

print("ml_baselines imported.")


def evaluate_model(model, name, V_train, P_train, V_test, P_true_100m3):
    model.fit(V_train, P_train)
    P_pred = model.predict(V_test).flatten()

    rmse_vec = np.sqrt(mean_squared_error(P_true_100m3, P_pred))
    rmse_cmax   = np.abs(P_pred[0] - P_true_100m3[0])
    rmse_mu     = np.abs(P_pred[1] - P_true_100m3[1])
    rmse_lambda = np.abs(P_pred[2] - P_true_100m3[2])

    print(f"\n{name}")
    print(f"Predicted parameters: {P_pred}")
    print(f"RMSE vector (all params): {rmse_vec:.4f}")
    print(f"  Error Cmax:   {rmse_cmax:.4f}")
    print(f"  Error mu:     {rmse_mu:.4f}")
    print(f"  Error lambda: {rmse_lambda:.4f}")

    return rmse_vec


def run_ml_baselines(V_train, P_train, V_test, P_true_100m3):
    rmse_dict = {}

    kernel = C(1.0, (1e-3, 1e4)) * RBF(length_scale=20.0, length_scale_bounds=(1e-3, 1e3))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)
    rmse_dict["GPR"] = evaluate_model(gpr, "Gaussian Process Regression",
                                      V_train, P_train, V_test, P_true_100m3)

    rf = RandomForestRegressor(n_estimators=500, max_depth=None)
    rmse_dict["RF"] = evaluate_model(rf, "Random Forest Regression",
                                     V_train, P_train, V_test, P_true_100m3)

    krr = KernelRidge(kernel='rbf', alpha=1e-6, gamma=0.1)
    rmse_dict["KRR"] = evaluate_model(krr, "Kernel Ridge Regression",
                                      V_train, P_train, V_test, P_true_100m3)

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation='relu',
        solver='adam',
        max_iter=20000,
        learning_rate_init=0.001
    )
    rmse_dict["MLP"] = evaluate_model(mlp, "Neural Network (MLP)",
                                      V_train, P_train, V_test, P_true_100m3)

    print("\n======================")
    print("SUMMARY OF BASELINES")
    print("======================")
    for k, v in rmse_dict.items():
        print(f"{k} RMSE: {v:.4f}")

    return rmse_dict
