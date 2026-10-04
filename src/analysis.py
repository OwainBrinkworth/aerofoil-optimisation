import os
import numpy as np
import pandas as pd
import time

from defaults import params, post, param_bounds
from matplotlib import pyplot as plt
from aerofoil_initialisation import getAerofoilsAsDataframe
from compile_results import extract_cl_values
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from paths import get_path

def analysis(model, training_df, training_time, X_test=None, y_test=None):
    analysis_path = get_path('analysis')

    csv_path = os.path.join(analysis_path, "runs.csv")
    # --- Flatten params ---
    flat_params = flatten_params(params)

    # --- Load or create dataframe ---
    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    # --- Determine next run number ---
    if "run" in df.columns and len(df) > 0:
        last_run = df["run"].iloc[-1]
        next_run_num = int(last_run) + 1
    else:
        next_run_num = 1

    run_id = f"{next_run_num:03d}"

    run_dir = os.path.join(analysis_path, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Extract training results
    training_df_clean = training_df.dropna(subset=["CL"])
    if len(training_df_clean) == 0:
        raise ValueError("No valid training CL values found!")
    X_train = training_df_clean.drop(columns=["CL"])
    y_train = training_df_clean["CL"]

    # Prefer the actual holdout split created during training.
    if X_test is None or y_test is None:
        testing_df = getAerofoilsAsDataframe(test=True)
        testing_df = extract_cl_values(testing_df, test=True)
        testing_df_clean = testing_df.dropna(subset=["CL"])
        if len(testing_df_clean) == 0:
            raise ValueError("No valid testing CL values found!")
        X_test = testing_df_clean.drop(columns=["CL"])
        y_test = testing_df_clean["CL"]
    else:
        # Keep the actual test set from the split
        testing_df_clean = pd.concat([X_test, y_test], axis=1)

    results_df = pd.concat([training_df_clean, testing_df_clean], ignore_index=True)

    # --- Predictions ---
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    # --- Metrics for analysis ---

    R2_train = model.score(X_train, y_train)
    R2_test = model.score(X_test, y_test)

    print(f"Test score (R^2): {R2_test:.3f}")
    print(f"Train score (R^2): {R2_train:.3f}")

    mean_abs_error = mean_absolute_error(y_test, y_pred)
    mean_sqd_error = mean_squared_error(y_test, y_pred)
    mean_abs_percent_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Cross-validation
    scores = cross_val_score(model, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), cv=5, scoring="r2")
    cross_val_mean = scores.mean()
    cross_val_std_dev = scores.std()
    print(cross_val_mean, cross_val_std_dev)

    # =========================
    # 1. True vs predicted CL
    # =========================

    save_path = os.path.join(run_dir, "predicted_vs_true.png")

    # --- Axis limits (same scale) ---
    min_val = min(min(y_test), min(y_pred)) - 0.02
    max_val = max(max(y_test), max(y_pred)) + 0.02

    # --- Plot ---
    plt.figure()

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.6
    )

    # Perfect prediction line
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

    plt.xlabel("True CL")
    plt.ylabel("Predicted CL")
    plt.title("Surrogate Model Performance")

    # --- Equal axes ---
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')

    # --- Save ---
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # 2. Error distribution
    # =========================

    error = y_test - y_pred

    plt.figure()
    plt.hist(error, bins=50, density=True)

    plt.xlabel("Error (True - Predicted CL)")
    plt.ylabel("Probability Density")
    plt.title("Normalised Error Distribution")

    plt.xlim(-0.5, 0.5)

    plt.savefig(os.path.join(run_dir, "error_hist.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    bias = np.mean(error)

    # Geometry based plots
    x_suc_all = results_df["x_SUC"]
    y_suc_all = results_df["y_SUC"]
    cl_all = results_df["CL"]

    test_idx = X_test.index

    x_suc_test = testing_df_clean["x_SUC"]
    y_suc_test = testing_df_clean["y_SUC"]

    # =========================
    # Global limits (shared)
    # =========================
    if not params['test']['interpolation']:
        expansion_ratio = params['test']['expansion_ratio']
    else:
        expansion_ratio = 1
    x_min, x_max = expand_bounds(*param_bounds["x_SUC"], expansion_ratio)
    y_min, y_max = expand_bounds(*param_bounds["y_SUC"], expansion_ratio)

    # Shared colour scale for CL
    cl_min = min(cl_all.min(), y_pred.min())
    cl_max = max(cl_all.max(), y_pred.max())

    # Error (test only)
    error = y_test - y_pred
    err_max = np.max(np.abs(error))  # symmetric range

    # =========================
    # 3. TRUE CL (full dataset)
    # =========================
    plt.figure()

    sc = plt.scatter(
        x_suc_all, y_suc_all,
        c=cl_all,
        vmin=cl_min, vmax=cl_max
    )

    # Highlight test points (black outline)
    plt.scatter(
        x_suc_test, y_suc_test,
        facecolors='none',
        edgecolors='black',
        linewidths=1,
        label="Test points"
    )

    plt.colorbar(sc, label="True CL")

    plt.xlabel("x_SUC")
    plt.ylabel("y_SUC")
    plt.title("True CL across geometry space")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.legend()

    plt.savefig(os.path.join(run_dir, "true_CL_map.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # =========================
    # 2. PREDICTED CL (test set)
    # =========================
    plt.figure()

    # --- Background: all points (faint grey) ---
    plt.scatter(
        x_suc_all, y_suc_all,
        color='lightgrey',
        alpha=0.3,
        label="All samples"
    )

    # --- Test points coloured by prediction ---
    sc = plt.scatter(
        x_suc_test, y_suc_test,
        c=y_pred,
        vmin=cl_min, vmax=cl_max,
        edgecolors='black',
        linewidths=1.0,
        label="Test points"
    )

    plt.colorbar(sc, label="Predicted CL")

    plt.xlabel("x_SUC")
    plt.ylabel("y_SUC")
    plt.title("Predicted CL (test set)")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.legend()

    plt.savefig(os.path.join(run_dir, "predicted_CL_map.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


    # =========================
    # 3. ERROR (True - Predicted)
    # =========================
    plt.figure()

    sc = plt.scatter(
        x_suc_test, y_suc_test,
        c=error,
        cmap="coolwarm",
        vmin=-err_max,
        vmax=err_max,
        edgecolors='black',
        linewidths=1.0
    )

    plt.colorbar(sc, label="CL Error (True - Predicted)")

    plt.xlabel("x_SUC")
    plt.ylabel("y_SUC")
    plt.title("Prediction Error (test set)")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.savefig(os.path.join(run_dir, "error_map.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # Plotting each variable vs the error
    abs_error = np.abs(error)
    param_names = [col for col in results_df.columns if col != "CL"]
    for param in param_names:
        x_all = results_df[param]
        x_train = X_train[param]
        x_test = X_test[param]
        # =========================
        # Extract values
        # =========================
        cl_all = results_df["CL"]
        cl_test = y_test
        cl_train = y_train

        # Bounds
        x_min, x_max = param_bounds[param]
        cl_min, cl_max = post['CL']
        err_min, err_max = post['error']

        # =========================
        # 1. Error vs parameter
        # =========================
        plt.figure()

        plt.scatter(x_test, abs_error, alpha=0.6)
        plt.axhline(0, linestyle='--')

        plt.xlabel(param)
        plt.ylabel("Absolute error (True - Predicted CL)")
        plt.title(f"Absolute error vs {param}")

        plt.xlim(x_min, x_max)
        plt.ylim(err_min, err_max)

        safe_param = param.replace("/", "_")
        plt.savefig(os.path.join(run_dir, f"error_vs_{safe_param}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        # =========================
        # 2. True CL vs parameter
        # =========================
        plt.figure()

        plt.scatter(x_all, cl_all, color='lightgrey', alpha=0.3, label="All data")
        plt.scatter(x_train, cl_train, alpha=0.6, label="Train")

        plt.scatter(
            x_test, cl_test,
            facecolors='none',
            edgecolors='black',
            linewidths=1,
            label="Test"
        )

        plt.xlabel(param)
        plt.ylabel("CL")
        plt.title(f"True CL vs {param}")

        plt.xlim(x_min, x_max)
        plt.ylim(cl_min, cl_max)

        plt.legend()
        plt.savefig(os.path.join(run_dir, f"true_CL_vs_{safe_param}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        # =========================
        # 3. Predicted CL vs parameter
        # =========================
        plt.figure()

        plt.scatter(x_all, cl_all, color='lightgrey', alpha=0.3, label="All data")

        plt.scatter(
            x_test, y_pred,
            edgecolors='black',
            linewidths=0.5,
            label="Predicted (test)"
        )

        plt.xlabel(param)
        plt.ylabel("CL")
        plt.title(f"Predicted CL vs {param}")

        plt.xlim(x_min, x_max)
        plt.ylim(cl_min, cl_max)

        plt.legend()

        plt.savefig(os.path.join(run_dir, f"predicted_CL_vs_{safe_param}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    plt.figure()
    # Train values
    plt.hist(y_train, bins=50, density=True, alpha=0.5, label="Train")

    # Test values
    plt.hist(y_test, bins=50, density=True, alpha=0.5, label="Test")

    plt.xlabel("CL")
    plt.ylabel("Density")
    plt.title("CL Distribution (Train vs Test)")

    plt.legend()

    save_path = os.path.join(run_dir, "CL_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    sampling_time_file = get_path('sampling_time')
    with open(sampling_time_file, "r") as f:
        sampling_time = float(f.read())

    keys_to_round = ["sampling_time", "training_time", "prediction_time"]

    results_dict = {
        "R2_train": R2_train,
        "R2_test": R2_test,
        "MAE": mean_abs_error,
        "RMSE": np.sqrt(mean_sqd_error),
        "MAPE": mean_abs_percent_error,
        "max_error": err_max,
        "bias": bias,
        "cross_val_mean": cross_val_mean,
        "cross_val_std_dev": cross_val_std_dev,
        "sampling_time": sampling_time,
        "training_time": training_time,
        "prediction_time": prediction_time,
        "samples": len(X_train),
    }

    # Round some of the fields
    for key in keys_to_round:
        if key in results_dict:
            results_dict[key] = round(results_dict[key], 3)

    # Save csv
    # --- Combine data ---
    new_row = {
        "run": run_id,
        **flat_params,
        **results_dict
    }

    # --- Append ---
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # --- Save ---
    df.to_csv(csv_path, index=False)

    print(f"Run {run_id} saved to {csv_path}")

def flatten_params(params, parent_key='', sep='_'):
    """
    Flatten nested dictionary into single level.
    Example: params['sample']['n_samples'] → sample_n_samples
    """
    items = []
    for k, v in params.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_params(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def expand_bounds(min_val, max_val, expansion_ratio):
    centre = 0.5 * (min_val + max_val)
    half_range = 0.5 * (max_val - min_val)

    new_half = half_range * expansion_ratio

    return centre - new_half, centre + new_half
