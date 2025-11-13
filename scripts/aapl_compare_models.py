"""Compare volatility forecasting using both signature methods."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from output import SAVE_PATH
from signature_vol.calibration.signature_sde_calibration import calibrate_signature_sde
from signature_vol.exact.signature import Signature

matplotlib.use("Agg")


# region Data Loading and Preparation
# Load data from Yahoo Finance
ticker = "AAPL"
df = yf.download(ticker, start="2020-01-01", end="2025-01-01")
df = df.reset_index()
df = df.rename(columns={"Date": "timestamp", "Close": "price"})

# Compute log-returns
df["logret"] = np.log(df["price"]).diff().fillna(0)

# Create sliding windows of paths and target volatility
window = 50
horizon = 10
paths = []
targets = []
target_dates = []

for i in range(len(df) - window - horizon + 1):
    segment = df["logret"].values[i : i + window]
    path = np.column_stack([segment, np.abs(segment), np.cumsum(segment)])
    fut = df["logret"].values[i + window : i + window + horizon]
    realized_vol = np.sqrt(np.mean(fut**2))
    targets.append(realized_vol)
    paths.append(path)
    target_dates.append(df["timestamp"].iloc[i + window + horizon - 1])

targets = np.array(targets)
target_dates = np.array(target_dates)

# Compute log-targets
eps = 1e-8
log_targets = np.log(targets + eps)

# Specify truncation levels to test
truncation_levels = [2, 3, 4]

# Store results for both methods
results = {"SDE": {}, "Signature": {}}
# endregion

# region Model Training and Evaluation
# Process each truncation level with both methods
for sig_order in truncation_levels:
    print(f"\n{'=' * 60}")
    print(f"Processing truncation level {sig_order}")
    print(f"{'=' * 60}")

    # region SDE Calibration Method
    # Method 1: SDE Calibration
    print(f"\n[1/2] Computing features with calibrate_signature_sde...")
    X_sde_list = []
    for path_segment in paths:
        times_segment = np.arange(len(path_segment))
        cumsum_col = path_segment[:, 2]

        beta, alpha, sigs = calibrate_signature_sde(
            X=cumsum_col, times=times_segment, sig_order=sig_order, window_length=None
        )

        signature_features = sigs.array[-1, :]
        X_sde_list.append(signature_features)

    X_sde = np.vstack(X_sde_list)

    # Train SDE model
    tscv = TimeSeriesSplit(n_splits=5)
    model_sde = Pipeline([("scale", StandardScaler()), ("clf", LinearRegression())])

    mse_scores_sde = []
    r2_scores_sde = []
    all_predictions_sde = np.full(len(targets), np.nan)
    for train_idx, test_idx in tscv.split(X_sde):
        model_sde.fit(X_sde[train_idx], log_targets[train_idx])
        pred_log = model_sde.predict(X_sde[test_idx])
        pred = np.exp(pred_log)
        all_predictions_sde[test_idx] = pred
        mse_scores_sde.append(mean_squared_error(targets[test_idx], pred))
        r2_scores_sde.append(r2_score(targets[test_idx], pred))

    results["SDE"][sig_order] = {
        "predictions": all_predictions_sde,
        "mse_mean": np.mean(mse_scores_sde),
        "mse_std": np.std(mse_scores_sde),
        "r2_mean": np.mean(r2_scores_sde),
        "r2_std": np.std(r2_scores_sde),
    }

    print(
        f"SDE Method - CV MSE: {np.mean(mse_scores_sde):.6f}, std: {np.std(mse_scores_sde):.6f}, "
        f"R²: {np.mean(r2_scores_sde):.6f}, std: {np.std(r2_scores_sde):.6f}"
    )
    # endregion

    # region Pure Signature Method
    # Method 2: Pure Signature
    print(f"\n[2/2] Computing features with Signature.from_path...")
    signatures = [Signature.from_path(p, sig_order) for p in paths]
    X_sig = np.vstack([sig.array for sig in signatures])

    # Train Signature model
    model_sig = Pipeline([("scale", StandardScaler()), ("clf", LinearRegression())])

    mse_scores_sig = []
    r2_scores_sig = []
    all_predictions_sig = np.full(len(targets), np.nan)
    for train_idx, test_idx in tscv.split(X_sig):
        model_sig.fit(X_sig[train_idx], log_targets[train_idx])
        pred_log = model_sig.predict(X_sig[test_idx])
        pred = np.exp(pred_log)
        all_predictions_sig[test_idx] = pred
        mse_scores_sig.append(mean_squared_error(targets[test_idx], pred))
        r2_scores_sig.append(r2_score(targets[test_idx], pred))

    results["Signature"][sig_order] = {
        "predictions": all_predictions_sig,
        "mse_mean": np.mean(mse_scores_sig),
        "mse_std": np.std(mse_scores_sig),
        "r2_mean": np.mean(r2_scores_sig),
        "r2_std": np.std(r2_scores_sig),
    }

    print(
        f"Signature Method - CV MSE: {np.mean(mse_scores_sig):.6f}, std: {np.std(mse_scores_sig):.6f}, "
        f"R²: {np.mean(r2_scores_sig):.6f}, std: {np.std(r2_scores_sig):.6f}"
    )
    # endregion

    # region Individual Plots for Each Method
    # Create individual plot for SDE method
    fig_sde, ax_sde = plt.subplots(figsize=(12, 6))
    ax_sde.plot(
        target_dates,
        targets,
        label="Actual realized volatility",
        color="black",
        linewidth=2,
        alpha=0.7,
    )
    ax_sde.plot(
        target_dates,
        all_predictions_sde,
        label=f"Out-of-sample forecast (truncation level {sig_order})",
        color="dodgerblue",
        linewidth=2,
        alpha=0.8,
    )

    # Mark fold boundaries for SDE plot
    fold_sizes = []
    for train_idx, test_idx in tscv.split(X_sde):
        fold_sizes.append((train_idx[0], train_idx[-1], test_idx[0], test_idx[-1]))

    blue_colors = ["lightblue", "cornflowerblue"]

    for i, (train_start, train_end, test_start, test_end) in enumerate(fold_sizes):
        ax_sde.axvline(
            x=target_dates[test_start],
            color="gray",
            linestyle="--",
            alpha=0.5,
            linewidth=1,
        )
        ax_sde.axvline(
            x=target_dates[train_end],
            color="darkblue",
            linestyle="-",
            alpha=0.6,
            linewidth=1.5,
        )

        color = blue_colors[i % 2]
        if i == 0:
            ax_sde.axvspan(
                target_dates[train_start],
                target_dates[train_end],
                alpha=0.15,
                color=color,
                label="Training folds",
            )
        else:
            ax_sde.axvspan(
                target_dates[train_start],
                target_dates[train_end],
                alpha=0.15,
                color=color,
            )

    ax_sde.set_xlabel("Date")
    ax_sde.set_ylabel("Volatility")
    ax_sde.set_title(
        f"Out-of-Sample Volatility Forecasts (5-Fold CV, Level {sig_order})\nMean MSE: {np.mean(mse_scores_sde):.6f}"
    )
    ax_sde.legend()
    ax_sde.grid(True, alpha=0.3)
    fig_sde.autofmt_xdate()

    save_path_sde = SAVE_PATH / f"aapl_signature_forecast_oos_level_{sig_order}.png"
    plt.savefig(save_path_sde, dpi=150, bbox_inches="tight")
    print(f"Saved SDE out-of-sample forecast plot to {save_path_sde}")
    plt.close(fig_sde)

    # Create individual plot for Signature method
    fig_sig, ax_sig = plt.subplots(figsize=(12, 6))
    ax_sig.plot(
        target_dates,
        targets,
        label="Actual realized volatility",
        color="black",
        linewidth=2,
        alpha=0.7,
    )
    ax_sig.plot(
        target_dates,
        all_predictions_sig,
        label=f"Out-of-sample forecast (truncation level {sig_order})",
        color="crimson",
        linewidth=2,
        alpha=0.8,
    )

    # Mark fold boundaries for Signature plot
    for i, (train_start, train_end, test_start, test_end) in enumerate(fold_sizes):
        ax_sig.axvline(
            x=target_dates[test_start],
            color="gray",
            linestyle="--",
            alpha=0.5,
            linewidth=1,
        )
        ax_sig.axvline(
            x=target_dates[train_end],
            color="darkblue",
            linestyle="-",
            alpha=0.6,
            linewidth=1.5,
        )

        color = blue_colors[i % 2]
        if i == 0:
            ax_sig.axvspan(
                target_dates[train_start],
                target_dates[train_end],
                alpha=0.15,
                color=color,
                label="Training folds",
            )
        else:
            ax_sig.axvspan(
                target_dates[train_start],
                target_dates[train_end],
                alpha=0.15,
                color=color,
            )

    ax_sig.set_xlabel("Date")
    ax_sig.set_ylabel("Volatility")
    ax_sig.set_title(
        f"Out-of-Sample Volatility Forecasts (5-Fold CV, Level {sig_order})\nMean MSE: {np.mean(mse_scores_sig):.6f}"
    )
    ax_sig.legend()
    ax_sig.grid(True, alpha=0.3)
    fig_sig.autofmt_xdate()

    save_path_sig = SAVE_PATH / f"aapl_pure_data_forecast_oos_level_{sig_order}.png"
    plt.savefig(save_path_sig, dpi=150, bbox_inches="tight")
    print(f"Saved Signature out-of-sample forecast plot to {save_path_sig}")
    plt.close(fig_sig)
    # endregion

    # region Comparison Plot for Current Truncation Level
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot actual volatility
    ax.plot(
        target_dates,
        targets,
        label="Actual realized volatility",
        color="black",
        linewidth=2.5,
        alpha=0.8,
        zorder=3,
    )

    # Plot SDE predictions
    ax.plot(
        target_dates,
        all_predictions_sde,
        label=f"SDE Calibration (MSE: {np.mean(mse_scores_sde):.6f})",
        color="dodgerblue",
        linewidth=2,
        alpha=0.8,
        linestyle="-",
        zorder=2,
    )

    # Plot Signature predictions
    ax.plot(
        target_dates,
        all_predictions_sig,
        label=f"Pure Signature (MSE: {np.mean(mse_scores_sig):.6f})",
        color="crimson",
        linewidth=2,
        alpha=0.8,
        linestyle="--",
        zorder=2,
    )

    # Mark fold boundaries
    fold_sizes = []
    for train_idx, test_idx in tscv.split(X_sde):
        fold_sizes.append((train_idx[0], train_idx[-1], test_idx[0], test_idx[-1]))

    blue_colors = ["lightblue", "cornflowerblue"]

    for i, (train_start, train_end, test_start, test_end) in enumerate(fold_sizes):
        # Mark test fold start
        ax.axvline(
            x=target_dates[test_start],
            color="gray",
            linestyle="--",
            alpha=0.5,
            linewidth=1,
        )

        # Mark training regime end
        ax.axvline(
            x=target_dates[train_end],
            color="darkblue",
            linestyle="-",
            alpha=0.6,
            linewidth=1.5,
        )

        color = blue_colors[i % 2]
        if i == 0:
            ax.axvspan(
                target_dates[train_start],
                target_dates[train_end],
                alpha=0.15,
                color=color,
                label="Training folds",
                zorder=0,
            )
        else:
            ax.axvspan(
                target_dates[train_start],
                target_dates[train_end],
                alpha=0.15,
                color=color,
                zorder=0,
            )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Volatility", fontsize=12)
    ax.set_title(
        f"Model Comparison: Out-of-Sample Volatility Forecasts\n"
        f"Truncation Level {sig_order}, 5-Fold CV",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    # Save comparison plot
    save_path = SAVE_PATH / f"aapl_model_comparison_level_{sig_order}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot to {save_path}")
    plt.close()
    # endregion
# endregion

# region Aggregate Metrics Analysis
levels = sorted(results["SDE"].keys())
sde_mse = [results["SDE"][lvl]["mse_mean"] for lvl in levels]
sde_r2 = [results["SDE"][lvl]["r2_mean"] for lvl in levels]
sig_mse = [results["Signature"][lvl]["mse_mean"] for lvl in levels]
sig_r2 = [results["Signature"][lvl]["r2_mean"] for lvl in levels]

# Compute 1 - R²
sde_one_minus_r2 = [1 - r2 for r2 in sde_r2]
sig_one_minus_r2 = [1 - r2 for r2 in sig_r2]

# Create single-axis plot with unified log scale for both MSE and 1-R²
fig, ax = plt.subplots(figsize=(10, 6))

# Plot both MSE and 1-R² on unified log scale
ax.set_xlabel("Signature Truncation Depth", fontsize=12)
ax.set_ylabel("Metric Value (log scale)", fontsize=12)
ax.plot(
    levels,
    sde_mse,
    "o-",
    color="tab:blue",
    label="SDE Calibration MSE",
    linewidth=2,
    markersize=8,
)
ax.plot(
    levels,
    sig_mse,
    "s--",
    color="tab:cyan",
    label="Pure Signature MSE",
    linewidth=2,
    markersize=8,
)
ax.plot(
    levels,
    sde_one_minus_r2,
    "^-",
    color="tab:orange",
    label="SDE Calibration 1-R²",
    linewidth=2,
    markersize=8,
)
ax.plot(
    levels,
    sig_one_minus_r2,
    "d--",
    color="tab:red",
    label="Pure Signature 1-R²",
    linewidth=2,
    markersize=8,
)
ax.set_yscale("log")
ax.set_xticks(levels)
ax.grid(True, alpha=0.3, which="both")
ax.legend(loc="best", fontsize=10)

fig.suptitle(
    "Model Performance vs Signature Truncation Depth",
    fontsize=14,
    fontweight="bold",
)
fig.tight_layout()

save_path_metrics = SAVE_PATH / "aapl_model_metrics_vs_depth.png"
plt.savefig(save_path_metrics, dpi=150, bbox_inches="tight")
print(f"Saved metrics vs depth plot to {save_path_metrics}")
plt.close()

# Save metrics to text file
metrics_txt_path = SAVE_PATH / "aapl_model_metrics.txt"
with open(metrics_txt_path, "w") as f:
    f.write("Model Performance Comparison: MSE and R² Values\n")
    f.write("=" * 60 + "\n\n")

    for level in levels:
        f.write(f"Truncation Level {level}:\n")
        f.write("-" * 40 + "\n")

        f.write(f"SDE Calibration Method:\n")
        f.write(
            f"  MSE: {results['SDE'][level]['mse_mean']:.6f} ± {results['SDE'][level]['mse_std']:.6f}\n"
        )
        f.write(
            f"  R²:  {results['SDE'][level]['r2_mean']:.6f} ± {results['SDE'][level]['r2_std']:.6f}\n"
        )
        f.write(f"  1-R²: {1 - results['SDE'][level]['r2_mean']:.6f}\n\n")

        f.write(f"Pure Signature Method:\n")
        f.write(
            f"  MSE: {results['Signature'][level]['mse_mean']:.6f} ± {results['Signature'][level]['mse_std']:.6f}\n"
        )
        f.write(
            f"  R²:  {results['Signature'][level]['r2_mean']:.6f} ± {results['Signature'][level]['r2_std']:.6f}\n"
        )
        f.write(f"  1-R²: {1 - results['Signature'][level]['r2_mean']:.6f}\n\n")

        f.write("\n")

print(f"Saved metrics to text file: {metrics_txt_path}")
# endregion
