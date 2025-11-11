"""Volatility forecasting using calibrate_signature_sde for feature extraction."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from output import SAVE_PATH
from signature_vol.calibration.signature_sde_calibration import calibrate_signature_sde

matplotlib.use("Agg")


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
    # Store the date corresponding to this target (end of the forecast horizon)
    target_dates.append(df["timestamp"].iloc[i + window + horizon - 1])

targets = np.array(targets)
target_dates = np.array(target_dates)

# Compute signatures using calibrate_signature_sde
eps = 1e-8
log_targets = np.log(targets + eps)

# Specify truncation levels to test
truncation_levels = [2, 3, 4]

# Process each truncation level
for sig_order in truncation_levels:
    print(f"\nProcessing truncation level {sig_order}...")

    X_list = []
    for path_segment in paths:
        times_segment = np.arange(len(path_segment))
        cumsum_col = path_segment[:, 2]

        beta, alpha, sigs = calibrate_signature_sde(
            X=cumsum_col, times=times_segment, sig_order=sig_order, window_length=None
        )

        signature_features = sigs.array[-1, :]
        X_list.append(signature_features)

    X = np.vstack(X_list)

    # Train model with time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    model = Pipeline([("scale", StandardScaler()), ("clf", LinearRegression())])

    mse_scores = []
    all_predictions = np.full(len(targets), np.nan)
    for train_idx, test_idx in tscv.split(X):
        model.fit(X[train_idx], log_targets[train_idx])
        pred_log = model.predict(X[test_idx])
        pred = np.exp(pred_log)
        all_predictions[test_idx] = pred
        mse_scores.append(mean_squared_error(targets[test_idx], pred))

    print(
        f"Truncation level {sig_order} - CV MSE: {np.mean(mse_scores):.6f}, std: {np.std(mse_scores):.6f}"
    )

    # Visualization: Out-of-sample forecasts from cross-validation
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(
        target_dates,
        targets,
        label="Actual realized volatility",
        color="black",
        linewidth=2,
        alpha=0.7,
    )
    ax2.plot(
        target_dates,
        all_predictions,
        label=f"Out-of-sample forecast (truncation level {sig_order})",
        color="dodgerblue",
        linewidth=2,
        alpha=0.8,
    )

    # Mark the fold boundaries with alternating blue shades for training periods
    fold_sizes = []
    for train_idx, test_idx in tscv.split(X):
        fold_sizes.append((train_idx[0], train_idx[-1], test_idx[0], test_idx[-1]))

    # Define two alternating shades of blue
    blue_colors = ["lightblue", "cornflowerblue"]

    for i, (train_start, train_end, test_start, test_end) in enumerate(fold_sizes):
        # Mark the start of each test fold with a dashed line
        ax2.axvline(
            x=target_dates[test_start],
            color="gray",
            linestyle="--",
            alpha=0.5,
            linewidth=1,
        )

        # Add small tick marks at the end of each training regime
        ax2.axvline(
            x=target_dates[train_end],
            color="darkblue",
            linestyle="-",
            alpha=0.6,
            linewidth=1.5,
        )

        color = blue_colors[i % 2]
        if i == 0:
            ax2.axvspan(
                target_dates[train_start],
                target_dates[train_end],
                alpha=0.15,
                color=color,
                label="Training folds",
            )
        else:
            ax2.axvspan(
                target_dates[train_start],
                target_dates[train_end],
                alpha=0.15,
                color=color,
            )

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Volatility")
    ax2.set_title(
        f"Out-of-Sample Volatility Forecasts (5-Fold CV, Level {sig_order})\nMean MSE: {np.mean(mse_scores):.6f}"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.autofmt_xdate()

    save_path_oos = SAVE_PATH / f"aapl_signature_forecast_oos_level_{sig_order}.png"
    plt.savefig(save_path_oos, dpi=150, bbox_inches="tight")
    print(f"Saved out-of-sample forecast plot to {save_path_oos}")
