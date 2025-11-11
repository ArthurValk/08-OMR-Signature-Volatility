# scripts/train_signature_model.py
from output import SAVE_PATH
from signature_vol.exact.signature import Signature

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf

import matplotlib

matplotlib.use("Agg")

# --- 1. Load data directly from Yahoo Finance ---
ticker = "AAPL"  # Apple, for example
df = yf.download(ticker, start="2020-01-01", end="2025-01-01")
df = df.reset_index()

# Rename columns for compatibility
df = df.rename(columns={"Date": "timestamp", "Close": "price"})

# --- 2. Compute log-returns ---
df["logret"] = np.log(df["price"]).diff().fillna(0)

# --- 2. Create sliding windows of paths and target volatility ---
window = 50  # length of path (timesteps)
horizon = 10  # predict next-horizon realized vol?
paths = []
targets = []
target_dates = []
for i in range(len(df) - window - horizon + 1):
    segment = df["logret"].values[i : i + window]
    # richer 3D path: returns, abs(returns), cumulative sum
    path = np.column_stack([segment, np.abs(segment), np.cumsum(segment)])
    # target: realized volatility over next horizon (e.g. std of future returns)
    fut = df["logret"].values[i + window : i + window + horizon]
    realized_vol = np.sqrt(np.mean(fut**2))
    targets.append(np.sqrt(np.mean(fut**2)))  # RMSE-style realized vol
    paths.append(path)
    # Store the date corresponding to this target (end of the forecast horizon)
    target_dates.append(df["timestamp"].iloc[i + window + horizon - 1])

targets = np.array(targets)
target_dates = np.array(target_dates)

# --- 3. compute signatures for each path ---
# --- 3. Predict log-vol instead of vol ---
eps = 1e-8
log_targets = np.log(targets + eps)

# Specify truncation levels to test
truncation_levels = [2, 3, 4]

import matplotlib.pyplot as plt

# Process each truncation level
for depth in truncation_levels:
    print(f"\nProcessing truncation level {depth}...")

    # Compute signatures using Signature.from_path
    signatures = [Signature.from_path(p, depth) for p in paths]
    X = np.vstack([sig.array for sig in signatures])

    # --- 4. Train a model with time-series cross-validation ---
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
        f"Truncation level {depth} - CV MSE: {np.mean(mse_scores):.6f}, std: {np.std(mse_scores):.6f}"
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
        label=f"Out-of-sample forecast (truncation level {depth})",
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
        f"Out-of-Sample Volatility Forecasts (5-Fold CV, Level {depth})\nMean MSE: {np.mean(mse_scores):.6f}"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.autofmt_xdate()

    save_path_oos = SAVE_PATH / f"aapl_pure_data_forecast_oos_level_{depth}.png"
    plt.savefig(save_path_oos, dpi=150, bbox_inches="tight")
    print(f"Saved out-of-sample forecast plot to {save_path_oos}")

# Approximation quality
rng = np.random.default_rng(0)

# 1. simulate random 2D paths (e.g. random walk increments)
n_paths = 500
T = 30
paths = []
targets = []

for _ in range(n_paths):
    incr = rng.normal(scale=0.1, size=(T, 2))  # Gaussian steps in R^2
    path = np.cumsum(incr, axis=0)  # cumulative sum -> path
    paths.append(path)

    # define a nonlinear functional of the whole path
    xcoord = path[:, 0]
    functional_value = (
        xcoord.max() - xcoord.min()
    ) ** 2  # squared range of first coord
    targets.append(functional_value)

paths = np.array(paths, dtype=object)
targets = np.array(targets)


# helper: given depth m, build signature feature matrix
def build_sig_features(paths, depth):
    signatures = [Signature.from_path(p, depth) for p in paths]
    X = np.vstack([sig.array for sig in signatures])
    return X


# --- 3. Loop over depths, fit linear model, store metrics ---
depths, mse_vals, r2_vals = [], [], []

for depth in [1, 2, 3, 4, 5, 6, 7, 8]:
    X = build_sig_features(paths, depth)
    reg = LinearRegression().fit(X, targets)
    pred = reg.predict(X)

    mse = mean_squared_error(targets, pred)
    r2 = r2_score(targets, pred)

    depths.append(depth)
    mse_vals.append(mse)
    r2_vals.append(r2)

    print(f"depth={depth}  mse={mse:.4e}  R²={r2:.4f}")

# --- 4. Plot MSE and R² vs. depth ---
fig, ax1 = plt.subplots(figsize=(7, 4))

color1 = "tab:blue"
ax1.plot(depths, mse_vals, "o-", color=color1, label="MSE")
ax1.set_xlabel("Signature depth")
ax1.set_ylabel("Mean Squared Error", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_xticks(depths)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color2 = "tab:orange"
ax2.plot(depths, r2_vals, "s--", color=color2, label="$R^2$")
ax2.set_ylabel("$R^2$", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)
ax2.set_ylim(0, 1)

fig.suptitle("Approximation quality vs. signature truncation depth")
fig.tight_layout()
save_path = SAVE_PATH / "approximation_quality.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Saved approximation quality plot to {save_path}")
