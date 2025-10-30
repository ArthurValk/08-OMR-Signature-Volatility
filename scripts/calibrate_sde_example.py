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

for i in range(len(df) - window - horizon + 1):
    segment = df["logret"].values[i : i + window]
    path = np.column_stack([segment, np.abs(segment), np.cumsum(segment)])
    fut = df["logret"].values[i + window : i + window + horizon]
    realized_vol = np.sqrt(np.mean(fut**2))
    targets.append(realized_vol)
    paths.append(path)

targets = np.array(targets)

# Compute signatures using calibrate_signature_sde
eps = 1e-8
log_targets = np.log(targets + eps)
sig_order = 3

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
for train_idx, test_idx in tscv.split(X):
    model.fit(X[train_idx], targets[train_idx])
    pred = model.predict(X[test_idx])
    mse_scores.append(mean_squared_error(targets[test_idx], pred))

print("CV MSE:", np.mean(mse_scores), "std:", np.std(mse_scores))

# Final fit on full data
model.fit(X, log_targets)
pred_log = model.predict(X)
pred = np.exp(pred_log)

# Visualization
plt.figure(figsize=(10, 5))
plt.xlim(0, 200)
plt.plot(targets, label="Actual realized volatility", color="black", linewidth=2)
plt.plot(
    pred,
    label="Predicted volatility (signature-SDE model)",
    color="dodgerblue",
    linewidth=2,
    alpha=0.8,
)
plt.xlabel("Time index")
plt.ylabel("Volatility")
plt.title("Signature-SDE Volatility Model vs Actual Volatility")
plt.legend()
plt.grid(True, alpha=0.3)

save_path = SAVE_PATH / "aapl_signature_sde_calibration.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
