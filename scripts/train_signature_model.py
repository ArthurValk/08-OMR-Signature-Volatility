# scripts/train_signature_model.py
import numpy as np
from signature_vol.exact.signature import Signature  # your class file

paths = [
    np.random.randn(100, 2),  # example 2D paths
    np.random.randn(120, 2),
]

signatures = [Signature.from_path(path, level=3) for path in paths]

# Example: access signature array of the first path
print(signatures[0].array)






import pandas as pd
import numpy as np
import iisignature     # pip install iisignature
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import yfinance as yf

# --- 1. Load data directly from Yahoo Finance ---
ticker = "AAPL"  # Apple, for example
df = yf.download(ticker, start="2020-01-01", end="2025-01-01")
df = df.reset_index()

# Rename columns for compatibility
df = df.rename(columns={"Date": "timestamp", "Close": "price"})

# --- 2. Compute log-returns ---
df["logret"] = np.log(df["price"]).diff().fillna(0)

# --- 2. Create sliding windows of paths and target volatility ---
window = 50   # length of path (timesteps)
horizon = 10  # predict next-horizon realized vol?
paths = []
targets = []
for i in range(len(df) - window - horizon + 1):
    segment = df["logret"].values[i : i + window]
    # richer 3D path: returns, abs(returns), cumulative sum
    path = np.column_stack([
        segment,
        np.abs(segment),
        np.cumsum(segment)
    ])
    # target: realized volatility over next horizon (e.g. std of future returns)
    fut = df["logret"].values[i + window : i + window + horizon]
    realized_vol = np.sqrt(np.mean(fut**2))
    targets.append(realized_vol)  # RMSE-style realized vol
    paths.append(path)

targets = np.array(targets)

# --- 3. compute signatures for each path ---
# --- 3. Predict log-vol instead of vol ---
eps = 1e-8
log_targets = np.log(targets + eps)
# choose signature depth
depth = 3
# compute signature length needed and create a signature transform
sig_dim = paths[0].shape[1]
sig_len = iisignature.siglength(sig_dim, depth)
X = np.vstack([iisignature.sig(p, depth) for p in paths])


# --- 4. Train a model with time-series cross-validation ---
tscv = TimeSeriesSplit(n_splits=5)
model = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LinearRegression())
])

mse_scores = []
for train_idx, test_idx in tscv.split(X):
    model.fit(X[train_idx], log_targets[train_idx])
    pred_log = model.predict(X[test_idx])
    pred = np.exp(pred_log)
    mse_scores.append(mean_squared_error(targets[test_idx], pred))

print("CV MSE:", np.mean(mse_scores), "std:", np.std(mse_scores))

# Final fit on full data
model.fit(X, log_targets)
pred_log = model.predict(X)
pred = np.exp(pred_log)



import matplotlib.pyplot as plt
import numpy as np

# Suppose you already have:
# targets  = np.array([...])  # actual realized volatilities
# pred     = model.predict(X) # predicted volatilities

plt.figure(figsize=(10, 5))
plt.xlim(0,200)
plt.plot(targets, label="Actual realized volatility", color="black", linewidth=2)
plt.plot(pred, label="Predicted volatility (signature model)", color="dodgerblue", linewidth=2, alpha=0.8)
plt.xlabel("Time index")
plt.ylabel("Volatility")
plt.title("Signature-based Volatility Model vs Actual Volatility")

plt.legend()
plt.show()

#Approximation quality
rng = np.random.default_rng(0)

# 1. simulate random 2D paths (e.g. random walk increments)
n_paths = 500
T = 30
paths = []
targets = []

for _ in range(n_paths):
    incr = rng.normal(scale=0.1, size=(T, 2))            # Gaussian steps in R^2
    path = np.cumsum(incr, axis=0)                       # cumulative sum -> path
    paths.append(path)

    # define a nonlinear functional of the whole path
    xcoord = path[:,0]
    functional_value = (xcoord.max() - xcoord.min())**2  # squared range of first coord
    targets.append(functional_value)

paths = np.array(paths, dtype=object)
targets = np.array(targets)

# helper: given depth m, build signature feature matrix
def build_sig_features(paths, depth):
    d = paths[0].shape[1]
    feat_dim = iisignature.siglength(d, depth)
    X = np.zeros((len(paths), feat_dim))
    for i, p in enumerate(paths):
        p = np.asarray(p, dtype=float)  # <--- ensure numeric array
        X[i, :] = iisignature.sig(p, depth)
    return X

# --- 3. Loop over depths, fit linear model, store metrics ---
depths, mse_vals, r2_vals = [], [], []

for depth in [1, 2, 3, 4,5,6,7,8]:
    X = build_sig_features(paths, depth)
    reg = LinearRegression().fit(X, targets)
    pred = reg.predict(X)

    mse = mean_squared_error(targets, pred)
    r2  = r2_score(targets, pred)

    depths.append(depth)
    mse_vals.append(mse)
    r2_vals.append(r2)

    print(f"depth={depth}  mse={mse:.4e}  R²={r2:.4f}")

# --- 4. Plot MSE and R² vs. depth ---
fig, ax1 = plt.subplots(figsize=(7,4))

color1 = 'tab:blue'
ax1.plot(depths, mse_vals, 'o-', color=color1, label='MSE')
ax1.set_xlabel('Signature depth')
ax1.set_ylabel('Mean Squared Error', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(depths)

ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.plot(depths, r2_vals, 's--', color=color2, label='$R^2$')
ax2.set_ylabel('$R^2$', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 1)

fig.suptitle('Approximation quality vs. signature truncation depth')
fig.tight_layout()
plt.show()