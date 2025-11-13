"""Signature approximation quality analysis on synthetic data."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from output import SAVE_PATH
from signature_vol.exact.signature import Signature

matplotlib.use("Agg")

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
