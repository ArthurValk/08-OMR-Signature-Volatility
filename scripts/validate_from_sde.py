"""Validation of from_sde linear operator construction.

Verifies that operators constructed via from_sde satisfy their SDEs
and that X_t = ⟨ℓ, Sig[Ŵ]⟩ matches analytical solutions.
"""

import numpy as np
import matplotlib.pyplot as plt

from signature_vol.exact.linear_operator import SignatureLinearOperatorBuilder
from signature_vol.exact.batch_signature import BatchSignature
from scripts import (
    analytical_solution_gbm,
    analytical_solution_ou,
    index_to_word,
)
from output import SAVE_PATH


def main():
    np.random.seed(42)

    # Simulation parameters
    N = 1000
    T = 1.0
    times = np.linspace(0, T, N)
    dt = T / N

    # Simulate Brownian motion
    dW = np.random.normal(0, np.sqrt(dt), N - 1)
    W = np.zeros(N)
    W[1:] = np.cumsum(dW)

    # Create time-augmented path [t, W(t)]
    time_brownian_path = np.column_stack([times, W])

    print("=" * 70)
    print("VALIDATION OF from_sde OPERATOR CONSTRUCTION")
    print("=" * 70)
    print()

    # Test 1: Geometric Brownian Motion (GBM)
    print("Test 1: Geometric Brownian Motion")
    print("-" * 70)

    # Parameters for log-GBM: dX = (mu - sigma^2/2)dt + sigma*dW
    mu = 0.05
    sigma = 0.2
    x0_gbm = 0.0

    print(f"SDE: dX = ({mu} - {sigma}^2/2)dt + {sigma}dW, X_0 = {x0_gbm}")
    print()

    # For GBM, we have a=mu-sigma^2/2, b=0, alpha=sigma, beta=0
    a_gbm = mu - 0.5 * sigma**2

    # Test at different truncation levels
    levels = [1, 2, 3]

    # Store results for all levels
    results = []

    for level in levels:
        print(f"  Level {level}:")

        # Construct operator using from_sde
        ell_builder = SignatureLinearOperatorBuilder.from_sde(
            dimension=2, level=level, x0=x0_gbm, a=a_gbm, b=0.0, alpha=sigma, beta=0.0
        )
        ell = ell_builder.build()

        print(f"    Operator coefficients: {ell.coeffs[: min(7, len(ell.coeffs))]}")

        # Compute signatures
        sigs = BatchSignature.from_streaming_path(time_brownian_path, level=level)

        # Evaluate X_t = ⟨ℓ, Sig[Ŵ]⟩
        X_signature = sigs.array @ ell.coeffs

        # Analytical solution
        X_analytical = analytical_solution_gbm(times[1:], W[1:], x0_gbm, mu, sigma)

        # Compute error
        error = np.abs(X_signature - X_analytical)
        max_error = np.max(error)
        mean_error = np.mean(error)

        print(f"    Max error: {max_error:.6e}")
        print(f"    Mean error: {mean_error:.6e}")
        print()

        results.append(
            {
                "level": level,
                "X_analytical": X_analytical,
                "X_signature": X_signature,
                "error": error,
                "max_error": max_error,
                "ell": ell,
            }
        )

    # Create one figure with three subplots in a row
    # Define colors and line styles for different levels
    colors = ["blue", "green", "red"]
    linestyles = ["-", "--", ":"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: Path comparison - all levels on one plot
    ax1 = axes[0]
    # Plot analytical solution (same for all levels, so plot once)
    ax1.plot(
        times[1:], results[0]["X_analytical"], "k-", label="Analytical", linewidth=2
    )
    # Plot signature reconstruction for each level
    for idx, res in enumerate(results):
        ax1.plot(
            times[1:],
            res["X_signature"],
            color=colors[idx],
            linestyle=linestyles[idx],
            label=f"Signature (Level {res['level']})",
            linewidth=1.5,
            alpha=0.8,
        )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("X(t)")
    ax1.set_title("GBM: Path Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Error plots - all levels on one plot
    ax2 = axes[1]
    for idx, res in enumerate(results):
        ax2.plot(
            times[1:],
            res["error"],
            color=colors[idx],
            linestyle=linestyles[idx],
            label=f"Level {res['level']} (Max: {res['max_error']:.2e})",
            linewidth=1.5,
        )
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Absolute Error")
    ax2.set_title("GBM: Reconstruction Error")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Operator coefficients - all levels combined
    ax3 = axes[2]
    # Determine the maximum number of coefficients across all levels
    max_coeffs = max(len(res["ell"].coeffs) for res in results)

    # Width of bars and positions
    bar_width = 0.25
    x_positions = np.arange(max_coeffs)

    # Plot bars for each level
    for idx, res in enumerate(results):
        n_coeffs = len(res["ell"].coeffs)
        offset = (idx - 1) * bar_width  # Center the bars
        ax3.bar(
            x_positions[:n_coeffs] + offset,
            np.abs(res["ell"].coeffs),
            width=bar_width,
            color=colors[idx],
            alpha=0.7,
            label=f"Level {res['level']}",
        )

    ax3.set_xlabel("Coefficient Index")
    ax3.set_ylabel("Absolute Value")
    ax3.set_title("Linear Operator Structure")
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # Generate word labels using index_to_word function
    dimension = 2  # Time-augmented path has dimension 2
    max_level = max(res["level"] for res in results)
    # Show labels for all coefficients
    word_labels = [index_to_word(i, dimension, max_level) for i in range(max_coeffs)]
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(word_labels, rotation=45, ha="right")

    plt.tight_layout()
    save_path_gbm = SAVE_PATH / "validate_from_sde_gbm.png"
    plt.savefig(save_path_gbm, dpi=150, bbox_inches="tight")
    print(f"GBM validation plot saved to: {save_path_gbm}")
    plt.close(fig)
    print()

    # Test 2: Ornstein-Uhlenbeck Process
    np.random.seed(42)
    print("Test 2: Ornstein-Uhlenbeck Process")
    print("-" * 70)

    # Parameters: dX = -theta*X*dt + sigma*dW
    theta = 0.5
    sigma_ou = 0.3
    x0_ou = 1.0

    print(f"SDE: dX = -{theta}*X*dt + {sigma_ou}*dW, X_0 = {x0_ou}")
    print()

    # For OU, we have a=0, b=-theta, alpha=sigma, beta=0

    # Test at different truncation levels (including level 4 for OU)
    levels_ou = [1, 2, 3, 4]

    # Store results for all levels
    results_ou = []

    for level in levels_ou:
        print(f"  Level {level}:")

        # Construct operator using from_sde
        ell_builder = SignatureLinearOperatorBuilder.from_sde(
            dimension=2,
            level=level,
            x0=x0_ou,
            a=0.0,
            b=-theta,
            alpha=sigma_ou,
            beta=0.0,
        )
        ell = ell_builder.build()

        print(f"    Operator coefficients: {ell.coeffs[: min(7, len(ell.coeffs))]}")

        # Compute signatures
        sigs = BatchSignature.from_streaming_path(time_brownian_path, level=level)

        # Evaluate X_t = ⟨ℓ, Sig[Ŵ]⟩
        X_signature = sigs.array @ ell.coeffs

        # Analytical solution (using Euler for OU)
        X_analytical = analytical_solution_ou(times, W, x0_ou, theta, sigma_ou)
        X_analytical = X_analytical[1:]  # Match signature array length

        # Compute error
        error = np.abs(X_signature - X_analytical)
        max_error = np.max(error)
        mean_error = np.mean(error)

        print(f"    Max error: {max_error:.6e}")
        print(f"    Mean error: {mean_error:.6e}")
        print()

        results_ou.append(
            {
                "level": level,
                "X_analytical": X_analytical,
                "X_signature": X_signature,
                "error": error,
                "max_error": max_error,
                "ell": ell,
            }
        )

    # Create one figure with three subplots in a row
    # Define colors and line styles for OU (4 levels)
    colors_ou = ["blue", "green", "red", "purple"]
    linestyles_ou = ["-", "--", ":", "-."]

    fig_ou, axes_ou = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: Path comparison - all levels on one plot
    ax1_ou = axes_ou[0]
    # Plot analytical solution (same for all levels, so plot once)
    ax1_ou.plot(
        times[1:], results_ou[0]["X_analytical"], "k-", label="Analytical", linewidth=2
    )
    # Plot signature reconstruction for each level
    for idx, res in enumerate(results_ou):
        ax1_ou.plot(
            times[1:],
            res["X_signature"],
            color=colors_ou[idx],
            linestyle=linestyles_ou[idx],
            label=f"Signature (Level {res['level']})",
            linewidth=1.5,
            alpha=0.8,
        )
    ax1_ou.axhline(y=0, color="gray", linestyle=":", alpha=0.5, label="Mean level")
    ax1_ou.set_xlabel("Time")
    ax1_ou.set_ylabel("X(t)")
    ax1_ou.set_title("OU Process: Path Comparison")
    ax1_ou.legend()
    ax1_ou.grid(True, alpha=0.3)

    # Subplot 2: Error plots - all levels on one plot
    ax2_ou = axes_ou[1]
    for idx, res in enumerate(results_ou):
        ax2_ou.plot(
            times[1:],
            res["error"],
            color=colors_ou[idx],
            linestyle=linestyles_ou[idx],
            label=f"Level {res['level']} (Max: {res['max_error']:.2e})",
            linewidth=1.5,
        )
    ax2_ou.set_xlabel("Time")
    ax2_ou.set_ylabel("Absolute Error")
    ax2_ou.set_title("OU Process: Reconstruction Error")
    ax2_ou.set_yscale("log")
    ax2_ou.legend()
    ax2_ou.grid(True, alpha=0.3)

    # Subplot 3: Operator coefficients - all levels combined
    ax3_ou = axes_ou[2]
    # Determine the maximum number of coefficients across all levels
    max_coeffs_ou = max(len(res["ell"].coeffs) for res in results_ou)

    # Width of bars and positions
    bar_width_ou = 0.2  # Slightly narrower bars for 4 levels
    x_positions_ou = np.arange(max_coeffs_ou)

    # Plot bars for each level
    for idx, res in enumerate(results_ou):
        n_coeffs = len(res["ell"].coeffs)
        offset = (idx - 1.5) * bar_width_ou  # Center the bars around each position
        ax3_ou.bar(
            x_positions_ou[:n_coeffs] + offset,
            np.abs(res["ell"].coeffs),
            width=bar_width_ou,
            color=colors_ou[idx],
            alpha=0.7,
            label=f"Level {res['level']}",
        )

    ax3_ou.set_xlabel("Coefficient Index")
    ax3_ou.set_ylabel("Absolute Value")
    ax3_ou.set_title("Linear Operator Structure")
    ax3_ou.set_yscale("log")
    ax3_ou.legend()
    ax3_ou.grid(True, alpha=0.3, axis="y")

    # Generate word labels using index_to_word function
    max_level_ou = max(res["level"] for res in results_ou)
    # Show labels for all coefficients
    word_labels_ou = [
        index_to_word(i, dimension, max_level_ou) for i in range(max_coeffs_ou)
    ]
    ax3_ou.set_xticks(x_positions_ou)
    ax3_ou.set_xticklabels(word_labels_ou, rotation=45, ha="right")

    plt.tight_layout()
    save_path_ou = SAVE_PATH / "validate_from_sde_ou.png"
    plt.savefig(save_path_ou, dpi=150, bbox_inches="tight")
    print(f"OU validation plot saved to: {save_path_ou}")
    plt.close(fig_ou)


if __name__ == "__main__":
    main()
