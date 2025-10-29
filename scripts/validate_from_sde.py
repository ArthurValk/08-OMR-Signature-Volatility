"""Validation of from_sde linear operator construction.

Verifies that operators constructed via from_sde satisfy their SDEs
and that X_t = ⟨ℓ, Sig[Ŵ]⟩ matches analytical solutions.
"""

import numpy as np
import matplotlib.pyplot as plt

from output import SAVE_PATH
from scripts import analytical_solution_gbm, analytical_solution_ou
from signature_vol.exact.linear_operator import SignatureLinearOperatorBuilder
from signature_vol.exact.batch_signature import BatchSignature


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

    fig, axes = plt.subplots(3, len(levels), figsize=(18, 14))

    for idx, level in enumerate(levels):
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

        # Plot path comparison
        axes[0, idx].plot(
            times[1:], X_analytical, "k-", label="Analytical", linewidth=2
        )
        axes[0, idx].plot(
            times[1:], X_signature, "r--", label="Signature", linewidth=1.5, alpha=0.7
        )
        axes[0, idx].set_xlabel("Time")
        axes[0, idx].set_ylabel("X(t)")
        axes[0, idx].set_title(f"GBM: Level {level}")
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3)

        # Plot error
        axes[1, idx].plot(times[1:], error, "b-", linewidth=1)
        axes[1, idx].set_xlabel("Time")
        axes[1, idx].set_ylabel("Absolute Error")
        axes[1, idx].set_title(f"Error (Max: {max_error:.2e})")
        axes[1, idx].grid(True, alpha=0.3)
        if max_error > 0:
            axes[1, idx].set_yscale("log")

        # Plot operator coefficients
        coeff_indices = np.arange(len(ell.coeffs))
        axes[2, idx].bar(coeff_indices, np.abs(ell.coeffs), color="green", alpha=0.7)
        axes[2, idx].set_xlabel("Coefficient Index")
        axes[2, idx].set_ylabel("Absolute Value")
        axes[2, idx].set_title("Linear Operator Structure")
        axes[2, idx].grid(True, alpha=0.3, axis="y")
        if np.any(ell.coeffs != 0):
            axes[2, idx].set_yscale("log")

        # Add signature term labels
        if level == 1:
            labels = ["ø", "1", "2"]
            axes[2, idx].set_xticks(coeff_indices)
            axes[2, idx].set_xticklabels(labels)
        elif level == 2:
            labels = ["ø", "1", "2", "11", "12", "21", "22"]
            axes[2, idx].set_xticks(coeff_indices)
            axes[2, idx].set_xticklabels(labels, rotation=45)

    plt.tight_layout()
    save_path_gbm = SAVE_PATH / "validate_from_sde_gbm.png"
    plt.savefig(save_path_gbm, dpi=150, bbox_inches="tight")
    print(f"GBM validation plot saved to: {save_path_gbm}")
    print()

    # Test 2: Ornstein-Uhlenbeck Process
    print("Test 2: Ornstein-Uhlenbeck Process")
    print("-" * 70)

    # Parameters: dX = -theta*X*dt + sigma*dW
    theta = 0.5
    sigma_ou = 0.3
    x0_ou = 1.0

    print(f"SDE: dX = -{theta}*X*dt + {sigma_ou}*dW, X_0 = {x0_ou}")
    print()

    # For OU, we have a=0, b=-theta, alpha=sigma, beta=0

    fig, axes = plt.subplots(3, len(levels), figsize=(18, 14))

    for idx, level in enumerate(levels):
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

        # Plot path comparison
        axes[0, idx].plot(
            times[1:], X_analytical, "k-", label="Analytical", linewidth=2
        )
        axes[0, idx].plot(
            times[1:], X_signature, "r--", label="Signature", linewidth=1.5, alpha=0.7
        )
        axes[0, idx].axhline(
            y=0, color="gray", linestyle=":", alpha=0.5, label="Mean level"
        )
        axes[0, idx].set_xlabel("Time")
        axes[0, idx].set_ylabel("X(t)")
        axes[0, idx].set_title(f"OU Process: Level {level}")
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3)

        # Plot error
        axes[1, idx].plot(times[1:], error, "b-", linewidth=1)
        axes[1, idx].set_xlabel("Time")
        axes[1, idx].set_ylabel("Absolute Error")
        axes[1, idx].set_title(f"Error (Max: {max_error:.2e})")
        axes[1, idx].grid(True, alpha=0.3)
        if max_error > 0:
            axes[1, idx].set_yscale("log")

        # Plot operator coefficients
        coeff_indices = np.arange(len(ell.coeffs))
        axes[2, idx].bar(coeff_indices, np.abs(ell.coeffs), color="purple", alpha=0.7)
        axes[2, idx].set_xlabel("Coefficient Index")
        axes[2, idx].set_ylabel("Absolute Value")
        axes[2, idx].set_title("Linear Operator Structure")
        axes[2, idx].grid(True, alpha=0.3, axis="y")
        if np.any(ell.coeffs != 0):
            axes[2, idx].set_yscale("log")

        # Add signature term labels
        if level == 1:
            labels = ["ø", "1", "2"]
            axes[2, idx].set_xticks(coeff_indices)
            axes[2, idx].set_xticklabels(labels)
        elif level == 2:
            labels = ["ø", "1", "2", "11", "12", "21", "22"]
            axes[2, idx].set_xticks(coeff_indices)
            axes[2, idx].set_xticklabels(labels, rotation=45)
        elif level == 3:
            # For level 3, just show indices
            axes[2, idx].set_xticks(coeff_indices[::2])  # Show every other index

    plt.tight_layout()
    save_path_ou = SAVE_PATH / "validate_from_sde_ou.png"
    plt.savefig(save_path_ou, dpi=150, bbox_inches="tight")
    print(f"OU validation plot saved to: {save_path_ou}")


if __name__ == "__main__":
    main()
