"""Validation of calibrate_linear_operator_from_path.

Verifies that the calibration method recovers the same linear operator
as the theoretical from_sde construction.
"""

import numpy as np
import matplotlib

from scripts import (
    analytical_solution_gbm,
    analytical_solution_ou,
    index_to_word,
)
from output import SAVE_PATH

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from signature_vol.exact.linear_operator import SignatureLinearOperatorBuilder
from signature_vol.calibration.signature_sde_calibration import (
    calibrate_linear_operator_from_path,
)
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
    path = np.column_stack([times, W])

    print("=" * 70)
    print("VALIDATION: calibrate_linear_operator_from_path vs from_sde")
    print("=" * 70)
    print()

    # Test 1: Geometric Brownian Motion (GBM)
    print("Test 1: Geometric Brownian Motion (log-space)")
    print("-" * 70)

    # Parameters for log-GBM: dX = (mu - sigma^2/2)dt + sigma*dW
    mu = 0.05
    sigma = 0.2
    x0_gbm = 0.0
    a_gbm = mu - 0.5 * sigma**2

    print(f"SDE: dX = ({a_gbm:.4f})dt + {sigma}dW, X_0 = {x0_gbm}")
    print()

    # Simulate the process
    X_gbm = analytical_solution_gbm(times, W, x0_gbm, mu, sigma)

    # Test at different truncation levels
    levels = [1, 2, 3]
    fig, axes = plt.subplots(2, len(levels), figsize=(18, 10))

    for idx, level in enumerate(levels):
        print(f"  Level {level}:")

        # Method 1: Theoretical construction
        ell_theoretical = SignatureLinearOperatorBuilder.from_sde(
            dimension=2, level=level, x0=x0_gbm, a=a_gbm, b=0.0, alpha=sigma, beta=0.0
        ).build()

        # Method 2: Calibrate from path
        ell_calibrated = calibrate_linear_operator_from_path(
            path=path, X_values=X_gbm, sig_order=level, window_length=None
        )

        # Compare coefficients
        coeff_diff = ell_theoretical.coeffs - ell_calibrated.coeffs
        max_diff = np.max(np.abs(coeff_diff))
        rel_error = max_diff / (np.max(np.abs(ell_theoretical.coeffs)) + 1e-10)

        print(
            f"    Theoretical coeffs: {ell_theoretical.coeffs[: min(5, len(ell_theoretical.coeffs))]}"
        )
        print(
            f"    Calibrated coeffs:  {ell_calibrated.coeffs[: min(5, len(ell_calibrated.coeffs))]}"
        )
        print(f"    Max absolute diff:  {max_diff:.6e}")
        print(f"    Relative error:     {rel_error:.6e}")

        # Verify predictions match
        sigs = BatchSignature.from_streaming_path(path, level=level)
        X_theoretical = sigs.array @ ell_theoretical.coeffs
        X_calibrated = sigs.array @ ell_calibrated.coeffs

        pred_diff = np.max(np.abs(X_theoretical - X_calibrated))
        print(f"    Max prediction diff: {pred_diff:.6e}")
        print()

        # Plot coefficient comparison
        indices = np.arange(len(ell_theoretical.coeffs))
        width = 0.35

        axes[0, idx].bar(
            indices - width / 2,
            ell_theoretical.coeffs,
            width,
            label="from_sde",
            alpha=0.8,
            color="blue",
        )
        axes[0, idx].bar(
            indices + width / 2,
            ell_calibrated.coeffs,
            width,
            label="calibrated",
            alpha=0.8,
            color="red",
        )

        # Generate word labels for x-axis
        dimension = 2  # Time-augmented path has dimension 2
        word_labels = [index_to_word(i, dimension, level) for i in indices]

        axes[0, idx].set_xlabel("Signature Term")
        axes[0, idx].set_ylabel("Value")
        axes[0, idx].set_title(f"GBM Level {level}: Operator Coefficients")
        axes[0, idx].set_xticks(indices)
        axes[0, idx].set_xticklabels(word_labels, rotation=45, ha="right")
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3, axis="y")

        # Plot coefficient differences
        axes[1, idx].bar(indices, np.abs(coeff_diff), color="purple", alpha=0.7)
        axes[1, idx].set_xlabel("Signature Term")
        axes[1, idx].set_ylabel("Absolute Difference")
        axes[1, idx].set_title(f"Max Diff: {max_diff:.2e}")
        axes[1, idx].set_xticks(indices)
        axes[1, idx].set_xticklabels(word_labels, rotation=45, ha="right")
        axes[1, idx].grid(True, alpha=0.3, axis="y")
        if max_diff > 0:
            axes[1, idx].set_yscale("log")

    plt.tight_layout()
    save_path_gbm = SAVE_PATH / "validate_path_calibration_gbm.png"
    plt.savefig(save_path_gbm, dpi=150, bbox_inches="tight")
    print(f"GBM validation plot saved to: {save_path_gbm}")
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

    # Simulate the process
    X_ou = analytical_solution_ou(times, W, x0_ou, theta, sigma_ou)

    fig, axes = plt.subplots(2, len(levels), figsize=(18, 10))

    for idx, level in enumerate(levels):
        print(f"  Level {level}:")

        # Method 1: Theoretical construction
        ell_theoretical = SignatureLinearOperatorBuilder.from_sde(
            dimension=2,
            level=level,
            x0=x0_ou,
            a=0.0,
            b=-theta,
            alpha=sigma_ou,
            beta=0.0,
        ).build()

        # Method 2: Calibrate from path
        ell_calibrated = calibrate_linear_operator_from_path(
            path=path, X_values=X_ou, sig_order=level, window_length=None
        )

        # Compare coefficients
        coeff_diff = ell_theoretical.coeffs - ell_calibrated.coeffs
        max_diff = np.max(np.abs(coeff_diff))
        rel_error = max_diff / (np.max(np.abs(ell_theoretical.coeffs)) + 1e-10)

        print(
            f"    Theoretical coeffs: {ell_theoretical.coeffs[: min(5, len(ell_theoretical.coeffs))]}"
        )
        print(
            f"    Calibrated coeffs:  {ell_calibrated.coeffs[: min(5, len(ell_calibrated.coeffs))]}"
        )
        print(f"    Max absolute diff:  {max_diff:.6e}")
        print(f"    Relative error:     {rel_error:.6e}")

        # Verify predictions match
        sigs = BatchSignature.from_streaming_path(path, level=level)
        X_theoretical = sigs.array @ ell_theoretical.coeffs
        X_calibrated = sigs.array @ ell_calibrated.coeffs

        pred_diff = np.max(np.abs(X_theoretical - X_calibrated))
        print(f"    Max prediction diff: {pred_diff:.6e}")
        print()

        # Plot coefficient comparison
        indices = np.arange(len(ell_theoretical.coeffs))
        width = 0.35

        axes[0, idx].bar(
            indices - width / 2,
            ell_theoretical.coeffs,
            width,
            label="from_sde",
            alpha=0.8,
            color="blue",
        )
        axes[0, idx].bar(
            indices + width / 2,
            ell_calibrated.coeffs,
            width,
            label="calibrated",
            alpha=0.8,
            color="red",
        )

        # Generate word labels for x-axis
        dimension = 2  # Time-augmented path has dimension 2
        word_labels = [index_to_word(i, dimension, level) for i in indices]

        axes[0, idx].set_xlabel("Signature Term")
        axes[0, idx].set_ylabel("Value")
        axes[0, idx].set_title(f"OU Level {level}: Operator Coefficients")
        axes[0, idx].set_xticks(indices)
        axes[0, idx].set_xticklabels(word_labels, rotation=45, ha="right")
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3, axis="y")

        # Plot coefficient differences
        axes[1, idx].bar(indices, np.abs(coeff_diff), color="purple", alpha=0.7)
        axes[1, idx].set_xlabel("Signature Term")
        axes[1, idx].set_ylabel("Absolute Difference")
        axes[1, idx].set_title(f"Max Diff: {max_diff:.2e}")
        axes[1, idx].set_xticks(indices)
        axes[1, idx].set_xticklabels(word_labels, rotation=45, ha="right")
        axes[1, idx].grid(True, alpha=0.3, axis="y")
        if max_diff > 0:
            axes[1, idx].set_yscale("log")

    plt.tight_layout()
    save_path_ou = SAVE_PATH / "validate_path_calibration_ou.png"
    plt.savefig(save_path_ou, dpi=150, bbox_inches="tight")
    print(f"OU validation plot saved to: {save_path_ou}")


if __name__ == "__main__":
    main()
