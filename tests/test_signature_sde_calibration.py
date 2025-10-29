"""
Tests for signature SDE calibration methods.

This module tests the private helper functions used in the signature-driven
SDE calibration pipeline.
"""

import numpy as np
import pytest

from signature_vol.calibration.signature_sde_calibration import (
    _compute_windowed_signatures,
    _estimate_drift_functional,
    _estimate_diffusion_functional,
    calibrate_signature_sde,
    evaluate_coefficients,
    calibrate_linear_operator_from_path,
)
from signature_vol.exact.batch_signature import BatchSignature
from signature_vol.exact.brownian_motion import generate_brownian_motion
from signature_vol.exact.linear_operator import SignatureLinearOperator


class TestComputeStreamingSignatures:
    """Tests for _compute_streaming_signatures function."""

    def test_basic_computation(self):
        """Test that streaming signatures are computed correctly."""
        # Simple path: linear in time
        times = np.array([0.0, 1.0, 2.0, 3.0])
        X = np.array([0.0, 1.0, 2.0, 3.0])

        sigs = _compute_windowed_signatures(X, times, order=2)

        # Check type
        assert isinstance(sigs, BatchSignature)

        # Check dimensions
        assert sigs.d == 2  # Time-augmented (time, value)
        assert sigs.level == 2
        assert sigs.n_samples == len(times) - 1  # n-1 signatures

    def test_output_shape(self):
        """Test that output has correct shape."""
        times = np.linspace(0, 1, 10)
        X = np.random.randn(10)

        sigs = _compute_windowed_signatures(X, times, order=2)

        # Should have 9 signatures for 10 time points
        assert sigs.n_samples == 9
        assert sigs.array.shape[0] == 9

    def test_different_orders(self):
        """Test computation with different signature orders."""
        times = np.linspace(0, 1, 5)
        X = np.random.randn(5)

        for order in [1, 2, 3]:
            sigs = _compute_windowed_signatures(X, times, order=order)
            assert sigs.level == order
            assert sigs.n_samples == 4

    def test_constant_path(self):
        """Test with a constant path (no movement)."""
        times = np.array([0.0, 1.0, 2.0])
        X = np.array([1.0, 1.0, 1.0])

        sigs = _compute_windowed_signatures(X, times, order=2)

        # Should still compute signatures
        assert sigs.n_samples == 2
        assert not np.any(np.isnan(sigs.array))


class TestEstimateDriftFunctional:
    """Tests for _estimate_drift_functional function."""

    def test_zero_drift(self):
        """Test estimation with zero drift (pure diffusion)."""
        N = 100
        T = 1.0

        # Generate pure Brownian motion (zero drift)
        path = generate_brownian_motion(n_steps=N - 1, T=T, seed=42)
        times = path[:, 0]
        X = path[:, 1]

        # Compute signatures
        sigs = _compute_windowed_signatures(X, times, order=2)

        # Estimate drift
        beta = _estimate_drift_functional(X, times, sigs)

        # Check type
        assert isinstance(beta, SignatureLinearOperator)
        assert beta.level == 2
        assert beta.d == 2

        # Check that coefficients are finite (not NaN or inf)
        assert np.all(np.isfinite(beta.coeffs))

    def test_constant_drift(self):
        """Test estimation with constant drift."""
        N = 200
        times = np.linspace(0, 1, N)
        dt = 1 / (N - 1)
        drift = 2.0

        # Generate process with constant drift
        X = np.zeros(N)
        for i in range(1, N):
            X[i] = X[i - 1] + drift * dt + 0.1 * np.random.randn() * np.sqrt(dt)

        sigs = _compute_windowed_signatures(X, times, order=2)
        beta = _estimate_drift_functional(X, times, sigs)

        # The constant term should capture the drift
        # (first coefficient corresponds to constant term)
        assert isinstance(beta, SignatureLinearOperator)
        assert len(beta.coeffs) > 0

    def test_output_dimensions(self):
        """Test that output has correct dimensions."""
        times = np.linspace(0, 1, 50)
        X = np.random.randn(50)

        sigs = _compute_windowed_signatures(X, times, order=2)
        beta = _estimate_drift_functional(X, times, sigs)

        # Check that coefficients match signature dimension
        assert len(beta.coeffs) == sigs.sig_length
        assert beta.d == sigs.d
        assert beta.level == sigs.level

    def test_handles_different_time_spacing(self):
        """Test with non-uniform time spacing."""
        # Non-uniform times
        times = np.array([0.0, 0.1, 0.3, 0.7, 1.0])
        X = np.array([0.0, 0.2, 0.5, 0.9, 1.1])

        sigs = _compute_windowed_signatures(X, times, order=2)
        beta = _estimate_drift_functional(X, times, sigs)

        assert isinstance(beta, SignatureLinearOperator)
        assert not np.any(np.isnan(beta.coeffs))


class TestEstimateDiffusionFunctional:
    """Tests for _estimate_diffusion_functional function."""

    def test_constant_diffusion(self):
        """Test estimation with constant diffusion."""
        N = 200
        T = 1.0
        sigma = 0.3

        # Generate Brownian motion with constant volatility
        path = generate_brownian_motion(n_steps=N - 1, T=T, seed=42)
        times = path[:, 0]
        X = path[:, 1] * sigma  # Scale by volatility

        sigs = _compute_windowed_signatures(X, times, order=2)
        beta = _estimate_drift_functional(X, times, sigs)
        alpha = _estimate_diffusion_functional(X, times, sigs, beta)

        # Check type
        assert isinstance(alpha, SignatureLinearOperator)
        assert alpha.level == 2
        assert alpha.d == 2

        # Diffusion coefficients should be positive (or zero)
        # LinearRegression with positive=True ensures this
        assert np.all(alpha.coeffs >= 0)

    def test_requires_drift_estimate(self):
        """Test that diffusion estimation uses drift estimate."""
        times = np.linspace(0, 1, 50)
        X = np.random.randn(50)

        sigs = _compute_windowed_signatures(X, times, order=2)
        beta = _estimate_drift_functional(X, times, sigs)

        # Should not raise an error
        alpha = _estimate_diffusion_functional(X, times, sigs, beta)

        assert isinstance(alpha, SignatureLinearOperator)

    def test_output_dimensions(self):
        """Test that output has correct dimensions."""
        times = np.linspace(0, 1, 50)
        X = np.random.randn(50)

        sigs = _compute_windowed_signatures(X, times, order=2)
        beta = _estimate_drift_functional(X, times, sigs)
        alpha = _estimate_diffusion_functional(X, times, sigs, beta)

        # Check dimensions match
        assert len(alpha.coeffs) == sigs.sig_length
        assert alpha.d == sigs.d
        assert alpha.level == sigs.level

    def test_positivity_constraint(self):
        """Test that diffusion coefficients are non-negative."""
        np.random.seed(123)
        times = np.linspace(0, 1, 100)
        X = np.cumsum(np.random.randn(100) * 0.2)

        sigs = _compute_windowed_signatures(X, times, order=2)
        beta = _estimate_drift_functional(X, times, sigs)
        alpha = _estimate_diffusion_functional(X, times, sigs, beta)

        # All coefficients should be non-negative due to positive=True
        assert np.all(alpha.coeffs >= 0)

    def test_handles_different_time_spacing(self):
        """Test with non-uniform time spacing."""
        times = np.array([0.0, 0.1, 0.3, 0.7, 1.0])
        X = np.random.randn(5)

        sigs = _compute_windowed_signatures(X, times, order=2)
        beta = _estimate_drift_functional(X, times, sigs)
        alpha = _estimate_diffusion_functional(X, times, sigs, beta)

        assert isinstance(alpha, SignatureLinearOperator)
        assert not np.any(np.isnan(alpha.coeffs))


class TestCalibrateSignatureSDE:
    """Tests for the main calibrate_signature_sde function."""

    def test_full_calibration_pipeline(self):
        """Test the complete calibration pipeline."""
        np.random.seed(42)
        N = 100
        times = np.linspace(0, 1, N)

        # Generate synthetic data
        X = np.cumsum(np.random.randn(N) * 0.2)

        # Run calibration
        beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=2)

        # Check all outputs
        assert isinstance(beta, SignatureLinearOperator)
        assert isinstance(alpha, SignatureLinearOperator)
        assert isinstance(sigs, BatchSignature)

        # Check consistency
        assert beta.level == 2
        assert alpha.level == 2
        assert sigs.level == 2

    def test_different_signature_orders(self):
        """Test calibration with different signature orders."""
        np.random.seed(42)
        times = np.linspace(0, 1, 50)
        X = np.random.randn(50)

        for order in [1, 2, 3]:
            beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=order)

            assert beta.level == order
            assert alpha.level == order
            assert sigs.level == order

    def test_ornstein_uhlenbeck_process(self):
        """Test calibration on an Ornstein-Uhlenbeck process."""
        np.random.seed(42)
        N = 500
        T = 1.0
        times = np.linspace(0, T, N)
        dt = T / (N - 1)

        # OU parameters
        theta = 0.5
        sigma = 0.2
        X0 = 1.0

        # Simulate OU process
        X = np.zeros(N)
        X[0] = X0
        for i in range(1, N):
            dW = np.random.randn() * np.sqrt(dt)
            X[i] = X[i - 1] - theta * X[i - 1] * dt + sigma * dW

        # Calibrate
        beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=2)

        # Should successfully calibrate (even if not perfectly matching OU)
        assert isinstance(beta, SignatureLinearOperator)
        assert isinstance(alpha, SignatureLinearOperator)

        # Diffusion should be positive
        assert np.all(alpha.coeffs >= 0)

    def test_minimum_data_requirements(self):
        """Test with minimal amount of data."""
        # Need at least 3 points for calibration
        times = np.array([0.0, 0.5, 1.0])
        X = np.array([0.0, 0.5, 1.0])

        beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=1)

        assert isinstance(beta, SignatureLinearOperator)
        assert isinstance(alpha, SignatureLinearOperator)

    def test_consistency_with_returned_signatures(self):
        """Test that returned signatures match internal computation."""
        times = np.linspace(0, 1, 50)
        X = np.random.randn(50)

        beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=2)

        # Recompute signatures independently
        sigs_check = _compute_windowed_signatures(X, times, order=2)

        # Should match
        np.testing.assert_array_almost_equal(sigs.array, sigs_check.array)

    def test_non_uniform_time_spacing(self):
        """Test calibration with non-uniform time spacing."""
        # Exponentially spaced times
        times = np.concatenate([[0], np.cumsum(np.exp(np.linspace(-2, 0, 50)))])
        times = times / times[-1]  # Normalize to [0, 1]
        X = np.random.randn(51)

        beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=2)

        assert isinstance(beta, SignatureLinearOperator)
        assert isinstance(alpha, SignatureLinearOperator)
        assert not np.any(np.isnan(beta.coeffs))
        assert not np.any(np.isnan(alpha.coeffs))


class TestEdgeCases:
    """Tests for edge cases and potential failure modes."""

    def test_very_short_path(self):
        """Test with the minimum viable path length."""
        times = np.array([0.0, 0.5, 1.0])
        X = np.array([0.0, 1.0, 2.0])

        # Should work with minimal data
        beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=1)

        assert beta is not None
        assert alpha is not None

    def test_large_increments(self):
        """Test with large path increments."""
        times = np.linspace(0, 1, 20)
        X = np.cumsum(np.random.randn(20) * 10)  # Large volatility

        beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=2)

        # Should handle large values
        assert np.all(np.isfinite(beta.coeffs))
        assert np.all(np.isfinite(alpha.coeffs))

    def test_near_constant_path(self):
        """Test with a path that barely moves."""
        times = np.linspace(0, 1, 50)
        X = np.ones(50) + np.random.randn(50) * 1e-6  # Very small noise

        beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=2)

        # Should still compute (even if coefficients are small)
        assert np.all(np.isfinite(beta.coeffs))
        assert np.all(np.isfinite(alpha.coeffs))


class TestEvaluateCoefficients:
    """Tests for the evaluate_coefficients function."""

    def test_basic_evaluation(self):
        """Test basic evaluation of coefficients along a path."""
        np.random.seed(42)
        N = 50
        times = np.linspace(0, 1, N)
        X = np.cumsum(np.random.randn(N) * 0.2)

        # Calibrate
        beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=2)

        # Evaluate coefficients
        drift_coeffs, diffusion_coeffs = evaluate_coefficients(beta, alpha, X, times)

        # Check output shapes
        # Should have n-2 coefficients (same as in calibration)
        assert len(drift_coeffs) == N - 2
        assert len(diffusion_coeffs) == N - 2

    def test_output_types(self):
        """Test that outputs are numpy arrays."""
        times = np.linspace(0, 1, 30)
        X = np.random.randn(30)

        beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=2)
        drift_coeffs, diffusion_coeffs = evaluate_coefficients(beta, alpha, X, times)

        assert isinstance(drift_coeffs, np.ndarray)
        assert isinstance(diffusion_coeffs, np.ndarray)

    def test_finiteness(self):
        """Test that evaluated coefficients are finite."""
        np.random.seed(123)
        times = np.linspace(0, 1, 40)
        X = np.cumsum(np.random.randn(40) * 0.3)

        beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=2)
        drift_coeffs, diffusion_coeffs = evaluate_coefficients(beta, alpha, X, times)

        # All values should be finite (no NaN or inf)
        assert np.all(np.isfinite(drift_coeffs))
        assert np.all(np.isfinite(diffusion_coeffs))

    def test_diffusion_positivity(self):
        """Test that diffusion coefficients are non-negative."""
        np.random.seed(42)
        times = np.linspace(0, 1, 100)
        X = np.cumsum(np.random.randn(100) * 0.2)

        beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=2)
        drift_coeffs, diffusion_coeffs = evaluate_coefficients(beta, alpha, X, times)

        # Diffusion coefficients should be non-negative
        # (since alpha coefficients are constrained to be positive)
        assert np.all(diffusion_coeffs >= 0)

    def test_consistency_with_calibration(self):
        """Test that evaluated coefficients are consistent with calibration."""
        np.random.seed(42)
        times = np.linspace(0, 1, 50)
        X = np.cumsum(np.random.randn(50) * 0.2)

        # Calibrate
        beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=2)

        # Evaluate on same path
        drift_coeffs, diffusion_coeffs = evaluate_coefficients(beta, alpha, X, times)

        # The evaluated drift should roughly match the observed drift
        # Compute observed drift
        dt = np.diff(times)
        dX = np.diff(X)
        observed_drift = dX / dt

        # Check that evaluated drift and observed drift are in similar range
        # (not exact match due to signature approximation)
        assert drift_coeffs.min() <= observed_drift[:-1].max()
        assert drift_coeffs.max() >= observed_drift[:-1].min()

    def test_constant_path(self):
        """Test evaluation on a constant path."""
        times = np.linspace(0, 1, 20)
        X = np.ones(20)

        beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=2)
        drift_coeffs, diffusion_coeffs = evaluate_coefficients(beta, alpha, X, times)

        # For constant path, drift should be near zero
        assert np.abs(drift_coeffs).max() < 1e-10

    def test_different_signature_levels(self):
        """Test evaluation with different signature levels."""
        times = np.linspace(0, 1, 40)
        X = np.random.randn(40)

        for level in [1, 2, 3]:
            beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=level)
            drift_coeffs, diffusion_coeffs = evaluate_coefficients(
                beta, alpha, X, times
            )

            assert len(drift_coeffs) == len(times) - 2
            assert len(diffusion_coeffs) == len(times) - 2

    def test_linear_drift_path(self):
        """Test evaluation on a path with linear drift."""
        N = 100
        times = np.linspace(0, 1, N)
        drift_true = 2.0
        X = drift_true * times + 0.1 * np.random.randn(N)

        beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=2)
        drift_coeffs, diffusion_coeffs = evaluate_coefficients(beta, alpha, X, times)

        # Mean drift should be close to true drift
        assert np.abs(drift_coeffs.mean() - drift_true) < 0.5

    def test_ou_process_evaluation(self):
        """Test evaluation on an Ornstein-Uhlenbeck process."""
        np.random.seed(42)
        N = 200
        T = 1.0
        times = np.linspace(0, T, N)
        dt = T / (N - 1)

        # OU parameters
        theta = 0.5
        sigma = 0.2
        X0 = 1.0

        # Simulate OU process
        X = np.zeros(N)
        X[0] = X0
        for i in range(1, N):
            dW = np.random.randn() * np.sqrt(dt)
            X[i] = X[i - 1] - theta * X[i - 1] * dt + sigma * dW

        beta, alpha, sigs = calibrate_signature_sde(X, times, sig_order=2)
        drift_coeffs, diffusion_coeffs = evaluate_coefficients(beta, alpha, X, times)

        # Diffusion should be roughly constant around sigma^2
        mean_diffusion = np.mean(diffusion_coeffs)
        assert 0.5 * sigma**2 < mean_diffusion < 2.0 * sigma**2


class TestWindowedSignatures:
    """Tests for windowed signature computation."""

    def test_windowed_basic(self):
        """Test basic windowed signature computation."""
        times = np.linspace(0, 1, 20)
        X = np.cumsum(np.random.randn(20) * 0.2)
        window_length = 5

        sigs = _compute_windowed_signatures(
            X, times, order=2, window_length=window_length
        )

        assert isinstance(sigs, BatchSignature)
        assert sigs.n_samples == len(times) - window_length

    def test_window_length_validation_too_small(self):
        """Test that window_length < 2 raises ValueError."""
        times = np.linspace(0, 1, 10)
        X = np.random.randn(10)

        with pytest.raises(ValueError, match="must be at least 2"):
            _compute_windowed_signatures(X, times, order=2, window_length=1)

    def test_window_length_validation_too_large(self):
        """Test that window_length >= n raises ValueError."""
        times = np.linspace(0, 1, 10)
        X = np.random.randn(10)

        with pytest.raises(ValueError, match="less than number of data points"):
            _compute_windowed_signatures(X, times, order=2, window_length=10)

    def test_windowed_drift_estimation(self):
        """Test drift estimation with windowed signatures."""
        N = 100
        times = np.linspace(0, 1, N)
        X = np.cumsum(np.random.randn(N) * 0.2)
        window_length = 20

        sigs = _compute_windowed_signatures(
            X, times, order=2, window_length=window_length
        )
        beta = _estimate_drift_functional(X, times, sigs, window_length=window_length)

        assert isinstance(beta, SignatureLinearOperator)
        assert np.all(np.isfinite(beta.coeffs))

    def test_windowed_diffusion_estimation(self):
        """Test diffusion estimation with windowed signatures."""
        N = 100
        times = np.linspace(0, 1, N)
        X = np.cumsum(np.random.randn(N) * 0.2)
        window_length = 20

        sigs = _compute_windowed_signatures(
            X, times, order=2, window_length=window_length
        )
        beta = _estimate_drift_functional(X, times, sigs, window_length=window_length)
        alpha = _estimate_diffusion_functional(
            X, times, sigs, beta, window_length=window_length
        )

        assert isinstance(alpha, SignatureLinearOperator)
        assert np.all(alpha.coeffs >= 0)

    def test_windowed_full_calibration(self):
        """Test full calibration with windowing."""
        N = 100
        times = np.linspace(0, 1, N)
        X = np.cumsum(np.random.randn(N) * 0.2)
        window_length = 30

        beta, alpha, sigs = calibrate_signature_sde(
            X, times, sig_order=2, window_length=window_length
        )

        assert isinstance(beta, SignatureLinearOperator)
        assert isinstance(alpha, SignatureLinearOperator)
        assert isinstance(sigs, BatchSignature)

    def test_insufficient_data_windowed(self):
        """Test that insufficient data with windowing raises ValueError."""
        times = np.linspace(0, 1, 30)
        X = np.random.randn(30)
        window_length = 25

        with pytest.raises(ValueError, match="Insufficient data"):
            calibrate_signature_sde(X, times, sig_order=2, window_length=window_length)


class TestCalibrateLinearOperatorFromPath:
    """Tests for calibrate_linear_operator_from_path function."""

    def test_basic_calibration(self):
        """Test basic linear operator calibration from path."""
        N = 100
        T = 1.0
        path = generate_brownian_motion(n_steps=N - 1, T=T, seed=42)
        times = path[:, 0]
        W = path[:, 1]

        # GBM parameters
        mu = 0.05
        sigma = 0.2
        x0 = 0.0
        X_values = x0 + (mu - 0.5 * sigma**2) * times + sigma * W

        path_augmented = np.column_stack([times, W])
        ell = calibrate_linear_operator_from_path(path_augmented, X_values, sig_order=2)

        assert isinstance(ell, SignatureLinearOperator)
        assert ell.level == 2

    def test_path_length_mismatch(self):
        """Test that path and X_values length mismatch raises ValueError."""
        path = np.random.randn(50, 2)
        X_values = np.random.randn(40)

        with pytest.raises(ValueError, match="path length.*must match"):
            calibrate_linear_operator_from_path(path, X_values, sig_order=2)

    def test_windowed_calibration(self):
        """Test calibration with windowed signatures."""
        N = 100
        path = generate_brownian_motion(n_steps=N - 1, T=1.0, seed=42)
        times = path[:, 0]
        W = path[:, 1]
        X_values = 0.1 * times + 0.2 * W

        path_augmented = np.column_stack([times, W])
        window_length = 30

        ell = calibrate_linear_operator_from_path(
            path_augmented, X_values, sig_order=2, window_length=window_length
        )

        assert isinstance(ell, SignatureLinearOperator)
        assert ell.level == 2

    def test_window_length_validation_small(self):
        """Test that window_length < 2 raises ValueError."""
        path = np.random.randn(50, 2)
        X_values = np.random.randn(50)

        with pytest.raises(ValueError, match="must be at least 2"):
            calibrate_linear_operator_from_path(
                path, X_values, sig_order=2, window_length=1
            )

    def test_window_length_validation_large(self):
        """Test that window_length >= n raises ValueError."""
        path = np.random.randn(50, 2)
        X_values = np.random.randn(50)

        with pytest.raises(ValueError, match="less than data length"):
            calibrate_linear_operator_from_path(
                path, X_values, sig_order=2, window_length=50
            )

    def test_insufficient_data_windowed(self):
        """Test that insufficient data with windowing raises ValueError."""
        path = np.random.randn(40, 2)
        X_values = np.random.randn(40)
        window_length = 35

        with pytest.raises(ValueError, match="Insufficient data"):
            calibrate_linear_operator_from_path(
                path, X_values, sig_order=2, window_length=window_length
            )

    def test_different_signature_orders(self):
        """Test calibration with different signature orders."""
        N = 80
        path = generate_brownian_motion(n_steps=N - 1, T=1.0, seed=42)
        times = path[:, 0]
        W = path[:, 1]
        X_values = 0.1 * times + 0.2 * W

        path_augmented = np.column_stack([times, W])

        for order in [1, 2, 3]:
            ell = calibrate_linear_operator_from_path(
                path_augmented, X_values, sig_order=order
            )
            assert ell.level == order

    def test_operator_evaluation(self):
        """Test that calibrated operator can evaluate on signatures."""
        N = 100
        path = generate_brownian_motion(n_steps=N - 1, T=1.0, seed=42)
        times = path[:, 0]
        W = path[:, 1]
        X_values = 0.1 * times + 0.2 * W

        path_augmented = np.column_stack([times, W])
        ell = calibrate_linear_operator_from_path(path_augmented, X_values, sig_order=2)

        # Evaluate on path
        sigs = BatchSignature.from_streaming_path(path_augmented, level=2)
        predictions = sigs.array @ ell.coeffs

        # Predictions should be close to X_values[1:]
        error = np.abs(predictions - X_values[1:]).mean()
        assert error < 0.1  # Should fit reasonably well
