"""Signature-driven SDE calibration methods.

Implements calibration of SDEs where drift and diffusion coefficients
are linear functionals of the path signature.
"""

import numpy as np
import iisignature  # type: ignore
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional

from signature_vol.exact.batch_signature import BatchSignature
from signature_vol.exact.linear_operator import SignatureLinearOperator


def _compute_windowed_signatures(
    X: np.ndarray,
    times: np.ndarray,
    order: int = 2,
    window_length: Optional[int] = None,
) -> BatchSignature:
    """
    Compute path signatures over sliding windows or from the start.

    Parameters:
    ----------
    X : np.ndarray
        Observed path values, shape (n,)
    times : np.ndarray
        Time points corresponding to observations, shape (n,)
    order : int
        Truncation order for signature computation (default: 2)
    window_length : Optional[int]
        Number of points in sliding window. If None, uses streaming signatures
        from start. Must be >= 2.

    Returns:
    -------
    BatchSignature
        Batch of signature vectors. If window_length is None: (n-1) signatures.
        If windowed: (n - window_length) signatures.

    Raises:
    ------
    ValueError
        If window_length < 2 or window_length >= n

    Notes:
    -----
    Windowing prevents signature saturation for long paths by focusing on
    recent history. Initial partial windows are skipped to ensure consistent
    lookback period.
    """
    n = len(X)

    if window_length is not None:
        if window_length < 2:
            raise ValueError(
                f"window_length must be at least 2 (need minimum 2 points for signature), "
                f"got {window_length}"
            )
        if window_length >= n:
            raise ValueError(
                f"window_length ({window_length}) must be less than number of data points ({n})"
            )

    # Create time-augmented path
    path = np.column_stack([times, X])

    # Streaming signatures from start
    if window_length is None:
        return BatchSignature.from_streaming_path(path, level=order)

    # Windowed signatures over sliding windows
    dimension = 2
    expected_sig_length = iisignature.siglength(dimension, order) + 1
    n_full_windows = n - window_length
    signatures = np.zeros((n_full_windows, expected_sig_length))

    for i in range(window_length, n):
        window_start = i - window_length
        window_end = i + 1

        path_window = path[window_start:window_end]
        sig = iisignature.sig(path_window, order)

        idx = i - window_length
        signatures[idx, 0] = 1.0
        signatures[idx, 1:] = sig

    return BatchSignature(signatures, dimension, order)


def _estimate_drift_functional(
    X: np.ndarray,
    times: np.ndarray,
    sigs: BatchSignature,
    window_length: Optional[int] = None,
) -> SignatureLinearOperator:
    """
    Estimate drift functional β such that b(Sig_t) = ⟨β, Sig_t⟩.

    Uses linear regression on path increments to estimate the drift coefficient
    as a linear functional of the signature.

    Parameters:
    ----------
    X : np.ndarray
        Observed path values, shape (n,)
    times : np.ndarray
        Time points, shape (n,)
    sigs : BatchSignature
        Precomputed signatures
    window_length : Optional[int]
        If provided, aligns increments with windowed signatures

    Returns:
    -------
    SignatureLinearOperator
        Drift functional
    """
    dt = np.diff(times)
    dX = np.diff(X)
    y = dX / dt

    if window_length is not None:
        y = y[window_length - 1 :]

    X_design = sigs.array
    y = y[: len(X_design)]

    model = LinearRegression(fit_intercept=False)
    model.fit(X_design, y)

    beta_coeffs = model.coef_

    return SignatureLinearOperator(beta_coeffs, sigs.d, sigs.level)


def _estimate_diffusion_functional(
    X: np.ndarray,
    times: np.ndarray,
    sigs: BatchSignature,
    beta: SignatureLinearOperator,
    window_length: Optional[int] = None,
) -> SignatureLinearOperator:
    """
    Estimate diffusion functional α such that a(Sig_t) = ⟨α, Sig_t⟩.

    Uses linear regression on squared residuals (after drift removal) to estimate
    the diffusion coefficient as a linear functional of the signature.

    Parameters:
    ----------
    X : np.ndarray
        Observed path values, shape (n,)
    times : np.ndarray
        Time points, shape (n,)
    sigs : BatchSignature
        Precomputed signatures
    beta : SignatureLinearOperator
        Previously estimated drift functional
    window_length : Optional[int]
        If provided, aligns increments with windowed signatures

    Returns:
    -------
    SignatureLinearOperator
        Diffusion functional
    """
    dt = np.diff(times)
    dX = np.diff(X)

    if window_length is not None:
        dt = dt[window_length - 1 :]
        dX = dX[window_length - 1 :]

    X_design = sigs.array
    drift_prediction = X_design @ beta.coeffs

    dt_aligned = dt[: len(X_design)]
    dX_aligned = dX[: len(X_design)]

    residuals = dX_aligned - drift_prediction * dt_aligned
    y = residuals**2 / dt_aligned

    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(X_design, y)

    alpha_coeffs = model.coef_

    return SignatureLinearOperator(alpha_coeffs, sigs.d, sigs.level)


def calibrate_signature_sde(
    X: np.ndarray,
    times: np.ndarray,
    sig_order: int = 2,
    window_length: Optional[int] = None,
) -> Tuple[SignatureLinearOperator, SignatureLinearOperator, BatchSignature]:
    """
    Calibrate signature-driven SDE: dX_t = ⟨β, Sig_t⟩ dt + √⟨α, Sig_t⟩ dW_t.

    Parameters:
    ----------
    X : np.ndarray
        Observed path, shape (n,)
    times : np.ndarray
        Observation times, shape (n,)
    sig_order : int
        Signature truncation order (default: 2)
    window_length : Optional[int]
        Number of points in sliding window. If None, uses streaming signatures.
        Must be >= 2 and leave sufficient data for calibration.

    Returns:
    -------
    beta : SignatureLinearOperator
        Drift functional
    alpha : SignatureLinearOperator
        Diffusion functional
    sigs : BatchSignature
        Computed signatures

    Raises:
    ------
    ValueError
        If window_length is invalid or insufficient data remains after windowing

    Notes:
    -----
    Windowing is recommended for long time series or non-stationary processes.
    Typical window sizes: 30-50 observations for GBM-like processes, or
    approximately 1/κ time units for mean-reverting processes.
    """
    n = len(X)
    dimension = 2

    if window_length is not None:
        n_effective = n - window_length - 2
        sig_dim = iisignature.siglength(dimension, sig_order) + 1
        min_required = max(3 * sig_dim, 20)

        if n_effective < min_required:
            raise ValueError(
                f"Insufficient data after windowing: "
                f"window_length={window_length} leaves only {n_effective} effective observations "
                f"for calibration, but need at least {min_required} "
                f"(3x signature dimension {sig_dim} or minimum 20). "
                f"Either reduce window_length or provide more data (currently n={n})."
            )

    sigs = _compute_windowed_signatures(
        X, times, order=sig_order, window_length=window_length
    )

    beta = _estimate_drift_functional(X, times, sigs, window_length=window_length)

    alpha = _estimate_diffusion_functional(
        X, times, sigs, beta, window_length=window_length
    )

    return beta, alpha, sigs


def evaluate_coefficients(
    beta: SignatureLinearOperator,
    alpha: SignatureLinearOperator,
    X: np.ndarray,
    times: np.ndarray,
    window_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate drift and diffusion coefficients along a given path.

    For diagnostic purposes: compute b(Sig_t) and a(Sig_t) at each point.

    Parameters:
    ----------
    beta : SignatureLinearOperator
        Drift functional
    alpha : SignatureLinearOperator
        Diffusion functional
    X : np.ndarray
        Path values
    times : np.ndarray
        Time points
    window_length : Optional[int]
        If provided, uses windowed signatures matching calibration setup

    Returns:
    -------
    drift_coeffs : np.ndarray
        Drift values b(Sig_t) along path
    diffusion_coeffs : np.ndarray
        Diffusion values a(Sig_t) along path
    """
    sigs = _compute_windowed_signatures(
        X, times, order=beta.level, window_length=window_length
    )

    # Compute inner products using efficient matrix multiplication
    drift_coeffs = sigs.array[:-1] @ beta.coeffs
    diffusion_coeffs = sigs.array[:-1] @ alpha.coeffs

    return drift_coeffs, diffusion_coeffs


def calibrate_linear_operator_from_path(
    path: np.ndarray,
    X_values: np.ndarray,
    sig_order: int = 2,
    window_length: Optional[int] = None,
) -> SignatureLinearOperator:
    """
    Calibrate linear operator ℓ such that X_t ≈ ⟨ℓ, Sig_t⟩.

    Estimates the operator by linear regression of X_values against signatures
    of the given path. Useful for validating from_sde construction or learning
    the solution operator from observed data.

    Parameters:
    ----------
    path : np.ndarray
        Time-augmented path, shape (n, d). For example: [time, W(t)]
    X_values : np.ndarray
        Observed values of the process X, shape (n,)
    sig_order : int
        Signature truncation order (default: 2)
    window_length : Optional[int]
        Number of points in sliding window. If None, uses streaming signatures.
        Must be >= 2 and leave sufficient data for calibration.

    Returns:
    -------
    SignatureLinearOperator
        Calibrated solution operator

    Raises:
    ------
    ValueError
        If path and X_values have incompatible shapes
        If window_length is invalid or insufficient data remains
    """
    n = len(X_values)
    if len(path) != n:
        raise ValueError(f"path length ({len(path)}) must match X_values length ({n})")

    if window_length is not None:
        if window_length < 2:
            raise ValueError(f"window_length must be at least 2, got {window_length}")
        if window_length >= n:
            raise ValueError(
                f"window_length ({window_length}) must be less than data length ({n})"
            )

        n_effective = n - window_length
        dimension = path.shape[1]
        sig_dim = iisignature.siglength(dimension, sig_order) + 1

        min_required = max(3 * sig_dim, 20)
        if n_effective < min_required:
            raise ValueError(
                f"Insufficient data after windowing: "
                f"window_length={window_length} leaves only {n_effective} observations, "
                f"but need at least {min_required} "
                f"(3x signature dimension {sig_dim} or minimum 20). "
                f"Either reduce window_length or provide more data."
            )

    if window_length is None:
        sigs = BatchSignature.from_streaming_path(path, level=sig_order)
        X_aligned = X_values[1:]
    else:
        dimension = path.shape[1]
        expected_sig_length = iisignature.siglength(dimension, sig_order) + 1
        n_full_windows = n - window_length
        signatures = np.zeros((n_full_windows, expected_sig_length))

        for i in range(window_length, n):
            window_start = i - window_length
            window_end = i + 1

            path_window = path[window_start:window_end]
            sig = iisignature.sig(path_window, sig_order)

            idx = i - window_length
            signatures[idx, 0] = 1.0
            signatures[idx, 1:] = sig

        sigs = BatchSignature(signatures, dimension, sig_order)
        X_aligned = X_values[window_length:]

    model = LinearRegression(fit_intercept=False)
    model.fit(sigs.array, X_aligned)

    ell_coeffs = model.coef_

    return SignatureLinearOperator(ell_coeffs, sigs.d, sigs.level)
