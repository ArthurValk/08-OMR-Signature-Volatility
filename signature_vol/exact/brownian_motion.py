"""Brownian motion generation."""

import numpy as np


def generate_brownian_motion(
    n_steps: int,
    T: float,
    seed=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generating a discretized realization of a Brownian motion with n_steps steps over [0,T].

    Parameters
    ----------
    n_steps : int
        Number of steps.
    T : float
        Time horizon.
    seed : int | None
        Random seed for reproducibility.
    Returns
    -------
    t : np.ndarray
        Time grid.

    Raises
    ------
    ValueError
        If T is not positive.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    rng = np.random.default_rng(seed)

    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)

    dW = rng.normal(0, np.sqrt(dt), n_steps)
    W = np.concatenate([[0], np.cumsum(dW)])

    W_hat_transposed = np.array([t, W])
    return np.transpose(W_hat_transposed)
