"""Testing the brownian_motion"""

import numpy as np
import pytest

from signature_vol.exact.brownian_motion import generate_brownian_motion


def test_generate_brownian_motion() -> None:
    """Testing the generate_brownian_motion function"""
    W_hat = generate_brownian_motion(n_steps=100, T=1.0, seed=42)

    assert isinstance(W_hat, np.ndarray)
    assert W_hat.shape == (101, 2)
    np.testing.assert_array_almost_equal(W_hat[0], np.array([0.0, 0.0]))


def test_generate_brownian_motion_raises() -> None:
    """Testing the generate_brownian_motion function with invalid T"""
    with pytest.raises(ValueError, match="T must be positive, got"):
        _ = generate_brownian_motion(n_steps=100, T=-1.0, seed=42)

    with pytest.raises(ValueError, match="n_steps must be positive, got"):
        _ = generate_brownian_motion(n_steps=-1, T=100.0, seed=42)
