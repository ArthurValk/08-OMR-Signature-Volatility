"""Testing whole integration of components."""

import numpy as np
import iisignature  # type: ignore
import pytest

from signature_vol.exact.brownian_motion import generate_brownian_motion
from signature_vol.exact.linear_operator import (
    SignatureLinearOperatorBuilder,
    inner,
)
from signature_vol.exact.signature import Signature


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_signature_from_path_with_operator(self):
        """Test computing signature from path and applying operator."""
        # Create a simple 2D path
        t = np.linspace(0, 1, 100)
        path = np.column_stack([t, np.sin(2 * np.pi * t)])

        level = 2
        sig = Signature.from_path(path, level)

        # Create operator with ones
        op = SignatureLinearOperatorBuilder.ones(2, level).build()

        # Apply operator
        result = inner(op, sig)

        assert isinstance(result, (float, np.floating))
        # Result should be sum of signature coefficients
        expected = np.sum(sig.array)
        np.testing.assert_almost_equal(result, expected)

    def test_truncate_signature_and_operator(self):
        """Test truncating both signature and operator to same level."""
        d, m = 3, 4
        path = np.random.randn(100, d)

        sig = Signature.from_path(path, m)
        op = SignatureLinearOperatorBuilder.ones(d, m).build()

        # Truncate both to same level
        new_level = 2
        sig_trunc = sig.truncate(new_level)
        op_trunc = op.truncate(new_level)

        # Should be compatible
        result = inner(op_trunc, sig_trunc)
        assert isinstance(result, (float, np.floating))

    def test_geometric_brownian_motion_signature_integration(self):
        """Testing the integration of GBM with signature computation and operator application."""
        T = 1
        W_hat = generate_brownian_motion(n_steps=50, T=T)

        level, dimension = 2, 2
        sig = Signature.from_path(W_hat, level)
        mu = 0.1
        sigma = 0.2
        x0 = 0

        linear_operator_builder = SignatureLinearOperatorBuilder.from_sde(
            dimension=dimension,
            level=level,
            x0=x0,
            a=mu - sigma**2 / 2,
            b=0,
            alpha=sigma,
            beta=0,
        )
        linear_operator = linear_operator_builder.build()
        sig_X = inner(linear_operator, sig)

        # calculating analytic solution
        analytic_X = x0 + (mu - 0.5 * sigma**2) * W_hat[-1, 0] + sigma * W_hat[-1, 1]
        assert analytic_X == pytest.approx(sig_X, rel=1e-6)
