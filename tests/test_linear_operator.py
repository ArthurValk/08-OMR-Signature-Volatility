"""Tests for linear_operator.py"""

import numpy as np
import pytest
import iisignature  # type: ignore

from signature_vol.exact import (
    SignatureLinearOperator,
    SignatureLinearOperatorBuilder,
    Signature,
    inner,
)


class TestSignatureLinearOperator:
    """Test suite for SignatureLinearOperator class."""

    def test_init_valid(self):
        """Test initialization with valid coefficients."""
        d, level = 2, 2
        expected_length = iisignature.siglength(d, level) + 1  # Include constant term
        coeffs = np.random.randn(expected_length)

        op = SignatureLinearOperator(coeffs, d, level)

        assert op.d == d
        assert op.level == level
        assert len(op.coeffs) == expected_length

    def test_init_invalid_length(self):
        """Test initialization with wrong length."""
        d, level = 2, 2
        coeffs = np.random.randn(10)  # Wrong length

        with pytest.raises(ValueError, match="doesn't match expected signature length"):
            SignatureLinearOperator(coeffs, d, level)

    def test_truncate_valid(self):
        """Test truncation to lower level."""
        d, level = 2, 4
        coeffs = np.random.randn(
            iisignature.siglength(d, level) + 1
        )  # Include constant term
        op = SignatureLinearOperator(coeffs, d, level)

        new_level = 2
        truncated = op.truncate(new_level)

        assert truncated.level == new_level
        assert truncated.d == d
        assert (
            len(truncated.coeffs) == iisignature.siglength(d, new_level) + 1
        )  # Include constant term
        # Check that truncated coeffs match first part of original
        np.testing.assert_array_equal(
            truncated.coeffs, op.coeffs[: len(truncated.coeffs)]
        )

    def test_truncate_invalid(self):
        """Test truncation to higher level raises error."""
        d, level = 2, 2
        coeffs = np.random.randn(
            iisignature.siglength(d, level) + 1
        )  # Include constant term
        op = SignatureLinearOperator(coeffs, d, level)

        with pytest.raises(ValueError, match="Cannot truncate to level"):
            op.truncate(3)

    def test_repr(self):
        """Test string representation."""
        d, level = 2, 2
        coeffs = np.random.randn(
            iisignature.siglength(d, level) + 1
        )  # Include constant term
        op = SignatureLinearOperator(coeffs, d, level)

        repr_str = repr(op)
        assert "SignatureLinearOperator" in repr_str
        assert f"dimension={d}" in repr_str
        assert f"level={level}" in repr_str


class TestSignatureLinearOperatorBuilder:
    """Test suite for SignatureLinearOperatorBuilder class."""

    def test_from_array_valid(self):
        """Test building from array."""
        d, level = 2, 2
        expected_length = iisignature.siglength(d, level) + 1  # Include constant term
        coeffs = np.random.randn(expected_length)

        builder = SignatureLinearOperatorBuilder.from_array(coeffs, d, level)
        op = builder.build()

        assert op.d == d
        assert op.level == level
        np.testing.assert_array_equal(op.coeffs, coeffs)

    def test_from_array_invalid_length(self):
        """Test building from array with wrong length."""
        d, level = 2, 2
        coeffs = np.random.randn(10)  # Wrong length

        with pytest.raises(ValueError, match="Expected .* coefficients"):
            SignatureLinearOperatorBuilder.from_array(coeffs, d, level)

    def test_from_levels_valid(self):
        """Test building from levels dictionary."""
        d, level = 2, 3

        coeffs_by_level = {
            1: np.random.randn(d**1),
            2: np.random.randn(d**2),
            3: np.random.randn(d**3),
        }

        builder = SignatureLinearOperatorBuilder.from_levels(coeffs_by_level, d, level)
        op = builder.build()

        assert op.d == d
        assert op.level == level
        assert (
            len(op.coeffs) == iisignature.siglength(d, level) + 1
        )  # Include constant term

        # Verify concatenation is correct: constant term (1.0) + level coefficients
        expected = np.concatenate(
            [[1.0]] + [coeffs_by_level[k] for k in range(1, level + 1)]
        )
        np.testing.assert_array_equal(op.coeffs, expected)
        # Verify first element is the constant term = 1
        assert op.coeffs[0] == 1.0

    def test_from_levels_missing_level(self):
        """Test building from levels with missing level."""
        d, level = 2, 3

        coeffs_by_level = {
            1: np.random.randn(d**1),
            3: np.random.randn(d**3),  # Missing level 2
        }

        with pytest.raises(ValueError, match="Missing coefficients for level"):
            SignatureLinearOperatorBuilder.from_levels(coeffs_by_level, d, level)

    def test_from_levels_wrong_size(self):
        """Test building from levels with wrong size at a level."""
        d, level = 2, 2

        coeffs_by_level = {
            1: np.random.randn(d**1),
            2: np.random.randn(10),  # Wrong size
        }

        with pytest.raises(ValueError, match="expected .* coefficients"):
            SignatureLinearOperatorBuilder.from_levels(coeffs_by_level, d, level)

    def test_zeros(self):
        """Test building zeros operator."""
        d, level = 2, 2

        builder = SignatureLinearOperatorBuilder.zeros(d, level)
        op = builder.build()

        assert op.d == d
        assert op.level == level
        expected_length = iisignature.siglength(d, level) + 1  # Include constant term
        assert len(op.coeffs) == expected_length
        np.testing.assert_array_equal(op.coeffs, np.zeros(expected_length))

    def test_ones(self):
        """Test building ones operator."""
        d, level = 2, 2

        builder = SignatureLinearOperatorBuilder.ones(d, level)
        op = builder.build()

        assert op.d == d
        assert op.level == level
        expected_length = iisignature.siglength(d, level) + 1  # Include constant term
        assert len(op.coeffs) == expected_length
        np.testing.assert_array_equal(op.coeffs, np.ones(expected_length))

    def test_build_without_coefficients(self):
        """Test building without setting coefficients."""
        d, level = 2, 2

        builder = SignatureLinearOperatorBuilder(d, level)

        with pytest.raises(ValueError, match="Must set coefficients before building"):
            builder.build()

    def test_build_at_level_valid(self):
        """Test building at specific level."""
        d, level = 2, 4
        target_level = 2

        builder = SignatureLinearOperatorBuilder.ones(d, level)
        op = builder.build_at_level(target_level)

        assert op.d == d
        assert op.level == target_level
        assert (
            len(op.coeffs) == iisignature.siglength(d, target_level) + 1
        )  # Include constant term

    def test_build_at_level_exceeds_max(self):
        """Test building at level exceeding builder level."""
        d, level = 2, 2

        builder = SignatureLinearOperatorBuilder.ones(d, level)

        with pytest.raises(ValueError, match="exceeds builder level"):
            builder.build_at_level(3)

    def test_build_at_level_without_coefficients(self):
        """Test building at level without setting coefficients."""
        d, level = 2, 2

        builder = SignatureLinearOperatorBuilder(d, level)

        with pytest.raises(ValueError, match="Must set coefficients before building"):
            builder.build_at_level(1)

    def test_chaining(self):
        """Test that classmethods return builder which can call build()."""
        d, level = 2, 2

        op = SignatureLinearOperatorBuilder.zeros(d, level).build()

        assert isinstance(op, SignatureLinearOperator)

    def test_from_sde_initial_condition_only(self) -> None:
        """Testing the from_sde method with a trivial SDE.

        We expect l=x0ø
        """
        d, level = 2, 2

        # Testing with only initial condition
        x0 = 2
        builder = SignatureLinearOperatorBuilder.from_sde(
            x0=x0,
            a=0,
            alpha=0,
            b=0,
            beta=0,
            dimension=d,
            level=level,
        )
        op = builder.build()

        assert op.d == d
        assert op.level == level
        expected_length = iisignature.siglength(d, level) + 1
        assert op.length == expected_length
        assert op[0] == x0
        for coeff in op.coeffs[1:]:
            assert coeff == 0

    def test_from_sde_only_constant_drift(self) -> None:
        """Testing the from_sde method with a!=0, all other coeffs zero

        We expect l=a1, with all other terms equal to zero.
        """
        d, level = 2, 2

        x0 = 0
        a = 0.5
        alpha = 0
        b = 0
        beta = 0

        builder = SignatureLinearOperatorBuilder.from_sde(
            x0=x0,
            a=a,
            alpha=alpha,
            b=b,
            beta=beta,
            dimension=d,
            level=level,
        )
        op = builder.build()

        assert op.d == d
        assert op.level == level
        expected_length = iisignature.siglength(d, level) + 1
        assert op.length == expected_length
        # assert 'drift term' is exactly a
        assert op[1] == a
        assert op[0] == 0
        for alpha in op[2:]:
            assert alpha == 0

    def test_from_sde_only_constant_diffusion(self) -> None:
        """Testing the from_sde method with b!=0, all other coeffs zero.
        We expect l = b2 to capture the diffusion effect in level 2.
        """
        d, level = 2, 2

        x0 = 0
        a = 0
        alpha = 1
        b = 0
        beta = 0

        builder = SignatureLinearOperatorBuilder.from_sde(
            x0=x0,
            a=a,
            alpha=alpha,
            b=b,
            beta=beta,
            dimension=d,
            level=level,
        )
        op = builder.build()

        assert op.d == d
        assert op.level == level
        expected_length = iisignature.siglength(d, level) + 1
        assert op.length == expected_length
        # assert initial condtion is properly handled
        assert op[0] == x0
        # assert 'diffusion term' is exactly alpha at appropriate index
        assert op[2] == alpha
        for coeff in op[3:]:
            assert coeff == 0
        for coeff in op[:2]:
            assert coeff == 0

    def test_from_sde_generic_sde(self) -> None:
        """Testing the from_sde method with generic SDE coefficients

        Expected coefficients were verified by manual calculation to ensure correctness.
        """
        d, level = 2, 2

        x0 = 1.0
        a = 0.5
        alpha = 0.1
        b = -0.3
        beta = 0.05

        builder = SignatureLinearOperatorBuilder.from_sde(
            x0=x0,
            a=a,
            alpha=alpha,
            b=b,
            beta=beta,
            dimension=d,
            level=level,
        )
        op = builder.build()

        coeff0 = x0
        coeff1 = a + beta**2 / 2 + b * x0
        coeff2 = alpha + beta * x0
        coeff11 = b * (a + beta**2 / 2) + x0 * b**2
        coeff12 = beta * (a + beta**2 / 2) + x0 * b * beta
        coeff21 = b * alpha + x0 * b * beta
        coeff22 = alpha * beta + x0 * beta**2

        assert op[0] == coeff0
        assert op[1] == coeff1
        assert op[2] == coeff2
        assert op[3] == coeff11
        assert op[4] == coeff12
        assert op[5] == coeff21
        assert op[6] == coeff22


class TestInner:
    """Tests for the inner function"""

    def test_inner_product_valid(self):
        """Test inner product with compatible signature."""
        d, level = 2, 2
        expected_length = iisignature.siglength(d, level) + 1  # Include constant term

        coeffs = np.random.randn(expected_length)
        sig_array = np.random.randn(expected_length)
        sig_array[0] = 1.0  # ensure zeroth term is 1

        op = SignatureLinearOperator(coeffs, d, level)
        sig = Signature(sig_array, d, level)

        result = inner(op, sig)

        # Verify it's a float
        assert isinstance(result, (float, np.floating))
        # Verify correct computation
        expected = np.dot(sig_array, coeffs)
        np.testing.assert_almost_equal(result, expected)

    def test_inner_product_shape_mismatch(self):
        """Test inner product with incompatible signature."""
        d1, level1 = 2, 2
        d2, level2 = 2, 3  # Different level

        coeffs = np.random.randn(
            iisignature.siglength(d1, level1) + 1
        )  # Include constant term
        sig_array = np.random.randn(
            iisignature.siglength(d2, level2) + 1
        )  # Include constant term
        sig_array[0] = 1.0  # ensure zeroth term is 1

        op = SignatureLinearOperator(coeffs, d1, level1)
        sig = Signature(sig_array, d2, level2)

        with pytest.raises(ValueError, match="doesn't match"):
            inner(op, sig)
