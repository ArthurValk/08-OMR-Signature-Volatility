"""Tests for the signature.py module"""

import numpy as np
import pytest

from signature_vol.exact.signature import Signature
import iisignature  # type: ignore


class TestSignature:
    """Test suite for Signature class."""

    def test_init_valid_1d(self):
        """Test initialization with valid 1D array."""
        d, level = 2, 2
        expected_length = iisignature.siglength(d, level) + 1  # Include constant term
        array = np.random.randn(expected_length)
        array[0] = 1.0  # Set constant term
        sig = Signature(array, d, level)

        assert sig.d == d
        assert sig.level == level
        assert len(sig.array) == expected_length
        assert sig.length == expected_length

    def test_init_invalid_ndim(self):
        """Test initialization with wrong number of dimensions."""
        d, level = 2, 2
        expected_length = iisignature.siglength(d, level) + 1  # Include constant term
        array = np.random.randn(3, expected_length)  # 2D instead of 1D

        with pytest.raises(ValueError, match="must be 1D"):
            Signature(array, d, level)

    def test_init_invalid_length(self):
        """Test initialization with wrong length."""
        d, level = 2, 2
        array = np.random.randn(10)  # Wrong length

        with pytest.raises(ValueError, match="doesn't match expected length"):
            Signature(array, d, level)

    def test_init_invalid_zeroth_term(self):
        """Test initialization with zeroth term != 1."""
        d, level = 2, 2
        expected_length = iisignature.siglength(d, level) + 1
        array = np.random.randn(expected_length)
        array[0] = 0.5  # Invalid: zeroth term must be 1

        with pytest.raises(ValueError, match="Zeroth term .* must equal 1"):
            Signature(array, d, level)

    def test_from_path_1d(self):
        """Test from_path with 1D path."""
        path = np.random.randn(100)  # 100 time steps, 1 dimension
        level = 2

        sig = Signature.from_path(path, level)

        assert sig.d == 1
        assert sig.level == level
        expected_length = iisignature.siglength(1, level) + 1  # Include constant term
        assert sig.length == expected_length
        # Verify first element is the constant term = 1
        assert sig.array[0] == 1.0

        # Testing with known values
        level = 3
        test_path = np.array([1, 2, 3, 4, 5, 6])
        sig = Signature.from_path(test_path, level)

        # Verify first element is the constant term = 1
        assert sig.array[0] == 1.0
        np.testing.assert_array_almost_equal(
            sig.array, np.array([1.0, 5.0, 12.5, 20.83333333])
        )
        assert sig.d == 1
        assert sig.level == 3
        assert sig.length == 4

    def test_from_path_2d(self):
        """Test from_path with 2D path."""
        path = np.random.randn(100, 3)  # 100 time steps, 3 dimensions
        level = 2

        sig = Signature.from_path(path, level)

        assert sig.d == 3
        assert sig.level == level
        expected_length = iisignature.siglength(3, level) + 1  # Include constant term
        assert sig.length == expected_length
        # Verify first element is the constant term = 1
        assert sig.array[0] == 1.0

        # Test with known values
        path_2d = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 1],
                [2, 2],
                [1, 3],
                [0, 3],
                [-1, 2],
                [-1, 1],
                [0, 0.5],
                [1, 1],
            ]
        )
        signature_2d = Signature.from_path(path_2d, level=3)

        assert signature_2d.d == 2
        assert signature_2d.level == 3
        assert signature_2d.length == 15
        np.testing.assert_array_almost_equal(
            signature_2d.array,
            np.array(
                [
                    1,
                    1,
                    1,
                    0.5,
                    7,
                    -6,
                    0.5,
                    0.16666667,
                    3.66666667,
                    -0.33333333,
                    -3.58333333,
                    -2.83333333,
                    14.16666667,
                    -10.08333333,
                    0.16666667,
                ]
            ),
        )

    def test_length_property(self):
        """Test length property."""
        d, level = 2, 3
        expected_length = iisignature.siglength(d, level) + 1  # Include constant term
        array = np.random.randn(expected_length)
        array[0] = 1.0  # Set constant term
        sig = Signature(array, d, level)

        assert sig.length == expected_length
        assert len(sig) == expected_length

    def test_truncate_valid(self):
        """Test truncation to lower level."""
        d, level = 2, 4
        array = np.random.randn(
            iisignature.siglength(d, level) + 1
        )  # Include constant term
        array[0] = 1.0  # Set constant term
        sig = Signature(array, d, level)

        new_level = 2
        truncated = sig.truncate(new_level)

        assert truncated.level == new_level
        assert truncated.d == d
        assert (
            truncated.length == iisignature.siglength(d, new_level) + 1
        )  # Include constant term
        assert truncated.array[0] == 1.0
        # Check that truncated array matches first part of original
        np.testing.assert_array_equal(truncated.array, sig.array[: truncated.length])

        # Test with known values
        path_2d = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 1],
                [2, 2],
                [1, 3],
                [0, 3],
                [-1, 2],
                [-1, 1],
                [0, 0.5],
                [1, 1],
            ]
        )
        signature_2d = Signature.from_path(path_2d, level=3)
        truncated_signature = signature_2d.truncate(2)

        assert truncated_signature.d == 2
        assert truncated_signature.level == 2
        assert truncated_signature.length == 7
        np.testing.assert_array_almost_equal(
            truncated_signature.array,
            np.array(
                [
                    1,
                    1,
                    1,
                    0.5,
                    7,
                    -6,
                    0.5,
                ]
            ),
        )

    def test_truncate_invalid(self):
        """Test truncation to higher level raises error."""
        d, level = 2, 2
        array = np.random.randn(
            iisignature.siglength(d, level) + 1
        )  # Include constant term
        array[0] = 1.0  # Set constant term
        sig = Signature(array, d, level)

        with pytest.raises(ValueError, match="Cannot truncate to level"):
            sig.truncate(3)

    def test_repr(self):
        """Test string representation."""
        d, level = 2, 2
        array = np.random.randn(
            iisignature.siglength(d, level) + 1
        )  # Include constant term
        array[0] = 1.0  # Set constant term
        sig = Signature(array, d, level)

        repr_str = repr(sig)
        assert "Signature" in repr_str
        assert f"dimension={d}" in repr_str
        assert f"level={level}" in repr_str

    def test_stratonovich(self) -> None:
        """Test that the signature is computed in the sense of Stratonovich"""
        # We will try the simplest possible example:
        X_hat = np.array([[0, 0], [1, 1]])

        level = 2
        sig = Signature.from_path(X_hat, level)

        # If the signature is computed in the sense of Stratonovich, we'd expect (1,1,1,1/2,1/2,1/2,1/2,...)
        np.testing.assert_array_almost_equal(
            sig.array,
            np.array([1, 1, 1, 0.5, 0.5, 0.5, 0.5]),
        )

        # If it were Itô, we'd instead expect (1,1,1,0,0,0,0,...),
        # since the covariation is exactly 1, and the correction is -1/2 x covariation
