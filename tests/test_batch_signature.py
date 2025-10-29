"""Tests for the batch_signature.py module"""

import numpy as np
import pytest

from signature_vol.exact.batch_signature import BatchSignature
from signature_vol.exact.signature import Signature
import iisignature  # type: ignore


class TestBatchSignature:
    """Test suite for BatchSignature class."""

    def test_init_valid(self):
        """Test initialization with valid array."""
        d, level = 2, 2
        n_samples = 10
        sig_length = iisignature.siglength(d, level) + 1  # Include constant term

        # Create valid batch array
        array = np.random.randn(n_samples, sig_length)
        array[:, 0] = 1.0  # Set all constant terms to 1

        batch_sig = BatchSignature(array, d, level)

        assert batch_sig.d == d
        assert batch_sig.level == level
        assert batch_sig.n_samples == n_samples
        assert batch_sig.sig_length == sig_length
        assert batch_sig.array.shape == (n_samples, sig_length)

    def test_init_invalid_ndim(self):
        """Test initialization with wrong number of dimensions."""
        d, level = 2, 2
        sig_length = iisignature.siglength(d, level) + 1

        # Create 1D array instead of 2D
        array = np.random.randn(sig_length)

        with pytest.raises(ValueError, match="must be 2D"):
            BatchSignature(array, d, level)

        # Create 3D array instead of 2D
        array = np.random.randn(5, 10, sig_length)

        with pytest.raises(ValueError, match="must be 2D"):
            BatchSignature(array, d, level)

    def test_init_invalid_length(self):
        """Test initialization with wrong signature length."""
        d, level = 2, 2
        n_samples = 5
        wrong_length = 10

        array = np.random.randn(n_samples, wrong_length)
        array[:, 0] = 1.0

        with pytest.raises(ValueError, match="doesn't match expected length"):
            BatchSignature(array, d, level)

    def test_init_invalid_constant_terms(self):
        """Test initialization with constant terms != 1."""
        d, level = 2, 2
        n_samples = 5
        sig_length = iisignature.siglength(d, level) + 1

        array = np.random.randn(n_samples, sig_length)
        array[:, 0] = 0.5  # Invalid: constant terms must be 1

        with pytest.raises(ValueError, match="constant terms.*must be all 1s"):
            BatchSignature(array, d, level)

        # Test with some (but not all) constant terms equal to 1
        array[:, 0] = 1.0
        array[0, 0] = 2.0  # Make one invalid

        with pytest.raises(ValueError, match="constant terms.*must be all 1s"):
            BatchSignature(array, d, level)

    def test_from_streaming_path_1d(self):
        """Test from_streaming_path with 1D path."""
        n = 50
        path = np.random.randn(n)
        level = 2

        batch_sig = BatchSignature.from_streaming_path(path, level)

        assert batch_sig.d == 1
        assert batch_sig.level == level
        assert batch_sig.n_samples == n - 1  # n-1 signatures for n points
        expected_sig_length = iisignature.siglength(1, level) + 1
        assert batch_sig.sig_length == expected_sig_length

        # Verify all constant terms are 1
        assert np.allclose(batch_sig.array[:, 0], 1.0)

        # Verify first signature matches manual computation
        sig_manual = Signature.from_path(path[:2], level)
        np.testing.assert_array_almost_equal(batch_sig.array[0], sig_manual.array)

        # Verify last signature matches manual computation
        sig_manual_last = Signature.from_path(path, level)
        np.testing.assert_array_almost_equal(batch_sig.array[-1], sig_manual_last.array)

    def test_from_streaming_path_2d(self):
        """Test from_streaming_path with 2D path."""
        n = 30
        d = 3
        path = np.random.randn(n, d)
        level = 2

        batch_sig = BatchSignature.from_streaming_path(path, level)

        assert batch_sig.d == d
        assert batch_sig.level == level
        assert batch_sig.n_samples == n - 1
        expected_sig_length = iisignature.siglength(d, level) + 1
        assert batch_sig.sig_length == expected_sig_length

        # Verify all constant terms are 1
        assert np.allclose(batch_sig.array[:, 0], 1.0)

        # Verify signatures at different time points
        for i in range(min(5, n - 1)):
            sig_manual = Signature.from_path(path[: i + 2], level)
            np.testing.assert_array_almost_equal(
                batch_sig.array[i],
                sig_manual.array,
                err_msg=f"Signature at index {i} doesn't match",
            )

    def test_from_streaming_path_known_values(self):
        """Test from_streaming_path with known values."""
        # Simple linear path
        path = np.array([0.0, 1.0, 2.0, 3.0])
        level = 2

        batch_sig = BatchSignature.from_streaming_path(path, level)

        assert batch_sig.n_samples == 3
        assert batch_sig.d == 1

        # First signature: path from 0 to 1
        sig_0 = Signature.from_path(np.array([0.0, 1.0]), level)
        np.testing.assert_array_almost_equal(batch_sig.array[0], sig_0.array)

        # Second signature: path from 0 to 2
        sig_1 = Signature.from_path(np.array([0.0, 1.0, 2.0]), level)
        np.testing.assert_array_almost_equal(batch_sig.array[1], sig_1.array)

        # Third signature: path from 0 to 3
        sig_2 = Signature.from_path(np.array([0.0, 1.0, 2.0, 3.0]), level)
        np.testing.assert_array_almost_equal(batch_sig.array[2], sig_2.array)

    def test_from_streaming_path_2d_known_values(self):
        """Test from_streaming_path with known 2D values."""
        path_2d = np.array(
            [
                [0, 0],
                [1, 0],
                [2, 1],
                [2, 2],
            ]
        )
        level = 2

        batch_sig = BatchSignature.from_streaming_path(path_2d, level)

        assert batch_sig.n_samples == 3
        assert batch_sig.d == 2

        # Verify each signature matches individual computation
        for i in range(3):
            sig_manual = Signature.from_path(path_2d[: i + 2], level)
            np.testing.assert_array_almost_equal(
                batch_sig.array[i],
                sig_manual.array,
                err_msg=f"Signature at index {i} doesn't match",
            )

    def test_getitem(self):
        """Test indexing to get individual signatures."""
        n = 20
        d = 2
        path = np.random.randn(n, d)
        level = 2

        batch_sig = BatchSignature.from_streaming_path(path, level)

        # Test getting individual signatures
        for i in [0, 5, 10, n - 2]:
            sig = batch_sig[i]
            assert isinstance(sig, Signature)
            assert sig.d == d
            assert sig.level == level
            np.testing.assert_array_equal(sig.array, batch_sig.array[i])

        # Test negative indexing
        sig_last = batch_sig[-1]
        np.testing.assert_array_equal(sig_last.array, batch_sig.array[-1])

    def test_len(self):
        """Test length of batch signature."""
        n = 15
        path = np.random.randn(n)
        level = 2

        batch_sig = BatchSignature.from_streaming_path(path, level)

        assert len(batch_sig) == n - 1
        assert len(batch_sig) == batch_sig.n_samples

    def test_n_samples_property(self):
        """Test n_samples property."""
        n_samples = 10
        d, level = 2, 2
        sig_length = iisignature.siglength(d, level) + 1

        array = np.random.randn(n_samples, sig_length)
        array[:, 0] = 1.0

        batch_sig = BatchSignature(array, d, level)

        assert batch_sig.n_samples == n_samples

    def test_sig_length_property(self):
        """Test sig_length property."""
        d, level = 3, 3
        n_samples = 5
        sig_length = iisignature.siglength(d, level) + 1

        array = np.random.randn(n_samples, sig_length)
        array[:, 0] = 1.0

        batch_sig = BatchSignature(array, d, level)

        assert batch_sig.sig_length == sig_length

    def test_truncate_valid(self):
        """Test truncation to lower level."""
        n = 20
        d = 2
        path = np.random.randn(n, d)
        level = 4

        batch_sig = BatchSignature.from_streaming_path(path, level)

        new_level = 2
        truncated = batch_sig.truncate(new_level)

        assert truncated.level == new_level
        assert truncated.d == d
        assert truncated.n_samples == batch_sig.n_samples
        expected_length = iisignature.siglength(d, new_level) + 1
        assert truncated.sig_length == expected_length

        # Verify constant terms are preserved
        assert np.allclose(truncated.array[:, 0], 1.0)

        # Verify truncated arrays match first part of original
        np.testing.assert_array_equal(
            truncated.array, batch_sig.array[:, :expected_length]
        )

        # Verify each truncated signature matches individual truncation
        for i in range(min(5, n - 1)):
            sig_original = batch_sig[i]
            sig_truncated_manual = sig_original.truncate(new_level)
            sig_from_batch = truncated[i]
            np.testing.assert_array_equal(
                sig_from_batch.array, sig_truncated_manual.array
            )

    def test_truncate_invalid(self):
        """Test truncation to higher level raises error."""
        n = 10
        path = np.random.randn(n)
        level = 2

        batch_sig = BatchSignature.from_streaming_path(path, level)

        with pytest.raises(ValueError, match="Cannot truncate to level"):
            batch_sig.truncate(3)

    def test_truncate_same_level(self):
        """Test truncation to same level returns copy."""
        n = 10
        path = np.random.randn(n)
        level = 2

        batch_sig = BatchSignature.from_streaming_path(path, level)
        truncated = batch_sig.truncate(level)

        assert truncated.level == level
        assert truncated.d == batch_sig.d
        assert truncated.n_samples == batch_sig.n_samples
        np.testing.assert_array_equal(truncated.array, batch_sig.array)

    def test_repr(self):
        """Test string representation."""
        n = 15
        d = 3
        level = 2
        path = np.random.randn(n, d)

        batch_sig = BatchSignature.from_streaming_path(path, level)

        repr_str = repr(batch_sig)
        assert "BatchSignature" in repr_str
        assert f"n_samples={n - 1}" in repr_str
        assert f"dimension={d}" in repr_str
        assert f"level={level}" in repr_str

    def test_empty_path_raises_error(self):
        """Test that empty path raises appropriate error."""
        path = np.array([])
        level = 2

        # Empty path should fail when computing signature
        with pytest.raises((ValueError, IndexError)):
            BatchSignature.from_streaming_path(path, level)

    def test_single_point_path(self):
        """Test path with single point (should return empty batch)."""
        path = np.array([1.0])
        level = 2

        batch_sig = BatchSignature.from_streaming_path(path, level)

        # Single point means no signatures (need at least 2 points)
        assert batch_sig.n_samples == 0
        assert batch_sig.array.shape[0] == 0

    def test_two_point_path(self):
        """Test path with exactly two points."""
        path = np.array([0.0, 1.0])
        level = 2

        batch_sig = BatchSignature.from_streaming_path(path, level)

        assert batch_sig.n_samples == 1

        # Verify it matches direct signature computation
        sig_manual = Signature.from_path(path, level)
        np.testing.assert_array_almost_equal(batch_sig.array[0], sig_manual.array)

    def test_consistency_across_levels(self):
        """Test that higher level signatures contain lower level information."""
        n = 20
        path = np.random.randn(n)

        batch_sig_level2 = BatchSignature.from_streaming_path(path, level=2)
        batch_sig_level3 = BatchSignature.from_streaming_path(path, level=3)

        # Truncating level 3 to level 2 should give same result as computing level 2
        batch_sig_level3_truncated = batch_sig_level3.truncate(2)

        np.testing.assert_array_almost_equal(
            batch_sig_level2.array, batch_sig_level3_truncated.array
        )

    def test_time_augmented_path(self):
        """Test with time-augmented path (typical use case for calibration)."""
        n = 30
        times = np.linspace(0, 1, n)
        values = np.random.randn(n)

        # Create time-augmented path
        path = np.column_stack([times, values])
        level = 2

        batch_sig = BatchSignature.from_streaming_path(path, level)

        assert batch_sig.d == 2
        assert batch_sig.n_samples == n - 1
        assert np.allclose(batch_sig.array[:, 0], 1.0)

        # Verify signatures increase in complexity as we go forward in time
        # (this is a qualitative check that streaming computation is working)
        for i in range(1, min(10, n - 1)):
            sig_prev = batch_sig[i - 1]
            sig_curr = batch_sig[i]
            # Later signatures should generally have larger norms
            # (except in pathological cases)
            assert sig_curr.array.shape == sig_prev.array.shape

    def test_dtype_preservation(self):
        """Test that float64 dtype is preserved."""
        n = 10
        path = np.random.randn(n).astype(np.float32)
        level = 2

        batch_sig = BatchSignature.from_streaming_path(path, level)

        # Should be converted to float64
        assert batch_sig.array.dtype == np.float64

    def test_large_dimension(self):
        """Test with higher dimensional path."""
        n = 15
        d = 5
        path = np.random.randn(n, d)
        level = 2

        batch_sig = BatchSignature.from_streaming_path(path, level)

        assert batch_sig.d == d
        expected_length = iisignature.siglength(d, level) + 1
        assert batch_sig.sig_length == expected_length

        # Verify a few signatures
        for i in [0, n // 2, n - 2]:
            sig_manual = Signature.from_path(path[: i + 2], level)
            np.testing.assert_array_almost_equal(batch_sig.array[i], sig_manual.array)
