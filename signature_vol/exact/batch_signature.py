"""Batch signature class for efficient computation of multiple signatures."""

import numpy as np
import iisignature  # type: ignore
from typing_extensions import Self

from signature_vol.exact.signature import Signature


class BatchSignature:
    """
    Efficient storage for multiple signatures sharing the same dimension and level.

    This class is designed for streaming signature computation where we need
    signatures at multiple time points. Instead of creating individual Signature
    objects (which would have memory overhead), we store all signatures in a
    single contiguous array while maintaining the same validation guarantees.
    """

    def __init__(
        self,
        array: np.ndarray,
        dimension: int,
        level: int,
    ) -> None:
        """
        Direct construction (prefer using from_streaming_path).

        Parameters:
        -----------
        array : np.ndarray
            2D array of shape (n_samples, sig_length) where
            sig_length = siglength(dimension, level) + 1
            First column should be all 1s (constant term)
        dimension : int
            Dimension of the path space
        level : int
            Truncation level

        Raises:
        ------
        ValueError
            If array shape doesn't match expected signature length
            If first column is not all 1s
        """
        self.array = np.asarray(array, dtype=np.float64)
        self.d = dimension
        self.level = level

        # Validate shape
        expected_sig_length = iisignature.siglength(dimension, level) + 1

        if self.array.ndim != 2:
            raise ValueError(
                f"BatchSignature array must be 2D, got shape {self.array.shape}"
            )

        if self.array.shape[1] != expected_sig_length:
            raise ValueError(
                f"Signature length {self.array.shape[1]} doesn't match "
                f"expected length {expected_sig_length} (including constant term) for "
                f"dimension {dimension} and level {level}"
            )

        # Validate that all constant terms equal 1
        if not np.allclose(self.array[:, 0], 1.0):
            raise ValueError(
                f"First column (constant terms) must be all 1s, got {self.array[:, 0]}"
            )

    @classmethod
    def from_streaming_path(cls, path: np.ndarray, level: int) -> Self:
        """
        Compute streaming signatures along a path.

        For each time point t_i (i >= 1), computes the signature of the path
        from the start up to t_i. This is the natural representation for
        signature-driven models where coefficients depend on path history.

        Parameters:
        -----------
        path : np.ndarray
            Shape (n, d) for path with n points in d-dimensional space
            First column should be time, remaining columns are coordinates
            If 1D array of shape (n,), will be reshaped to (n, 1)
        level : int
            Truncation level

        Returns:
        --------
        BatchSignature
            Contains n-1 signatures (one for each time point after the first)

        Notes:
        ------
        The path should be time-augmented: path[i] = [t_i, X_1(t_i), X_2(t_i), ...]
        where the first coordinate is time and remaining coordinates are state variables.
        """
        path_array = np.asarray(path, dtype=np.float64)

        # Handle 1D case: (n,) -> (n, 1)
        if path_array.ndim == 1:
            path_array = path_array.reshape(-1, 1)
            dimension = 1
        else:
            dimension = path_array.shape[1]

        n = len(path_array)
        expected_sig_length = iisignature.siglength(dimension, level) + 1

        # Preallocate array for all signatures
        signatures = np.zeros((n - 1, expected_sig_length))

        # Compute signature at each time point
        for i in range(1, n):
            # Path from start to current time
            path_segment = path_array[: i + 1]

            # Compute signature (without constant term)
            sig = iisignature.sig(path_segment, level)

            # Add constant term and store
            signatures[i - 1, 0] = 1.0
            signatures[i - 1, 1:] = sig

        return cls(signatures, dimension, level)

    def __getitem__(self, idx: int) -> Signature:
        """
        Get individual signature at given index.

        Parameters:
        -----------
        idx : int
            Index of the signature to retrieve

        Returns:
        --------
        Signature
            Individual signature object at that index
        """
        return Signature(self.array[idx], self.d, self.level)

    def __len__(self) -> int:
        """Return number of signatures in the batch."""
        return self.array.shape[0]

    @property
    def n_samples(self) -> int:
        """Number of signature samples in the batch."""
        return self.array.shape[0]

    @property
    def sig_length(self) -> int:
        """Length of each signature vector."""
        return self.array.shape[1]

    def truncate(self, new_level: int) -> Self:
        """
        Truncate all signatures to lower level.

        Parameters:
        -----------
        new_level : int
            Must be <= self.level

        Returns:
        --------
        BatchSignature
            New batch with all signatures truncated

        Raises:
        -------
        ValueError
            If new_level > self.level
        """
        if new_level > self.level:
            raise ValueError(
                f"Cannot truncate to level {new_level} > current level {self.level}"
            )

        # Include the constant term in the length
        new_length = iisignature.siglength(self.d, new_level) + 1
        truncated_array = self.array[:, :new_length]

        return BatchSignature(truncated_array, self.d, new_level)

    def __repr__(self) -> str:
        return (
            f"BatchSignature(\n"
            f"\tn_samples={self.n_samples},\n"
            f"\tdimension={self.d},\n"
            f"\tlevel={self.level},\n"
            f"\tsig_length={self.sig_length}\n"
            f")"
        )
