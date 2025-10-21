"""Signature class with validation and metadata."""

import numpy as np
from typing_extensions import Self
import iisignature  # type: ignore


class Signature:
    """
    Immutable signature object with validation and metadata.
    """

    def __init__(
        self,
        array: np.ndarray,
        dimension: int,
        level: int,
    ) -> None:
        """
        Parameters:
        -----------
        array : np.ndarray
            Signature coefficients, shape (siglength + 1,)
            First element is the constant term (always 1 for paths)
        dimension : int
            Dimension of the path space
        level : int
            Truncation level

        Raises:
        ------
        ValueError
            If array shape doesn't match expected signature length
            If zeroth term is not 1
        """
        self.array = np.asarray(array, dtype=np.float64)
        self.d = dimension
        self.level = level

        # Validate shape
        # Include the constant term (zeroth level)
        expected_length = iisignature.siglength(dimension, level) + 1

        if self.array.ndim != 1:
            raise ValueError(
                f"Signature array must be 1D, got shape {self.array.shape}"
            )
        if len(self.array) != expected_length:
            raise ValueError(
                f"Signature array length {len(self.array)} doesn't match "
                f"expected length {expected_length} (including constant term) for "
                f"dimension {dimension} and level {level}"
            )

        # Validate that the zeroth term (constant) equals 1
        if not np.isclose(self.array[0], 1.0):
            raise ValueError(
                f"Zeroth term (constant) must equal 1, got {self.array[0]}"
            )

    @classmethod
    def from_path(cls, path: np.array, level: int):
        """
        Compute signature from path.

        Parameters:
        -----------
        path : np.ndarray
            Shape (n, d) for path with n points in d-dimensional space
            If 1D array of shape (n,), will be reshaped to (n, 1)
        level : int
            Truncation level

        Returns:
        --------
        Signature
        """
        path_array = np.asarray(path, dtype=np.float64)

        # Handle 1D case: (n,) -> (n, 1)
        if path_array.ndim == 1:
            path_array = path_array.reshape(-1, 1)
            dimension = 1
        else:
            # For any nD array where n >= 2, dimension is last axis
            dimension = path_array.shape[-1]

        sig_array = iisignature.sig(path_array, level)
        # Prepend the constant term (zeroth level = 1)
        sig_array_with_constant = np.concatenate([[1.0], sig_array])
        return cls(sig_array_with_constant, dimension, level)

    @property
    def length(self) -> int:
        """Length of signature vector."""
        return len(self.array)

    def truncate(self, new_level: int) -> Self:
        """
        Truncate to lower level.

        Parameters:
        -----------
        new_level : int
            Must be <= self.m

        Returns:
        --------
        Signature

        Raises
        ------
        ValueError
            If new_level > self.m
        """
        if new_level > self.level:
            raise ValueError(
                f"Cannot truncate to level {new_level} > current level {self.level}"
            )

        # Include the constant term in the length
        new_length = iisignature.siglength(self.d, new_level) + 1
        truncated_array = self.array[:new_length]

        return Signature(truncated_array, self.d, new_level)

    def __repr__(self) -> str:
        return (
            f"Signature("
            f"\tdimension={self.d},\n"
            f"\tlevel={self.level},\n"
            f"\tlength={self.length},\n"
            f"\tarray={self.array},\n"
            f")"
        )

    def __len__(self) -> int:
        """Return length of signature vector."""
        return self.length
