"""File defining a linear operator on signature space."""

from typing import cast

from typing_extensions import Self

import iisignature  # type: ignore
import numpy as np

from signature_vol.exact.signature import Signature


def inner(operator: "SignatureLinearOperator", signature: Signature) -> float:
    """
    Take inner product of linear operator and signature.

    Parameters:
    -----------
    operator : SignatureLinearOperator
        Linear operator to apply
    signature : Signature
        Signature object to apply operator to

    Returns:
    --------
    float

    Raises:
    ------
    ValueError
        If signature shape is incompatible with operator
    """
    if signature.array.shape != operator.coeffs.shape:
        raise ValueError(
            f"Signature shape {signature.array.shape} doesn't match "
            f"operator shape {operator.coeffs.shape}"
        )
    return signature.array @ operator.coeffs


class SignatureLinearOperator:
    """Immutable linear operator on signature space.
    Use SignatureLinearOperatorBuilder to construct.
    """

    def __init__(
        self,
        coefficients: np.ndarray,
        dimension: int,
        level: int,
    ) -> None:
        """
        Direct construction (prefer using Builder).

        Parameters:
        -----------
        coefficients : np.ndarray
            1D array of length siglength(dimension, level) + 1
            First element is the constant coefficient
        dimension : int
            Dimension of the path space
        level : int
            Truncation level

        Raises:
        ------
        ValueError
            If coefficients length doesn't match expected signature length
        """
        self.coeffs = np.asarray(coefficients, dtype=np.float64)
        self.d = dimension
        self.level = level

        # Validate
        # Include the constant term (zeroth level)
        expected_length = iisignature.siglength(dimension, level) + 1
        if len(self.coeffs) != expected_length:
            raise ValueError(
                f"Coefficient array length {len(self.coeffs)} doesn't match "
                f"expected signature length {expected_length} (including constant term) for "
                f"dimension {dimension} and level {level}"
            )

    def truncate(self, new_level: int) -> Self:
        """
        Create new operator truncated at lower level.

        Parameters:
        -----------
        new_level : int
            Must be <= self.m

        Returns:
        --------
        SignatureLinearOperator
            New operator truncated at new_level

        Raises:
        -------
        ValueError
            If new_level > self.m
        """
        if new_level > self.level:
            raise ValueError(
                f"Cannot truncate to level {new_level} > current level {self.level}"
            )

        # Include the constant term in the length
        new_length = iisignature.siglength(self.d, new_level) + 1
        truncated_coeffs = self.coeffs[:new_length]

        return SignatureLinearOperator(truncated_coeffs, self.d, new_level)

    def __repr__(self) -> str:
        return (
            f"SignatureLinearOperator(dimension={self.d}, level={self.level}, "
            f"coeffs_length={len(self.coeffs)})"
        )

    @property
    def length(self) -> int:
        """Length of coefficient vector."""
        return len(self.coeffs)

    def __getitem__(self, n: int) -> float:
        """Prevent accidental indexing."""
        return cast(float, self.coeffs[n])


class SignatureLinearOperatorBuilder:
    """
    Builder for constructing SignatureLinearOperator from various sources.
    """

    def __init__(self, dimension: int, level: int):
        """
        Parameters:
        -----------
        dimension : int
            Dimension of the path space
        level : int
            Maximum truncation level
        """
        self.d = dimension
        self.level = level
        # Include the constant term in signature length
        self.sig_length = iisignature.siglength(dimension, level) + 1
        self._coeffs: np.ndarray | None = None

    @classmethod
    def from_array(
        cls,
        coefficients: np.ndarray,
        dimension: int,
        level: int,
    ) -> Self:
        """
        Build from explicit coefficient array.

        Parameters:
        -----------
        coefficients : np.ndarray
            Length must be siglength(dimension, level)
        dimension : int
            Dimension of the path space
        level : int
            Truncation level

        Returns:
        --------
        SignatureLinearOperatorBuilder (ready to build)

        Raises:
        -------
        ValueError
            If coefficients length doesn't match expected signature length
        """
        builder = cls(dimension, level)
        builder._coeffs = np.asarray(coefficients, dtype=np.float64)
        if len(builder._coeffs) != builder.sig_length:
            raise ValueError(
                f"Expected {builder.sig_length} coefficients, got {len(builder._coeffs)}"
            )
        return builder

    @classmethod
    def from_levels(
        cls,
        coeffs_by_level: dict[int, np.ndarray],
        dimension: int,
        level: int,
    ) -> Self:
        """
        Build from dictionary mapping level to coefficients.

        Parameters:
        -----------
        coeffs_by_level : dict[int, array-like]
            Keys are levels 1, 2, ..., m (implicitly, zero-th level is automatically 1)
            Values are coefficient arrays of length d^k for level k
        dimension : int
            Dimension of the path space
        level : int
            Truncation level

        Returns:
        --------
        SignatureLinearOperatorBuilder (ready to build)

        Raises:
        -------
        ValueError
            If any level is missing or has incorrect number of coefficients
        """
        builder = cls(dimension, level)
        # Start with the constant term (zeroth level = 1)
        level_arrays = [np.array([1.0])]

        for k in range(1, builder.level + 1):
            if k not in coeffs_by_level:
                raise ValueError(f"Missing coefficients for level {k}")

            level_coeffs = np.asarray(coeffs_by_level[k], dtype=np.float64)
            expected_length = builder.d**k
            if len(level_coeffs) != expected_length:
                raise ValueError(
                    f"Level {k}: expected {expected_length} coefficients, "
                    f"got {len(level_coeffs)}"
                )
            level_arrays.append(level_coeffs)

        builder._coeffs = np.concatenate(level_arrays)
        return builder

    @classmethod
    def zeros(cls, dimension: int, level: int) -> Self:
        """
        Initialize with zeros.

        Parameters:
        -----------
        dimension : int
            Dimension of the path space
        level : int
            Truncation level

        Returns:
        --------
        SignatureLinearOperatorBuilder (ready to build)
        """
        builder = cls(dimension, level)
        builder._coeffs = np.zeros(builder.sig_length)
        return builder

    @classmethod
    def ones(cls, dimension: int, level: int) -> Self:
        """
        Initialize with ones.

        Parameters:
        -----------
        dimension : int
            Dimension of the path space
        level : int
            Truncation level

        Returns:
        --------
        SignatureLinearOperatorBuilder (ready to build)
        """
        builder = cls(dimension, level)
        builder._coeffs = np.ones(builder.sig_length)
        return builder

    def build(self) -> SignatureLinearOperator:
        """
        Construct the final immutable operator.

        Returns:
        --------
        SignatureLinearOperator

        Raises
        ------
        ValueError
            If coefficients have not been set
        """
        if self._coeffs is None:
            raise ValueError("Must set coefficients before building")

        return SignatureLinearOperator(self._coeffs, self.d, self.level)

    def build_at_level(self, target_level: int) -> SignatureLinearOperator:
        """
        Build operator truncated at specific level.

        Parameters:
        -----------
        target_level : int
            Must be <= self.m

        Returns:
        --------
        SignatureLinearOperator

        Raises:
        -------
        ValueError
            If target_level > self.m or coefficients not set
        """
        if target_level > self.level:
            raise ValueError(
                f"Target level {target_level} exceeds builder level {self.level}"
            )

        if self._coeffs is None:
            raise ValueError("Must set coefficients before building")

        # Include the constant term in the length
        truncated_length = iisignature.siglength(self.d, target_level) + 1
        truncated_coeffs = self._coeffs[:truncated_length]

        return SignatureLinearOperator(truncated_coeffs, self.d, target_level)

    # <editor-fold desc="from_sde method and helpers">

    @classmethod
    def from_sde(
        cls,
        dimension: int,
        level: int,
        x0: float,
        a: float,
        b: float,
        alpha: float,
        beta: float,
    ):
        """
        Construct operator ℓ = p(δ - q)^(-1) from scalar SDE parameters.

        For the SDE:
            dXₜ = (a + bXₜ)dt + (α + βXₜ)dWₜ,  X₀ = x₀

        Constructs p and q according to equation (5):
            p = x₀ø + (a + ½β²)𝟙 + α𝟚
            q = b𝟙 + β𝟚

        Then solves ℓ = p ⊗ (ø + q + q⊗q + q⊗q⊗q + ...) using pure concatenation.

        Parameters:
        -----------
        dimension : int
            Dimension of the path space (for time-space augmentation)
        level : int
            Truncation level
        x0 : float
            Initial condition
        a : float
            Constant drift coefficient
        b : float
            Linear drift coefficient (feedback)
        alpha : float
            Constant diffusion coefficient
        beta : float
            Linear diffusion coefficient

        Returns:
        --------
        SignatureLinearOperatorBuilder (ready to build)

        Notes:
        ------
        𝟙 represents coordinate 0 (time/dt)
        𝟚 represents coordinate 1 (Brownian/dW)
        All operations use pure tensor algebra concatenation, no iisignature.
        """
        builder = cls(dimension, level)

        # We'll work with a dictionary mapping word tuples to coefficients
        # Word tuples are like: () for ø, (0,) for 𝟙, (1,) for 𝟚, (0,0) for 𝟙𝟙, etc.

        # Construct p = x₀ø + (a + ½β²)𝟙 + α𝟚
        p = {}
        p[()] = x0  # Empty word (ø)
        p[(0,)] = a + 0.5 * beta**2  # Coordinate 0 (time)
        if dimension >= 2:
            p[(1,)] = alpha  # Coordinate 1 (Brownian)

        # Construct q = b𝟙 + β𝟚
        q = {}
        q[(0,)] = b
        if dimension >= 2:
            q[(1,)] = beta

        # Compute resolvent = ø + q + q⊗q + q⊗q⊗q + ...
        resolvent = cls._concatenation_series(q, dimension, level)

        # Compute ℓ = p ⊗ resolvent
        ell = cls._concatenation_product_dict(p, resolvent, level)

        # Convert dictionary to array format expected by the builder
        coeffs_array = cls._dict_to_array(ell, dimension, level)

        builder._coeffs = coeffs_array
        return builder

    @staticmethod
    def _concatenation_series(q: dict, dimension: int, level: int) -> dict:
        """
        Compute (ø - q)^(-1) = ø + q + q⊗q + q⊗q⊗q + ...

        Parameters:
        -----------
        q : dict
            Dictionary mapping word tuples to coefficients
        dimension : int
            Dimension
        level : int
            Truncation level

        Returns:
        --------
        dict
            Resolvent as dictionary
        """
        result = {(): 1.0}  # Start with ø (identity)

        # Add q
        q_power = q.copy()
        result = SignatureLinearOperatorBuilder._add_dicts(result, q_power)

        # Add q⊗q, q⊗q⊗q, etc.
        for n in range(2, level + 1):
            q_power = SignatureLinearOperatorBuilder._concatenation_product_dict(
                q_power, q, level
            )
            result = SignatureLinearOperatorBuilder._add_dicts(result, q_power)

        return result

    @staticmethod
    def _concatenation_product_dict(a: dict, b: dict, max_level: int) -> dict:
        """
        Compute concatenation product a ⊗ b where both are dictionaries.

        For word w in a and word v in b:
            w ⊗ v = wv (concatenation of tuples)

        Parameters:
        -----------
        a : dict
            First operand (word -> coefficient)
        b : dict
            Second operand (word -> coefficient)
        max_level : int
            Maximum level (word length) to keep

        Returns:
        --------
        dict
            Product a ⊗ b
        """
        result = {}

        for word_a, coeff_a in a.items():
            for word_b, coeff_b in b.items():
                # Concatenate words
                concatenated_word = word_a + word_b

                # Only keep if within truncation level
                if len(concatenated_word) <= max_level:
                    if concatenated_word not in result:
                        result[concatenated_word] = 0.0
                    result[concatenated_word] += coeff_a * coeff_b

        return result

    @staticmethod
    def _add_dicts(a: dict, b: dict) -> dict:
        """Add two dictionaries (linear combinations)."""
        result = a.copy()
        for word, coeff in b.items():
            if word not in result:
                result[word] = 0.0
            result[word] += coeff
        return result

    @staticmethod
    def _dict_to_array(word_dict: dict, dimension: int, level: int) -> np.ndarray:
        """
        Convert dictionary representation to array.

        Array format: [level_0, level_1_coords..., level_2_coords..., ...]

        Parameters:
        -----------
        word_dict : dict
            Dictionary mapping word tuples to coefficients
        dimension : int
            Dimension
        level : int
            Truncation level

        Returns:
        --------
        np.ndarray
            Coefficients in array format
        """
        # Calculate total length
        total_length = 1  # Level 0
        for lev in range(1, level + 1):
            total_length += dimension**lev

        result = np.zeros(total_length)

        # Map words to indices
        index = 0

        # Level 0: empty word
        if () in word_dict:
            result[0] = word_dict[()]
        index = 1

        # Levels 1 to max_level
        for lev in range(1, level + 1):
            # Generate all words of length lev in lexicographic order
            for word in SignatureLinearOperatorBuilder._generate_words(dimension, lev):
                if word in word_dict:
                    result[index] = word_dict[word]
                index += 1

        return result

    @staticmethod
    def _generate_words(dimension: int, length: int):
        """
        Generate all words of given length in lexicographic order.

        Parameters:
        -----------
        dimension : int
            Number of coordinates (alphabet size)
        length : int
            Word length

        Yields:
        -------
        tuple
            Words as tuples of coordinates
        """
        if length == 0:
            yield ()
        else:
            for coord in range(dimension):
                for rest in SignatureLinearOperatorBuilder._generate_words(
                    dimension, length - 1
                ):
                    yield (coord,) + rest

    # </editor-fold>

    @classmethod
    def drift_functional_from_sde(
        cls,
        dimension: int,
        level: int,
        x0: float,
        a: float,
        b: float,
        alpha: float,
        beta: float,
    ):
        """
        Construct the analytical drift functional β such that:
            drift(Sig_t) = ⟨β, Sig_t⟩ = a + b·X_t

        where X_t = ⟨ℓ_X, Sig_t⟩ is given by the from_sde operator.

        For the SDE:
            dXₜ = (a + bXₜ)dt + (α + βXₜ)dWₜ,  X₀ = x₀

        The drift functional is:
            β = a·[1,0,0,...] + b·ℓ_X

        where ℓ_X is the solution operator from from_sde and [1,0,0,...] is
        the constant functional.

        Parameters:
        -----------
        dimension : int
            Dimension of the path space (for time-space augmentation)
        level : int
            Truncation level
        x0 : float
            Initial condition
        a : float
            Constant drift coefficient
        b : float
            Linear drift coefficient (feedback)
        alpha : float
            Constant diffusion coefficient
        beta : float
            Linear diffusion coefficient

        Returns:
        --------
        SignatureLinearOperatorBuilder
            Builder for the drift functional
        """
        # First, get the solution operator ℓ_X for X_t
        ell_X_builder = cls.from_sde(dimension, level, x0, a, b, alpha, beta)
        ell_X = ell_X_builder.build()

        # Construct drift functional: β = a·e_0 + b·ℓ_X
        # where e_0 = [1, 0, 0, ...] is the constant functional
        builder = cls(dimension, level)
        drift_coeffs = np.zeros(builder.sig_length)

        # Constant term: a times the constant functional
        drift_coeffs[0] = a

        # Linear term: b times the solution operator
        drift_coeffs += b * ell_X.coeffs

        builder._coeffs = drift_coeffs
        return builder

    @classmethod
    def diffusion_functional_from_sde(
        cls,
        dimension: int,
        level: int,
        x0: float,
        a: float,
        b: float,
        alpha: float,
        beta: float,
    ):
        """
        Construct the analytical diffusion functional α such that:
            diffusion²(Sig_t) = ⟨α, Sig_t⟩ = (α + β·X_t)²

        where X_t = ⟨ℓ_X, Sig_t⟩ is given by the from_sde operator.

        For the SDE:
            dXₜ = (a + bXₜ)dt + (α + βXₜ)dWₜ,  X₀ = x₀

        The diffusion functional (variance) is:
            α_func = α² + 2·α·β·ℓ_X + β²·(ℓ_X ⊗ ℓ_X)

        However, the last term (ℓ_X ⊗ ℓ_X) involves products of signature terms,
        which is a QUADRATIC functional, not linear. This is only exact for β=0.

        For β ≠ 0, we approximate by ignoring the quadratic term:
            α_func ≈ α² + 2·α·β·ℓ_X

        Parameters:
        -----------
        dimension : int
            Dimension of the path space (for time-space augmentation)
        level : int
            Truncation level
        x0 : float
            Initial condition
        a : float
            Constant drift coefficient
        b : float
            Linear drift coefficient (feedback)
        alpha : float
            Constant diffusion coefficient
        beta : float
            Linear diffusion coefficient

        Returns:
        --------
        SignatureLinearOperatorBuilder
            Builder for the (linearized) diffusion functional

        Warnings:
        ---------
        This is only exact for β=0 (constant volatility). For β≠0, the true
        diffusion functional is quadratic in the signature.
        """
        # Get the solution operator ℓ_X for X_t
        ell_X_builder = cls.from_sde(dimension, level, x0, a, b, alpha, beta)
        ell_X = ell_X_builder.build()

        # Construct diffusion functional: α_func ≈ α² + 2·α·β·ℓ_X
        builder = cls(dimension, level)
        diffusion_coeffs = np.zeros(builder.sig_length)

        # Constant term: α²
        diffusion_coeffs[0] = alpha**2

        # Linear term: 2·α·β·ℓ_X
        if beta != 0:
            import warnings

            warnings.warn(
                f"Diffusion functional for β={beta}≠0 is approximate. "
                f"True functional is quadratic, not linear.",
                UserWarning,
            )
            diffusion_coeffs += 2 * alpha * beta * ell_X.coeffs

        builder._coeffs = diffusion_coeffs
        return builder

    # </editor-fold>
