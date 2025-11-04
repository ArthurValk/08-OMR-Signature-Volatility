"""Examples"""

import numpy as np
from numpy.typing import NDArray


# Formulae for analytical solutions used in examples
def analytical_solution_ou(
    t: NDArray[np.floating],
    W: NDArray[np.floating],
    x0: float,
    theta: float,
    sigma: float,
) -> NDArray[np.floating]:
    """
    Analytical solution for OU process: dX = -theta*X*dt + sigma*dW

    Solution: X_t = x_0 * e^(-theta*t) + sigma * int_0^t e^(-theta*(t-s)) dW_s
    """
    # For discrete time points, we approximate the integral
    # This is exact for the discretization we use
    dt = t[1] - t[0]
    X = np.zeros_like(t)
    X[0] = x0

    for i in range(len(t) - 1):
        # Exact solution for one step
        X[i + 1] = X[i] * np.exp(-theta * dt) + sigma * (W[i + 1] - W[i])

    return X


def analytical_solution_gbm(
    t: NDArray[np.floating],
    W: NDArray[np.floating],
    x0: float,
    mu: float,
    sigma: float,
) -> NDArray[np.floating]:
    """
    Analytical solution for (log of) GBM: dX = (mu - sigma^2/2)*dt + sigma*dW

    Solution: X_t = x_0 + (mu - sigma^2/2)*t + sigma*W_t
    """
    return x0 + (mu - 0.5 * sigma**2) * t + sigma * W


def index_to_word(index: int, dimension: int, level: int) -> str:
    """Convert signature term index to word representation.

    Parameters
    ----------
    index : int
        Index in the signature array
    dimension : int
        Dimension of the path space
    level : int
        Truncation level

    Returns
    -------
    str
        Word representation (e.g., "∅", "1", "2", "11", "12", etc.)
    """
    if index == 0:
        return "∅"

    # Count terms at each level to find which level this index belongs to
    current_index = 1  # Start after the emptyset term
    for word_length in range(1, level + 1):
        terms_at_level = dimension**word_length
        if index < current_index + terms_at_level:
            # This index is at this level
            # Find which word within this level
            position = index - current_index

            # Convert position to base-dimension representation
            word = []
            for _ in range(word_length):
                word.append(str((position % dimension) + 1))
                position //= dimension
            return "".join(reversed(word))

        current_index += terms_at_level

    return str(index)  # Fallback
