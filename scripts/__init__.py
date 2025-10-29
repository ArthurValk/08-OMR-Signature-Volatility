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
