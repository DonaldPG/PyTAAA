"""
Monte Carlo simulation engine for asymmetric volatility modeling.
Simulates asset paths with conditional volatility amplification on downside moves.
"""

import numpy as np

def simulate_paths(S0, mu, sigma, T, steps, n_paths, lambda_down=2.0):
    dt = T / steps
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0

    for t in range(1, steps + 1):
        z = np.random.normal(size=n_paths)
        shock = np.where(z < 0, lambda_down * z, z)
        paths[:, t] = paths[:, t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shock)

    return paths
