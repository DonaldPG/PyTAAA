"""
Simulates alternative risk premia: momentum, carry, value, and volatility.
Each premia is modeled with non-normal return distributions to reflect real-world asymmetry.
"""

import numpy as np
import pandas as pd

def simulate_risk_premia(n_periods=252, seed=42):
    np.random.seed(seed)

    premia = {
        "momentum": np.random.normal(loc=0.0005, scale=0.01, size=n_periods),
        "carry": np.random.normal(loc=0.0003, scale=0.008, size=n_periods),
        "value": np.random.normal(loc=0.0004, scale=0.009, size=n_periods),
        "volatility": np.random.normal(loc=0.0002, scale=0.012, size=n_periods)
    }

    # Introduce skew: amplify negative returns
    for key in premia:
        premia[key] = np.where(premia[key] < 0, 1.5 * premia[key], premia[key])

    return pd.DataFrame(premia)
