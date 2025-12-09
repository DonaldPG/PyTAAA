"""
Portfolio optimizer using semi-variance and CVaR to penalize downside risk.
"""

import numpy as np
from scipy.optimize import minimize

def semi_variance(returns):
    downside = returns[returns < 0]
    return np.var(downside) if len(downside) > 0 else 0

def portfolio_objective(weights, returns, method="semi"):
    portfolio_returns = returns @ weights
    if method == "semi":
        return semi_variance(portfolio_returns)
    elif method == "cvar":
        alpha = 0.05
        threshold = np.percentile(portfolio_returns, alpha * 100)
        tail_losses = portfolio_returns[portfolio_returns <= threshold]
        return -np.mean(tail_losses)  # minimize expected shortfall
    else:
        raise ValueError("Unknown method")

def optimize_portfolio(returns, method="semi"):
    n_assets = returns.shape[1]
    init_weights = np.ones(n_assets) / n_assets
    bounds = [(0, 1)] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(portfolio_objective, init_weights,
                      args=(returns, method),
                      bounds=bounds,
                      constraints=constraints)

    return result.x
