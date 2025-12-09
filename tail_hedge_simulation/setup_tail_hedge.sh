#!/bin/bash

# Set project path
PROJECT_PATH="/Users/donaldpg/PyProjects/tail_hedge_simulation"

# Create folders
mkdir -p "$PROJECT_PATH"/{data,notebooks,src,tests,config}

# Create README
cat > "$PROJECT_PATH/README.md" <<EOF
# Tail Hedge Simulation Framework

This project models asymmetric risk and tail-hedging strategies for portfolio construction using Monte Carlo simulation and behavioral overlays.

Modules:
- risk_premia.py: Simulates alternative risk premia.
- asymmetric_sim.py: Monte Carlo engine with volatility asymmetry.
- optimizer.py: Portfolio optimizer using semi-variance and CVaR.
- hedge_overlay.py: Protective put strategy simulation.
- reporting.py: Performance metrics and visualization.

Run simulations from notebooks/exploratory_analysis.ipynb
EOF

# Create requirements.txt
cat > "$PROJECT_PATH/requirements.txt" <<EOF
numpy
pandas
matplotlib
scipy
pyyaml
EOF

# Create starter Python files
touch "$PROJECT_PATH/src/__init__.py"

cat > "$PROJECT_PATH/src/risk_premia.py" <<EOF
import numpy as np
import pandas as pd

def simulate_risk_premia(n_periods=252, seed=42):
    np.random.seed(seed)
    premia = {
        "momentum": np.random.normal(0.0005, 0.01, n_periods),
        "carry": np.random.normal(0.0003, 0.008, n_periods),
        "value": np.random.normal(0.0004, 0.009, n_periods),
        "volatility": np.random.normal(0.0002, 0.012, n_periods)
    }
    for key in premia:
        premia[key] = np.where(premia[key] < 0, 1.5 * premia[key], premia[key])
    return pd.DataFrame(premia)
EOF

cat > "$PROJECT_PATH/src/optimizer.py" <<EOF
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
        return -np.mean(tail_losses)
    else:
        raise ValueError("Unknown method")

def optimize_portfolio(returns, method="semi"):
    n_assets = returns.shape[1]
    init_weights = np.ones(n_assets) / n_assets
    bounds = [(0, 1)] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    result = minimize(portfolio_objective, init_weights, args=(returns, method),
                      bounds=bounds, constraints=constraints)
    return result.x
EOF

cat > "$PROJECT_PATH/src/hedge_overlay.py" <<EOF
import numpy as np
import pandas as pd

def simulate_hedge_overlay(market_returns, hedge_allocation=0.03, strike_offset=-0.05, leverage=5):
    hedge_returns = []
    for r in market_returns:
        if r < strike_offset:
            payoff = abs(r - strike_offset) * leverage
            hedge_returns.append(payoff - hedge_allocation)
        else:
            hedge_returns.append(-hedge_allocation)
    hedge_returns = np.array(hedge_returns)
    traditional_returns = market_returns * (1 - hedge_allocation)
    combined_returns = traditional_returns + hedge_returns
    return pd.DataFrame({
        "Traditional": traditional_returns,
        "Hedge": hedge_returns,
        "Combined": combined_returns
    })
EOF

cat > "$PROJECT_PATH/notebooks/exploratory_analysis.ipynb" <<EOF
{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 2
}
EOF

echo "✅ Project scaffolded at $PROJECT_PATH"
