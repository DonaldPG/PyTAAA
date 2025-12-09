"""
Simulates a protective put hedge overlay for a traditional portfolio.
Models monthly cost and payoff behavior under market drawdowns.
"""

import numpy as np
import pandas as pd

def simulate_hedge_overlay(market_returns, hedge_allocation=0.03, strike_offset=-0.05, leverage=5):
    """
    Parameters:
    - market_returns: Series of monthly market returns
    - hedge_allocation: % of portfolio allocated to hedge (e.g., 0.03 = 3%)
    - strike_offset: % below current price for put strike (e.g., -0.05 = 5% OTM)
    - leverage: payoff multiplier during crash

    Returns:
    - DataFrame with traditional, hedge, and combined portfolio returns
    """
    hedge_returns = []

    for r in market_returns:
        if r < strike_offset:
            # Market dropped below strike → hedge pays off
            payoff = abs(r - strike_offset) * leverage
            hedge_returns.append(payoff - hedge_allocation)  # net of cost
        else:
            # Put expires worthless → lose premium
            hedge_returns.append(-hedge_allocation)

    hedge_returns = np.array(hedge_returns)
    traditional_returns = market_returns * (1 - hedge_allocation)
    combined_returns = traditional_returns + hedge_returns

    return pd.DataFrame({
        "Traditional": traditional_returns,
        "Hedge": hedge_returns,
        "Combined": combined_returns
    })
