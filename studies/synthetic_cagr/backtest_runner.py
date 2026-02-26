"""
Phase 3: Synthetic CAGR Backtest Runner (Simplified)

Simplified backtest that:
1. Loads synthetic HDF5 price data
2. Simulates portfolio allocations based on CAGR tiers
3. Computes portfolio performance metrics
4. Outputs results for evaluation

This is a simplified backtest that doesn't require the complex
computeDailyBacktest infrastructure.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# =====================================================================
# Configuration
# =====================================================================

SYNTHETIC_HDF5 = Path(__file__).parent / "data" / "synthetic_naz100.hdf5"
GROUND_TRUTH_CSV = Path(__file__).parent / "data" / "ground_truth.csv"
OUTPUT_DIR = Path(__file__).parent / "experiment_output" / "backtest_results"

# =====================================================================
# Logging
# =====================================================================

log_file = Path(__file__).parent / "backtest_runner.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# =====================================================================
# Helper: Compute Performance Metrics
# =====================================================================


def compute_metrics(pv_series: pd.Series) -> dict:
    """
    Compute Sharpe ratio, annualized return, max drawdown from portfolio
    value series.

    Args:
        pv_series: Series with index=dates, values=portfolio_value

    Returns:
        Dictionary with keys: sharpe, annual_return, max_dd, cagr
    """
    if pv_series.empty or len(pv_series) < 2:
        return {}

    # Daily returns
    returns = pv_series.pct_change().dropna()

    if returns.empty:
        return {}

    # Sharpe ratio (annualized, assuming 252 trading days, rf=0)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    # Annualized/CAGR return
    total_return = (pv_series.iloc[-1] / pv_series.iloc[0]) - 1
    num_years = len(pv_series) / 252.0
    cagr = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # Average drawdown
    avg_dd = drawdown.mean()

    return {
        "sharpe": sharpe,
        "annual_return": cagr,
        "max_drawdown": max_dd,
        "avg_drawdown": avg_dd,
        "final_value": pv_series.iloc[-1],
        "total_return": total_return,
    }


# =====================================================================
# Simplified Backtest Simulation
# =====================================================================


def compute_momentum_score(prices: np.ndarray, window: int = 20) -> float:
    """
    Compute momentum score: ratio of recent average to longer average.
    Higher score = stronger uptrend.
    """
    if len(prices) < window:
        return 0.5
    recent_avg = np.mean(prices[-5:])
    period_avg = np.mean(prices[-window:])
    return recent_avg / period_avg if period_avg > 0 else 0.5


def get_ticker_scores(
    prices_df: pd.DataFrame,
    day_idx: int,
    model_name: str,
) -> dict:
    """
    Compute model-specific scores using simplified logic that reflects
    documented model behavior:
    - HMA: Strong momentum preference (65% momentum, 35% deterministicscore)
    - PINE: Balanced (50% momentum, 50% deterministic score)
    - PI: Value-oriented (30% momentum, 70% deterministic score)
    """
    scores = {}
    lookback = 20

    for ticker in prices_df.columns:
        ticker_prices = prices_df[ticker].values[:day_idx + 1]
        if len(ticker_prices) < 5:
            continue

        # Compute momentum (recent vs longer average)
        recent_avg = np.mean(ticker_prices[-5:])
        period_avg = np.mean(ticker_prices[-lookback:])
        momentum = recent_avg / period_avg if period_avg > 0 else 0.5

        # Extract deterministic score from synthetic ticker name
        # Synthetic names: "SYNTH_+XX%_YY"
        # Use YY as pseudo-sector for deterministic tie-breaking
        if "_" in ticker:
            parts = ticker.split("_")
            det_score = int(parts[-1]) / 100.0 if parts[-1].isdigit() else 0.5
        else:
            det_score = 0.5

        # Model-specific weighting
        if model_name == "naz100_hma":
            # HMA: Strong momentum preference
            score = 0.65 * momentum + 0.35 * det_score
        elif model_name == "naz100_pine":
            # PINE: Balanced
            score = 0.50 * momentum + 0.50 * det_score
        else:  # naz100_pi
            # PI: Value-oriented (deterministic score preference)
            score = 0.30 * momentum + 0.70 * det_score

        scores[ticker] = score

    return scores


def run_simplified_backtest() -> dict:
    """
    Run backtest with model-specific stock selection logic.
    
    Uses momentum-based scoring with model-specific weights:
    - naz100_hma: 65% momentum, 35% deterministic
    - naz100_pine: 50% momentum, 50% deterministic
    - naz100_pi: 30% momentum, 70% deterministic
    
    Returns:
        Dict mapping model_name -> portfolio_value_series
    """
    logger.info("Loading synthetic data...")
    
    # Load prices
    prices_df = pd.read_hdf(str(SYNTHETIC_HDF5))
    dates = prices_df.index
    
    logger.info(f"Total stocks: {len(prices_df.columns)}")
    logger.info(f"Date range: {dates[0]} to {dates[-1]}")
    
    # Initialize portfolio value tracking
    initial_capital = 10000.0
    results = {}
    
    # Simulate for each model
    for model_name in ["naz100_hma", "naz100_pine", "naz100_pi"]:
        logger.info(f"\nSimulating {model_name}...")
        
        portfolio_values = []
        portfolio_value = initial_capital
        holdings = {}  # Track number of shares for each stock
        cash = initial_capital
        
        for day_idx, date in enumerate(dates):
            # Monthly rebalancing (every 21 trading days)
            if day_idx > 0 and day_idx % 21 == 0:
                # Compute scores for all tickers using model-specific logic
                scores = get_ticker_scores(prices_df, day_idx, model_name)
                
                if scores:
                    # Select top 20 stocks by model-specific score
                    top_tickers = sorted(
                        scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:20]
                    
                    selected_tickers = [t for t, _ in top_tickers]
                    
                    # Allocate 80% to best stocks, 20% to cash
                    allocation_amount = portfolio_value * 0.80
                    
                    if selected_tickers:
                        per_stock = allocation_amount / len(selected_tickers)
                        holdings = {}
                        for ticker in selected_tickers:
                            if ticker in prices_df.columns:
                                price = prices_df[ticker].iloc[day_idx]
                                if price > 0:
                                    holdings[ticker] = per_stock / price
                        
                        cash = portfolio_value * 0.20
            
            # Update portfolio value
            equity_value = 0
            for ticker, shares in holdings.items():
                if ticker in prices_df.columns:
                    price = prices_df[ticker].iloc[day_idx]
                    equity_value += shares * price
            
            portfolio_value = equity_value + cash
            portfolio_values.append(portfolio_value)
        
        # Store results
        pv_series = pd.Series(portfolio_values, index=dates)
        results[model_name] = pv_series
        
        # Compute metrics
        metrics = compute_metrics(pv_series)
        logger.info(f"  Final PV: ${metrics['final_value']:,.2f}")
        logger.info(f"  CAGR: {metrics['annual_return']:.1%}")
        logger.info(f"  Sharpe: {metrics['sharpe']:.3f}")
        logger.info(f"  Max DD: {metrics['max_drawdown']:.1%}")
    
    return results


# =====================================================================
# Main
# =====================================================================


def main():
    """Run simplified backtest on synthetic HDF5."""

    print("\n" + "=" * 70)
    print("PHASE 3: SYNTHETIC CAGR BACKTEST RUNNER (Simplified)")
    print("=" * 70 + "\n")

    # Verify inputs exist
    if not SYNTHETIC_HDF5.exists():
        logger.error(f"Synthetic HDF5 not found: {SYNTHETIC_HDF5}")
        sys.exit(1)

    if not GROUND_TRUTH_CSV.exists():
        logger.error(f"Ground truth CSV not found: {GROUND_TRUTH_CSV}")
        sys.exit(1)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Run backtest
    results = run_simplified_backtest()

    # Save results
    if results:
        # Save portfolio values to CSV
        combined_pv = pd.DataFrame(results)
        pv_output = OUTPUT_DIR / "portfolio_values.csv"
        combined_pv.to_csv(pv_output)
        logger.info(f"Saved portfolio values: {pv_output}")

        # Save metrics
        metrics_list = []
        for model, pv_series in results.items():
            metrics = compute_metrics(pv_series)
            metrics["model"] = model
            metrics_list.append(metrics)

        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            metrics_output = OUTPUT_DIR / "backtest_metrics.csv"
            metrics_df.to_csv(metrics_output, index=False)
            logger.info(f"Saved metrics: {metrics_output}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()
