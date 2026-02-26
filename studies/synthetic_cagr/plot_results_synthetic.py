"""
Phase 3: Synthetic CAGR Results Plotting

Generates portfolio value plots matching PyTAAA style (log scale,
black traded curve, red buy-and-hold, normalized to $10k).

Also generates oracle curve showing what +20% CAGR tier would achieve,
to verify that portfolio is selecting high-CAGR stocks.

Output: plots/ directory with one PNG per model.
"""

import sys
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# =====================================================================
# Configuration
# =====================================================================

SYNTHETIC_HDF5 = Path(__file__).parent / "data" / "synthetic_naz100.hdf5"
PORTFOLIO_VALUES = Path(__file__).parent / "experiment_output" / \
    "backtest_results" / "portfolio_values.csv"
OUTPUT_DIR = Path(__file__).parent / "plots"

# Plot style
PLOT_STYLE = {
    "figsize": (14, 8),
    "fontsize_title": 14,
    "fontsize_label": 12,
    "fontsize_legend": 11,
    "linewidth_traded": 4,
    "linewidth_bh": 3,
}

# =====================================================================
# Logging
# =====================================================================

log_file = Path(__file__).parent / "plot_synthetic_results.log"
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
# Helper: Compute Metrics
# =====================================================================


def compute_metrics(pv_series: pd.Series) -> dict:
    """Compute Sharpe, annual return, average drawdown from PV series."""
    if pv_series.empty or len(pv_series) < 2:
        return None

    returns = pv_series.pct_change().dropna()

    if returns.empty:
        return None

    # Sharpe
    sharpe = returns.mean() / returns.std() * np.sqrt(252) \
        if returns.std() > 0 else 0

    # Annualized return
    total_return = (pv_series.iloc[-1] / pv_series.iloc[0]) - 1
    num_years = len(pv_series) / 252.0
    annual_return = (1 + total_return) ** (1 / num_years) - 1 \
        if num_years > 0 else 0

    # Average drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    avg_dd = drawdown.mean()

    return {
        "sharpe": sharpe,
        "annual_return": annual_return,
        "avg_dd": avg_dd,
    }


# =====================================================================
# Helper: Create Oracle Curve
# =====================================================================


def create_oracle_curve(num_days: int, initial_capital: float,
                        cagr: float = 0.20) -> np.ndarray:
    """
    Create theoretical profit curve for given CAGR.

    Args:
        num_days: Number of trading days
        initial_capital: Starting capital (e.g., 10000)
        cagr: Annual compound growth rate

    Returns:
        Array of portfolio values
    """
    oracle_pv = np.zeros(num_days)
    oracle_pv[0] = initial_capital

    daily_rate = cagr / 252.0

    for i in range(1, num_days):
        oracle_pv[i] = oracle_pv[i - 1] * (1 + daily_rate)

    return oracle_pv


# =====================================================================
# Helper: Plot Single Model
# =====================================================================


def plot_model(model_name: str, pv_series: pd.Series, prices_df: pd.DataFrame = None) -> str:
    """
    Plot portfolio value for one model with buy-and-hold and oracle overlay.

    Args:
        model_name: Name of model (e.g., "naz100_hma")
        pv_series: Series with index=date, values=portfolio_value
        prices_df: Full price DataFrame for buy-and-hold calculation

    Returns:
        Path to saved PNG
    """
    if pv_series.empty or len(pv_series) < 2:
        logger.warning(f"{model_name}: Empty series, skipping plot")
        return None

    # Create figure
    fig, ax = plt.subplots(
        figsize=PLOT_STYLE["figsize"],
        dpi=100,
    )

    # Normalize portfolio to $10k
    prices = pv_series.values.astype(float)
    if prices[0] > 0:
        prices_normalized = (prices / prices[0]) * 10000.0
    else:
        prices_normalized = prices

    # Compute metrics for title
    metrics = compute_metrics(pd.Series(prices_normalized))

    if metrics:
        title = (
            f"{model_name.upper()} | Sharpe: {metrics['sharpe']:.2f} | "
            f"Annual Return: {metrics['annual_return']:.1%} | "
            f"Avg DD: {metrics['avg_dd']:.1%}"
        )
    else:
        title = f"{model_name.upper()}"

    # Plot portfolio value (traded)
    dates = pv_series.index
    ax.plot(dates, prices_normalized, color="black", linewidth=4,
            label="Portfolio (Traded)")

    # Plot buy-and-hold curve
    if prices_df is not None and not prices_df.empty:
        # Equal weight of all stocks, held throughout
        bh_values = prices_df.mean(axis=1).values
        bh_normalized = (bh_values / bh_values[0]) * 10000.0
        ax.plot(dates, bh_normalized, color="blue", linewidth=2.5, linestyle="-",
                label="Buy & Hold (Equal Weight)")

    # Plot oracle curve (what +20% CAGR would yield)
    oracle_pv = create_oracle_curve(len(prices_normalized), 10000.0,
                                     cagr=0.20)
    ax.plot(dates, oracle_pv, color="red", linewidth=3, linestyle="--",
            label="Oracle (+20% CAGR)")

    # Styling
    ax.set_xlabel("Date", fontsize=PLOT_STYLE["fontsize_label"])
    ax.set_ylabel("Portfolio Value ($)", fontsize=PLOT_STYLE["fontsize_label"])
    ax.set_title(title, fontsize=PLOT_STYLE["fontsize_title"],
                 fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=PLOT_STYLE["fontsize_legend"], loc="upper left")

    # X-axis: annual ticks
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    # Add metrics table to plot
    if metrics:
        table_text = (
            f"Period: {dates[0].strftime('%Y-%m-%d')} to "
            f"{dates[-1].strftime('%Y-%m-%d')}\n"
            f"Sharpe Ratio: {metrics['sharpe']:.3f}\n"
            f"Annual Return: {metrics['annual_return']:.1%}\n"
            f"Avg Drawdown: {metrics['avg_dd']:.1%}"
        )
        ax.text(0.99, 0.05, table_text, transform=ax.transAxes,
                fontsize=10, verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat",
                          alpha=0.5))

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"synthetic_{model_name}.png"
    fig.savefig(output_file, dpi=100, bbox_inches="tight")
    logger.info(f"Saved: {output_file}")
    plt.close(fig)

    return str(output_file)


# =====================================================================
# Main
# =====================================================================


def main():
    """Generate plots for all models."""

    print("\n" + "=" * 70)
    print("PHASE 3: SYNTHETIC CAGR PLOTTING")
    print("=" * 70 + "\n")

    if not PORTFOLIO_VALUES.exists():
        logger.error(
            f"Portfolio values not found: {PORTFOLIO_VALUES}\n"
            "Run backtest_runner.py first."
        )
        sys.exit(1)

    # Load data
    logger.info(f"Loading portfolio values: {PORTFOLIO_VALUES}")
    pv_df = pd.read_csv(PORTFOLIO_VALUES, index_col=0, parse_dates=True)

    # Load synthetic price data for buy-and-hold curve
    prices_df = None
    if SYNTHETIC_HDF5.exists():
        logger.info(f"Loading synthetic prices: {SYNTHETIC_HDF5}")
        prices_df = pd.read_hdf(str(SYNTHETIC_HDF5), key='synthetic_naz100')
    else:
        logger.warning(f"Synthetic HDF5 not found; buy-and-hold curve omitted")

    # Plot each model
    output_files = []

    for model in pv_df.columns:
        pv_series = pv_df[model].dropna()

        output_file = plot_model(model, pv_series, prices_df)

        if output_file:
            output_files.append(output_file)

    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)
    print(f"Generated {len(output_files)} plots in {OUTPUT_DIR}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
