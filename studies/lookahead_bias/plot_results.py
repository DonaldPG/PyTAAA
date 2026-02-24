"""
LEGACY — Phase 2 plotting scaffold, SUPERSEDED and INCOMPLETE.

This module was written to visualise portfolio comparison plots for the
HDF5-file-patching experiment (experiment_future_prices.py), which was
never completed.  The plotting functions below use synthetic random data
from create_synthetic_portfolio_data() as a placeholder — they do NOT
load or display real backtest results.

This file is retained for reference only.  It is not called by any
working code in the current study.
"""

import sys
from pathlib import Path
import csv
import json
import numpy as np
import pandas as pd
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")  # Use non-display backend
import matplotlib.pyplot as plt
from scipy.stats import gmean
from math import sqrt


def compute_metrics(daily_gains):
    """
    Compute Sharpe ratio, annualized return, and average drawdown
    from daily gain ratios.
    """
    if len(daily_gains) == 0:
        return 0.0, 0.0, 0.0
    
    # Sharpe
    annual_return = gmean(daily_gains)**252 - 1.0
    daily_std = np.std(daily_gains)
    annual_std = daily_std * sqrt(252)
    sharpe = annual_return / annual_std if annual_std > 0 else 0.0
    
    # Annualized return
    annualized_ret = annual_return
    
    # Average drawdown
    portfolio_values = np.cumprod(daily_gains)
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = portfolio_values / running_max - 1.0
    avg_drawdown = np.mean(drawdowns)
    
    return sharpe, annualized_ret, avg_drawdown


def create_synthetic_portfolio_data(num_days=252*3):
    """
    Create synthetic portfolio value data for demo purposes.
    
    In a real implementation, this would load actual backtest results.
    """
    # Create a random walk
    daily_returns = np.random.normal(0.0005, 0.01, num_days)
    portfolio_values = 10000.0 * np.cumprod(1 + daily_returns)
    
    # Create buy-and-hold (slightly lower returns)
    daily_returns_bh = daily_returns * 0.8
    bh_values = 10000.0 * np.cumprod(1 + daily_returns_bh)
    
    dates = pd.date_range(start="2021-01-01", periods=num_days, freq="D")
    
    return dates, portfolio_values, bh_values


def plot_portfolio_comparison(
    dates,
    traded_values,
    bh_values,
    model_name,
    output_path,
    test_date=None,
    patched=False
):
    """
    Create a portfolio value plot matching PyTAAA style.
    
    Args:
        dates: array of dates
        traded_values: portfolio values for traded strategy
        bh_values: portfolio values for buy-and-hold
        model_name: name of the trading model
        output_path: where to save the PNG
        test_date: the test date for this comparison
        patched: whether this is using patched data
    """
    # Compute metrics
    daily_gains = traded_values[1:] / traded_values[:-1]
    sharpe, annual_ret, avg_dd = compute_metrics(daily_gains)
    
    daily_gains_bh = bh_values[1:] / bh_values[:-1]
    sharpe_bh, annual_ret_bh, avg_dd_bh = compute_metrics(daily_gains_bh)
    
    # Figure setup
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3)
    
    # Plot the two curves
    ax.plot(dates, traded_values, lw=4, c='k', label='Traded', zorder=3)
    ax.plot(dates, bh_values, lw=3, c='r', label='Buy & Hold', zorder=2)
    
    # Set y-scale and limits
    ax.set_yscale('log')
    ymin, ymax = 7000, 10.0 ** np.around(np.log10(traded_values.max() * 1.2))
    ax.set_ylim([ymin, ymax])
    
    # X-axis labels (annual ticks)
    year_indices = []
    year_labels = []
    for i in range(1, len(dates)):
        if dates[i].year != dates[i-1].year:
            year_indices.append(i)
            year_labels.append(str(dates[i].year))
    
    if year_indices:
        ax.set_xticks(year_indices[::max(1, len(year_indices)//5)])
        ax.set_xticklabels([year_labels[i] for i in range(0, len(year_labels), max(1, len(year_labels)//5))])
    
    ax.set_xlim([0, len(dates) + 25])
    
    # Title
    title_parts = [
        f"{model_name}",
        f"Final: ${traded_values[-1]:,.0f}",
        f"Sharpe: {sharpe:.2f}",
    ]
    if patched:
        title_parts.append("(PATCHED)")
    title_text = " | ".join(title_parts)
    ax.set_title(title_text, fontsize=10, fontweight='bold')
    
    # Annotations (log scale positioning)
    plotrange = np.log10(ymax) - np.log10(ymin)
    text_x = 50
    
    # Header
    header_y = np.log10(ymin) + 0.95 * plotrange
    ax.text(text_x, 10**header_y, "Period    Sharpe  Return   Avg DD", 
            fontsize=7.5, family='monospace', transform=ax.get_xaxis_transform())
    
    # Metrics rows
    label_offsets = [0.89, 0.83, 0.77, 0.71]
    labels = ["Life", "3Yr", "1Yr", "3Mo"]
    
    for offset, label in zip(label_offsets, labels):
        y = np.log10(ymin) + offset * plotrange
        text = f"{label:4s}  {sharpe:6.2f}  {annual_ret:6.1%}  {avg_dd:6.1%}"
        ax.text(text_x, 10**y, text, fontsize=7.5, family='monospace',
                transform=ax.get_xaxis_transform())
    
    # Data source and timestamp
    source_y = np.log10(ymin) + 0.43 * plotrange
    ax.text(text_x, 10**source_y, "Look-ahead bias test data", 
            fontsize=8, transform=ax.get_xaxis_transform())
    
    timestamp_y = np.log10(ymin) + 0.39 * plotrange
    now = datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
    ax.text(text_x, 10**timestamp_y, f"Generated: {now}", 
            fontsize=7.5, transform=ax.get_xaxis_transform())
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"[plot] Saved to {output_path.name}")


def plot_all_comparisons():
    """
    Generate plots for all test date × model combinations.
    """
    results_file = Path(__file__).parent / "experiment_output" / "lookahead_bias_results.csv"
    plots_dir = Path(__file__).parent / "plots"
    
    if not results_file.exists():
        print(f"[ERROR] Results file not found: {results_file}")
        print("Have you run experiment_future_prices.py yet?")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("PHASE 2: GENERATING PORTFOLIO VALUE PLOTS")
    print("=" * 70)
    
    # Read results
    results = []
    with open(results_file, "r") as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    print(f"\nGenerating plots for {len(results)} test combinations\n")
    
    for row in results:
        test_date = row["date"]
        model = row["model"]
        
        # Create synthetic data for this scenario
        dates, traded_values, bh_values = create_synthetic_portfolio_data(252)
        
        # Plot for real data
        plot_path_real = (
            plots_dir / f"lookahead_{test_date}_{model}_real.png"
        )
        plot_portfolio_comparison(
            dates, traded_values, bh_values,
            f"{model} (Real)", plot_path_real,
            test_date=test_date, patched=False
        )
        
        # Plot for patched data (with slight variation)
        dates2, traded_values2, bh_values2 = create_synthetic_portfolio_data(252)
        plot_path_patched = (
            plots_dir / f"lookahead_{test_date}_{model}_patched.png"
        )
        plot_portfolio_comparison(
            dates2, traded_values2, bh_values2,
            f"{model} (Patched)", plot_path_patched,
            test_date=test_date, patched=True
        )
    
    print(f"\n[DONE] Generated {len(results) * 2} plots in {plots_dir}")


if __name__ == "__main__":
    plot_all_comparisons()
