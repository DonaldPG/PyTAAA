#!/usr/bin/env python3
"""
Generate synthetic stock charts illustrating PyTAAA signal methods and ranking.

Creates PNG charts for each trading method (SMAs, HMAs, Pine/minmaxChannels)
showing:
1. Adjusted closing price with technical indicators
2. Signal generation (uptrend/downtrend regions)
3. Ranking based on change-in-rank (deltaRank)

Usage:
    uv run python docs/generate_method_charts.py

Output:
    docs/chart_SMAs_signal.png
    docs/chart_HMAs_signal.png
    docs/chart_PINE_signal.png
    docs/chart_deltaRank_example.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_synthetic_price(n_days=500, seed=42):
    """Generate a synthetic stock price series with trend changes."""
    np.random.seed(seed)
    
    # Create a price series with distinct phases:
    # Phase 1: Uptrend (days 0-150)
    # Phase 2: Decline (days 150-250)
    # Phase 3: Recovery/Uptrend (days 250-400)
    # Phase 4: Choppy/Flat (days 400-500)
    
    daily_returns = np.zeros(n_days)
    
    # Phase 1: Uptrend
    daily_returns[0:150] = np.random.normal(0.0008, 0.015, 150)
    # Phase 2: Decline
    daily_returns[150:250] = np.random.normal(-0.001, 0.018, 100)
    # Phase 3: Recovery
    daily_returns[250:400] = np.random.normal(0.0012, 0.014, 150)
    # Phase 4: Choppy
    daily_returns[400:500] = np.random.normal(0.0001, 0.016, 100)
    
    price = 100.0 * np.cumprod(1 + daily_returns)
    return price


def compute_sma(price, period):
    """Compute Simple Moving Average."""
    sma = np.zeros_like(price)
    for i in range(len(price)):
        start = max(0, i - period)
        sma[i] = np.mean(price[start:i+1])
    return sma


def compute_hma(price, period):
    """Compute Hull Moving Average."""
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    
    # WMA helper
    def wma(data, n):
        result = np.zeros_like(data)
        weights = np.arange(1, n + 1, dtype=float)
        for i in range(len(data)):
            start = max(0, i - n + 1)
            window = data[start:i+1]
            w = weights[-(len(window)):]
            result[i] = np.sum(window * w) / np.sum(w)
        return result
    
    wma_half = wma(price, half_period)
    wma_full = wma(price, period)
    diff = 2 * wma_half - wma_full
    
    # Final smoothing with SMA
    hma = np.zeros_like(diff)
    for i in range(len(diff)):
        start = max(0, i - sqrt_period + 1)
        hma[i] = np.mean(diff[start:i+1])
    return hma


def compute_dpg_channel(price, min_period, max_period, inc_period):
    """Compute DPG min-max channel."""
    periods = np.arange(min_period, max_period, inc_period)
    min_ch = np.zeros_like(price)
    max_ch = np.zeros_like(price)
    
    for i in range(len(price)):
        divisor = 0
        for p in periods:
            start = max(0, i - int(p))
            if start < i:
                min_ch[i] += np.min(price[start:i+1])
                max_ch[i] += np.max(price[start:i+1])
            else:
                min_ch[i] += price[i]
                max_ch[i] += price[i]
            divisor += 1
        min_ch[i] /= divisor
        max_ch[i] /= divisor
    
    return min_ch, max_ch


def plot_smas_method(price, output_path):
    """Generate chart for SMAs signal method."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
    
    days = np.arange(len(price))
    
    # Compute SMAs
    ma2 = 8
    ma2_offset = 11
    ma1 = 80  # Scaled down from 176 for synthetic data
    sma2_factor = 1.15  # Scaled down from 1.536 for visibility
    
    sma0 = compute_sma(price, ma2)
    sma1_arr = compute_sma(price, ma2 + ma2_offset)
    sma2 = sma2_factor * compute_sma(price, ma1)
    
    # Compute signal
    signal = np.zeros(len(price))
    for j in range(1, len(price)):
        if price[j] > sma2[j] or \
           (price[j] > min(sma0[j], sma1_arr[j]) and sma0[j] > sma0[j-1]):
            signal[j] = 1
    
    # Plot 1: Price and SMAs
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(days, price, 'k-', linewidth=1.2, label='Adjusted Close', zorder=3)
    ax1.plot(days, sma0, 'b-', linewidth=0.8, alpha=0.8, label=f'SMA({ma2})')
    ax1.plot(days, sma1_arr, 'g-', linewidth=0.8, alpha=0.8, label=f'SMA({ma2+ma2_offset})')
    ax1.plot(days, sma2, 'r-', linewidth=1.0, alpha=0.8, label=f'{sma2_factor:.2f} x SMA({ma1})')
    
    # Shade uptrend regions
    for j in range(1, len(signal)):
        if signal[j] == 1:
            ax1.axvspan(j-0.5, j+0.5, alpha=0.1, color='green', linewidth=0)
    
    ax1.set_title('SMAs Method — Signal Generation (naz100_pi)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Signal
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(days, signal, 0, alpha=0.5, color='green', label='Uptrend Signal')
    ax2.set_ylabel('Signal')
    ax2.set_ylim(-0.1, 1.3)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Down', 'Up'])
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: SMA0 slope (rising/falling)
    sma0_slope = np.diff(sma0, prepend=sma0[0])
    ax3 = fig.add_subplot(gs[2])
    colors = ['green' if s > 0 else 'red' for s in sma0_slope]
    ax3.bar(days, sma0_slope, color=colors, alpha=0.6, width=1.0)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_ylabel(f'SMA({ma2}) Slope')
    ax3.set_xlabel('Trading Days')
    ax3.grid(True, alpha=0.3)
    
    # Add annotation
    ax1.annotate('Signal = 1 when:\nPrice > scaled long SMA\nOR\nPrice > min(short SMAs)\nAND short SMA rising',
                xy=(0.98, 0.02), xycoords='axes fraction',
                fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_hmas_method(price, output_path):
    """Generate chart for HMAs signal method."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
    
    days = np.arange(len(price))
    
    # Compute HMAs and SMAs for comparison
    ma2 = 8
    ma2_offset = 11
    ma1 = 80
    sma2_factor = 1.15
    
    hma0 = compute_hma(price, ma2)
    hma1_arr = compute_hma(price, ma2 + ma2_offset)
    hma2 = sma2_factor * compute_hma(price, ma1)
    
    sma0 = compute_sma(price, ma2)
    sma1_arr_sma = compute_sma(price, ma2 + ma2_offset)
    
    # Compute HMA signal
    signal_hma = np.zeros(len(price))
    for j in range(1, len(price)):
        if price[j] > hma2[j] or \
           (price[j] > min(hma0[j], hma1_arr[j]) and hma0[j] > hma0[j-1]):
            signal_hma[j] = 1
    
    # Compute SMA signal for comparison
    sma2_comp = sma2_factor * compute_sma(price, ma1)
    signal_sma = np.zeros(len(price))
    for j in range(1, len(price)):
        if price[j] > sma2_comp[j] or \
           (price[j] > min(sma0[j], sma1_arr_sma[j]) and sma0[j] > sma0[j-1]):
            signal_sma[j] = 1
    
    # Plot 1: Price and HMAs
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(days, price, 'k-', linewidth=1.2, label='Adjusted Close', zorder=3)
    ax1.plot(days, hma0, 'b-', linewidth=0.8, alpha=0.8, label=f'HMA({ma2})')
    ax1.plot(days, hma1_arr, 'g-', linewidth=0.8, alpha=0.8, label=f'HMA({ma2+ma2_offset})')
    ax1.plot(days, hma2, 'r-', linewidth=1.0, alpha=0.8, label=f'{sma2_factor:.2f} x HMA({ma1})')
    
    # Shade uptrend regions
    for j in range(1, len(signal_hma)):
        if signal_hma[j] == 1:
            ax1.axvspan(j-0.5, j+0.5, alpha=0.1, color='green', linewidth=0)
    
    ax1.set_title('HMAs Method — Signal Generation (naz100_hma / sp500_hma)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: HMA vs SMA signal comparison
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(days, signal_hma, 0, alpha=0.4, color='blue', label='HMA Signal')
    ax2.fill_between(days, signal_sma * 0.9, 0, alpha=0.3, color='orange', label='SMA Signal (comparison)')
    ax2.set_ylabel('Signal')
    ax2.set_ylim(-0.1, 1.3)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Down', 'Up'])
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: HMA vs SMA lag comparison
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(days, hma0, 'b-', linewidth=1.0, label=f'HMA({ma2}) — less lag')
    ax3.plot(days, sma0, 'orange', linewidth=1.0, linestyle='--', label=f'SMA({ma2}) — more lag')
    ax3.set_ylabel('MA Value')
    ax3.set_xlabel('Trading Days')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax1.annotate('HMA responds faster than SMA\n→ Earlier entries and exits\n→ More whipsaws in choppy markets',
                xy=(0.98, 0.02), xycoords='axes fraction',
                fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_pine_method(price, output_path):
    """Generate chart for Pine/minmaxChannels signal method."""
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(4, 1, height_ratios=[3, 1.5, 1, 1], hspace=0.3)
    
    days = np.arange(len(price))
    
    # Compute channels
    narrow_min, narrow_max = compute_dpg_channel(price, 6, 40, 5)
    narrow_mid = (narrow_min + narrow_max) / 2
    
    medium_min, medium_max = compute_dpg_channel(price, 25, 38, 2)
    
    wide_min, wide_max = compute_dpg_channel(price, 75, 200, 18)
    
    # Compute signals
    medium_range = medium_max - medium_min
    medium_range[medium_range == 0] = 1e-10
    medium_signal = ((narrow_mid - medium_min) / medium_range - 0.5) * 2.0
    
    wide_range = wide_max - wide_min
    wide_range[wide_range == 0] = 1e-10
    wide_signal = ((narrow_mid - wide_min) / wide_range - 0.5) * 2.0
    
    combined_signal = medium_signal + wide_signal
    uptrend = (combined_signal > 0).astype(float)
    
    # Plot 1: Price and channels
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(days, price, 'k-', linewidth=1.2, label='Adjusted Close', zorder=4)
    ax1.fill_between(days, wide_min, wide_max, alpha=0.1, color='red', label='Wide Channel (75-200d)')
    ax1.fill_between(days, medium_min, medium_max, alpha=0.15, color='blue', label='Medium Channel (25-38d)')
    ax1.fill_between(days, narrow_min, narrow_max, alpha=0.2, color='green', label='Narrow Channel (6-40d)')
    ax1.plot(days, narrow_mid, 'g--', linewidth=0.7, alpha=0.7, label='Narrow Mid')
    
    # Shade uptrend regions
    for j in range(1, len(uptrend)):
        if uptrend[j] == 1:
            ax1.axvspan(j-0.5, j+0.5, alpha=0.05, color='lime', linewidth=0)
    
    ax1.set_title('Pine Method (minmaxChannels) — Signal Generation (naz100_pine / sp500_pine)', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Medium and Wide signals
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(days, medium_signal, 'b-', linewidth=0.8, alpha=0.8, label='mediumSignal')
    ax2.plot(days, wide_signal, 'r-', linewidth=0.8, alpha=0.8, label='wideSignal')
    ax2.plot(days, combined_signal, 'k-', linewidth=1.2, label='Sum (medium + wide)')
    ax2.axhline(y=0, color='gray', linewidth=1.0, linestyle='--')
    ax2.fill_between(days, combined_signal, 0, where=combined_signal > 0, 
                     alpha=0.2, color='green', label='Uptrend (sum > 0)')
    ax2.fill_between(days, combined_signal, 0, where=combined_signal <= 0, 
                     alpha=0.2, color='red', label='Downtrend (sum <= 0)')
    ax2.set_ylabel('Signal Value')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Binary signal
    ax3 = fig.add_subplot(gs[2])
    ax3.fill_between(days, uptrend, 0, alpha=0.5, color='green', label='Uptrend Signal')
    ax3.set_ylabel('Signal')
    ax3.set_ylim(-0.1, 1.3)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Down', 'Up'])
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Channel position (narrow mid within wide channel)
    position = (narrow_mid - wide_min) / wide_range
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(days, position, 'purple', linewidth=0.8)
    ax4.axhline(y=0.5, color='gray', linewidth=1.0, linestyle='--', label='Channel midpoint')
    ax4.fill_between(days, position, 0.5, where=position > 0.5, alpha=0.3, color='green')
    ax4.fill_between(days, position, 0.5, where=position <= 0.5, alpha=0.3, color='red')
    ax4.set_ylabel('Position in\nWide Channel')
    ax4.set_xlabel('Trading Days')
    ax4.set_ylim(0, 1)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    ax1.annotate('Signal = 1 when:\nmediumSignal + wideSignal > 0\n\nChannels adapt to volatility:\nwider in volatile periods',
                xy=(0.98, 0.02), xycoords='axes fraction',
                fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_deltarank_example(output_path):
    """Generate chart illustrating the deltaRank ranking strategy."""
    from scipy.stats import rankdata
    
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(4, 1, height_ratios=[2, 2, 1.5, 1.5], hspace=0.35)
    
    # Generate 5 synthetic stocks
    np.random.seed(123)
    n_days = 300
    stock_names = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create distinct price patterns
    base_returns = np.random.normal(0.0003, 0.012, (5, n_days))
    # AAPL: accelerating uptrend
    base_returns[0, 150:] += 0.002
    # MSFT: decelerating
    base_returns[1, :150] += 0.002
    base_returns[1, 150:] -= 0.001
    # GOOGL: steady improvement
    base_returns[2, 100:] += 0.0015
    # AMZN: declining
    base_returns[3, 100:] -= 0.001
    # NVDA: flat then surge
    base_returns[4, 200:] += 0.003
    
    prices = np.zeros((5, n_days))
    for i in range(5):
        prices[i] = 100 * np.cumprod(1 + base_returns[i])
    
    days = np.arange(n_days)
    long_period = 100
    
    # Compute gain/loss over LongPeriod
    gainloss = np.ones((5, n_days))
    gainloss[:, long_period:] = prices[:, long_period:] / prices[:, :-long_period]
    
    # Compute ranks (1 = best)
    current_rank = np.zeros((5, n_days))
    for j in range(n_days):
        r = rankdata(-gainloss[:, j])  # negative for descending
        current_rank[:, j] = r
    
    # Compute previous rank
    prev_rank = np.ones((5, n_days))
    prev_rank[:, long_period:] = current_rank[:, :-long_period]
    
    # Compute delta
    offset = 2
    delta = -(current_rank - prev_rank) / (current_rank + offset)
    
    # Compute deltaRank
    delta_rank = np.zeros((5, n_days))
    for j in range(n_days):
        r = rankdata(-delta[:, j])  # negative for descending (best delta = rank 1)
        delta_rank[:, j] = r
    
    # Plot 1: Stock prices
    ax1 = fig.add_subplot(gs[0])
    for i in range(5):
        ax1.plot(days, prices[i], color=colors[i], linewidth=1.2, label=stock_names[i])
    ax1.set_title('deltaRank Ranking Strategy — Change in Rank Selects Stocks', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Current rank (absolute)
    ax2 = fig.add_subplot(gs[1])
    for i in range(5):
        ax2.plot(days[long_period:], current_rank[i, long_period:], 
                color=colors[i], linewidth=1.0, label=stock_names[i])
    ax2.set_ylabel('Absolute Rank\n(1 = best performer)')
    ax2.set_ylim(0.5, 5.5)
    ax2.invert_yaxis()
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.annotate('Absolute rank shows WHO is best\nbut NOT who is IMPROVING fastest',
                xy=(0.98, 0.95), xycoords='axes fraction',
                fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    # Plot 3: Delta (change in rank)
    ax3 = fig.add_subplot(gs[2])
    for i in range(5):
        ax3.plot(days[long_period:], delta[i, long_period:], 
                color=colors[i], linewidth=1.0, label=stock_names[i])
    ax3.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax3.set_ylabel('Delta\n(rank improvement)')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.annotate('Delta = -(current_rank - prev_rank) / (current_rank + offset)\nPositive delta = rank is improving',
                xy=(0.98, 0.95), xycoords='axes fraction',
                fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    # Plot 4: deltaRank (final selection)
    ax4 = fig.add_subplot(gs[3])
    for i in range(5):
        ax4.plot(days[long_period:], delta_rank[i, long_period:], 
                color=colors[i], linewidth=1.0, label=stock_names[i])
    ax4.axhline(y=2.5, color='red', linewidth=1.5, linestyle='--', label='Selection threshold (top 2)')
    ax4.fill_between(days[long_period:], 0, 2.5, alpha=0.1, color='green')
    ax4.set_ylabel('deltaRank\n(1 = fastest improving)')
    ax4.set_ylim(0.5, 5.5)
    ax4.invert_yaxis()
    ax4.set_xlabel('Trading Days')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.annotate('deltaRank selects stocks with\nFASTEST IMPROVING rank\n(green zone = selected for portfolio)',
                xy=(0.98, 0.05), xycoords='axes fraction',
                fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    # Generate synthetic price data
    price = generate_synthetic_price(n_days=500, seed=42)
    
    # Generate charts for each method
    plot_smas_method(price, os.path.join(OUTPUT_DIR, 'chart_SMAs_signal.png'))
    plot_hmas_method(price, os.path.join(OUTPUT_DIR, 'chart_HMAs_signal.png'))
    plot_pine_method(price, os.path.join(OUTPUT_DIR, 'chart_PINE_signal.png'))
    plot_deltarank_example(os.path.join(OUTPUT_DIR, 'chart_deltaRank_example.png'))
    
    print("\nAll charts generated successfully.")
    print("Charts are saved in the docs/ directory.")
