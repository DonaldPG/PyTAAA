"""Download and plot a stock ticker versus major market indices.

Usage examples:
    cd studies
    uv run python ticker_history_plot.py XQQI
    uv run python ticker_history_plot.py MSFT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


INDEX_TICKERS = {
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq 100",
}


def download_adj_close(symbol: str, period: str = "10y") -> pd.DataFrame:
    """Download daily adjusted-close history for a symbol."""

    data = yf.Ticker(symbol).history(
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=False,
    )

    if data.empty:
        raise ValueError(f"No price history returned for {symbol}.")

    frame = data[["Adj Close"]].copy()
    frame.columns = ["adj_close"]
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index().dropna()
    return frame


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Compute a simple moving average for the given window size."""

    return series.rolling(window=window, min_periods=window).mean()


def normalize_to_start_value(series: pd.Series, start_value: float = 10000.0) -> pd.Series:
    """Map a series to a starting value of $10,000."""

    series = pd.to_numeric(series, errors="coerce")
    first_valid = series.dropna().iloc[0] if not series.dropna().empty else None
    if first_valid is None or first_valid == 0:
        return pd.Series(index=series.index, dtype=float)
    return (series / first_valid) * start_value


def build_plot_frame(ticker: str) -> pd.DataFrame:
    """Merge ticker and benchmark histories into a single frame."""

    ticker_hist = download_adj_close(ticker)
    ticker_name = ticker.upper()
    ticker_hist = ticker_hist.rename(columns={"adj_close": ticker_name})

    combined = ticker_hist
    for index_symbol, label in INDEX_TICKERS.items():
        index_hist = download_adj_close(index_symbol)
        combined = combined.join(index_hist.rename(columns={"adj_close": label}), how="inner")

    combined[ticker_name] = pd.to_numeric(combined[ticker_name], errors="coerce")
    for label in INDEX_TICKERS.values():
        combined[label] = pd.to_numeric(combined[label], errors="coerce")

    combined[ticker_name] = normalize_to_start_value(combined[ticker_name])
    for label in INDEX_TICKERS.values():
        combined[label] = normalize_to_start_value(combined[label])

    combined["sma_50"] = compute_sma(combined[ticker_name], 50)
    combined["sma_200"] = compute_sma(combined[ticker_name], 200)
    combined = combined.dropna(subset=[ticker_name])
    return combined


def make_summary_rows(frame: pd.DataFrame, ticker: str) -> list[tuple[str, float, float]]:
    """Summarize final value and percent gain for each tracked series."""

    ticker_name = ticker.upper()
    rows = []
    series_map = {
        ticker_name: frame[ticker_name],
        "S&P 500 (^GSPC)": frame["S&P 500"],
        "Nasdaq 100 (^IXIC)": frame["Nasdaq 100"],
    }

    for name, values in series_map.items():
        final_value = float(values.dropna().iloc[-1]) if not values.dropna().empty else 0.0
        pct_gain = (final_value / 10000.0 - 1.0) * 100.0
        rows.append((name, final_value, pct_gain))

    return rows


def plot_history(ticker: str, output_dir: Path | None = None) -> Path:
    """Create a chart comparing the ticker versus the main indexes."""

    frame = build_plot_frame(ticker)
    ticker_name = ticker.upper()

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(
        frame.index,
        frame[ticker_name],
        label=ticker_name,
        linewidth=2,
        color="tab:blue",
    )
    ax.plot(
        frame.index,
        frame["S&P 500"],
        label="S&P 500 (^GSPC)",
        linewidth=1.5,
        alpha=0.8,
        color="tab:green",
    )
    ax.plot(
        frame.index,
        frame["Nasdaq 100"],
        label="Nasdaq 100 (^IXIC)",
        linewidth=1.5,
        alpha=0.8,
        color="tab:orange",
    )
    ax.plot(
        frame.index,
        frame["sma_50"],
        label="50-day SMA",
        linestyle="--",
        linewidth=1.5,
        color="tab:red",
    )
    ax.plot(
        frame.index,
        frame["sma_200"],
        label="200-day SMA",
        linestyle=":",
        linewidth=2,
        color="tab:purple",
    )

    ax.set_title(f"Performance of $10,000 Invested in {ticker_name} vs S&P 500 and Nasdaq 100")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, alpha=0.25)

    summary_rows = make_summary_rows(frame, ticker)
    summary_text = "\n".join(
        f"{name}: ${final:,.0f} ({pct:+.1f}%)"
        for name, final, pct in summary_rows
    )
    ax.text(
        0.02,
        0.02,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent

    output_path = output_dir / f"{ticker_name}_history.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved chart to {output_path}")
    print(f"Data period: {frame.index.min().date()} to {frame.index.max().date()}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download a ticker's adjusted-close history and compare it against "
            "the S&P 500 and Nasdaq 100 with moving averages."
        )
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default="XQQI",
        help="Ticker symbol to plot, such as XQQI, MSFT, or AAPL.",
    )
    parser.add_argument(
        "--period",
        default="3y",
        help="YFinance history period, such as 1y, 3y, 5y, or 10y.",
    )
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()
    print(f"Downloading history for {ticker} and benchmark indexes...")

    # Reuse the same period for all symbols.
    stock_hist = download_adj_close(ticker, period=args.period)
    # Keep each series aligned by using the same daily index window.
    combined = stock_hist.rename(columns={"adj_close": ticker})
    for index_symbol, index_name in INDEX_TICKERS.items():
        index_hist = download_adj_close(index_symbol, period=args.period)
        combined = combined.join(index_hist.rename(columns={"adj_close": index_name}), how="inner")

    combined[ticker] = pd.to_numeric(combined[ticker], errors="coerce")
    for label in INDEX_TICKERS.values():
        combined[label] = pd.to_numeric(combined[label], errors="coerce")

    combined[ticker] = normalize_to_start_value(combined[ticker])
    for label in INDEX_TICKERS.values():
        combined[label] = normalize_to_start_value(combined[label])
    combined["sma_50"] = compute_sma(combined[ticker], 50)
    combined["sma_200"] = compute_sma(combined[ticker], 200)

    if combined.empty:
        raise ValueError(f"No overlapping history was found for {ticker} and the benchmark indices.")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(combined.index, combined[ticker], label=ticker, linewidth=2, color="tab:blue")
    ax.plot(
        combined.index,
        combined["S&P 500"],
        label="S&P 500 (^GSPC)",
        linewidth=1.5,
        alpha=0.8,
        color="tab:green",
    )
    ax.plot(
        combined.index,
        combined["Nasdaq 100"],
        label="Nasdaq 100 (^IXIC)",
        linewidth=1.5,
        alpha=0.8,
        color="tab:orange",
    )
    ax.plot(
        combined.index,
        combined["sma_50"],
        label="50-day SMA",
        linestyle="--",
        linewidth=1.5,
        color="tab:red",
    )
    ax.plot(
        combined.index,
        combined["sma_200"],
        label="200-day SMA",
        linestyle=":",
        linewidth=2,
        color="tab:purple",
    )
    ax.set_title(f"Performance of $10,000 invested in {ticker} vs Benchmarks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, alpha=0.25)

    summary_rows = make_summary_rows(combined, ticker)
    summary_text = "\n".join(
        f"{name}: ${final:,.0f} ({pct:+.1f}%)"
        for name, final, pct in summary_rows
    )
    ax.text(
        0.02,
        0.02,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()

    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / f"{ticker}_history.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved chart to {output_path}")
    print(f"Data period: {combined.index.min().date()} to {combined.index.max().date()}")


if __name__ == "__main__":
    main()
