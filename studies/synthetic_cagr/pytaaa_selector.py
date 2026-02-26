"""
PyTAAA Model Selector Wrapper

Provides unified interface to get stock selections for each PyTAAA model
(naz100_hma, naz100_pine, naz100_pi) using documented model specifications.

Model Specifications (from docs):
- naz100_hma: Hull Moving Averages signal + momentum-based ranking
- naz100_pine: Min-max channels signal + momentum-based ranking
- naz100_pi: Simple Moving Averages signal + momentum-based ranking

All models use deltaRank (change in momentum rank) for stock selection.
"""

import sys
import logging
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


class SimpleMA:
    """Simple Moving Average."""

    @staticmethod
    def compute(prices: np.ndarray, period: int) -> np.ndarray:
        """Compute SMA with NaN padding."""
        result = np.full_like(prices, np.nan, dtype=float)
        for i in range(period - 1, len(prices)):
            result[i] = np.nanmean(prices[i - period + 1 : i + 1])
        return result


class HullMA:
    """Hull Moving Average - faster responsive MA."""

    @staticmethod
    def compute(prices: np.ndarray, period: int) -> np.ndarray:
        """
        Compute HMA using formula:
        HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
        """
        # Use simple WMA approximation: weighted average with more recent bias
        result = np.full_like(prices, np.nan, dtype=float)
        for i in range(period - 1, len(prices)):
            window = prices[i - period + 1 : i + 1]
            weights = np.linspace(1, period, period)
            result[i] = np.nansum(window * weights) / np.nansum(weights)
        return result


class PyTAAAselector:
    """
    Selector for PyTAAA models using documented trading specifications.
    """

    def __init__(self, model_name: str):
        """
        Initialize selector for a specific model.

        Args:
            model_name: One of 'naz100_hma', 'naz100_pine', 'naz100_pi'
        """
        self.model_name = model_name

        # Model-specific parameters (from PyTAAA docs)
        if model_name == "naz100_hma":
            self.signal_method = "HMA"
            self.num_stocks = 7
            self.ma_short = 8
            self.ma_long = 176
            self.lookback_momentum = 600
        elif model_name == "naz100_pine":
            self.signal_method = "minmax"
            self.num_stocks = 7
            self.narrow_days = (6, 40)
            self.medium_days = (25, 38)
            self.wide_days = (75, 512)
            self.lookback_momentum = 600
        elif model_name == "naz100_pi":
            self.signal_method = "SMA"
            self.num_stocks = 7
            self.ma_short = 8
            self.ma_medium_offset = 11
            self.ma_long = 176
            self.lookback_momentum = 600
        else:
            raise ValueError(f"Unknown model: {model_name}")

        logger.info(f"Initialized {model_name} selector (signal={self.signal_method})")

    def get_stocks_for_date(
        self, prices_df: pd.DataFrame, date_idx: int
    ) -> List[str]:
        """
        Get ranked stock selections for a given date.

        Args:
            prices_df: DataFrame with index=dates, columns=tickers, values=prices
            date_idx: Index of the date in prices_df

        Returns:
            List of selected stock tickers, sorted by rank (best first)
        """
        if date_idx < 100:
            # Need sufficient history
            return []

        # Extract price array and tickers
        prices_array = prices_df.iloc[:date_idx+1, :].values.T  # (tickers, dates)
        tickers = list(prices_df.columns)

        # Compute uptrend signals
        signals = self._compute_signals(prices_array)

        # Find uptrending stocks with their indices
        uptrend_mask = signals[:, date_idx] > 0
        uptrend_indices = [i for i in range(len(tickers)) if uptrend_mask[i]]
        uptrend_tickers = [tickers[i] for i in uptrend_indices]

        if not uptrend_tickers:
            return []

        # Rank uptrending stocks by momentum
        # Create price subarray for just uptrending stocks
        uptrend_prices = prices_array[uptrend_indices, :]
        ranks = self._rank_by_momentum(uptrend_prices, uptrend_tickers, date_idx)

        # Select top N stocks
        top_tickers = sorted(ranks.items(), key=lambda x: x[1])[: self.num_stocks]
        selected = [t for t, _ in top_tickers]

        return selected

    def _compute_signals(self, prices_array: np.ndarray) -> np.ndarray:
        """Compute uptrend signals based on model's method."""

        if self.signal_method == "HMA":
            return self._signal_hma(prices_array)
        elif self.signal_method == "minmax":
            return self._signal_minmax(prices_array)
        elif self.signal_method == "SMA":
            return self._signal_sma(prices_array)

    def _signal_hma(self, prices_array: np.ndarray) -> np.ndarray:
        """Hull MA signal: price > HMA(long) or rising fast."""
        signals = np.zeros_like(prices_array, dtype=float)

        for ii in range(prices_array.shape[0]):
            prices = prices_array[ii, :]
            if np.isnan(prices).all():
                continue

            hma_short = HullMA.compute(prices, self.ma_short)
            hma_long = HullMA.compute(prices, self.ma_long)

            for jj in range(self.ma_long, len(prices)):
                if (prices[jj] > hma_long[jj]) or (
                    prices[jj] > hma_short[jj] and hma_short[jj] > hma_short[jj - 1]
                ):
                    signals[ii, jj] = 1

        return signals

    def _signal_minmax(self, prices_array: np.ndarray) -> np.ndarray:
        """Min-max channels signal (Pine method)."""
        signals = np.zeros_like(prices_array, dtype=float)

        # Simplified: use percentile-based channels
        for ii in range(prices_array.shape[0]):
            prices = prices_array[ii, :]
            if np.isnan(prices).all():
                continue

            for jj in range(60, len(prices)):
                # Look at 30-day range
                window = prices[jj - 30 : jj]
                if len(window) < 20:
                    continue
                p_min = np.nanpercentile(window, 20)
                p_max = np.nanpercentile(window, 80)
                pos = (prices[jj] - p_min) / (p_max - p_min + 1e-10)
                # Signal: in upper half of channel
                signals[ii, jj] = 1 if pos > 0.5 else 0

        return signals

    def _signal_sma(self, prices_array: np.ndarray) -> np.ndarray:
        """Three SMA signal (Pi method)."""
        signals = np.zeros_like(prices_array, dtype=float)

        for ii in range(prices_array.shape[0]):
            prices = prices_array[ii, :]
            if np.isnan(prices).all():
                continue

            sma_short = SimpleMA.compute(prices, self.ma_short)
            sma_medium = SimpleMA.compute(
                prices, self.ma_short + self.ma_medium_offset
            )
            sma_long = SimpleMA.compute(prices, self.ma_long)

            for jj in range(self.ma_long, len(prices)):
                if prices[jj] > sma_long[jj] or (
                    prices[jj] > min(sma_short[jj], sma_medium[jj])
                    and sma_short[jj] > sma_short[jj - 1]
                ):
                    signals[ii, jj] = 1

        return signals

    def _rank_by_momentum(
        self, prices_array: np.ndarray, tickers: List[str], date_idx: int
    ) -> dict:
        """
        Rank stocks by gain over lookback period (momentum).
        Higher gain = better rank = lower numeric value.
        
        Args:
            prices_array: (n_stocks, n_dates) array for the given tickers
            tickers: List of ticker symbols
            date_idx: Current date index
        """
        ranks = {}
        lookback = min(self.lookback_momentum, date_idx - 10)

        if lookback < 10:
            # Use simple gain/loss ranking with short history
            for ii, ticker in enumerate(tickers):
                if ii < prices_array.shape[0]:
                    price_now = prices_array[ii, date_idx]
                    price_past = prices_array[ii, max(0, date_idx - 20)]
                    if price_past > 0 and not np.isnan(price_now) and not np.isnan(price_past):
                        gain = (price_now - price_past) / price_past
                        ranks[ticker] = gain
            return ranks

        # Compute momentum based on lookback period
        gains = {}
        for ii, ticker in enumerate(tickers):
            if ii < prices_array.shape[0]:
                price_start = prices_array[ii, date_idx - lookback]
                price_now = prices_array[ii, date_idx]

                if price_start > 0 and not np.isnan(price_now) and not np.isnan(price_start):
                    recent_gain = (price_now - price_start) / price_start
                    gains[ticker] = recent_gain

        # Rank: lower rank = better (fastest gaining)
        if gains:
            sorted_by_gain = sorted(gains.items(), key=lambda x: x[1], reverse=True)
            for rank_idx, (ticker, _) in enumerate(sorted_by_gain):
                ranks[ticker] = rank_idx

        return ranks

