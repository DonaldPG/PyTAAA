"""
Compute new stock allocations from current holdings and target percentages.

This module provides functions to load holdings and target data from CSV files
and compute new share allocations.

Usage:
    # Load from CSV files
    computer = AllocationComputer.from_csv(
        holdings_csv='current_holdings.csv',
        targets_csv='target_allocations.csv'
    )
    computer.compute_allocations(targets_df)
    print(computer.format_output())

    # Or use DataFrames directly
    computer = AllocationComputer()
    computer.set_portfolio_value_from_holdings(holdings_df)
    computer.compute_allocations(targets_df)
    print(computer.format_output(targets_df))
"""

import pandas as pd
import os
from datetime import datetime
from typing import Dict, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


class AllocationComputer:
    """Compute new stock allocations from target percentages."""

    def __init__(self, portfolio_value: Optional[float] = None):
        """
        Initialize the allocation computer.

        Args:
            portfolio_value: Total portfolio value. If None, will be calculated
                           from current holdings.
        """
        self.portfolio_value = portfolio_value
        self.allocations = {}
        self.targets_df = None

    @classmethod
    def from_csv(cls, holdings_csv: str, targets_csv: str) -> 'AllocationComputer':
        """
        Create AllocationComputer from CSV files.

        Args:
            holdings_csv: Path to CSV with current holdings (must have 'Value ($)' column)
            targets_csv: Path to CSV with target allocations (must have columns:
                        symbol, percentage, current_price)

        Returns:
            AllocationComputer instance with portfolio value initialized
        """
        if not os.path.exists(holdings_csv):
            raise FileNotFoundError(f"Holdings CSV not found: {holdings_csv}")
        if not os.path.exists(targets_csv):
            raise FileNotFoundError(f"Targets CSV not found: {targets_csv}")

        holdings_df = pd.read_csv(holdings_csv)
        targets_df = pd.read_csv(targets_csv)

        computer = cls()
        computer.set_portfolio_value_from_holdings(holdings_df)
        computer.targets_df = targets_df

        return computer

    def set_portfolio_value_from_holdings(self, holdings_df: pd.DataFrame):
        """
        Calculate total portfolio value from current holdings.

        Args:
            holdings_df: DataFrame with current holdings containing 'Value ($)' column
        """
        if 'Value ($)' not in holdings_df.columns:
            raise ValueError("Holdings DataFrame must contain 'Value ($)' column")

        self.portfolio_value = holdings_df['Value ($)'].sum()
        logger.info(f"Portfolio value calculated: ${self.portfolio_value:,.2f}")

    def compute_allocations(
        self,
        targets_df: pd.DataFrame,
        rounding: str = "floor"
    ) -> Dict[str, float]:
        """
        Compute new share counts for each symbol based on target percentages.

        Args:
            targets_df: DataFrame with required columns:
                - symbol: Stock ticker symbol
                - percentage: Target allocation percentage (0-1)
                - current_price: Current price per share

                Optional columns (will be used if present):
                - rank: Rank/priority of the symbol
                - company_name: Full company name

            rounding: How to handle fractional shares
                - "floor": Round down to nearest integer
                - "round": Round to nearest integer
                - "none": Keep fractional shares

        Returns:
            Dictionary mapping symbol to number of shares

        Raises:
            ValueError: If required columns missing or portfolio value not set
        """
        if self.portfolio_value is None:
            raise ValueError("Portfolio value not set. Call set_portfolio_value_from_holdings().")

        required_cols = ['symbol', 'percentage', 'current_price']
        missing_cols = [col for col in required_cols if col not in targets_df.columns]
        if missing_cols:
            raise ValueError(f"Targets DataFrame missing required columns: {missing_cols}")

        self.allocations = {}
        self.targets_df = targets_df

        # Calculate shares for each stock
        for _, row in targets_df.iterrows():
            symbol = row['symbol']
            percentage = float(row['percentage'])
            current_price = float(row['current_price'])

            target_value = self.portfolio_value * percentage
            shares = target_value / current_price

            if rounding == "floor":
                shares = int(shares)
            elif rounding == "round":
                shares = round(shares)
            # else keep fractional

            self.allocations[symbol] = shares
            logger.info(
                f"{symbol}: {percentage*100:>6.2f}% of ${self.portfolio_value:,.2f} = "
                f"${target_value:,.2f} = {shares:>10.2f} shares @ ${current_price:>8.2f}"
            )

        return self.allocations

    def compute_remaining_cash(self) -> float:
        """
        Calculate remaining cash after stock allocations.

        Returns:
            Remaining cash value
        """
        if self.targets_df is None:
            raise ValueError("Must call compute_allocations() first")

        stock_value = 0.0
        for _, row in self.targets_df.iterrows():
            symbol = row['symbol']
            current_price = float(row['current_price'])
            shares = self.allocations.get(symbol, 0)
            stock_value += shares * current_price

        cash_remaining = self.portfolio_value - stock_value
        logger.info(f"Remaining cash: ${cash_remaining:,.2f}")
        return cash_remaining

    def format_output(
        self,
        targets_df: Optional[pd.DataFrame] = None,
        trade_date: Optional[str] = None,
        trading_model: str = "sp500_pine",
        cash_price: float = 1.0
    ) -> str:
        """
        Format the output in the requested format.

        Format:
            TradeDate: YYYY-MM-DD
            trading_model: model_name
            stocks:      CASH  SYM1  SYM2  ...
            shares:      xxx   xxx   xxx   ...
            buyprice:    1.0   xxx   xxx   ...

        Args:
            targets_df: Target allocations DataFrame. If None, uses the one from
                       compute_allocations()
            trade_date: Trade date (YYYY-MM-DD format). Defaults to today.
            trading_model: Name of trading model
            cash_price: Price per unit of cash (default 1.0)

        Returns:
            Formatted output string

        Raises:
            ValueError: If targets_df not provided and not set from compute_allocations()
        """
        if targets_df is None:
            if self.targets_df is None:
                raise ValueError(
                    "Must provide targets_df or call compute_allocations() first"
                )
            targets_df = self.targets_df

        if trade_date is None:
            trade_date = datetime.now().strftime("%Y-%m-%d")

        # Get sorted symbols
        symbols = list(targets_df['symbol'].values)
        prices = dict(zip(targets_df['symbol'], targets_df['current_price']))

        # Build output
        output = f"TradeDate: {trade_date}\n"
        output += f"trading_model: {trading_model}\n"

        # Add CASH first
        cash_shares = self.compute_remaining_cash() / cash_price
        output += "stocks:      CASH"
        output += "".join([f"  {s:>8}" for s in symbols])
        output += "\n"

        output += f"shares:      {cash_shares:>8.2f}"
        output += "".join([f"  {self.allocations[s]:>8.2f}" for s in symbols])
        output += "\n"

        output += f"buyprice:    {cash_price:>8.2f}"
        output += "".join([f"  {prices[s]:>8.2f}" for s in symbols])
        output += "\n"

        return output

    def print_summary(self, targets_df: Optional[pd.DataFrame] = None):
        """
        Print detailed allocation summary.

        Args:
            targets_df: Target allocations DataFrame. If None, uses the one from
                       compute_allocations()
        """
        if targets_df is None:
            if self.targets_df is None:
                raise ValueError(
                    "Must provide targets_df or call compute_allocations() first"
                )
            targets_df = self.targets_df

        print("\nAllocation Summary:")
        print("-" * 80)
        print(f"{'Symbol':<8} {'Shares':>12} {'Price':>10} {'Value':>15} {'Allocation %':>15}")
        print("-" * 80)

        for symbol in targets_df['symbol'].values:
            price = targets_df[targets_df['symbol'] == symbol]['current_price'].values[0]
            shares = self.allocations[symbol]
            value = shares * price
            pct = (value / self.portfolio_value) * 100
            print(f"{symbol:<8} {shares:>12.2f} ${price:>9.2f} ${value:>14,.2f} {pct:>14.2f}%")

        cash = self.compute_remaining_cash()
        cash_pct = (cash / self.portfolio_value) * 100
        print("-" * 80)
        print(
            f"{'CASH':<8} {cash:>12.2f} ${1.00:>9.2f} ${cash:>14,.2f} {cash_pct:>14.2f}%"
        )
        print("-" * 80)
        print(f"{'TOTAL':<8} {'':<12} {'':<10} ${self.portfolio_value:>14,.2f} {'100.00%':>14}")
        print("-" * 80)


def main():
    """Example usage demonstrating the AllocationComputer."""

    # Example 1: Using DataFrames directly
    print("\n" + "="*80)
    print("EXAMPLE 1: Using DataFrames directly")
    print("="*80)

    current_holdings_data = {
        'symbol': ['CASH', 'STX', 'STX', 'AVGO', 'PLTR', 'RL', 'STX', 'WBD'],
        'shares': [895, 4, 8, 70, 53, 14, 8, 53],
        'Value ($)': [895.06, 1150.16, 2300.32, 24333.40, 8896.58,
                      5075.42, 2300.32, 1511.03]
    }
    current_holdings_df = pd.DataFrame(current_holdings_data)

    target_allocations_data = {
        'symbol': ['GEV', 'WBD', 'SNDK', 'FIX', 'STX'],
        'rank': [1, 2, 3, 4, 5],
        'percentage': [0.3620, 0.0267, 0.1188, 0.3839, 0.1086],
        'current_price': [679.55, 28.51, 275.24, 1003.64, 287.54],
        'trend': ['up', 'up', 'up', 'up', 'up'],
        'company_name': ['GE Vernova', 'Warner Bros. Discovery',
                         'Sandisk', 'Comfort Systems USA', 'Seagate Technology']
    }
    targets_df = pd.DataFrame(target_allocations_data)

    computer = AllocationComputer()
    computer.set_portfolio_value_from_holdings(current_holdings_df)
    computer.compute_allocations(targets_df, rounding="floor")

    print("\nNEW ALLOCATIONS (Formatted Output)")
    print("-" * 80)
    output = computer.format_output(
        targets_df,
        trade_date="2026-01-02",
        trading_model="sp500_pine"
    )
    print(output)

    computer.print_summary(targets_df)


if __name__ == "__main__":
    main()
