"""
Compute new stock allocations based on target percentages and current portfolio value.

This script reads:
1. Current holdings (symbols, shares, current prices, values)
2. Target allocations (symbols, percentages, current prices)

And outputs:
- New number of shares for each symbol
- Remaining cash position
- Trade date and model information
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AllocationComputer:
    """Compute new stock allocations from target percentages."""

    def __init__(self, portfolio_value: float = None):
        """
        Initialize the allocation computer.

        Args:
            portfolio_value: Total portfolio value. If None, will be calculated
                           from current holdings.
        """
        self.portfolio_value = portfolio_value
        self.allocations = {}

    def set_portfolio_value_from_holdings(self, holdings_df: pd.DataFrame):
        """
        Calculate total portfolio value from current holdings.

        Args:
            holdings_df: DataFrame with current holdings containing 'Value ($)' column
        """
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
            targets_df: DataFrame with columns:
                - symbol: Stock ticker symbol
                - percentage: Target allocation percentage (0-1)
                - current_price: Current price per share

            rounding: How to handle fractional shares
                - "floor": Round down to nearest integer
                - "round": Round to nearest integer
                - "none": Keep fractional shares

        Returns:
            Dictionary mapping symbol to number of shares
        """
        if self.portfolio_value is None:
            raise ValueError("Portfolio value not set. Call set_portfolio_value_from_holdings().")

        self.allocations = {}

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

    def compute_remaining_cash(self, targets_df: pd.DataFrame) -> float:
        """
        Calculate remaining cash after stock allocations.

        Args:
            targets_df: DataFrame with current_price column

        Returns:
            Remaining cash value
        """
        stock_value = 0.0
        for _, row in targets_df.iterrows():
            symbol = row['symbol']
            current_price = float(row['current_price'])
            shares = self.allocations.get(symbol, 0)
            stock_value += shares * current_price

        cash_remaining = self.portfolio_value - stock_value
        logger.info(f"Remaining cash: ${cash_remaining:,.2f}")
        return cash_remaining

    def format_output(
        self,
        targets_df: pd.DataFrame,
        trade_date: str = None,
        trading_model: str = "sp500_pine",
        cash_price: float = 1.0
    ) -> str:
        """
        Format the output in the requested format.

        Args:
            targets_df: Target allocations DataFrame
            trade_date: Trade date (YYYY-MM-DD format). Defaults to today.
            trading_model: Name of trading model
            cash_price: Price per unit of cash (default 1.0)

        Returns:
            Formatted output string
        """
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y-%m-%d")

        # Get sorted symbols
        symbols = list(targets_df['symbol'].values)
        prices = dict(zip(targets_df['symbol'], targets_df['current_price']))

        # Build output
        output = f"TradeDate: {trade_date}\n"
        output += f"trading_model: {trading_model}\n"

        # Add CASH first
        cash_shares = self.compute_remaining_cash(targets_df) / cash_price
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


def main():
    """Example usage of AllocationComputer."""

    # Current holdings (updated)
    current_holdings_data = {
        'symbol': ['INTC', 'WBD', 'WBD', 'ASML', 'AVGO', 'CASH', 'INTC', 'KLAC', 'LRCX', 'MNST', 'WBD'],
        'shares': [270, 40, 49, 6, 246, 1002, 397, 18, 68, 85, 14],
        'Value ($)': [10632.60, 1140.40, 1396.99, 6982.68, 85514.52, 1002.09,
                      15633.86, 22940.46, 12584.08, 6473.60, 399.14]
    }
    current_holdings_df = pd.DataFrame(current_holdings_data)

    # Example: Target allocations
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

    # Create computer and compute allocations
    computer = AllocationComputer()
    computer.set_portfolio_value_from_holdings(current_holdings_df)
    computer.compute_allocations(targets_df, rounding="floor")

    # Print formatted output
    output = computer.format_output(
        targets_df,
        trade_date="2026-01-02",
        trading_model="sp500_pine"
    )
    print("\n" + "="*80)
    print("NEW ALLOCATIONS")
    print("="*80)
    print(output)
    print("="*80)

    # Print summary
    print("\nAllocation Summary:")
    print("-" * 60)
    for symbol in targets_df['symbol'].values:
        price = targets_df[targets_df['symbol'] == symbol]['current_price'].values[0]
        shares = computer.allocations[symbol]
        value = shares * price
        pct = (value / computer.portfolio_value) * 100
        print(f"{symbol:>6} {shares:>10.2f} shares @ ${price:>8.2f} = ${value:>12,.2f} ({pct:>6.2f}%)")

    cash = computer.compute_remaining_cash(targets_df)
    cash_pct = (cash / computer.portfolio_value) * 100
    print(f"{'CASH':>6} {cash:>10.2f} units  @ $1.00 = ${cash:>12,.2f} ({cash_pct:>6.2f}%)")
    print("-" * 60)
    print(f"{'TOTAL':>6} {'':<10}       = ${computer.portfolio_value:>12,.2f} (100.00%)")


if __name__ == "__main__":
    main()
