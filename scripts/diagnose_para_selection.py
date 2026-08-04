"""
diagnose_para_selection.py — Diagnose why PARA appears in new stock selections.

Three checks are performed:

1. HDF5 quotes: print the last 15 daily prices for PARA stored in the
   SP500 HDF5 file, so we can see whether the trailing prices are real
   or infilled (constant).

2. yfinance quotes: download PARA directly from Yahoo Finance to compare
   with what is in the HDF5.  PARA / PARAA were delisted on 2025-08-08
   after the Skydance merger, so yfinance should return no data (or a
   very short history ending on that date).

3. active_mask check: run load_quotes_for_analysis with the real JSON
   config and check whether PARA is masked out on the last date, and
   whether it still appears in the derived stock selection list.

Usage:
    cd /Users/donaldpg/PyProjects/worktree2/PyTAAA
    PYTHONPATH=$(pwd) uv run python scripts/diagnose_para_selection.py \
        --json /Users/donaldpg/pytaaa_data/sp500_hma/pytaaa_sp500_hma.json
"""

import argparse
import sys
import datetime

import numpy as np
import pandas as pd


##############################################################################
# CLI
##############################################################################

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose PARA appearing in SP500 stock selections."
    )
    parser.add_argument(
        "--json",
        dest="json_fn",
        required=True,
        help="Path to the PyTAAA JSON config (e.g. pytaaa_sp500_hma.json)",
    )
    parser.add_argument(
        "--symbol",
        dest="symbol",
        default="PARA",
        help="Ticker to investigate (default: PARA)",
    )
    parser.add_argument(
        "--tail",
        dest="tail",
        type=int,
        default=15,
        help="Number of trailing HDF5 rows to display (default: 15)",
    )
    return parser.parse_args()


##############################################################################
# Check 1 — HDF5 stored quotes
##############################################################################

def check_hdf5_quotes(
    json_fn: str, symbol: str, tail: int = 15
) -> None:
    """Load HDF5 and print the last *tail* prices for *symbol*."""
    from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
    from functions.GetParams import get_symbols_file

    symbols_file = get_symbols_file(json_fn)
    print(f"\n{'='*70}")
    print(f"CHECK 1 — HDF5 stored quotes for {symbol}")
    print(f"{'='*70}")
    print(f"  symbols_file : {symbols_file}")

    _x, symbols, datearray, quote_df, listname = loadQuotes_fromHDF(
        symbols_file, json_fn
    )

    if symbol not in symbols:
        print(f"  >>> {symbol} is NOT present in the HDF5 symbols list.")
        print(f"  Total symbols in HDF5: {len(symbols)}")
        return

    idx = symbols.index(symbol)
    prices = _x[idx, :]  # Shape: (n_days,)

    print(f"\n  {symbol} found at index {idx} in HDF5  "
          f"({len(datearray)} total dates)")
    print(f"\n  Last {tail} stored prices:")
    print(f"  {'Date':<14}  {'AdjClose':>10}  {'Daily change':>14}")
    print(f"  {'-'*14}  {'-'*10}  {'-'*14}")
    for i in range(max(0, len(datearray) - tail), len(datearray)):
        price = prices[i]
        if i > 0 and prices[i - 1] != 0 and not np.isnan(prices[i - 1]):
            chg = (price / prices[i - 1] - 1.0) * 100.0
            chg_str = f"{chg:+.4f}%"
        else:
            chg_str = "      n/a"
        nan_flag = "  <-- NaN" if np.isnan(price) else ""
        print(f"  {str(datearray[i]):<14}  {price:>10.4f}  "
              f"{chg_str:>14}{nan_flag}")

    # Count trailing constant values (infill indicator).
    n_trailing_const = 0
    for i in range(len(prices) - 1, 0, -1):
        if abs(prices[i] - prices[i - 1]) < 1e-7:
            n_trailing_const += 1
        else:
            break
    print(f"\n  Trailing constant days (infill indicator): {n_trailing_const}")

    # Also check the raw quote DataFrame for the same symbol.
    if symbol in quote_df.columns:
        raw_series = quote_df[symbol]
        n_trailing_nan = int(raw_series.isna()[::-1].cumprod().sum())
        print(f"  Trailing NaN days in raw HDF5 DataFrame : {n_trailing_nan}")
        last_valid_date = raw_series.last_valid_index()
        print(f"  Last valid (non-NaN) date in HDF5       : {last_valid_date}")
    else:
        print(f"  {symbol} not found as a column in raw quote DataFrame.")


##############################################################################
# Check 2 — yfinance live data
##############################################################################

def check_yfinance_quotes(symbol: str) -> None:
    """Download recent yfinance data for *symbol* and summarise."""
    import yfinance as yf

    print(f"\n{'='*70}")
    print(f"CHECK 2 — yfinance live quotes for {symbol}")
    print(f"{'='*70}")

    # Request a generous window: from before the delisting date.
    start = "2025-07-01"
    end = str(datetime.date.today() + datetime.timedelta(days=1))

    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start, end=end, auto_adjust=True)

    if hist.empty:
        print(f"\n  yfinance returned NO data for {symbol} "
              f"(period {start} to {end}).")
        print(f"  This is consistent with {symbol} having been delisted.")
    else:
        print(f"\n  yfinance returned {len(hist)} trading days "
              f"({start} to {end})")
        print(f"\n  Last 15 rows from yfinance:")
        pd.set_option("display.float_format", "{:.4f}".format)
        print(hist.tail(15).to_string())
        last_date = hist.index[-1].date()
        print(f"\n  Last available date : {last_date}")
        if last_date < datetime.date(2025, 8, 9):
            print(f"  >>> Data ends before/on delisting date (2025-08-08). "
                  f"Consistent with delisting.")
        else:
            print(f"  >>> Data extends PAST 2025-08-08 — unexpected for a "
                  f"delisted stock. Investigate further.")

    # Also check fast_info to get any available metadata.
    try:
        fi = ticker.fast_info
        print(f"\n  fast_info.exchange      : {fi.exchange}")
        print(f"  fast_info.quote_type    : {fi.quote_type}")
        print(f"  fast_info.last_price    : {fi.last_price}")
    except Exception as exc:
        print(f"\n  fast_info unavailable: {exc}")


##############################################################################
# Check 3 — active_mask and stock selection
##############################################################################

def check_active_mask_and_selection(json_fn: str, symbol: str) -> None:
    """Run load_quotes_for_analysis and verify *symbol* is masked out."""
    from functions.data_loaders import load_quotes_for_analysis
    from functions.GetParams import get_symbols_file

    symbols_file = get_symbols_file(json_fn)

    print(f"\n{'='*70}")
    print(f"CHECK 3 — active_mask and stock selection logic for {symbol}")
    print(f"{'='*70}")
    print(f"  Loading quotes via load_quotes_for_analysis ...")

    adjClose, symbols, datearray, active_mask = load_quotes_for_analysis(
        symbols_file, json_fn, verbose=True, include_active_mask=True
    )

    if symbol not in symbols:
        print(f"\n  {symbol} is not present in the symbols list at all.")
        return

    idx = symbols.index(symbol)
    is_active_last = bool(active_mask[idx, -1])
    print(f"\n  {symbol} index in symbols list : {idx}")
    print(f"  active_mask[{symbol}, last date] : {is_active_last}")
    print(f"  Last date in datearray           : {datearray[-1]}")

    if is_active_last:
        print(f"\n  >>> BUG CONFIRMED: {symbol} is still marked ACTIVE on the "
              f"last date even though it is not in the current index file.")
        print(f"  >>> It will appear in stock selections until the fix in "
              f"data_loaders.py is applied and quotes are re-run.")
    else:
        print(f"\n  OK: {symbol} is correctly marked INACTIVE on the last date.")
        print(f"  The active_mask fix is working; {symbol} will be excluded "
              f"from new stock selections.")

    # Simulate the Phase 7 selection loop from output_generators.py.
    # monthgainlossweight is not available here, but we can check that
    # the last-date active_mask column properly excludes the symbol.
    active_now = [s for s, a in zip(symbols, active_mask[:, -1]) if a]
    inactive_now = [s for s, a in zip(symbols, active_mask[:, -1]) if not a]

    print(f"\n  Active symbols on last date  : {len(active_now)}")
    print(f"  Inactive symbols on last date: {len(inactive_now)}")

    # Show whether other well-known removed symbols are also flagged.
    candidates = ["PARA", "PARAA", "VIAC", "VIAB", "FB", "TWTR", "WBD"]
    print(f"\n  Status of known-removed symbols:")
    for sym in candidates:
        if sym in symbols:
            i = symbols.index(sym)
            status = "ACTIVE  " if active_mask[i, -1] else "inactive"
            last_price = adjClose[i, -1]
            print(f"    {sym:<8}  {status}  last price={last_price:.4f}")
        else:
            print(f"    {sym:<8}  (not in HDF5)")

    # Check the symbols_file to see if PARA is there.
    print(f"\n  Is {symbol} in {symbols_file}?")
    try:
        with open(symbols_file) as f:
            index_symbols = {line.strip() for line in f if line.strip()}
        if symbol in index_symbols:
            print(f"  >>> YES — {symbol} is still in the index symbols file. "
                  f"Remove it to prevent it from being downloaded and selected.")
        else:
            print(f"  OK: {symbol} is NOT in the current index symbols file.")
    except OSError as exc:
        print(f"  Could not read symbols file: {exc}")


##############################################################################
# Entry point
##############################################################################

def main() -> None:
    args = _parse_args()

    check_hdf5_quotes(args.json_fn, args.symbol, tail=args.tail)
    check_yfinance_quotes(args.symbol)
    check_active_mask_and_selection(args.json_fn, args.symbol)

    print(f"\n{'='*70}")
    print("Diagnosis complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
