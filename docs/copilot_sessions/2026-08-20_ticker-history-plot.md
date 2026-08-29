# Ticker History Plot Session Summary

## Date and Context

Date: 2026-08-20

This session created a study script to visualize daily adjusted-close history for a single ticker alongside the S&P 500 and Nasdaq 100 indices.

## Problem Statement

The goal was to add a lightweight plotting utility in the studies folder that accepts a ticker symbol as an argument, downloads the relevant historical data from yfinance, and generates a PNG chart with the requested moving averages.

## Solution Overview

A new script at [studies/ticker_history_plot.py](../studies/ticker_history_plot.py) downloads adjusted-close data for the requested ticker and the benchmark indexes, aligns the series, computes 50-day and 200-day SMAs, and saves a chart to the studies directory.

## Key Changes

- Added [studies/ticker_history_plot.py](../studies/ticker_history_plot.py)
- Supports ticker arguments such as XQQI, MSFT, and AAPL
- Downloads ^GSPC and ^IXIC alongside the selected ticker
- Plots date on the x-axis and adjusted close on the y-axis
- Draws 50-day and 200-day moving averages

## Technical Details

- Uses yfinance to fetch daily adjusted-close history
- Forces the non-interactive Matplotlib backend to avoid headless-environment issues
- Saves output as PNG, using the ticker symbol in the filename

## Testing

Verified with:

```bash
cd /Users/donaldpg/PyProjects/worktree2/PyTAAA/studies && uv run python ticker_history_plot.py XQQI
```

This produced a chart file successfully and printed the saved path and date range.

## Follow-up Items

- Consider adding command-line options for custom output paths or longer/shorter windows.
- Consider plotting normalized price series if a more direct comparison across assets is desired.
