"""Monte Carlo backtest orchestration for PyTAAA trading strategies.

This module provides the high-level ``run_monte_carlo_backtest`` function
that coordinates parameter generation, single-realization execution, metric
collection, CSV output, and checkpoint saving.

The heavy numerical work is delegated to
``PyTAAA_backtest_sp500_pine_refactored`` so that this runner remains a
thin orchestration layer.
"""

import os
from typing import Any

import numpy as np
from scipy.stats import gmean

from functions.GetParams import get_json_params
from functions.data_loaders import load_quotes_for_analysis
from functions.backtesting.parameter_exploration import generate_random_parameters
from functions.backtesting.output_writers import (
    write_csv_header,
    append_csv_row,
    format_csv_row,
    export_optimized_parameters,
)
from functions.logger_config import get_logger

# NOTE: PyTAAA_backtest_sp500_pine_refactored contains module-level code
# that executes on import.  All symbols are imported lazily inside
# run_monte_carlo_backtest() to avoid triggering that code at import time.

logger = get_logger(__name__, log_file="pytaaa_backtest_montecarlo.log")

#############################################################################
# Trading day constants (mirror TradingConstants from the refactored script)
#############################################################################

_TRADING_DAYS_PER_YEAR = 252


def _calculate_cagr(
    end_value: float, start_value: float, days: int
) -> float:
    """Compute Compound Annual Growth Rate.

    Args:
        end_value: Portfolio value at end of period.
        start_value: Portfolio value at start of period.
        days: Number of trading days in the period.

    Returns:
        CAGR as a decimal (e.g. 0.125 for 12.5 % annual growth), or
        ``0.0`` when inputs are invalid.
    """
    if start_value <= 0 or end_value <= 0 or days <= 0:
        return 0.0
    try:
        return (end_value / start_value) ** (_TRADING_DAYS_PER_YEAR / days) - 1.0
    except (ZeroDivisionError, ValueError, OverflowError):
        return 0.0

#############################################################################
# Symbol-file → runnum + holdMonths mapping (mirrors the original script)
#############################################################################

_SYMBOL_FILE_MAP: dict = {
    "symbols.txt": ("run2501a", [1, 2, 3, 4, 6, 12]),
    "Naz100_Symbols.txt": (
        "run250b",
        [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 6, 12],
    ),
    "biglist.txt": ("run2503", [1, 2, 3, 4, 6, 12]),
    "ProvidentFundSymbols.txt": ("run2504", [4, 6, 12]),
    "sp500_symbols.txt": ("run2505", [1, 2, 3, 4, 6, 12]),
    "cmg_symbols.txt": ("run2507", [3, 4, 6, 12]),
    "SP500_Symbols.txt": ("run2506", [1, 2, 3, 4, 6, 12]),
}
_DEFAULT_HOLD_MONTHS = [1, 2, 3, 4, 6, 12]
_DEFAULT_RUNNUM = "run2508d"

_INDEX_15YR = 3780
_INDEX_10YR = 2520
_INDEX_5YR = 1260
_INDEX_3YR = 756
_INDEX_2YR = 504
_INDEX_1YR = 252


def _sharpe(gains: np.ndarray, n_days: int, window: int) -> float:
    """Compute annualised Sharpe ratio over the last ``window`` trading days.

    Args:
        gains: Array of daily portfolio gain ratios (length n_days).
        n_days: Total number of daily gains available.
        window: Number of trading days for this period.

    Returns:
        Annualised Sharpe ratio, or ``nan`` if insufficient history.
    """
    if n_days < window:
        return float("nan")
    g = gains[-window:]
    std = float(np.std(g))
    if std == 0.0:
        return float("nan")
    return float((gmean(g) ** 252 - 1.0) / (std * np.sqrt(252)))


def _return_ratio(values: np.ndarray, n_days: int, window: int) -> float:
    """Compute annualised return ratio over the last ``window`` days.

    Args:
        values: Portfolio value array.
        n_days: Total number of daily gains (len(values) - 1).
        window: Number of trading days for this period.

    Returns:
        Annualised return ratio, or ``nan`` if insufficient history.
    """
    if n_days < window:
        return float("nan")
    power = {
        _INDEX_15YR: 252.0 / _INDEX_15YR,
        _INDEX_10YR: 1 / 10.0,
        _INDEX_5YR: 1 / 5.0,
        _INDEX_3YR: 1 / 3.0,
        _INDEX_2YR: 1 / 2.0,
        _INDEX_1YR: 1.0,
    }.get(window, 252.0 / window)
    return float((values[-1] / values[-window]) ** power)


def _compute_metrics(
    portfolio_value: np.ndarray,
    bh_value: np.ndarray,
    bh_sharpes: dict,
    bh_returns: dict,
    bh_cagrs: dict,
    bh_drawdowns: dict,
) -> dict:
    """Compute all performance metrics for a single trial's portfolio.

    Args:
        portfolio_value: 1-D array of daily portfolio values.
        bh_value: 1-D array of equal-weighted buy-and-hold portfolio values.
        bh_sharpes: Pre-computed buy-and-hold Sharpe ratios by period key.
        bh_returns: Pre-computed buy-and-hold return ratios by period key.
        bh_cagrs: Pre-computed buy-and-hold CAGR values by period key.
        bh_drawdowns: Pre-computed buy-and-hold drawdowns by period key.

    Returns:
        Metrics dictionary consumed by ``format_csv_row``.
    """
    gains = portfolio_value[1:] / portfolio_value[:-1]
    n_days = len(gains)
    index = min(_INDEX_15YR, len(gains))

    #######################################################################
    # Sharpe ratios
    #######################################################################
    sharpe15 = _sharpe(gains, n_days, index)
    sharpe10 = _sharpe(gains, n_days, _INDEX_10YR)
    sharpe5 = _sharpe(gains, n_days, _INDEX_5YR)
    sharpe3 = _sharpe(gains, n_days, _INDEX_3YR)
    sharpe2 = _sharpe(gains, n_days, _INDEX_2YR)
    sharpe1 = _sharpe(gains, n_days, _INDEX_1YR)

    portfolio_sharpe = _sharpe(gains, n_days, n_days) if n_days > 0 else 0.0
    portfolio_std = float(np.std(gains) * np.sqrt(252))

    #######################################################################
    # Return ratios (annualised)
    #######################################################################
    ret15 = _return_ratio(portfolio_value, n_days, index)
    ret10 = _return_ratio(portfolio_value, n_days, _INDEX_10YR)
    ret5 = _return_ratio(portfolio_value, n_days, _INDEX_5YR)
    ret3 = _return_ratio(portfolio_value, n_days, _INDEX_3YR)
    ret2 = _return_ratio(portfolio_value, n_days, _INDEX_2YR)
    ret1 = _return_ratio(portfolio_value, n_days, _INDEX_1YR)

    #######################################################################
    # CAGR
    #######################################################################
    cagr15 = (
        _calculate_cagr(portfolio_value[-1], portfolio_value[-index], index)
        if n_days >= index else float("nan")
    )
    cagr10 = (
        _calculate_cagr(
            portfolio_value[-1], portfolio_value[-_INDEX_10YR], _INDEX_10YR
        )
        if n_days >= _INDEX_10YR else float("nan")
    )
    cagr5 = (
        _calculate_cagr(
            portfolio_value[-1], portfolio_value[-_INDEX_5YR], _INDEX_5YR
        )
        if n_days >= _INDEX_5YR else float("nan")
    )
    cagr3 = (
        _calculate_cagr(
            portfolio_value[-1], portfolio_value[-_INDEX_3YR], _INDEX_3YR
        )
        if n_days >= _INDEX_3YR else float("nan")
    )
    cagr2 = (
        _calculate_cagr(
            portfolio_value[-1], portfolio_value[-_INDEX_2YR], _INDEX_2YR
        )
        if n_days >= _INDEX_2YR else float("nan")
    )
    cagr1 = (
        _calculate_cagr(
            portfolio_value[-1], portfolio_value[-_INDEX_1YR], _INDEX_1YR
        )
        if n_days >= _INDEX_1YR else float("nan")
    )

    #######################################################################
    # Drawdown (average over period)
    #######################################################################
    max_val = np.maximum.accumulate(portfolio_value)
    drawdown = portfolio_value / max_val - 1.0
    dd15 = float(np.mean(drawdown[-index:])) if n_days >= index else float("nan")
    dd10 = float(np.mean(drawdown[-_INDEX_10YR:])) if n_days >= _INDEX_10YR else float("nan")
    dd5 = float(np.mean(drawdown[-_INDEX_5YR:])) if n_days >= _INDEX_5YR else float("nan")
    dd3 = float(np.mean(drawdown[-_INDEX_3YR:])) if n_days >= _INDEX_3YR else float("nan")
    dd2 = float(np.mean(drawdown[-_INDEX_2YR:])) if n_days >= _INDEX_2YR else float("nan")
    dd1 = float(np.mean(drawdown[-_INDEX_1YR:])) if n_days >= _INDEX_1YR else float("nan")

    #######################################################################
    # Beat buy-and-hold tests
    #######################################################################
    w = 1 / 15.0 + 1 / 10.0 + 1 / 5.0 + 1 / 3.0 + 1 / 2.0 + 1.0
    beat_bh = (
        (sharpe15 - bh_sharpes.get("15", 0.0)) / 15.0
        + (sharpe10 - bh_sharpes.get("10", 0.0)) / 10.0
        + (sharpe5 - bh_sharpes.get("5", 0.0)) / 5.0
        + (sharpe3 - bh_sharpes.get("3", 0.0)) / 3.0
        + (sharpe2 - bh_sharpes.get("2", 0.0)) / 2.0
        + (sharpe1 - bh_sharpes.get("1", 0.0)) / 1.0
    ) / w if w > 0 else 0.0

    beat_bh2 = 0.0
    for ret_port, ret_bh, weight in [
        (ret15, bh_returns.get("15", 0.0), 1.0),
        (ret10, bh_returns.get("10", 0.0), 1.0),
        (ret5, bh_returns.get("5", 0.0), 1.0),
        (ret3, bh_returns.get("3", 0.0), 1.5),
        (ret2, bh_returns.get("2", 0.0), 2.0),
        (ret1, bh_returns.get("1", 0.0), 2.5),
    ]:
        if ret_port > ret_bh:
            beat_bh2 += weight
        if ret_port > 0:
            beat_bh2 += weight
    for dd_port, dd_bh, weight in [
        (dd15, bh_drawdowns.get("15", 0.0), 1.0),
        (dd10, bh_drawdowns.get("10", 0.0), 1.0),
        (dd5, bh_drawdowns.get("5", 0.0), 1.0),
        (dd3, bh_drawdowns.get("3", 0.0), 1.5),
        (dd2, bh_drawdowns.get("2", 0.0), 2.0),
        (dd1, bh_drawdowns.get("1", 0.0), 2.5),
    ]:
        if dd_port > dd_bh:
            beat_bh2 += weight
    beat_bh2 /= 27.0

    return {
        "FinalValue": float(portfolio_value[-1]),
        "PortfolioStd": portfolio_std,
        "PortfolioSharpe": portfolio_sharpe,
        "targetdate": "",
        "AnnGainRecent": "",
        "SharpeRecent": "",
        "BHAnnGainRecent": "",
        "BHSharpeRecent": "",
        "Sharpe15Yr": sharpe15,
        "Sharpe10Yr": sharpe10,
        "Sharpe5Yr": sharpe5,
        "Sharpe3Yr": sharpe3,
        "Sharpe2Yr": sharpe2,
        "Sharpe1Yr": sharpe1,
        "Return15Yr": ret15,
        "Return10Yr": ret10,
        "Return5Yr": ret5,
        "Return3Yr": ret3,
        "Return2Yr": ret2,
        "Return1Yr": ret1,
        "CAGR15Yr": cagr15,
        "CAGR10Yr": cagr10,
        "CAGR5Yr": cagr5,
        "CAGR3Yr": cagr3,
        "CAGR2Yr": cagr2,
        "CAGR1Yr": cagr1,
        "BuyHoldCAGR15Yr": bh_cagrs.get("15", float("nan")),
        "BuyHoldCAGR10Yr": bh_cagrs.get("10", float("nan")),
        "BuyHoldCAGR5Yr": bh_cagrs.get("5", float("nan")),
        "BuyHoldCAGR3Yr": bh_cagrs.get("3", float("nan")),
        "BuyHoldCAGR2Yr": bh_cagrs.get("2", float("nan")),
        "BuyHoldCAGR1Yr": bh_cagrs.get("1", float("nan")),
        "Drawdown15Yr": dd15,
        "Drawdown10Yr": dd10,
        "Drawdown5Yr": dd5,
        "Drawdown3Yr": dd3,
        "Drawdown2Yr": dd2,
        "Drawdown1Yr": dd1,
        "beatBuyHoldTest": beat_bh,
        "beatBuyHoldTest2": beat_bh2,
    }


def _compute_bh_stats(
    value: np.ndarray,
) -> tuple[dict, dict, dict, dict]:
    """Compute buy-and-hold Sharpe, return, CAGR and drawdown metrics.

    Args:
        value: 2-D array of individual stock values (stocks × days).

    Returns:
        Tuple (bh_sharpes, bh_returns, bh_cagrs, bh_drawdowns), each a
        dict keyed by period label (``"15"``, ``"10"``, …, ``"1"``).
    """
    bh_portfolio = np.mean(value, axis=0)
    gains = bh_portfolio[1:] / bh_portfolio[:-1]
    n_days = len(gains)
    index = min(_INDEX_15YR, n_days)

    bh_sharpes = {
        "15": _sharpe(gains, n_days, index),
        "10": _sharpe(gains, n_days, _INDEX_10YR),
        "5": _sharpe(gains, n_days, _INDEX_5YR),
        "3": _sharpe(gains, n_days, _INDEX_3YR),
        "2": _sharpe(gains, n_days, _INDEX_2YR),
        "1": _sharpe(gains, n_days, _INDEX_1YR),
    }
    bh_returns = {
        "15": _return_ratio(bh_portfolio, n_days, index),
        "10": _return_ratio(bh_portfolio, n_days, _INDEX_10YR),
        "5": _return_ratio(bh_portfolio, n_days, _INDEX_5YR),
        "3": _return_ratio(bh_portfolio, n_days, _INDEX_3YR),
        "2": _return_ratio(bh_portfolio, n_days, _INDEX_2YR),
        "1": _return_ratio(bh_portfolio, n_days, _INDEX_1YR),
    }
    bh_cagrs = {
        "15": (
            _calculate_cagr(bh_portfolio[-1], bh_portfolio[-index], index)
            if n_days >= index else float("nan")
        ),
        "10": (
            _calculate_cagr(
                bh_portfolio[-1], bh_portfolio[-_INDEX_10YR], _INDEX_10YR
            )
            if n_days >= _INDEX_10YR else float("nan")
        ),
        "5": (
            _calculate_cagr(
                bh_portfolio[-1], bh_portfolio[-_INDEX_5YR], _INDEX_5YR
            )
            if n_days >= _INDEX_5YR else float("nan")
        ),
        "3": (
            _calculate_cagr(
                bh_portfolio[-1], bh_portfolio[-_INDEX_3YR], _INDEX_3YR
            )
            if n_days >= _INDEX_3YR else float("nan")
        ),
        "2": (
            _calculate_cagr(
                bh_portfolio[-1], bh_portfolio[-_INDEX_2YR], _INDEX_2YR
            )
            if n_days >= _INDEX_2YR else float("nan")
        ),
        "1": (
            _calculate_cagr(
                bh_portfolio[-1], bh_portfolio[-_INDEX_1YR], _INDEX_1YR
            )
            if n_days >= _INDEX_1YR else float("nan")
        ),
    }
    max_val = np.maximum.accumulate(bh_portfolio)
    dd = bh_portfolio / max_val - 1.0
    bh_drawdowns = {
        "15": float(np.mean(dd[-index:])) if n_days >= index else float("nan"),
        "10": float(np.mean(dd[-_INDEX_10YR:])) if n_days >= _INDEX_10YR else float("nan"),
        "5": float(np.mean(dd[-_INDEX_5YR:])) if n_days >= _INDEX_5YR else float("nan"),
        "3": float(np.mean(dd[-_INDEX_3YR:])) if n_days >= _INDEX_3YR else float("nan"),
        "2": float(np.mean(dd[-_INDEX_2YR:])) if n_days >= _INDEX_2YR else float("nan"),
        "1": float(np.mean(dd[-_INDEX_1YR:])) if n_days >= _INDEX_1YR else float("nan"),
    }
    return bh_sharpes, bh_returns, bh_cagrs, bh_drawdowns


def run_monte_carlo_backtest(
    json_fn: str,
    n_trials: int,
    output_paths: dict,
) -> dict:
    """Run the full Monte Carlo backtest loop and write results to CSV.

    Args:
        json_fn: Path to the JSON configuration file.
        n_trials: Number of Monte Carlo trials to run.
        output_paths: Dict with keys:
            - ``model_id``: Model identifier string.
            - ``outfiledir``: Directory for CSV and checkpoint files.
            - ``outfilename``: Full path to the output CSV file.
            - ``date_str``: Date string for filenames.
            - ``runnum``: Run identifier string.

    Returns:
        Summary dict with keys:
            - ``best_trial``: Index of the trial with the highest Sharpe.
            - ``best_sharpe``: Best Sharpe ratio found.
            - ``best_final_value``: Portfolio final value at best trial.
            - ``total_trials``: Total number of trials executed.
            - ``output_file``: Path to the CSV output file.

    Raises:
        FileNotFoundError: If the symbols file does not exist.
        ValueError: If quote data cannot be loaded.
    """
    #######################################################################
    # Lazy import: avoid module-level code in the refactored backtest
    # script running at import time.
    #######################################################################
    from PyTAAA_backtest_sp500_pine_refactored import (  # noqa: PLC0415
        run_single_monte_carlo_realization,
    )

    #######################################################################
    # Load configuration and quote data
    #######################################################################
    params = get_json_params(json_fn)
    symbols_file = params["symbols_file"]

    adjClose, symbols, datearray = load_quotes_for_analysis(
        symbols_file, json_fn, verbose=True
    )

    # Build gainloss, value and activeCount arrays (mirrors original script)
    gainloss = np.ones(adjClose.shape, dtype=float)
    gainloss[:, 1:] = adjClose[:, 1:] / adjClose[:, :-1]
    gainloss[np.isnan(gainloss)] = 1.0

    value = 10000.0 * np.cumprod(gainloss, axis=1)

    activeCount = np.zeros(adjClose.shape[1], dtype=float)
    for ii in range(adjClose.shape[0]):
        idx = np.argmax(
            np.clip(np.abs(gainloss[ii, :] - 1), 0, 1e-8)
        ) - 1
        activeCount[idx + 1:] += 1

    #######################################################################
    # Determine holdMonths from symbol-file basename
    #######################################################################
    basename = os.path.basename(symbols_file)
    runnum, hold_months = _SYMBOL_FILE_MAP.get(
        basename, (_DEFAULT_RUNNUM, _DEFAULT_HOLD_MONTHS)
    )
    # Prefer runnum from output_paths if provided
    runnum = output_paths.get("runnum", runnum)

    outfilename = output_paths["outfilename"]
    outfiledir = output_paths["outfiledir"]
    model_id = output_paths["model_id"]
    date_str = output_paths["date_str"]

    #######################################################################
    # Write CSV header and pre-compute buy-and-hold statistics
    #######################################################################
    write_csv_header(outfilename)

    bh_sharpes, bh_returns, bh_cagrs, bh_drawdowns = _compute_bh_stats(value)

    #######################################################################
    # Arrays to track best results
    #######################################################################
    all_sharpes = np.zeros(n_trials + 1, dtype=float)
    all_final_values = np.zeros(n_trials + 1, dtype=float)

    best_params: dict = {}
    best_sharpe = -np.inf
    best_iter = 0

    #######################################################################
    # Main Monte Carlo loop
    #######################################################################
    for iter_num in range(n_trials + 1):
        progress_pct = (iter_num / n_trials) * 100 if n_trials > 0 else 100.0
        if iter_num % max(1, n_trials // 20) == 0 or iter_num == 0:
            print(
                f"\nMonte Carlo Progress: Trial {iter_num}/{n_trials}"
                f" ({progress_pct:.1f}%)"
            )
            if iter_num > 0:
                print(
                    f"   Best Sharpe so far: "
                    f"{best_sharpe:.4f} (trial #{best_iter})"
                )

        ###################################################################
        # Generate parameters for this trial
        ###################################################################
        trial_params = generate_random_parameters(
            hold_months, iter_num, n_trials
        )

        ###################################################################
        # Run single realization
        ###################################################################
        try:
            results = run_single_monte_carlo_realization(
                json_fn,
                trial_params,
                iter_num,
                adjClose,
                symbols,
                datearray,
                gainloss,
                value,
                activeCount,
                hold_months,
                verbose=(iter_num <= 2),
            )
        except Exception as exc:
            logger.warning(
                "Trial %d failed: %s — skipping.", iter_num, exc
            )
            continue

        month_value = results.get("monthvalue", value)
        portfolio_value = np.average(month_value, axis=0)

        ###################################################################
        # Compute metrics
        ###################################################################
        metrics = _compute_metrics(
            portfolio_value,
            value,
            bh_sharpes,
            bh_returns,
            bh_cagrs,
            bh_drawdowns,
        )

        trial_sharpe = metrics["PortfolioSharpe"]
        trial_final = metrics["FinalValue"]

        all_sharpes[iter_num] = trial_sharpe
        all_final_values[iter_num] = trial_final

        if trial_sharpe > best_sharpe:
            best_sharpe = trial_sharpe
            best_iter = iter_num
            best_params = dict(trial_params)

        ###################################################################
        # Write CSV row
        ###################################################################
        row = format_csv_row(runnum, iter_num, trial_params, metrics)
        append_csv_row(outfilename, row)

        ###################################################################
        # Save checkpoint every 10 % of trials
        ###################################################################
        save_checkpoint = (
            iter_num > 0
            and iter_num % max(1, n_trials // 10) == 0
        ) or iter_num == n_trials - 1
        if save_checkpoint:
            try:
                ckpt_file = os.path.join(
                    outfiledir,
                    f"montecarlo_checkpoint_{iter_num}.npz",
                )
                np.savez_compressed(
                    ckpt_file,
                    PortfolioSharpe=all_sharpes[: iter_num + 1],
                    FinalTradedPortfolioValue=all_final_values[: iter_num + 1],
                    iteration=iter_num,
                )
                print(f"   Checkpoint saved: {ckpt_file}")
            except Exception as exc:
                logger.warning("Failed to save checkpoint: %s", exc)

    #######################################################################
    # Export optimized parameters
    #######################################################################
    if best_params:
        export_optimized_parameters(
            json_fn, best_params, outfiledir, model_id, date_str
        )

    logger.info(
        "Monte Carlo complete: best_sharpe=%.4f at trial %d",
        best_sharpe, best_iter,
    )

    return {
        "best_trial": best_iter,
        "best_sharpe": best_sharpe,
        "best_final_value": float(all_final_values[best_iter]),
        "total_trials": n_trials,
        "output_file": outfilename,
    }
