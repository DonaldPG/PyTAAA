"""Margin debt and portfolio overlay plotting utilities for PyTAAA."""

from __future__ import annotations

from functools import lru_cache
from io import StringIO
import json
import os

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests

from functions.GetParams import get_performance_store, get_webpage_store

if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


FRED_CSV_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_MARGIN_SERIES = "BOGZ1FL663067003Q"
FRED_GDP_SERIES = "GDP"
FINRA_MARGIN_CSV_URL = "https://www.finra.org/sites/default/files/MarginDebt.csv"
FINRA_MARGIN_XLSX_URL = "https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx"
FINRA_MARGIN_COLUMN = "Debit Balances in Customers' Securities Margin Accounts"

DEFAULT_SMA_MONTHS = 60
DEFAULT_RULE = "above_sma_to_cash"
DEFAULT_SMA_FACTOR = 1.0
EXTREMA_INTERVAL_PCT = 0.55
GDP_BILLIONS_TO_MILLIONS = 1000.0
EXTREMA_WINDOW_START_DAYS = 200
EXTREMA_WINDOW_END_DAYS = 2000
EXTREMA_WINDOW_GROWTH_FACTOR = 1.061991885
EXTREMA_WINDOW_COUNT = 7
EXTREMA_WINDOWS_OVERRIDE: tuple[int, ...] | None = (200, 278, 394, 570, 845, 1283, 2000)


def _build_extrema_windows() -> tuple[int, ...]:
    """Create geometric long-horizon windows for rolling extrema."""

    if EXTREMA_WINDOWS_OVERRIDE is not None:
        if len(EXTREMA_WINDOWS_OVERRIDE) != EXTREMA_WINDOW_COUNT:
            raise ValueError("EXTREMA_WINDOWS_OVERRIDE length must equal EXTREMA_WINDOW_COUNT")
        return tuple(int(value) for value in EXTREMA_WINDOWS_OVERRIDE)

    windows = [EXTREMA_WINDOW_START_DAYS]
    for _ in range(1, EXTREMA_WINDOW_COUNT):
        next_window = int(round(windows[-1] ** EXTREMA_WINDOW_GROWTH_FACTOR))
        windows.append(next_window)

    windows[0] = EXTREMA_WINDOW_START_DAYS
    windows[-1] = EXTREMA_WINDOW_END_DAYS

    # Ensure strictly increasing integer windows for rolling operations.
    for idx in range(1, len(windows)):
        if windows[idx] <= windows[idx - 1]:
            windows[idx] = windows[idx - 1] + 1

    return tuple(windows)


EXTREMA_WINDOWS = _build_extrema_windows()
EXTREMA_WINDOWS_LABEL = "/".join(str(window) for window in EXTREMA_WINDOWS)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIGNAL_PARAMS_PATH = os.path.join(REPO_ROOT, "studies", "margin_debt_signal_params.json")


@lru_cache(maxsize=16)
def _fred_series(series_id: str) -> pd.Series:
    """Fetch a public FRED time series as a pandas Series."""

    try:
        response = requests.get(
            FRED_CSV_BASE_URL,
            params={"id": series_id},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException:
        return pd.Series(dtype=float, name=series_id)

    frame = pd.read_csv(StringIO(response.text))
    if frame.shape[1] < 2:
        return pd.Series(dtype=float, name=series_id)

    date_col = frame.columns[0]
    value_col = frame.columns[1]
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    values = pd.to_numeric(frame[value_col], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates).dropna().sort_index()
    series.name = series_id
    return series


def _finra_margin_series() -> pd.Series:
    """Load FINRA margin debt as a monthly series in USD millions."""

    frame = None

    # Preferred source: FINRA xlsx workbook.
    try:
        frame = pd.read_excel(FINRA_MARGIN_XLSX_URL)
    except Exception:
        frame = None

    # Legacy fallback in case FINRA changes the workbook endpoint.
    if frame is None:
        try:
            response = requests.get(FINRA_MARGIN_CSV_URL, timeout=30)
            response.raise_for_status()
            frame = pd.read_csv(StringIO(response.text))
        except (requests.RequestException, pd.errors.ParserError):
            return pd.Series(dtype=float, name="MarginDebt_FINRA")

    if frame is None or frame.empty:
        return pd.Series(dtype=float, name="MarginDebt_FINRA")

    date_col = None
    for col in frame.columns:
        if str(col).strip().lower() in ("date", "month", "month ending", "year-month"):
            date_col = col
            break
    if date_col is None:
        date_col = frame.columns[0]

    value_col = None
    if FINRA_MARGIN_COLUMN in frame.columns:
        value_col = FINRA_MARGIN_COLUMN

    for col in frame.columns:
        if value_col is not None:
            break
        lowered = str(col).strip().lower()
        if "debit" in lowered and "margin" in lowered:
            value_col = col
            break
        if "margin" in lowered and "debt" in lowered:
            value_col = col
            break

    if value_col is None:
        numeric_cols = [
            col
            for col in frame.columns
            if col != date_col and pd.to_numeric(frame[col], errors="coerce").notna().any()
        ]
        if not numeric_cols:
            return pd.Series(dtype=float, name="MarginDebt_FINRA")
        value_col = numeric_cols[0]

    dates = pd.to_datetime(frame[date_col], errors="coerce")
    values = pd.to_numeric(frame[value_col], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates).dropna().sort_index()
    series.name = "MarginDebt_FINRA"
    return series


def _monthly_series(series: pd.Series) -> pd.Series:
    """Convert a time series to month-end frequency with forward fill."""

    if series.empty:
        return series
    monthly = series.sort_index().resample("ME").ffill().dropna()
    return monthly


def _gdp_monthly_with_extrapolation(
    gdp_series: pd.Series,
    *,
    target_end: pd.Timestamp,
) -> pd.Series:
    """Convert quarterly GDP to month-end and extrapolate to target end.

    The GDP series is interpreted at quarter-end dates. If FRED stamps
    quarters at the first day (for example 2026-01-01 for Q1 2026),
    values are shifted to the corresponding quarter-end date.
    Missing months after the latest known GDP value are filled by a
    quadratic fit over the last two years of quarterly GDP observations.
    """

    if gdp_series.empty:
        return gdp_series

    gdp = gdp_series.dropna().sort_index().copy()
    if gdp.empty:
        return pd.Series(dtype=float, name="GDP_USD")

    # Interpret quarterly GDP timestamps as quarter-end, not quarter-start.
    gdp.index = pd.to_datetime(gdp.index) + pd.offsets.QuarterEnd(0)
    gdp = gdp[~gdp.index.duplicated(keep="last")].sort_index()

    monthly = _monthly_series(gdp)
    monthly.name = "GDP_USD"

    if monthly.empty:
        return monthly

    target_end = pd.Timestamp(target_end)
    if monthly.index.max() >= target_end:
        return monthly

    fit_end = gdp.index.max()
    fit_start = fit_end - pd.DateOffset(years=2)
    fit_data = gdp.loc[gdp.index >= fit_start].dropna()
    if fit_data.shape[0] < 3:
        fit_data = gdp.dropna()

    future_idx = pd.date_range(
        monthly.index.max() + pd.offsets.MonthEnd(1),
        target_end,
        freq="ME",
    )
    if len(future_idx) == 0:
        return monthly

    if fit_data.shape[0] >= 3:
        origin = fit_data.index[0]
        x_fit = (fit_data.index - origin).days.to_numpy(dtype=float) / 30.4375
        y_fit = fit_data.to_numpy(dtype=float)
        coeffs = np.polyfit(x_fit, y_fit, deg=2)
        x_future = (future_idx - origin).days.to_numpy(dtype=float) / 30.4375
        y_future = np.polyval(coeffs, x_future)
    elif fit_data.shape[0] == 2:
        origin = fit_data.index[0]
        x_fit = (fit_data.index - origin).days.to_numpy(dtype=float) / 30.4375
        y_fit = fit_data.to_numpy(dtype=float)
        coeffs = np.polyfit(x_fit, y_fit, deg=1)
        x_future = (future_idx - origin).days.to_numpy(dtype=float) / 30.4375
        y_future = np.polyval(coeffs, x_future)
    else:
        y_future = np.full(len(future_idx), float(monthly.iloc[-1]))

    y_future = np.clip(y_future, 1.0, None)
    extrapolated = pd.Series(y_future, index=future_idx, name="GDP_USD")
    monthly = pd.concat([monthly, extrapolated])
    monthly = monthly[~monthly.index.duplicated(keep="last")].sort_index()
    return monthly


def _load_backtest_curves(json_fn: str) -> tuple[pd.Series, pd.Series]:
    """Load buy-and-hold and traded curves from the monthly backtest file."""

    performance_store = get_performance_store(json_fn)
    params_path = os.path.join(
        performance_store,
        "pyTAAAweb_backtestPortfolioValue.params",
    )
    if not os.path.isfile(params_path):
        return (
            pd.Series(dtype=float, name="Buy & Hold"),
            pd.Series(dtype=float, name="Trading System"),
        )

    dates = []
    buy_hold = []
    traded = []
    with open(params_path, "r") as handle:
        lines = handle.read().split("\n")

    for line in lines:
        parts = [item for item in line.split(" ") if item]
        if len(parts) < 3:
            continue
        try:
            dates.append(pd.to_datetime(parts[0]))
            buy_hold.append(float(parts[1]))
            traded.append(float(parts[2]))
        except (ValueError, TypeError):
            continue

    if not dates:
        return (
            pd.Series(dtype=float, name="Buy & Hold"),
            pd.Series(dtype=float, name="Trading System"),
        )

    index = pd.to_datetime(dates)
    buy_hold_series = pd.Series(buy_hold, index=index, name="Buy & Hold")
    traded_series = pd.Series(traded, index=index, name="Trading System")
    buy_hold_series = buy_hold_series.sort_index().dropna()
    traded_series = traded_series.sort_index().dropna()
    return buy_hold_series, traded_series


def _build_margin_gdp_frame() -> pd.DataFrame:
    """Build merged monthly frame with margin debt to GDP metrics."""

    finra_margin = _monthly_series(_finra_margin_series())
    fred_margin = _monthly_series(_fred_series(FRED_MARGIN_SERIES))

    margin_target = pd.concat([finra_margin, fred_margin], axis=1).dropna(how="all")
    if margin_target.empty:
        gdp_monthly = _monthly_series(_fred_series(FRED_GDP_SERIES))
    else:
        gdp_monthly = _gdp_monthly_with_extrapolation(
            _fred_series(FRED_GDP_SERIES),
            target_end=margin_target.index.max(),
        )

    # FINRA margin debt is reported in millions of USD, while FRED GDP is in
    # billions of USD. Convert GDP to millions so the ratio is dimensionless
    # and the plotted percentage reflects the true scale.
    gdp_monthly = gdp_monthly * GDP_BILLIONS_TO_MILLIONS

    merged = pd.concat(
        [
            finra_margin.rename("MarginDebt_FINRA"),
            fred_margin.rename("MarginDebt_FRED"),
            gdp_monthly.rename("GDP_USD"),
        ],
        axis=1,
        join="outer",
    ).sort_index()

    merged["MarginDebt_USD"] = merged["MarginDebt_FINRA"].combine_first(
        merged["MarginDebt_FRED"]
    )
    merged["MarginDebt_GDP"] = merged["MarginDebt_USD"] / merged["GDP_USD"]
    merged["Low10y"] = merged["MarginDebt_GDP"].rolling(window=120, min_periods=1).min()
    merged["Normalized"] = merged["MarginDebt_GDP"] / merged["Low10y"].replace(0.0, np.nan)
    return merged


def _load_signal_params() -> tuple[int, str, float]:
    """Load preferred signal parameters from studies output file."""

    sma_months = DEFAULT_SMA_MONTHS
    rule = DEFAULT_RULE
    sma_factor = DEFAULT_SMA_FACTOR
    try:
        with open(SIGNAL_PARAMS_PATH, "r") as handle:
            payload = json.load(handle)
        sma_months = int(payload.get("sma_months", DEFAULT_SMA_MONTHS))
        rule = str(payload.get("rule", DEFAULT_RULE))
        sma_factor = float(payload.get("sma_factor", DEFAULT_SMA_FACTOR))
    except Exception:
        pass

    sma_months = max(3, sma_months)
    if rule not in ("above_sma_to_cash", "above_sma_to_long"):
        rule = DEFAULT_RULE
    sma_factor = min(1.0, max(0.5, sma_factor))
    return sma_months, rule, sma_factor


def _compute_margin_sma_signal(
    frame: pd.DataFrame,
    *,
    sma_months: int,
    rule: str,
    sma_factor: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute ratio, SMA, and long/cash signal from margin-debt-to-GDP."""

    ratio = frame["MarginDebt_GDP"].astype(float)
    sma = ratio.rolling(window=sma_months, min_periods=3).mean()
    signal_line = sma * sma_factor

    if rule == "above_sma_to_long":
        is_long = ratio >= signal_line
    else:
        is_long = ratio <= signal_line

    is_long = is_long.fillna(False)
    signal = is_long.astype(float)
    ratio.name = "MarginDebt_GDP"
    signal_line.name = "MarginDebt_GDP_SMA_Adjusted"
    signal.name = "Signal_Long"
    return ratio, signal_line, signal


def _simulate_signal_curve(
    base_curve: pd.Series,
    monthly_signal: pd.Series,
) -> pd.Series:
    """Apply monthly long/cash signal to a base curve with 1-step delay."""

    if base_curve.empty:
        return pd.Series(dtype=float, name="Margin Debt SMA Signal")

    series = base_curve.dropna().sort_index().astype(float)
    if series.empty:
        return pd.Series(dtype=float, name="Margin Debt SMA Signal")

    signal = monthly_signal.reindex(series.index, method="ffill").fillna(0.0).astype(float)

    gain = (series / series.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    strategy = np.zeros(len(series), dtype=float)
    strategy[0] = float(series.iloc[0])

    for idx in range(1, len(series)):
        if signal.iloc[idx - 1] >= 0.5:
            strategy[idx] = strategy[idx - 1] * float(gain.iloc[idx])
        else:
            strategy[idx] = strategy[idx - 1]

    return pd.Series(strategy, index=series.index, name="Margin Debt SMA Signal")


def _daily_ratio_inputs(
    base_curve: pd.Series,
    ratio: pd.Series,
    ratio_sma: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Project monthly ratio inputs onto the daily backtest index."""

    if base_curve.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    daily_index = base_curve.dropna().sort_index().index
    ratio_daily = ratio.reindex(daily_index, method="ffill")
    ratio_sma_daily = ratio_sma.reindex(daily_index, method="ffill")
    return ratio_daily.astype(float), ratio_sma_daily.astype(float)


def _rolling_extrema_average(
    ratio_daily: pd.Series,
    windows: tuple[int, ...] = EXTREMA_WINDOWS,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average multiple long-horizon rolling minima and maxima curves."""

    if ratio_daily.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    min_curves = []
    max_curves = []
    for window in windows:
        min_curves.append(ratio_daily.rolling(window=window, min_periods=window).min())
        max_curves.append(ratio_daily.rolling(window=window, min_periods=window).max())

    min_frame = pd.concat(min_curves, axis=1)
    max_frame = pd.concat(max_curves, axis=1)
    # Require the full longest-history window before the averaged extrema exist.
    # This preserves the intended behavior: remain fully invested until the
    # longest configured rolling min/max can both be computed.
    avg_min = min_frame.mean(axis=1, skipna=False)
    avg_max = max_frame.mean(axis=1, skipna=False)
    pct_label = int(round(EXTREMA_INTERVAL_PCT * 100.0))
    # Reference line at EXTREMA_INTERVAL_PCT of the interval above moving minima.
    # line_pct = min + EXTREMA_INTERVAL_PCT * (max - min)
    line_40 = avg_min + EXTREMA_INTERVAL_PCT * (avg_max - avg_min)
    avg_min.name = "Moving Min Average"
    avg_max.name = "Moving Max Average"
    line_40.name = f"Moving Extrema {pct_label}pct"
    return avg_min, avg_max, line_40


def _simulate_extrema_reentry_signal(
    base_curve: pd.Series,
    ratio_daily: pd.Series,
    sma_daily: pd.Series,
    extrema_line_daily: pd.Series,
) -> pd.Series:
    """Backtest the long-term extrema re-entry experiment on the PyTAAA curve.

    The intent is asymmetric:
    - SELL when margin debt/GDP drops from above its short SMA trigger.
        - BUY when it later drops from above the long-horizon extrema threshold line.
        - Stay fully invested any time the ratio remains below that threshold line.
        - Also stay/go fully invested when the same line is uptrending for
          3 consecutive days.

    The BUY uses a downward cross intentionally. The rationale is that the
    ratio is no longer anomalously high once it falls back below the long-term
    extrema threshold line.
    """

    if base_curve.empty:
        return pd.Series(dtype=float, name="Margin Debt Extrema Signal")

    series = base_curve.dropna().sort_index().astype(float)
    if series.empty:
        return pd.Series(dtype=float, name="Margin Debt Extrema Signal")

    ratio_view = ratio_daily.reindex(series.index, method="ffill")
    sma_view = sma_daily.reindex(series.index, method="ffill")
    line_view = extrema_line_daily.reindex(series.index, method="ffill")
    gain = (series / series.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    strategy = np.zeros(len(series), dtype=float)
    invested = np.ones(len(series), dtype=bool)
    strategy[0] = float(series.iloc[0])

    for idx in range(1, len(series)):
        prev_ratio = ratio_view.iloc[idx - 1]
        curr_ratio = ratio_view.iloc[idx]
        prev_sma = sma_view.iloc[idx - 1]
        curr_sma = sma_view.iloc[idx]
        prev_line = line_view.iloc[idx - 1]
        curr_line = line_view.iloc[idx]
        prev2_line = line_view.iloc[idx - 2] if idx >= 2 else np.nan

        if pd.isna(curr_line) or pd.isna(prev_line):
            # Before the longest extrema window exists, stay fully invested.
            invested[idx] = True
        else:
            invested[idx] = invested[idx - 1]

            below_line = (
                pd.notna(curr_ratio)
                and pd.notna(curr_line)
                and curr_ratio <= curr_line
            )
            # Treat "uptrending for at least 3 days" robustly for step-like
            # series (flat most days with occasional upward moves): require a
            # non-decreasing 3-day path and a net increase over the window.
            line_up_3days = (
                pd.notna(prev2_line)
                and pd.notna(prev_line)
                and pd.notna(curr_line)
                and prev2_line <= prev_line <= curr_line
                and curr_line > prev2_line
            )
            sell_cross = (
                pd.notna(prev_ratio)
                and pd.notna(curr_ratio)
                and pd.notna(prev_sma)
                and pd.notna(curr_sma)
                and prev_ratio > prev_sma
                and curr_ratio <= curr_sma
            )
            buy_cross = (
                pd.notna(prev_ratio)
                and pd.notna(curr_ratio)
                and prev_ratio > prev_line
                and curr_ratio <= curr_line
            )

            if below_line or line_up_3days:
                # Once margin debt is back below the configured extrema line it is
                # no longer anomalously high, so remain fully invested.
                # Also force full investment when that line itself trends
                # upward for 3 consecutive days.
                invested[idx] = True
            elif sell_cross:
                invested[idx] = False
            elif buy_cross:
                invested[idx] = True

        if invested[idx - 1]:
            strategy[idx] = strategy[idx - 1] * float(gain.iloc[idx])
        else:
            strategy[idx] = strategy[idx - 1]

    return pd.Series(strategy, index=series.index, name="Margin Debt Extrema Signal")


def make_margin_debt_plot(json_fn: str) -> str:
    """Build the margin debt and portfolio chart and return HTML snippet."""

    webpage_dir = get_webpage_store(json_fn)
    figure_path = os.path.join(webpage_dir, "PyTAAA_marginDebt.png")

    buy_hold_curve, traded_curve = _load_backtest_curves(json_fn)
    bottom = _build_margin_gdp_frame()
    sma_months, rule, sma_factor = _load_signal_params()
    pct_label = int(round(EXTREMA_INTERVAL_PCT * 100.0))

    ratio = pd.Series(dtype=float)
    ratio_sma = pd.Series(dtype=float)
    signal_monthly = pd.Series(dtype=float)
    signal_curve = pd.Series(dtype=float)
    extrema_curve = pd.Series(dtype=float)
    ratio_daily = pd.Series(dtype=float)
    ratio_sma_daily = pd.Series(dtype=float)
    extrema_min_daily = pd.Series(dtype=float)
    extrema_max_daily = pd.Series(dtype=float)
    extrema_line_daily = pd.Series(dtype=float)
    if not bottom.empty:
        ratio, ratio_sma, signal_monthly = _compute_margin_sma_signal(
            bottom,
            sma_months=sma_months,
            rule=rule,
            sma_factor=sma_factor,
        )
        if not traded_curve.empty and not signal_monthly.empty:
            signal_curve = _simulate_signal_curve(traded_curve, signal_monthly)
        elif not buy_hold_curve.empty and not signal_monthly.empty:
            signal_curve = _simulate_signal_curve(buy_hold_curve, signal_monthly)

        extrema_base_curve = traded_curve if not traded_curve.empty else buy_hold_curve
        if not extrema_base_curve.empty:
            ratio_daily, ratio_sma_daily = _daily_ratio_inputs(
                extrema_base_curve,
                ratio,
                ratio_sma,
            )
            extrema_min_daily, extrema_max_daily, extrema_line_daily = _rolling_extrema_average(
                ratio_daily,
            )
            extrema_curve = _simulate_extrema_reentry_signal(
                extrema_base_curve,
                ratio_daily,
                ratio_sma_daily,
                extrema_line_daily,
            )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(9, 7),
        dpi=150,
        gridspec_kw={"height_ratios": [1, 1]},
        sharex=True,
    )
    ax_top = axes[0]
    ax_bottom = axes[1]

    x_start = None
    x_end = None

    if not buy_hold_curve.empty or not traded_curve.empty:
        top_start = None
        top_end = None
        if not buy_hold_curve.empty:
            top_start = buy_hold_curve.index[0]
            top_end = buy_hold_curve.index[-1]
        if not traded_curve.empty:
            if top_start is None:
                top_start = traded_curve.index[0]
                top_end = traded_curve.index[-1]
            else:
                top_start = min(top_start, traded_curve.index[0])
                top_end = max(top_end, traded_curve.index[-1])

        if not buy_hold_curve.empty:
            ax_top.plot(
                buy_hold_curve.index,
                buy_hold_curve.values,
                "r-",
                lw=1.25,
                label="Equal-Weight Index",
            )
        if not traded_curve.empty:
            ax_top.plot(
                traded_curve.index,
                traded_curve.values,
                "k-",
                lw=1.25,
                label="PyTAAA Portfolio",
            )
        if not signal_curve.empty:
            ax_top.plot(
                signal_curve.index,
                signal_curve.values,
                color="dodgerblue",
                lw=1.15,
                label="Margin Debt SMA Signal",
            )
        if not extrema_curve.empty:
            ax_top.plot(
                extrema_curve.index,
                extrema_curve.values,
                color="magenta",
                lw=1.1,
                label="Margin Debt Extrema Signal",
            )

        ax_top.set_yscale("log")
        ax_top.grid(True, alpha=0.3)
        ax_top.set_title("Monthly backtest series with Margin Debt signal")
        ax_top.set_ylabel("Value")
        ax_top.legend(loc="upper left")
        if top_start is not None and top_end is not None:
            x_start = top_start
            x_end = top_end

        latest_top_dates = []
        if not buy_hold_curve.empty:
            latest_top_dates.append(buy_hold_curve.index[-1])
        if not traded_curve.empty:
            latest_top_dates.append(traded_curve.index[-1])
        if latest_top_dates:
            ax_top.text(
                0.02,
                0.06,
                "Most recent data: "
                + pd.Timestamp(max(latest_top_dates)).strftime("%Y-%m-%d"),
                fontsize=8,
                transform=ax_top.transAxes,
                verticalalignment="bottom",
                bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
            )
        rule_text = "Above SMA -> Cash" if rule == "above_sma_to_cash" else "Above SMA -> Long"
        ax_top.text(
            0.02,
            0.16,
            f"Signal rule: {rule_text}, SMA={sma_months} months, factor={sma_factor:.3f}",
            fontsize=8,
            transform=ax_top.transAxes,
            verticalalignment="bottom",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
        )
        if not extrema_curve.empty:
            ax_top.text(
                0.02,
                0.26,
                f"Extrema rule: sell on drop below SMA, re-buy below {pct_label}% line",
                fontsize=8,
                transform=ax_top.transAxes,
                verticalalignment="bottom",
                bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
            )
    else:
        ax_top.text(0.5, 0.5, "No monthly backtest data available", ha="center", va="center")
        ax_top.grid(True, alpha=0.3)
        ax_top.set_title("Monthly backtest series")
        ax_top.set_ylabel("Value")

    if not bottom.empty:
        frame = bottom.dropna(subset=["MarginDebt_GDP"], how="all")
        if not frame.empty:
            if x_start is not None and x_end is not None:
                frame = frame.loc[(frame.index >= x_start) & (frame.index <= x_end)]

            ratio_view = ratio.reindex(frame.index)
            sma_view = ratio_sma.reindex(frame.index)
            ratio_pct = ratio_view * 100.0
            sma_pct = sma_view * 100.0

            ax_bottom.plot(
                frame.index,
                ratio_pct.values,
                color="steelblue",
                lw=1.5,
                label="Margin Debt / GDP (%)",
            )
            ax_bottom.plot(
                frame.index,
                sma_pct.values,
                color="darkred",
                lw=1.0,
                linestyle="--",
                label=f"SMA x factor ({sma_months}m, x{sma_factor:.3f})",
            )
            if not extrema_min_daily.empty:
                extrema_min_view = (extrema_min_daily.reindex(frame.index, method="ffill") * 100.0)
                extrema_max_view = (extrema_max_daily.reindex(frame.index, method="ffill") * 100.0)
                extrema_line_view = (extrema_line_daily.reindex(frame.index, method="ffill") * 100.0)
                ax_bottom.plot(
                    frame.index,
                    extrema_min_view.values,
                    color="orange",
                    lw=1.0,
                    linestyle=":",
                    label=f"Avg moving min ({EXTREMA_WINDOWS_LABEL}d)",
                )
                ax_bottom.plot(
                    frame.index,
                    extrema_max_view.values,
                    color="forestgreen",
                    lw=1.0,
                    linestyle=":",
                    label=f"Avg moving max ({EXTREMA_WINDOWS_LABEL}d)",
                )
                ax_bottom.plot(
                    frame.index,
                    extrema_line_view.values,
                    color="magenta",
                    lw=1.0,
                    linestyle="-.",
                    label=f"Extrema {pct_label}% line",
                )

            ax_bottom.legend(loc="upper left", fontsize=8)
            ax_bottom.set_title("Margin Debt / GDP Signal Input and Trend")
            ax_bottom.set_xlabel("Date")
            ax_bottom.set_ylabel("Percent (%)")
            ax_bottom.grid(True, alpha=0.3)

            rule_text = "Above SMA -> Cash" if rule == "above_sma_to_cash" else "Above SMA -> Long"
            ax_bottom.text(
                0.02,
                0.13,
                f"Signal rule: {rule_text}, factor={sma_factor:.3f}",
                fontsize=8,
                transform=ax_bottom.transAxes,
                verticalalignment="bottom",
                bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
            )
            ax_bottom.text(
                0.02,
                0.06,
                "Most recent data: " + frame.index[-1].strftime("%Y-%m-%d"),
                fontsize=8,
                transform=ax_bottom.transAxes,
                verticalalignment="bottom",
                bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
            )
        else:
            ax_bottom.text(0.5, 0.5, "No margin debt or GDP data available", ha="center", va="center")
            ax_bottom.set_title("Margin Debt / GDP Signal Input and Trend")
            ax_bottom.set_xlabel("Date")
            ax_bottom.set_ylabel("Percent (%)")
            ax_bottom.grid(True, alpha=0.3)
    else:
        ax_bottom.text(0.5, 0.5, "No margin debt or GDP data available", ha="center", va="center")
        ax_bottom.set_title("Margin Debt / GDP Signal Input and Trend")
        ax_bottom.set_xlabel("Date")
        ax_bottom.set_ylabel("Percent (%)")
        ax_bottom.grid(True, alpha=0.3)

    if x_start is None or x_end is None:
        candidate_indices = []
        if not buy_hold_curve.empty:
            candidate_indices.append((buy_hold_curve.index[0], buy_hold_curve.index[-1]))
        if not traded_curve.empty:
            candidate_indices.append((traded_curve.index[0], traded_curve.index[-1]))
        frame_for_window = bottom.dropna(subset=["MarginDebt_GDP"], how="all")
        if not frame_for_window.empty:
            candidate_indices.append((frame_for_window.index[0], frame_for_window.index[-1]))
        if candidate_indices:
            x_start = min(item[0] for item in candidate_indices)
            x_end = max(item[1] for item in candidate_indices)

    if x_start is not None and x_end is not None and x_end > x_start:
        ax_top.set_xlim(x_start, x_end)
        ax_bottom.set_xlim(x_start, x_end)

    # Keep major/minor date ticks and vertical gridlines identical in both panels.
    for axis in (ax_top, ax_bottom):
        axis.xaxis.set_major_locator(mdates.YearLocator(2))
        axis.xaxis.set_minor_locator(mdates.YearLocator(1))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.grid(which="major", axis="x", alpha=0.7, linewidth=0.9)
        axis.grid(which="minor", axis="x", alpha=0.45, linewidth=0.6)

    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)

    figure_name = "PyTAAA_marginDebt.png"
    rule_caption = "Above SMA -> Cash" if rule == "above_sma_to_cash" else "Above SMA -> Long"
    return (
        "\n<br><h3>Margin Debt and Portfolio Overlay</h3>\n"
        "<p>Upper subplot uses the same monthly backtest Buy & Hold and traded-series context, with "
        "additional blue and magenta strategy lines driven by margin-debt-to-GDP signals applied to "
        "the PyTAAA portfolio (black series).</p>\n"
        "<p>Lower subplot shows the signal input series (Margin Debt / GDP), its signal SMA, and "
        "long-horizon averaged moving extrema curves. "
        f"Current rule: <b>{rule_caption}</b>, SMA length: <b>{sma_months}</b> months, "
        f"factor: <b>{sma_factor:.3f}</b>.</p>\n"
        f"<br><img src=\"{figure_name}\" alt=\"Margin Debt and Portfolio Overlay\" "
        "width=\"850\" height=\"500\"><br>\n"
    )
