"""Liquidity proxy series and plot helpers for PyTAAA.

The current implementation uses public FRED CSV endpoints so it works
without an API key. The proxy is intentionally conservative: it focuses
on the Federal Reserve balance sheet and related reserve facility
series, and is structured so additional central-bank series can be
added later.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO, StringIO
from typing import Iterable

import os
import re
import zipfile

import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests

if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from functions.GetParams import get_performance_store, get_webpage_store

FRED_CSV_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
BOE_BANKSTATS_ZIP_URL = (
    "https://www.bankofengland.co.uk/-/media/boe/files/statistics/"
    "bankstats-latest-tables.zip"
)
RBA_BALANCE_SHEET_CSV_URL = (
    "https://www.rba.gov.au/statistics/tables/csv/a1-data.csv"
)
SNB_BALANCE_SHEET_URL = "https://data.snb.ch/json/file/cube"
PBOC_CSV_URL_ENV = "PBOC_LIQUIDITY_CSV_URL"
PBOC_CSV_PATH_ENV = "PBOC_LIQUIDITY_CSV_PATH"
PBOC_DATE_COLUMN_ENV = "PBOC_LIQUIDITY_DATE_COLUMN"
PBOC_VALUE_COLUMN_ENV = "PBOC_LIQUIDITY_VALUE_COLUMN"

FX_EUR_USD = "DEXUSEU"
FX_JPY_USD = "DEXJPUS"
FX_CNY_USD = "DEXCHUS"
FX_GBP_USD = "DEXUSUK"
FX_CHF_USD = "DEXCHUS"
FX_AUD_USD = "DEXUSAL"


@lru_cache(maxsize=16)
def fred_series(series_id: str) -> pd.Series:
    """Fetch a public FRED time series as a pandas Series."""

    response = requests.get(
        FRED_CSV_BASE_URL,
        params={"id": series_id},
        timeout=30,
    )
    response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text))
    if frame.shape[1] < 2:
        return pd.Series(dtype=float, name=series_id)

    date_column = frame.columns[0]
    value_column = frame.columns[1]
    values = pd.to_numeric(frame[value_column], errors="coerce")
    series = pd.Series(values.to_numpy(), index=pd.to_datetime(frame[date_column]))
    series = series.dropna()
    series.name = series_id
    return series


def _first_matching_column(frame: pd.DataFrame, choices: tuple[str, ...]) -> str | None:
    """Return the first matching column name from a frame."""

    lowered = {column.lower(): column for column in frame.columns}
    for choice in choices:
        column = lowered.get(choice.lower())
        if column is not None:
            return column
    return None


def _series_from_frame(
    frame: pd.DataFrame,
    date_column: str,
    value_column: str,
    *,
    name: str,
    dayfirst: bool = False,
) -> pd.Series:
    """Build a sorted numeric series from a tabular source."""

    dates = pd.to_datetime(frame[date_column], errors="coerce", dayfirst=dayfirst)
    values = pd.to_numeric(frame[value_column], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates)
    series = series.dropna().sort_index()
    series.name = name
    return series


def _series_from_csv_text(
    csv_text: str,
    *,
    name: str,
    date_column: str | None = None,
    value_column: str | None = None,
) -> pd.Series:
    """Load a series from CSV text using common date and value columns."""

    frame = pd.read_csv(StringIO(csv_text))
    if frame.empty:
        return pd.Series(dtype=float, name=name)

    if date_column is None:
        date_column = _first_matching_column(
            frame,
            ("date", "observation_date", "datetime", "time"),
        )
    if value_column is None:
        value_column = _first_matching_column(
            frame,
            ("value", "total_assets", "assets", "amount", "close", "last"),
        )

    if date_column is None or value_column is None:
        return pd.Series(dtype=float, name=name)

    return _series_from_frame(frame, date_column, value_column, name=name)


@lru_cache(maxsize=16)
def fx_series(series_id: str) -> pd.Series:
    """Fetch a public FX series from FRED."""

    return fred_series(series_id)


def _usd_billion_from_local_millions(
    local_series: pd.Series,
    fx_rate: pd.Series,
    *,
    rate_direction: str,
) -> pd.Series:
    """Convert a local-currency series in millions to USD billions."""

    if local_series.empty or fx_rate.empty:
        return pd.Series(dtype=float)

    frame = pd.concat([local_series.sort_index(), fx_rate.sort_index()], axis=1)
    frame = frame.ffill().dropna(how="any")
    if frame.empty:
        return pd.Series(dtype=float)

    local = frame.iloc[:, 0].astype(float)
    rate = frame.iloc[:, 1].astype(float)
    if rate_direction == "usd_per_local":
        usd_billion = local * rate / 1000.0
    elif rate_direction == "local_per_usd":
        usd_billion = local / rate / 1000.0
    else:
        raise ValueError(f"Unsupported rate_direction: {rate_direction}")

    usd_billion.name = local_series.name
    return usd_billion


def _usd_billion_from_local_units(
    local_series: pd.Series,
    fx_rate: pd.Series,
    *,
    local_units_per_observation: float,
    rate_direction: str,
) -> pd.Series:
    """Convert a local-currency series with non-million units to USD billions."""

    if local_series.empty or fx_rate.empty:
        return pd.Series(dtype=float)

    frame = pd.concat([local_series.sort_index(), fx_rate.sort_index()], axis=1)
    frame = frame.ffill().dropna(how="any")
    if frame.empty:
        return pd.Series(dtype=float)

    local = frame.iloc[:, 0].astype(float)
    rate = frame.iloc[:, 1].astype(float)
    if rate_direction == "usd_per_local":
        usd_billion = local * local_units_per_observation * rate / 1_000_000_000.0
    elif rate_direction == "local_per_usd":
        usd_billion = local * local_units_per_observation / rate / 1_000_000_000.0
    else:
        raise ValueError(f"Unsupported rate_direction: {rate_direction}")

    usd_billion.name = local_series.name
    return usd_billion


@lru_cache(maxsize=8)
def fed_balance_sheet_usd_billion() -> pd.Series:
    """Federal Reserve total assets in USD billions."""

    return fred_series("WALCL") / 1000.0


@lru_cache(maxsize=8)
def ecb_balance_sheet_usd_billion() -> pd.Series:
    """ECB consolidated balance sheet in USD billions."""

    eur_assets_millions = fred_series("ECBASSETSW")
    eur_usd = fx_series(FX_EUR_USD)
    return _usd_billion_from_local_millions(
        eur_assets_millions,
        eur_usd,
        rate_direction="usd_per_local",
    )


@lru_cache(maxsize=8)
def boj_balance_sheet_usd_billion() -> pd.Series:
    """Bank of Japan total assets in USD billions."""

    jpy_assets_100m = fred_series("JPNASSETS")
    jpy_usd = fx_series(FX_JPY_USD)
    return _usd_billion_from_local_units(
        jpy_assets_100m,
        jpy_usd,
        local_units_per_observation=100_000_000.0,
        rate_direction="local_per_usd",
    )


@lru_cache(maxsize=8)
def boe_balance_sheet_usd_billion() -> pd.Series:
    """Bank of England consolidated balance sheet in USD billions.

    The public FRED series `UKASSETS` covers the older history, while the
    current Bankstats table covers the latest years. The two are combined so
    the chart has a usable continuous series.
    """

    historical = fred_series("UKASSETS")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8",
        "Referer": "https://www.bankofengland.co.uk/statistics/tables",
    }
    response = requests.get(BOE_BANKSTATS_ZIP_URL, headers=headers, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        with archive.open("Latest Tables/TabB1.1.3.xls") as handle:
            frame = pd.read_excel(handle, header=None)

    data = frame.iloc[12:, [1, 26]].copy()
    data.columns = ["period", "total_assets"]

    dates: list[pd.Timestamp] = []
    values: list[float] = []
    current_year: int | None = None
    for period, value in zip(data["period"], data["total_assets"]):
        if pd.isna(period) or pd.isna(value):
            continue
        label = str(period).strip()
        match = re.fullmatch(r"(?:(\d{4})\s+)?Q([1-4])", label)
        if match is None:
            continue
        if match.group(1) is not None:
            current_year = int(match.group(1))
        if current_year is None:
            continue
        quarter = int(match.group(2))
        dates.append(pd.Period(f"{current_year}Q{quarter}", freq="Q").end_time.normalize())
        values.append(float(value))

    series = pd.Series(values, index=pd.to_datetime(dates), name="boe_balance_sheet_gbp_millions")
    if not historical.empty and not series.empty:
        historical = historical.rename("boe_balance_sheet_gbp_millions")
        series = pd.concat([historical, series]).sort_index()
        series = series[~series.index.duplicated(keep="last")]
    elif not historical.empty:
        series = historical.rename("boe_balance_sheet_gbp_millions")

    return _usd_billion_from_local_millions(
        series,
        fx_series(FX_GBP_USD),
        rate_direction="usd_per_local",
    )


@lru_cache(maxsize=8)
def snb_balance_sheet_usd_billion() -> pd.Series:
    """Swiss National Bank total assets in USD billions."""

    payload = {
        "selectedDimensionItems": [
            {
                "dimensionId": "D0",
                "selectedDimensionItemIds": [
                    "GFG",
                    "D",
                    "RIWF",
                    "IZ",
                    "W",
                    "FRGSF",
                    "FRGUSD",
                    "GSGSF",
                    "IG",
                    "GD",
                    "FI",
                    "WSF",
                    "DS",
                    "UA",
                    "T0",
                    "N",
                    "GB",
                    "VB",
                    "GBI",
                    "US",
                    "VRGSF",
                    "ES",
                    "UT",
                    "VF",
                    "AIWFS",
                    "SP",
                    "RE",
                    "T1",
                ],
            }
        ],
        "fromDate": "1996-12",
        "toDate": "2100-12",
        "getAllData": True,
        "pageViewTime": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
    }
    response = requests.post(
        SNB_BALANCE_SHEET_URL,
        params={
            "isWarehouse": "false",
            "cubeId": "snbbipo",
            "fileType": "CSV",
            "lang": "en",
            "pageViewTime": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        },
        json=payload,
        headers={"content-type": "application/json", "x-epb-ajax": "true"},
        timeout=60,
    )
    response.raise_for_status()

    lines = response.text.splitlines()
    start_row = next(
        index for index, line in enumerate(lines) if line.lstrip("\ufeff").startswith('"Date"')
    )
    frame = pd.read_csv(
        StringIO("\n".join(lines[start_row:])),
        sep=";",
        quotechar='"',
    )
    frame = frame.loc[frame["D0"] == "T0", ["Date", "Value"]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], format="%Y-%m", errors="coerce")
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    series = pd.Series(frame["Value"].to_numpy(), index=frame["Date"], name="snb_balance_sheet_chf_millions")
    return _usd_billion_from_local_millions(
        series,
        fx_series(FX_CHF_USD),
        rate_direction="local_per_usd",
    )


@lru_cache(maxsize=8)
def rba_balance_sheet_usd_billion() -> pd.Series:
    """Reserve Bank of Australia balance sheet in USD billions."""

    response = requests.get(RBA_BALANCE_SHEET_CSV_URL, timeout=60)
    response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text), header=None, skiprows=11)
    if frame.shape[1] < 2:
        return pd.Series(dtype=float)

    data = frame.iloc[:, [0, frame.shape[1] - 1]].copy()
    data.columns = ["date", "total_assets"]
    data["date"] = pd.to_datetime(data["date"], dayfirst=True, errors="coerce")
    data["total_assets"] = pd.to_numeric(data["total_assets"], errors="coerce")
    series = pd.Series(
        data["total_assets"].to_numpy(),
        index=data["date"],
        name="rba_balance_sheet_aud_millions",
    )
    return _usd_billion_from_local_millions(
        series,
        fx_series(FX_AUD_USD),
        rate_direction="usd_per_local",
    )


@lru_cache(maxsize=8)
def pboc_balance_sheet_cny_millions() -> pd.Series:
    """Optional user-supplied PBOC balance sheet in CNY millions."""

    csv_path = os.environ.get(PBOC_CSV_PATH_ENV, "").strip()
    csv_url = os.environ.get(PBOC_CSV_URL_ENV, "").strip()
    date_column = os.environ.get(PBOC_DATE_COLUMN_ENV, "").strip() or None
    value_column = os.environ.get(PBOC_VALUE_COLUMN_ENV, "").strip() or None

    if csv_path:
        frame = pd.read_csv(csv_path)
    elif csv_url:
        response = requests.get(csv_url, timeout=60)
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
    else:
        return pd.Series(dtype=float)

    if frame.empty:
        return pd.Series(dtype=float)

    if date_column is None:
        date_column = _first_matching_column(
            frame,
            ("date", "observation_date", "datetime", "time"),
        )
    if value_column is None:
        value_column = _first_matching_column(
            frame,
            ("value", "total_assets", "assets", "amount", "close", "last"),
        )

    if date_column is None or value_column is None:
        return pd.Series(dtype=float)

    series = _series_from_frame(
        frame,
        date_column,
        value_column,
        name="pboc_balance_sheet_cny_millions",
    )
    return series


def pboc_balance_sheet_usd_billion() -> pd.Series:
    """Optional PBOC balance sheet in USD billions."""

    return _usd_billion_from_local_millions(
        pboc_balance_sheet_cny_millions(),
        fx_series(FX_CNY_USD),
        rate_direction="local_per_usd",
    )


def _safe_series_fetch(name: str, loader) -> pd.Series:
    """Return an empty series instead of raising on transient source failures."""

    try:
        return loader()
    except Exception as exc:
        print(f" Warning: unable to load {name} liquidity series: {exc}")
        return pd.Series(dtype=float, name=name)


def central_bank_liquidity_components() -> pd.DataFrame:
    """Load the public central-bank balance sheets and convert them to USD."""

    series_map = {
        "Fed": _safe_series_fetch("Fed", fed_balance_sheet_usd_billion),
        "ECB": _safe_series_fetch("ECB", ecb_balance_sheet_usd_billion),
        "BOJ": _safe_series_fetch("BOJ", boj_balance_sheet_usd_billion),
        "BOE": _safe_series_fetch("BOE", boe_balance_sheet_usd_billion),
        "SNB": _safe_series_fetch("SNB", snb_balance_sheet_usd_billion),
        "RBA": _safe_series_fetch("RBA", rba_balance_sheet_usd_billion),
    }
    pboc_series = _safe_series_fetch("PBOC", pboc_balance_sheet_usd_billion)
    if not pboc_series.empty:
        series_map["PBOC"] = pboc_series

    frame = pd.concat(
        [series.rename(name) for name, series in series_map.items() if not series.empty],
        axis=1,
    ).sort_index()
    # Keep the union of dates so pre-2013 history is retained even when some
    # central-bank series start later than others.
    frame = frame.ffill()
    return frame


def global_liquidity_contributor_count(components: pd.DataFrame) -> pd.Series:
    """Return how many central-bank series contribute at each date."""

    if components.empty:
        return pd.Series(dtype=float)

    count = components.notna().sum(axis=1).astype(float)
    count.name = "contributor_count"
    return count


def fed_total_assets() -> pd.Series:
    """Federal Reserve total assets (WALCL) in millions of USD."""

    return fred_series("WALCL")


def fed_tga() -> pd.Series:
    """U.S. Treasury General Account balance at the Fed (WTREGEN)."""

    return fred_series("WTREGEN")


def fed_rrp() -> pd.Series:
    """Federal Reserve overnight reverse repo facility (RRPONTSYD)."""

    return fred_series("RRPONTSYD")


def _combine_series(series_list: Iterable[pd.Series]) -> pd.DataFrame:
    frame = pd.concat(list(series_list), axis=1).sort_index()
    frame = frame.ffill().dropna(how="any")
    return frame


def fed_net_liquidity() -> pd.Series:
    """Simple public U.S. liquidity proxy: WALCL - TGA - RRP."""

    frame = _combine_series([fed_total_assets(), fed_tga(), fed_rrp()])
    frame.columns = ["fed", "tga", "rrp"]
    net = frame["fed"] - frame["tga"] - frame["rrp"]
    net.name = "fed_net_liquidity"
    return net


def global_liquidity_full() -> pd.Series:
    """Aggregate public central-bank balance sheets in USD billions."""

    components = central_bank_liquidity_components()
    if components.empty:
        return pd.Series(dtype=float)

    liquidity = components.sum(axis=1, min_count=1)
    liquidity = liquidity.dropna()
    liquidity.name = "global_liquidity_usd_billions"
    return liquidity


def global_liquidity_flash() -> pd.Series:
    """A smoother short-term view of the global liquidity aggregate."""

    return global_liquidity_full().rolling(window="90D", min_periods=1).mean()


@lru_cache(maxsize=1)
def fed_stimulus_injection() -> pd.Series:
    """Fed stimulus as a ratio to a pre-crisis trend baseline.

    The baseline is an exponential trend fitted to 2003-2007 WALCL levels,
    then extrapolated across the full sample. Values above 1.0 indicate the
    Fed balance sheet is above that pre-crisis growth path.
    """

    walcl = fed_balance_sheet_usd_billion()
    if walcl.empty:
        return pd.Series(dtype=float)

    pre_crisis = walcl.loc[(walcl.index >= "2003-01-01") & (walcl.index <= "2007-12-31")]
    if pre_crisis.empty or len(pre_crisis) < 24:
        baseline = pd.Series(900.0, index=walcl.index)
    else:
        x0 = pre_crisis.index[0]
        x_pre = (pre_crisis.index - x0).days.to_numpy(dtype=float)
        y_pre = np.log(np.clip(pre_crisis.to_numpy(dtype=float), 1.0, None))
        slope, intercept = np.polyfit(x_pre, y_pre, 1)

        x_all = (walcl.index - x0).days.to_numpy(dtype=float)
        baseline_values = np.exp(intercept + slope * x_all)
        baseline = pd.Series(baseline_values, index=walcl.index)

    ratio = (walcl / baseline.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    ratio = ratio.dropna()
    ratio.name = "fed_stimulus_ratio_to_trend"
    return ratio


def liquidity_regime(series: pd.Series, lookback_days: int = 90) -> str:
    """Classify the short-term direction of a liquidity series."""

    cleaned = series.dropna().sort_index()
    if cleaned.empty or len(cleaned) < 2:
        return "neutral"

    cutoff = cleaned.index[-1] - pd.Timedelta(days=lookback_days)
    recent = cleaned.loc[cleaned.index >= cutoff]
    if len(recent) < 2:
        recent = cleaned.iloc[-min(len(cleaned), lookback_days):]

    if len(recent) < 2:
        return "neutral"

    elapsed_days = max((recent.index[-1] - recent.index[0]).days, 1)
    slope = (float(recent.iloc[-1]) - float(recent.iloc[0])) / elapsed_days
    if slope > 0:
        return "supportive"
    if slope < 0:
        return "restrictive"
    return "neutral"


def _load_buy_and_hold_curve(json_fn: str) -> pd.Series:
    """Load Buy-and-Hold curve from the existing backtest params file."""

    performance_store = get_performance_store(json_fn)
    params_path = os.path.join(
        performance_store,
        "pyTAAAweb_backtestPortfolioValue.params",
    )

    if not os.path.isfile(params_path):
        return pd.Series(dtype=float, name="Buy & Hold")

    dates = []
    buy_hold_values = []
    with open(params_path, "r") as handle:
        lines = handle.read().split("\n")
    for line in lines:
        parts = [item for item in line.split(" ") if item]
        if len(parts) < 2:
            continue
        try:
            dates.append(pd.to_datetime(parts[0]))
            buy_hold_values.append(float(parts[1]))
        except (ValueError, TypeError):
            continue

    if not dates:
        return pd.Series(dtype=float, name="Buy & Hold")

    series = pd.Series(buy_hold_values, index=pd.to_datetime(dates), name="Buy & Hold")
    series = series.sort_index().dropna()
    return series


def make_liquidity_plot(json_fn: str) -> str:
    """Build the liquidity chart and return the HTML snippet."""

    webpage_dir = get_webpage_store(json_fn)
    figure_path = os.path.join(webpage_dir, "PyTAAA_globalLiquidity.png")

    components = central_bank_liquidity_components()
    full = global_liquidity_full()
    flash = global_liquidity_flash()
    fed_stimulus = fed_stimulus_injection()
    contributor_count = global_liquidity_contributor_count(components)
    frame = pd.concat([full, flash], axis=1, join="inner").dropna()
    frame.columns = ["Global Liquidity", "90D Mean"]
    buy_hold_curve = _load_buy_and_hold_curve(json_fn)

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

    if not frame.empty:
        current_regime = liquidity_regime(frame["Global Liquidity"])
        latest_value = float(frame["Global Liquidity"].iloc[-1])
        created_at = pd.Timestamp.now().strftime("%Y-%m-%d %I:%M%p")

        plot_start = max(frame.index[0], pd.Timestamp("2003-01-01"))
        plot_end = frame.index[-1]

        if not buy_hold_curve.empty:
            buy_hold_view = buy_hold_curve.loc[
                (buy_hold_curve.index >= plot_start)
                & (buy_hold_curve.index <= plot_end)
            ]
            if not buy_hold_view.empty:
                ax_top.plot(
                    buy_hold_view.index,
                    buy_hold_view.values,
                    "r-",
                    lw=1.25,
                    label="Buy & Hold",
                )
                ax_top.set_yscale("log")
                ax_top.legend(loc="upper left")
            else:
                ax_top.text(
                    0.5,
                    0.5,
                    "Buy & Hold data unavailable for liquidity date range",
                    ha="center",
                    va="center",
                    transform=ax_top.transAxes,
                    fontsize=8,
                )
        else:
            ax_top.text(
                0.5,
                0.5,
                "Buy & Hold data unavailable",
                ha="center",
                va="center",
                transform=ax_top.transAxes,
                fontsize=8,
            )
        ax_top.set_title("Buy & Hold Stock Value")
        ax_top.set_ylabel("Value")
        ax_top.grid(True, alpha=0.3)

        frame_view = frame.loc[(frame.index >= plot_start) & (frame.index <= plot_end)]
        if frame_view.empty:
            frame_view = frame

        ax_bottom.plot(
            frame_view.index,
            frame_view["Global Liquidity"].values,
            color="orange",
            lw=1.5,
            label="Global Liquidity",
        )
        ax_bottom.plot(
            frame_view.index,
            frame_view["90D Mean"].values,
            color="gray",
            lw=1.0,
            label="90D Mean",
        )
        # Use true day-based rolling windows, not sample-count windows.
        full_sma_50 = frame["Global Liquidity"].rolling(window="50D", min_periods=1).mean()
        full_sma_250 = frame["Global Liquidity"].rolling(window="250D", min_periods=1).mean()
        sma_50_view = full_sma_50.reindex(frame_view.index)
        sma_250_view = full_sma_250.reindex(frame_view.index)
        ax_bottom.plot(
            frame_view.index,
            sma_50_view.values,
            color="dodgerblue",
            lw=1.0,
            label="50D SMA",
        )
        ax_bottom.plot(
            frame_view.index,
            sma_250_view.values,
            color="navy",
            lw=1.0,
            linestyle="--",
            label="250D SMA",
        )
        # Add secondary axis for Fed stimulus (log scale on right).
        ax_right = ax_bottom.twinx()
        if not fed_stimulus.empty:
            fed_view = fed_stimulus.loc[
                (fed_stimulus.index >= plot_start) & (fed_stimulus.index <= plot_end)
            ]
            if not fed_view.empty:
                ax_right.plot(
                    fed_view.index,
                    fed_view.values,
                    color="crimson",
                    lw=2.0,
                    alpha=0.95,
                    label="Fed vs Trend (R-axis)",
                    zorder=5,
                )
                ax_right.set_yscale("log")
                ax_right.set_ylabel("Fed / Pre-Crisis Trend (log)", color="crimson")
                ax_right.tick_params(axis="y", labelcolor="crimson")
        ax_bottom.set_title("Global Liquidity Index - Public Central Banks (USD bn)")
        ax_bottom.set_ylabel("US$ billions")
        ax_bottom.set_yscale("log")
        visible = np.concatenate(
            [
                frame_view["Global Liquidity"].to_numpy(dtype=float),
                frame_view["90D Mean"].to_numpy(dtype=float),
                sma_50_view.to_numpy(dtype=float),
                sma_250_view.to_numpy(dtype=float),
            ]
        )
        visible = visible[np.isfinite(visible) & (visible > 0.0)]
        if visible.size:
            y_min = visible.min() * 0.9
            y_max = visible.max() * 1.1
            if y_max > y_min:
                ax_bottom.set_ylim(y_min, y_max)
        lines_left, labels_left = ax_bottom.get_legend_handles_labels()
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        ax_bottom.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left", fontsize=9)
        ax_bottom.set_xlim(plot_start, plot_end)
        ax_top.set_xlim(plot_start, plot_end)
        # Major vertical grid every 2 years, minor every year.
        ax_bottom.xaxis.set_major_locator(mdates.YearLocator(2))
        ax_bottom.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_bottom.grid(which="major", axis="x", alpha=0.7, linewidth=0.9)
        ax_bottom.grid(which="minor", axis="x", alpha=0.45, linewidth=0.6)
        ax_bottom.grid(which="major", axis="y", alpha=0.3, linewidth=0.7)
        ax_bottom.text(
            0.02,
            0.58,
            (
                f"Current liquidity regime: {current_regime}\n"
                f"Latest global liquidity: {latest_value:,.0f} US$ bn\n"
                f"Contributing banks now: {int(contributor_count.iloc[-1]) if not contributor_count.empty else 0}\n"
                f"Sources: {', '.join(components.columns)}\n"
                f"Created: {created_at}"
            ),
            fontsize=8,
            transform=ax_bottom.transAxes,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
        )
    else:
        ax_top.text(0.5, 0.5, "No Buy & Hold data available", ha="center", va="center")
        ax_top.grid(True, alpha=0.3)
        ax_bottom.set_title("Global Liquidity Index - Public Central Banks (USD bn)")
        ax_bottom.text(0.5, 0.5, "No liquidity data available", ha="center", va="center")
        ax_bottom.xaxis.set_major_locator(mdates.YearLocator(2))
        ax_bottom.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_bottom.grid(which="major", axis="x", alpha=0.7, linewidth=0.9)
        ax_bottom.grid(which="minor", axis="x", alpha=0.45, linewidth=0.6)
        ax_bottom.grid(which="major", axis="y", alpha=0.3, linewidth=0.7)
        current_regime = "neutral"

    ax_bottom.set_xlabel("Date")
    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)

    figure_name = "PyTAAA_globalLiquidity.png"
    return (
        "\n<br><h3>Global Liquidity Proxy</h3>\n"
        "<p>This chart aggregates public balance-sheet data from the major central banks and converts "
        "each series into USD using same-date FX. The current public sources are the Fed, ECB, BOJ, "
        "BOE, SNB, and RBA. If you want PBOC included, supply a CNY balance-sheet CSV or URL via "
        "the PBOC_LIQUIDITY_CSV_PATH or PBOC_LIQUIDITY_CSV_URL environment variable.</p>\n"
        "<p><b>Liquidity trend framework:</b> rising liquidity often supports equities and "
        "compresses volatility; peak liquidity can hide growing risk; falling liquidity often "
        "coincides with weaker equities, wider credit spreads, and higher volatility; bottoming "
        "liquidity can align with durable market lows.</p>\n"
        "<p><b>General guidance (not investment advice):</b> expect more volatility, narrower "
        "leadership, earlier credit stress, possible policy response if tightening overshoots, "
        "and global divergence across regions. Liquidity is a meta-driver behind these shifts. "
        f"Current liquidity regime: <b>{current_regime}</b>.</p>\n"
        f"<br><img src=\"{figure_name}\" alt=\"Global Liquidity Proxy\" "
        "width=\"850\" height=\"500\"><br>\n"
    )
