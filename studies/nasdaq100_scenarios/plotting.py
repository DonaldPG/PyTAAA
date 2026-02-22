"""Plotting and output generation for NASDAQ100 oracle studies."""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from studies.nasdaq100_scenarios.portfolio_backtest import (
    compute_performance_metrics,
)

logger = logging.getLogger(__name__)


def plot_portfolio_histories(
    scenario_results: Dict[Tuple, Dict],
    datearray: List[date],
    output_path: str,
    title: str | None = None,
    log_scale: bool = False
) -> str:
    """Plot portfolio histories for all scenarios.

    Args:
        scenario_results: Dict mapping scenario key to result dict
        datearray: List of trading dates for x-axis
        output_path: File path for output PNG
        title: Optional plot title
        log_scale: Whether to plot y-axis in log scale

    Returns:
        Output path of the saved plot
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dates = pd.to_datetime(datearray)
    fig, ax = plt.subplots(figsize=(12, 7))

    keys_sorted = sorted(scenario_results.keys(), key=str)
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(keys_sorted))))

    for idx, key in enumerate(keys_sorted):
        result = scenario_results[key]
        values = result.get("portfolio_value")
        if values is None:
            continue

        label = _format_scenario_key(key)
        linewidth = 2.5 if label == "baseline" else 1.2
        alpha = 0.9 if label == "baseline" else 0.75

        ax.plot(
            dates,
            values,
            label=label,
            linewidth=linewidth,
            alpha=alpha,
            color=colors[idx % len(colors)]
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    if title:
        ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()

    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    logger.info("Saved portfolio history plot to %s", output_file)
    return str(output_file)


def plot_portfolio_histories_by_window_topn(
    scenario_results: Dict[Tuple, Dict],
    datearray: List[date],
    output_dir: str,
    study_name: str,
    analysis_start: date | None = None,
    analysis_stop: date | None = None,
    log_scale: bool = False
) -> List[str]:
    """Plot portfolio histories grouped by window and top_n.

    Args:
        scenario_results: Dict mapping scenario key to result dict
        datearray: List of trading dates for x-axis
        output_dir: Directory for output PNG files
        study_name: Study identifier used in file names
        analysis_start: Optional start date for x-axis limits
        analysis_stop: Optional stop date for x-axis limits
        log_scale: Whether to plot y-axis in log scale

    Returns:
        List of saved plot paths
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    windows, delays, top_ns = _extract_parameter_sets(scenario_results)
    if not windows or not delays or not top_ns:
        raise ValueError(
            "Scenario results must include window, delay, top_n keys"
        )

    baseline = _get_baseline_result(scenario_results)
    dates = pd.to_datetime(datearray)
    x_start = pd.Timestamp(analysis_start) if analysis_start else dates.min()
    x_stop = pd.Timestamp(analysis_stop) if analysis_stop else dates.max()
    x_start = max(x_start, dates.min())
    x_stop = min(x_stop, dates.max())

    saved_paths: List[str] = []
    delays_sorted = sorted(delays)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(delays_sorted))))
    y_min, y_max = _compute_portfolio_value_bounds(
        scenario_results,
        log_scale
    )

    for window in windows:
        for top_n in top_ns:
            fig, ax = plt.subplots(figsize=(12, 7))

            for idx, delay in enumerate(delays_sorted):
                key = (window, delay, top_n)
                result = scenario_results.get(key)
                if result is None:
                    continue

                values = result.get("portfolio_value")
                if values is None:
                    continue

                ax.plot(
                    dates,
                    values,
                    label=f"delay={delay}d",
                    linewidth=1.6,
                    alpha=0.85,
                    color=colors[idx % len(colors)]
                )

            if baseline is not None:
                baseline_values = baseline.get("portfolio_value")
                if baseline_values is not None:
                    ax.plot(
                        dates,
                        baseline_values,
                        label="buy_and_hold",
                        linewidth=2.6,
                        alpha=0.9,
                        color="black"
                    )

            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value")
            ax.set_title(f"Oracle Delay Study: {study_name}")
            if log_scale:
                ax.set_yscale("log")

            ax.set_xlim(x_start, x_stop)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.3)

            annotation = f"window={window}\ntop_n={top_n}"
            ax.text(
                0.98,
                0.02,
                annotation,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none"
                )
            )

            ax.legend(loc="best", fontsize=8, ncol=2)
            fig.tight_layout()

            output_file = output_root / (
                f"portfolio_histories_{study_name}_w{window}_n{top_n}.png"
            )
            fig.savefig(output_file, dpi=150)
            plt.close(fig)

            saved_paths.append(str(output_file))
            logger.info("Saved portfolio history plot to %s", output_file)

    return saved_paths


def generate_summary_json(
    scenario_results: Dict[Tuple, Dict],
    datearray: List[date],
    output_path: str
) -> str:
    """Generate summary JSON with performance metrics per scenario.

    Args:
        scenario_results: Dict mapping scenario key to result dict
        datearray: List of trading dates for CAGR computation
        output_path: File path for output JSON

    Returns:
        Output path of the saved JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    metrics_by_key = compute_performance_metrics(scenario_results)
    summary = {}

    start_date, end_date = _get_date_bounds(datearray)
    years = _compute_years_between(start_date, end_date)

    for key, result in scenario_results.items():
        scenario_key = _format_scenario_key(key)
        metrics = metrics_by_key.get(key, {})

        portfolio_value = result.get("portfolio_value")
        final_value = result.get("final_value")
        total_return = result.get("total_return")

        if final_value is None and portfolio_value is not None:
            final_value = float(portfolio_value[-1])
        if total_return is None and portfolio_value is not None:
            total_return = (portfolio_value[-1] / portfolio_value[0]) - 1.0

        initial_value = None
        if portfolio_value is not None and len(portfolio_value) > 0:
            initial_value = float(portfolio_value[0])
        elif final_value is not None and total_return is not None:
            initial_value = final_value / (1.0 + total_return)

        cagr = None
        if initial_value is not None and years > 0:
            cagr = (final_value / initial_value) ** (1.0 / years) - 1.0

        summary[scenario_key] = {
            "final_value": final_value,
            "total_return": total_return,
            "cagr": cagr,
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "volatility": metrics.get("volatility"),
            "max_drawdown": metrics.get("max_drawdown"),
        }

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    logger.info("Saved summary JSON to %s", output_file)
    return str(output_file)


def plot_parameter_sensitivity_panels(
    scenario_results: Dict[Tuple, Dict],
    output_path: str,
    metric: str = "total_return"
) -> str:
    """Plot parameter sensitivity panels by top_n.

    Args:
        scenario_results: Dict mapping (window, delay, top_n) to result dict
        output_path: File path for output PNG
        metric: Metric name to visualize

    Returns:
        Output path of the saved plot
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    windows, delays, top_ns = _extract_parameter_sets(scenario_results)
    if not windows or not delays or not top_ns:
        raise ValueError("Scenario results must include window, delay, top_n keys")

    panel_count = len(top_ns)
    grid_size = int(np.ceil(np.sqrt(panel_count)))

    global_min, global_max = _compute_metric_bounds(
        scenario_results,
        metric,
        windows,
        delays,
        top_ns
    )

    fig, axes = plt.subplots(
        nrows=grid_size,
        ncols=grid_size,
        figsize=(4 * grid_size, 3.5 * grid_size),
        squeeze=False
    )

    for idx, top_n in enumerate(top_ns):
        row = idx // grid_size
        col = idx % grid_size
        ax = axes[row][col]

        values = np.full((len(delays), len(windows)), np.nan)
        for i, delay in enumerate(delays):
            for j, window in enumerate(windows):
                key = (window, delay, top_n)
                result = scenario_results.get(key)
                if result is None:
                    continue

                if metric == "total_return":
                    values[i, j] = result.get("total_return")
                elif metric == "final_value":
                    values[i, j] = result.get("final_value")
                else:
                    values[i, j] = result.get(metric)

        im = ax.imshow(
            values,
            aspect="auto",
            origin="lower",
            vmin=global_min,
            vmax=global_max
        )
        ax.set_xticks(range(len(windows)), labels=windows)
        ax.set_yticks(range(len(delays)), labels=delays)
        ax.set_xlabel("Window")
        ax.set_ylabel("Delay")
        ax.set_title(f"top_n={top_n}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(panel_count, grid_size * grid_size):
        row = idx // grid_size
        col = idx % grid_size
        axes[row][col].axis("off")

    fig.suptitle(f"Parameter Sensitivity: {metric}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    logger.info("Saved parameter sensitivity panels to %s", output_file)
    return str(output_file)


def _format_scenario_key(key: Tuple) -> str:
    if key == ("baseline", 0, 0) or key == "baseline":
        return "baseline"
    if isinstance(key, tuple) and len(key) == 3:
        window, delay, top_n = key
        return f"w{window}_d{delay}_n{top_n}"
    return str(key)


def _get_date_bounds(datearray: Iterable[date]) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = [pd.Timestamp(d) for d in datearray]
    if not dates:
        raise ValueError("datearray is empty")
    return min(dates), max(dates)


def _compute_years_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    days = (end - start).days
    return max(days / 365.25, 0.0)


def _extract_parameter_sets(
    scenario_results: Dict[Tuple, Dict]
) -> tuple[List[int], List[int], List[int]]:
    windows = set()
    delays = set()
    top_ns = set()

    for key in scenario_results.keys():
        if isinstance(key, tuple) and len(key) == 3:
            window, delay, top_n = key
            if isinstance(window, int) and isinstance(delay, int) and isinstance(top_n, int):
                windows.add(window)
                delays.add(delay)
                top_ns.add(top_n)

    return sorted(windows), sorted(delays), sorted(top_ns)


def _compute_metric_bounds(
    scenario_results: Dict[Tuple, Dict],
    metric: str,
    windows: List[int],
    delays: List[int],
    top_ns: List[int]
) -> tuple[float, float]:
    values: List[float] = []
    for window in windows:
        for delay in delays:
            for top_n in top_ns:
                key = (window, delay, top_n)
                result = scenario_results.get(key)
                if result is None:
                    continue
                if metric == "total_return":
                    metric_value = result.get("total_return")
                elif metric == "final_value":
                    metric_value = result.get("final_value")
                else:
                    metric_value = result.get(metric)
                if metric_value is not None and np.isfinite(metric_value):
                    values.append(float(metric_value))

    if not values:
        return 0.0, 1.0

    return float(np.min(values)), float(np.max(values))


def _get_baseline_result(scenario_results: Dict[Tuple, Dict]) -> Dict | None:
    if ("baseline", 0, 0) in scenario_results:
        return scenario_results[("baseline", 0, 0)]
    if "baseline" in scenario_results:
        return scenario_results["baseline"]
    return None


def _compute_portfolio_value_bounds(
    scenario_results: Dict[Tuple, Dict],
    log_scale: bool
) -> tuple[float, float]:
    values: List[float] = []
    for result in scenario_results.values():
        series = result.get("portfolio_value")
        if series is None:
            continue
        series_values = np.asarray(series, dtype=float)
        values.extend(series_values[np.isfinite(series_values)].tolist())

    if not values:
        return 1.0, 1.0

    min_value = float(np.min(values))
    max_value = float(np.max(values))

    if log_scale:
        positive_values = [v for v in values if v > 0]
        if positive_values:
            min_value = float(np.min(positive_values))
        else:
            min_value = 1.0

    if min_value == max_value:
        max_value = min_value * 1.01

    return min_value, max_value
