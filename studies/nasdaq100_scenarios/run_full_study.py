"""End-to-end runner for NASDAQ100 oracle delay studies."""

import argparse
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Tuple

from studies.nasdaq100_scenarios.data_loader import load_nasdaq100_window
from studies.nasdaq100_scenarios.oracle_signals import generate_scenario_signals
from studies.nasdaq100_scenarios.portfolio_backtest import (
    compute_extrema_interpolated_series,
    compute_performance_metrics,
    simulate_buy_and_hold,
    simulate_monthly_portfolio,
)
from studies.nasdaq100_scenarios.plotting import (
    generate_summary_json,
    plot_parameter_sensitivity_panels,
    plot_portfolio_histories_by_window_topn,
)

logger = logging.getLogger(__name__)


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)

    log_level = config.get("execution_control", {}).get("log_level", "INFO")
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

    print("Starting oracle delay study...")
    print(f"Config: {args.config}")

    adjClose_full, symbols, datearray_full, tradable_mask_full = (
        load_nasdaq100_window(args.config)
    )

    oracle_params = config["oracle_parameters"]
    delay_params = config["delay_parameters"]
    portfolio_params = config["portfolio_parameters"]
    ranking_config = config.get("ranking_method", {})
    output_config = config.get("output_control", {})

    extrema_windows = oracle_params["extrema_windows"]
    delays = delay_params["days_delay"]
    top_n_list = portfolio_params["top_n_list"]

    initial_value = portfolio_params.get("initial_portfolio_value", 10000.0)
    transaction_cost = portfolio_params.get("transaction_cost_per_trade", 0.0)
    apply_costs = portfolio_params.get("enable_transaction_costs", False)

    ranking_method = _map_ranking_method(ranking_config.get("method"))
    apply_delay_to_ranking = ranking_config.get("apply_delay_to_ranking", False)

    if ranking_method == "oracle" and apply_delay_to_ranking:
        logger.warning(
            "apply_delay_to_ranking is ignored for forward-return ranking"
        )

    data_config = config["data_selection"]
    analysis_start = _parse_date(data_config["start_date"])
    analysis_stop = _parse_date(data_config["stop_date"])
    analysis_start_idx, analysis_stop_idx = _find_analysis_indices(
        datearray_full,
        analysis_start,
        analysis_stop
    )

    adjClose = adjClose_full[:, analysis_start_idx:analysis_stop_idx + 1]
    datearray = datearray_full[analysis_start_idx:analysis_stop_idx + 1]
    tradable_mask = (
        tradable_mask_full[:, analysis_start_idx:analysis_stop_idx + 1]
    )

    scenario_signals_full = generate_scenario_signals(
        adjClose_full,
        symbols,
        datearray_full,
        extrema_windows,
        delays,
    )
    scenario_signals = {}
    for key, signal in scenario_signals_full.items():
        scenario_signals[key] = (
            signal[:, analysis_start_idx:analysis_stop_idx + 1]
        )

    print("Running portfolio scenarios...")
    scenario_results = _run_scenarios(
        adjClose=adjClose,
        symbols=symbols,
        datearray=datearray,
        scenario_signals=scenario_signals,
        top_n_list=top_n_list,
        initial_value=initial_value,
        transaction_cost=transaction_cost,
        apply_costs=apply_costs,
        ranking_method=ranking_method,
        apply_delay_to_ranking=apply_delay_to_ranking,
    )

    baseline = simulate_buy_and_hold(
        adjClose=adjClose,
        datearray=datearray,
        symbols=symbols,
        initial_value=initial_value,
        tradable_mask=tradable_mask,
    )
    scenario_results[("baseline", 0, 0)] = baseline

    study_name = config.get("study_metadata", {}).get("name", "oracle_delay")
    output_root = Path("studies/nasdaq100_scenarios")

    if output_config.get("output_metrics", True):
        metrics_path = output_root / "results" / f"metrics_{study_name}.json"
        metrics = compute_performance_metrics(scenario_results)
        _write_json(metrics_path, _stringify_keys(metrics))

    if output_config.get("output_summary_json", True):
        summary_path = output_root / "results" / f"summary_{study_name}.json"
        generate_summary_json(scenario_results, datearray, str(summary_path))

    if output_config.get("output_holdings_log", False):
        holdings_path = output_root / "results" / f"holdings_{study_name}.json"
        _write_json(holdings_path, _extract_holdings(scenario_results))

    if output_config.get("output_plots", True):
        plots_dir = output_root / "plots"
        plot_portfolio_histories_by_window_topn(
            scenario_results,
            datearray,
            str(plots_dir),
            study_name=study_name,
            analysis_start=analysis_start,
            analysis_stop=analysis_stop,
            log_scale=True,
        )
        plot_parameter_sensitivity_panels(
            scenario_results,
            str(plots_dir / f"parameter_sensitivity_total_return_{study_name}.png"),
            metric="total_return",
        )

    print("Study complete.")


def _run_scenarios(
    adjClose,
    symbols,
    datearray,
    scenario_signals,
    top_n_list,
    initial_value,
    transaction_cost,
    apply_costs,
    ranking_method,
    apply_delay_to_ranking,
) -> Dict[Tuple[int, int, int], Dict]:
    results: Dict[Tuple[int, int, int], Dict] = {}

    for (window, delay), signal2D in scenario_signals.items():
        interpolated_series = None
        if ranking_method == "slope":
            logger.info(
                "Precomputing extrema interpolation for window=%d, delay=%d",
                window,
                delay,
            )
            interpolated_series = compute_extrema_interpolated_series(
                adjClose=adjClose,
                datearray=datearray,
                window_half_width=window,
            )

        for top_n in top_n_list:
            delay_days = delay if apply_delay_to_ranking else 0
            window_half_width = window if ranking_method == "slope" else 10

            result = simulate_monthly_portfolio(
                adjClose=adjClose,
                signal2D=signal2D,
                top_n=top_n,
                datearray=datearray,
                symbols=symbols,
                initial_value=initial_value,
                transaction_cost=transaction_cost,
                apply_costs=apply_costs,
                ranking_method=ranking_method,
                window_half_width=window_half_width,
                delay_days=delay_days,
                interpolated_series=interpolated_series,
            )
            results[(window, delay, top_n)] = result

    return results


def _map_ranking_method(method: str | None) -> str | None:
    if method is None:
        return None
    if method in ("forward_monthly_return", "oracle"):
        return "oracle"
    if method == "slope":
        return "slope"
    if method == "equal":
        return None
    return None


def _extract_holdings(results: Dict[Tuple[int, int, int], Dict]) -> Dict[str, List]:
    output: Dict[str, List] = {}
    for key, result in results.items():
        scenario_key = _format_scenario_key(key)
        holdings_log = result.get("holdings_log", [])
        output[scenario_key] = _serialize_holdings_log(holdings_log)
    return output


def _format_scenario_key(key: Tuple) -> str:
    if key == ("baseline", 0, 0):
        return "baseline"
    window, delay, top_n = key
    return f"w{window}_d{delay}_n{top_n}"


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _stringify_keys(metrics: Dict) -> Dict:
    return {_format_scenario_key(key): value for key, value in metrics.items()}


def _serialize_holdings_log(holdings_log: List) -> List[Dict[str, object]]:
    serialized: List[Dict[str, object]] = []
    for entry in holdings_log:
        if not isinstance(entry, tuple) or len(entry) != 2:
            continue

        entry_date, holdings = entry
        if isinstance(entry_date, date):
            entry_date = entry_date.isoformat()

        serialized.append({
            "date": entry_date,
            "holdings": holdings,
        })

    return serialized


def _load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NASDAQ100 oracle delay studies",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to scenario configuration JSON",
    )
    return parser.parse_args()


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _find_analysis_indices(
    datearray: List[date],
    analysis_start: date,
    analysis_stop: date
) -> tuple[int, int]:
    start_idx = None
    stop_idx = None

    for i, current_date in enumerate(datearray):
        if start_idx is None and current_date >= analysis_start:
            start_idx = i
        if current_date <= analysis_stop:
            stop_idx = i

    if start_idx is None or stop_idx is None or start_idx > stop_idx:
        raise ValueError(
            "Analysis date range has no overlap with available data."
        )

    return start_idx, stop_idx


if __name__ == "__main__":
    main()
