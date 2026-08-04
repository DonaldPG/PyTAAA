"""Monte Carlo search for margin-debt SMA signal parameters.

This study optimizes a long/cash trading rule using the margin-debt-to-GDP
series as indicator input and the Buy & Hold curve from
pyTAAAweb_backtestPortfolioValue.params as the tradable benchmark.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random

import numpy as np
import pandas as pd

from macro.margin_debt import (
    _build_margin_gdp_frame,
    _compute_margin_sma_signal,
    _load_backtest_curves,
    _simulate_signal_curve,
)


def _annualized_sharpe(value_curve: pd.Series) -> float:
    """Compute annualized Sharpe ratio from a value curve."""

    if value_curve.empty or len(value_curve) < 3:
        return 0.0
    returns = (value_curve / value_curve.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    returns = returns - 1.0
    if returns.empty:
        return 0.0
    vol = float(returns.std())
    if vol <= 0.0:
        return 0.0
    return float((returns.mean() / vol) * np.sqrt(252.0))


def run_search(
    json_fn: str,
    iterations: int,
    min_sma: int,
    max_sma: int,
    min_factor: float,
    max_factor: float,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    """Run Monte Carlo search and return best configuration and all trials."""

    random.seed(seed)

    buy_hold_curve, traded_curve = _load_backtest_curves(json_fn)
    base_curve = traded_curve if not traded_curve.empty else buy_hold_curve
    frame = _build_margin_gdp_frame()
    if base_curve.empty:
        raise RuntimeError("Base curve is empty; cannot run study.")
    if frame.empty or frame["MarginDebt_GDP"].dropna().empty:
        raise RuntimeError("MarginDebt_GDP series is empty; cannot run study.")

    trials = []
    best = None

    for idx in range(iterations):
        sma_months = random.randint(min_sma, max_sma)
        rule = random.choice(["above_sma_to_cash", "above_sma_to_long"])
        sma_factor = random.uniform(min_factor, max_factor)

        _, _, signal = _compute_margin_sma_signal(
            frame,
            sma_months=sma_months,
            rule=rule,
            sma_factor=sma_factor,
        )
        strategy = _simulate_signal_curve(base_curve, signal)
        if strategy.empty:
            continue

        final_value = float(strategy.iloc[-1])
        sharpe = _annualized_sharpe(strategy)
        base_final = float(base_curve.iloc[-1])
        excess = final_value / base_final if base_final > 0.0 else 0.0

        row = {
            "trial": idx + 1,
            "sma_months": sma_months,
            "sma_factor": sma_factor,
            "rule": rule,
            "final_value": final_value,
            "base_final": base_final,
            "excess_vs_base": excess,
            "sharpe": sharpe,
        }
        trials.append(row)

        if best is None:
            best = row
        else:
            # Primary objective: final value. Tie-breakers: excess, then sharpe.
            if (
                row["final_value"] > best["final_value"]
                or (
                    row["final_value"] == best["final_value"]
                    and row["excess_vs_base"] > best["excess_vs_base"]
                )
                or (
                    row["final_value"] == best["final_value"]
                    and row["excess_vs_base"] == best["excess_vs_base"]
                    and row["sharpe"] > best["sharpe"]
                )
            ):
                best = row

    if best is None:
        raise RuntimeError("No valid trials produced strategy curves.")

    results = pd.DataFrame(trials)
    return best, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo margin debt signal search")
    parser.add_argument("--json", required=True, help="Path to method JSON config")
    parser.add_argument("--iterations", type=int, default=2000, help="Number of Monte Carlo trials")
    parser.add_argument("--min-sma", type=int, default=24, help="Minimum SMA months")
    parser.add_argument("--max-sma", type=int, default=120, help="Maximum SMA months")
    parser.add_argument("--min-factor", type=float, default=0.5, help="Minimum SMA multiplier")
    parser.add_argument("--max-factor", type=float, default=1.0, help="Maximum SMA multiplier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    best, results = run_search(
        json_fn=args.json,
        iterations=args.iterations,
        min_sma=args.min_sma,
        max_sma=args.max_sma,
        min_factor=args.min_factor,
        max_factor=args.max_factor,
        seed=args.seed,
    )

    studies_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(studies_dir, "margin_debt_signal_monte_carlo_results.csv")
    json_path = os.path.join(studies_dir, "margin_debt_signal_params.json")

    results.sort_values(
        by=["final_value", "excess_vs_base", "sharpe"],
        ascending=[False, False, False],
        inplace=True,
    )
    results.to_csv(csv_path, index=False)

    payload = {
        "generated_at": dt.datetime.now().isoformat(),
        "json_fn": args.json,
        "iterations": args.iterations,
        "min_sma": args.min_sma,
        "max_sma": args.max_sma,
        "min_factor": args.min_factor,
        "max_factor": args.max_factor,
        "seed": args.seed,
        "sma_months": int(best["sma_months"]),
        "sma_factor": float(best["sma_factor"]),
        "rule": str(best["rule"]),
        "final_value": float(best["final_value"]),
        "base_final": float(best["base_final"]),
        "excess_vs_base": float(best["excess_vs_base"]),
        "sharpe": float(best["sharpe"]),
    }
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    print("Best margin-debt signal parameters")
    print(f"  SMA months        : {payload['sma_months']}")
    print(f"  SMA factor        : {payload['sma_factor']:.4f}")
    print(f"  Rule              : {payload['rule']}")
    print(f"  Final value       : {payload['final_value']:.2f}")
    print(f"  Base final        : {payload['base_final']:.2f}")
    print(f"  Excess multiple   : {payload['excess_vs_base']:.4f}")
    print(f"  Sharpe            : {payload['sharpe']:.4f}")
    print(f"Saved trials CSV    : {csv_path}")
    print(f"Saved params JSON   : {json_path}")


if __name__ == "__main__":
    main()
