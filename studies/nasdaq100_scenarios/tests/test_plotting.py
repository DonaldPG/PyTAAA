"""Unit tests for plotting utilities."""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from studies.nasdaq100_scenarios.plotting import (
    generate_summary_json,
    plot_parameter_sensitivity_panels,
    plot_portfolio_histories,
)


def _build_scenario_results(num_days: int = 30):
    base = np.linspace(10000.0, 11000.0, num_days)
    alt = np.linspace(10000.0, 10500.0, num_days)

    return {
        (5, 0, 2): {
            "portfolio_value": base,
            "final_value": float(base[-1]),
            "total_return": (base[-1] / base[0]) - 1.0,
        },
        (5, 5, 2): {
            "portfolio_value": alt,
            "final_value": float(alt[-1]),
            "total_return": (alt[-1] / alt[0]) - 1.0,
        },
        ("baseline", 0, 0): {
            "portfolio_value": base,
            "final_value": float(base[-1]),
            "total_return": (base[-1] / base[0]) - 1.0,
        },
    }


def _build_datearray(num_days: int = 30):
    start = date(2020, 1, 1)
    return [start + timedelta(days=i) for i in range(num_days)]


def test_plot_portfolio_histories(tmp_path):
    scenario_results = _build_scenario_results()
    datearray = _build_datearray()
    output_path = tmp_path / "portfolio_plot.png"

    saved_path = plot_portfolio_histories(
        scenario_results=scenario_results,
        datearray=datearray,
        output_path=str(output_path),
        title="Test Plot",
        log_scale=False,
    )

    assert Path(saved_path).exists()
    assert Path(saved_path).stat().st_size > 0


def test_generate_summary_json(tmp_path):
    scenario_results = _build_scenario_results()
    datearray = _build_datearray()
    output_path = tmp_path / "summary.json"

    saved_path = generate_summary_json(
        scenario_results=scenario_results,
        datearray=datearray,
        output_path=str(output_path),
    )

    assert Path(saved_path).exists()
    with open(saved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "w5_d0_n2" in data
    assert "baseline" in data
    assert "cagr" in data["w5_d0_n2"]


def test_plot_parameter_sensitivity_panels(tmp_path):
    scenario_results = _build_scenario_results()
    output_path = tmp_path / "sensitivity.png"

    saved_path = plot_parameter_sensitivity_panels(
        scenario_results=scenario_results,
        output_path=str(output_path),
        metric="total_return",
    )

    assert Path(saved_path).exists()
    assert Path(saved_path).stat().st_size > 0
