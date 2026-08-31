"""Tests for semantic production-baseline artifact comparison."""

import threading
from pathlib import Path

import scripts.refactor_baseline_gate as baseline_gate
from scripts.refactor_baseline_gate import (
    GATE_ENVIRONMENT,
    MAX_CONCURRENT_AFTER_TESTS,
    _resource_limited_command,
    capture_params,
    compare_artifacts,
)


def _write_params(root: Path, name: str, content: str) -> None:
    params_dir = root / "params_files"
    params_dir.mkdir(parents=True, exist_ok=True)
    (params_dir / name).write_text(content)


def test_append_only_diagnostic_compares_latest_record(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    record = (
        "currently held stocks: ['CASH']\n"
        "currently held shares: [1.]\n"
        "currently held buyprice: [1.]\n"
        "currently held nowprice: [1.]\n"
        "new stock selection: ['ABC']\n"
        "new stock weight: [1.]\n"
        "new stock nowprice: [10.]\n"
    )
    _write_params(
        before,
        "PyTAAA_diagnostic.params",
        "2026-08-30 10:00:00.123456\n" + record,
    )
    _write_params(
        after,
        "PyTAAA_diagnostic.params",
        "old record\n2026-08-30 11:00:00.654321\n" + record,
    )

    assert compare_artifacts(before, after) == []


def test_latest_rank_record_detects_selection_change(tmp_path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_params(
        before,
        "PyTAAA_ranks.params",
        "lastdate: 2026-08-30\nsymbols: ABC DEF\nranks: 1 2\n",
    )
    _write_params(
        after,
        "PyTAAA_ranks.params",
        "lastdate: 2026-08-30\nsymbols: DEF ABC\nranks: 1 2\n",
    )

    assert compare_artifacts(before, after) == [
        "content_mismatch: PyTAAA_ranks.params"
    ]


def test_after_tests_use_quiet_resource_limits():
    assert MAX_CONCURRENT_AFTER_TESTS == 2
    assert GATE_ENVIRONMENT["PYTAAA_GATE_MANAGED_BACKTEST"] == "1"
    assert GATE_ENVIRONMENT["PYTAAA_SKIP_PLOTS"] == "1"
    assert GATE_ENVIRONMENT["OMP_NUM_THREADS"] == "1"
    assert GATE_ENVIRONMENT["OPENBLAS_NUM_THREADS"] == "1"


def test_after_tests_use_background_qos_when_available(monkeypatch):
    monkeypatch.setattr(
        "scripts.refactor_baseline_gate.shutil.which",
        lambda _name: "/usr/sbin/taskpolicy",
    )

    assert _resource_limited_command(["uv", "run", "python"]) == [
        "/usr/sbin/taskpolicy",
        "-b",
        "uv",
        "run",
        "python",
    ]


def test_production_model_tests_peak_at_two(tmp_path, monkeypatch):
    active = 0
    peak = 0
    lock = threading.Lock()
    model_barrier = threading.Barrier(2)

    def fake_run(command, _log_path):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        if "pytaaa_main.py" in command:
            model_barrier.wait(timeout=1)
        with lock:
            active -= 1

    monkeypatch.setattr(
        baseline_gate,
        "MODEL_CONFIGS",
        (Path("model_one.json"), Path("model_two.json")),
    )
    monkeypatch.setattr(baseline_gate, "_run_command", fake_run)

    baseline_gate.run_production_commands(tmp_path)

    assert peak == 2


def test_each_model_after_test_awaits_backtest(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, log_path):
        commands.append((command, log_path))

    config = Path("model_one.json")
    monkeypatch.setattr(baseline_gate, "_run_command", fake_run)

    baseline_gate._run_model_after_test(config, tmp_path)

    assert commands[0][0][3] == "pytaaa_main.py"
    assert commands[1][0][3:6] == [
        "-m",
        "functions.background_montecarlo_runner",
        "--json-file",
    ]
    assert commands[1][1] == tmp_path / "model_one_backtest.log"


def test_capture_preserves_model_paths(tmp_path, monkeypatch):
    static_root = tmp_path / "static"
    first = static_root / "model_one/data_store/shared.params"
    second = static_root / "model_two/data_store/shared.params"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("model one\n")
    second.write_text("model two\n")
    monkeypatch.setattr(baseline_gate, "STATIC_DATA_ROOT", static_root)

    output_dir = tmp_path / "capture"
    capture_params(output_dir)

    assert (
        output_dir / "params_files/model_one/data_store/shared.params"
    ).read_text() == "model one\n"
    assert (
        output_dir / "params_files/model_two/data_store/shared.params"
    ).read_text() == "model two\n"