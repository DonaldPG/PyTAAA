"""Tests for semantic production-baseline artifact comparison."""

from pathlib import Path

from scripts.refactor_baseline_gate import compare_artifacts


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