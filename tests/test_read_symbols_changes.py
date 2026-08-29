import os

import pytest

from functions.readSymbols import get_symbols_changes


@pytest.mark.parametrize(
    "json_name, symbol_file_name, stock_list",
    [
        ("sp500_hma", "SP500_Symbols.txt", ["A", "B"]),
        ("naz100_pine", "Naz100_Symbols.txt", ["A", "B"]),
    ],
)
def test_get_symbols_changes_handles_no_changes(
    monkeypatch, tmp_path, json_name, symbol_file_name, stock_list
):
    """No-change comparisons should not fail when the change files are empty."""
    symbol_dir = tmp_path / "symbols"
    symbol_dir.mkdir()
    symbol_file = symbol_dir / symbol_file_name
    company_file = symbol_dir / (
        "SP500_companyNames.txt" if "SP500" in symbol_file_name else "Naz100_companyNames.txt"
    )
    change_file = symbol_dir / (
        "SP500_symbolsChanges.txt" if "SP500" in symbol_file_name else "Naz100_symbolsChanges.txt"
    )

    symbol_file.write_text("\n".join(stock_list) + "\n")
    company_file.write_text("A;Alpha\nB;Bravo\n")
    change_file.write_text("")

    json_path = tmp_path / f"{json_name}.json"
    json_path.write_text(
        '{"Valuation": {"symbols_file": "' + str(symbol_file) + '", "stockList": "' + ("SP500" if "SP500" in symbol_file_name else "Naz100") + '", "webpage": "' + str(tmp_path / "webpage") + '", "performance_store": "' + str(tmp_path / "data_store") + '"}}'
    )

    monkeypatch.setattr(
        "functions.readSymbols.read_company_names_local",
        lambda *_args, **_kwargs: (
            ["Alpha", "Bravo"],
            stock_list,
        ),
    )
    monkeypatch.setattr(
        "functions.readSymbols.read_symbols_list_web",
        lambda *_args, **_kwargs: (
            ["Alpha", "Bravo"],
            stock_list,
        ),
    )

    result = get_symbols_changes(str(json_path), verbose=False)
    assert result[0] == stock_list
    assert result[1] == []
    assert result[2] == []
