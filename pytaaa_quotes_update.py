import click

import os
from functions.GetParams import get_symbols_file, get_json_params
from functions.clean_quote_data import fix_quotes
from functions.readSymbols import get_symbols_changes


def start_quote_update(json_fn):
    # Placeholder function. Replace this with your actual function implementation.

    # get data file path
    _data_store = get_symbols_file(json_fn)
    _data_path = os.path.abspath(os.path.join(_data_store, "..", ".."))
    params = get_json_params(json_fn)
    stock_list = params["stockList"]

    print(f'Running quote_update with {_data_path}\nStock list: {stock_list}')

    # Refresh only the universe explicitly referenced by this JSON.
    symbols_file_lower = _data_store.lower()
    expected_token = {
        "Naz100": "naz100",
        "SP500": "sp500",
    }.get(stock_list)
    if expected_token and expected_token in symbols_file_lower:
        _, removed_tickers, added_tickers = get_symbols_changes(json_fn)
        print(
            "Updated symbols list before quote refresh: "
            f"stock_list={stock_list}, "
            f"added={len(added_tickers)}, removed={len(removed_tickers)}"
        )
    else:
        print(
            "Skipped symbols list refresh: "
            f"stock_list={stock_list}, symbols_file={_data_store}"
        )

    ### --------------------------------------
    ### clean quotes stored locally
    ### --------------------------------------

    fix_quotes(json_fn, _data_path, stockList=stock_list)

    return


@click.command()
@click.option(
	'--json', 'json_fn',
	type=click.Path(exists=True),
	help='Path to the JSON file with PyTAAA parameters'
)


def main(json_fn):
    if json_fn:
        start_quote_update(json_fn)
    else:
        click.echo('Please specify a JSON file with PyTAAA parameters using the --json tag.')


if __name__ == '__main__':
    main()