

cd /Users/donaldpg/PyProjects/PyTAAA.master
uv run pytaaa_quotes_update.py --json /Users/donaldpg/pyTAAA_data/naz100_pine/pytaaa_naz100_pine.json 2>&1 | tee naz100_quote_update.log
uv run pytaaa_quotes_update.py --json /Users/donaldpg/pyTAAA_data/sp500_hma/pytaaa_sp500_hma.json 2>&1 | tee sp500_quote_update.log


uv run pytaaa_main.py --json /Users/donaldpg/pyTAAA_data/naz100_pine/pytaaa_naz100_pine.json 2>&1 | tee pytaaa_naz100_pine.3.log
uv run pytaaa_main.py --json /Users/donaldpg/pyTAAA_data/naz100_hma/pytaaa_naz100_hma.json 2>&1 | tee pytaaa_naz100_hma.log
uv run pytaaa_main.py --json /Users/donaldpg/pyTAAA_data/naz100_pi/pytaaa_naz100_pi.json 2>&1 | tee pytaaa_naz100_pi.log
uv run pytaaa_main.py --json /Users/donaldpg/pytaaa_data/sp500_hma/pytaaa_sp500_hma.json 2>&1 | tee pytaaa_sp500.log


cd /Users/donaldpg/PyProjects/worktree2/PyTAAA
#./run_monte_carlo.sh 30 explore-exploit --reset
#./run_monte_carlo.sh 10 explore-exploit --reset --randomize --json=abacus_combined_PyTAAA_status.params.json
./run_monte_carlo.sh 30 explore-exploit --reset --randomize \
    --json=abacus_combined_PyTAAA_status.params.json \
    --fp-year-min=1995 --fp-year-max=2021 --fp-duration=5

cp abacus_combined_PyTAAA_status.params.json abacus_combined_PyTAAA_status.test_params.json
uv run python update_json_from_csv.py \
--xlsx abacus_best_performers_4.xlsx \
--row 19 \
--json abacus_combined_PyTAAA_status.test_params.json
      
cp abacus_combined_PyTAAA_status.test_params.json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json       
uv run python recommend_model.py \
	--json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json     



#cd /Users/donaldpg/PyProjects/worktree/PyTAAA.master

#uv run python recommend_model.py --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json --verbose
#uv run python recommend_model.py --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_model_switching_params.json --verbose

uv run python daily_abacus_update.py \
--json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json \
--verbose



pytaaa_quotes_update.py
pytaaa_main.py
clean_SP500_data.py
re-generateHDF5.py
PyTAAA_backtest_filtered.py
