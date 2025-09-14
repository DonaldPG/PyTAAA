# Simplified Implementation Plan: Model Switching Portfolio Tracker (naz100_sp500_abacus)

## Overview

- all the scripts should use a json file for parameters
- a single json should be used. it is at /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json
- if the json file needs to be updated, it should be copied to the local codebase folder, updated, and copied back to its original location with corrections

- the data file for the  4 stock trading models are at:
/Users/donaldpg/pyTAAA_data/naz100_hma/data_store/PyTAAA_status.params
/Users/donaldpg/pyTAAA_data/naz100_pine/data_store/PyTAAA_status.params
/Users/donaldpg/pyTAAA_data/naz100_pi/data_store/PyTAAA_status.params
/Users/donaldpg/pyTAAA_data/sp500_hma/data_store/PyTAAA_status.params

- there should be 5 stock trading models:
1. naz100_pine
2. naz100_hma
3. naz100_pi
4. sp500_hma
5. cash

- multiple versions of PyTAAA_status.params exist. one per stock trading model in each data_store, one per stock trading model

model_choices in run_monte_carlo.py should be input from the json file at /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json so that it can be shared by all the model switching scripts. these scripts are:

* run_monte_carlo.py which is called by run_monte_carlo.sh
* recommend_model.py
* run_abacus_daily.py

- i expect the scripts like this to 1) search for best model switching parameters, 2) monthly model recommendation, 3) daily updates to stock values and creation of web page html files
* ./run_monte_carlo.sh 30 --explore-exploit --reset
* uv run python recommend_model.py --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json --lookbacks "139, 149, 158"
* uv run python daily_abacus_update.py --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json --verbose

 