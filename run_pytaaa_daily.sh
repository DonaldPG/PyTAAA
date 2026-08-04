
# update hdf5 files with lastest quotes from web
cd /Users/donaldpg/PyProjects/worktree2/PyTAAA

# update hdf5 files with lastest quotes from web
uv run pytaaa_quotes_update.py \
--json /Users/donaldpg/pyTAAA_data/naz100_pine/pytaaa_naz100_pine.json 2>&1 | tee logs/snaz100_quote_update.log
uv run pytaaa_quotes_update.py \
--json /Users/donaldpg/pyTAAA_data/sp500_hma/pytaaa_sp500_hma.json 2>&1 | tee logs/sp500_quote_update.log

# worktree2 daily rebuild of files, plots, web pages
uv run pytaaa_main.py \
    --json /Users/donaldpg/pyTAAA_data/naz100_pine/pytaaa_naz100_pine.json 2>&1 | tee logs/pytaaa_naz100_pine.wt2-02.log&
uv run pytaaa_main.py \
    --json /Users/donaldpg/pyTAAA_data/naz100_hma/pytaaa_naz100_hma.json 2>&1 | tee logs/pytaaa_naz100_hma.wt2-02.log&
uv run pytaaa_main.py \
    --json /Users/donaldpg/pyTAAA_data/naz100_pi/pytaaa_naz100_pi.json 2>&1 | tee logs/pytaaa_naz100_pi.wt2-02.log&
uv run pytaaa_main.py \
    --json /Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json 2>&1 | tee logs/pytaaa_sp500_pine.wt2-02.log&
uv run pytaaa_main.py \
    --json /Users/donaldpg/pytaaa_data/sp500_hma/pytaaa_sp500_hma.json 2>&1 | tee logs/pytaaa_sp500_hma.wt2-02.log


# worktree2 abacus daily rebuild of files, plots, web pages 
uv run python daily_abacus_update.py \
--json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json \
--verbose 2>&1 | tee pytaaa_naz100_sp500_abacus.wt2.log

# update model-switching web pagesize
cd /Users/donaldpg/PyProjects/pytaaa_web
sh ./start_pytaaa_web_db.sh

# return to original folder
cd /Users/donaldpg/PyProjects/worktree2/PyTAAA