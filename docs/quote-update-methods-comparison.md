# Quote Update Methods Comparison

Date: 2026-08-19

## Scope

This report compares two quote-update methods currently present in this
repository:

1. Standalone method: pytaaa_quotes_update.py
2. Integrated method: pytaaa_main.py -> run_pytaaa.py, controlled by JSON
   stock_server.quote_download_server (mapped to params.quote_server)


## Executive Summary

The two methods are not independent implementations. They share the same
core updater function, UpdateHDF_yf, for downloading and writing quotes.
The standalone script adds a larger quote-quality-control and
reconciliation workflow around that shared core.

Because pytaaa_quotes_update.py already calls fix_quotes(), and fix_quotes()
calls get_stored_quotes(), and get_stored_quotes() calls UpdateHDF_yf(),
the standalone method can be made the sole daily updater. To do that cleanly,
the in-run update in run_pytaaa.py should be disabled or made optional.


## Call Chains

### Method A: Standalone pytaaa_quotes_update.py

Call chain:

- pytaaa_quotes_update.py:start_quote_update
- get_symbols_changes(json_fn) (universe refresh for Naz100/SP500)
- fix_quotes(json_fn, _data_path, stockList)
- get_stored_quotes(json_fn, stockList)
- UpdateHDF_yf(symbol_directory, symbols_file, json_fn)

Relevant files:

- pytaaa_quotes_update.py
- functions/clean_quote_data.py
- functions/UpdateSymbols_inHDF5.py
- functions/readSymbols.py


### Method B: Integrated update in pytaaa_main.py

Call chain:

- pytaaa_main.py:main
- run_pytaaa.py:run_pytaaa
- get_symbols_changes(json_fn) (if hour <= 17 and list is Naz100/SP500)
- Hostname gate via params.quote_server (from stock_server.quote_download_server)
- UpdateHDF_yf(symbol_directory, symbols_file, json_fn)

Additional branch:

- If quote_server != current machine name, run_pytaaa.py calls
  copy_updated_quotes(json_fn) to copy HDF from remote server (FTP/SFTP path).

Relevant files:

- pytaaa_main.py
- run_pytaaa.py
- functions/config_accessors.py
- functions/ftp_quotes.py
- functions/UpdateSymbols_inHDF5.py


## Minimum Requirements Analysis

### 1) HDF name for stored stock quotes

Both methods use the same HDF naming logic through loadQuotes_fromHDF and
UpdateHDF_yf.

- Default key/list name for Naz100 symbols file: Naz100_Symbols
- Default HDF file name: Naz100_Symbols_.hdf5
- Default path: same directory as the symbols file
- Optional override: Valuation.hdf_store in JSON config

Result: both methods point to the same quote store unless hdf_store override
directs them elsewhere.


### 2) Number of days (window) for updated stock quotes to download

Core updater behavior (shared by both methods via UpdateHDF_yf):

- Existing symbols:
  - start date is based on getLastDateFromHDF5()
  - end date is tomorrow (today + 1 day)
- New symbols:
  - start date fixed to 1991-01-01
  - end date today

Important current behavior:

- getLastDateFromHDF5() currently returns yesterday unconditionally
  (the function has an unconditional return yesterday before its intended
  time-window branch).
- Therefore, the effective incremental refresh window is approximately
  yesterday through tomorrow, not a long trailing window.

Method A additional fetches:

- fix_quotes() also downloads fresh comparison data (yfinance) for quality
  checks and reconciliation; this is broader than the narrow incremental
  update and can span much larger ranges depending on first stored date.


### 3) How comparison and merging is performed

Shared core merge in UpdateHDF_yf (both methods):

- quoteupdate DataFrame built from newly downloaded quotes
- updatedquotes = quoteupdate.combine_first(quote)
  - Fresh values win where present
  - Existing HDF values are kept where fresh values are missing
- Cleanup passes run per symbol (spike cleaning, interpolation,
  beginning-fill)
- For symbols removed from current index list, fresh-period dates are
  explicitly set to NaN so stale carry-forward values do not look active
- Full table written back to HDF with append=False

Method A additional reconciliation in fix_quotes:

- Builds merged stored + fresh comparison frame across dates/symbols
- Applies case-based per-date logic:
  - both present: keep fresh
  - fresh present, stored missing: keep fresh
  - fresh missing, stored present: keep stored (with adjustment continuity)
- Performs extra large single-day change diagnostics
- Rebuilds updated DataFrame and writes HDF in place


### 4) Handling days where downloaded quotes exist but no HDF quote exists,
###    or adjusted values differ

Both methods (via UpdateHDF_yf):

- If fresh date exists and stored date missing, combine_first inserts fresh
  values for that date.
- If fresh value differs from stored value at same date, fresh value wins.

Method A adds stronger handling:

- Explicit per-date, per-symbol reconciliation logic beyond simple combine_first
- Adjustment-factor-aware update of Close vs Adj Close continuity
- Additional diagnostics for suspicious daily differences


### 5) Are methods complementary, or does pytaaa_quotes_update.py replace
###    method invoked from pytaaa_main.py?

Conclusion:

- They are complementary today, with overlap in core updating.
- pytaaa_quotes_update.py does not bypass the old core; it uses it.
- pytaaa_quotes_update.py can replace daily quote updating done in
  run_pytaaa.py, but only after handling one missing operational feature:
  the remote-copy path used when quote_server != current host.

In other words:

- For local update behavior, pytaaa_quotes_update.py is a superset.
- For remote quote-distribution behavior, run_pytaaa.py still contains logic
  not inherently performed by the standalone updater unless added.


### 6) If pytaaa_quotes_update.py is not a full replacement, how to modify it
###    to become full replacement while keeping unique features

Gaps to close:

1. Remote-copy behavior
   - Add optional mode in pytaaa_quotes_update.py:
     - If quote_server == current host: perform local update (current behavior)
     - Else: run copy_updated_quotes(json_fn), optionally skip local update
2. Host-gating policy
   - Preserve existing host policy currently embedded in run_pytaaa.py,
     but move policy decision to pytaaa_quotes_update.py
3. Scheduling/idempotence policy
   - Add optional guard to prevent duplicate same-day updates if desired
   - Keep script idempotent when called repeatedly

Improvements recommended at the same time:

1. Fix getLastDateFromHDF5() control flow by removing the unconditional
   return yesterday and using explicit intended branch logic.
2. Keep UpdateHDF_yf as shared core, but ensure fix_quotes remains the
   authoritative post-update reconciliation/QC layer.
3. Add explicit logging summary:
   - HDF path
   - date window downloaded
   - inserted date count
   - replaced value count
   - removed-symbol NaN operations


## Feature Overlap Matrix

| Capability | pytaaa_quotes_update.py | pytaaa_main-integrated |
|---|---|---|
| Symbol list refresh (Naz100/SP500) | Yes | Yes |
| Core HDF updater (UpdateHDF_yf) | Yes | Yes |
| Full reconciliation/QC pass | Yes | No |
| Large-difference diagnostics | Yes | No |
| Remote copy when not quote server | Not built-in by default | Yes |
| Hostname-gated update policy | Not built-in by default | Yes |
| Full-table HDF rewrite | Yes | Yes (through shared core) |


## Proposed Plan: Make pytaaa_quotes_update.py the Only Daily Quote/HDF Updater

Goal:

- Daily quote downloads and HDF maintenance are performed only by
  pytaaa_quotes_update.py.
- pytaaa_main.py never performs quote updates directly.

### Phase 1: Implement parity features in pytaaa_quotes_update.py

1. Add quote-server policy mode:
   - Read quote_server from JSON
   - Compare with local hostname
   - If mismatch, optionally call copy_updated_quotes(json_fn)
2. Add CLI flags:
   - --mode auto|update|copy
   - --allow-local-update-on-nonserver true|false
3. Add run summary metrics and robust exit codes

### Phase 2: Disable integrated updating in run_pytaaa.py

1. Add config flag, for example:
   - stock_server.enable_quote_update_in_run_pytaaa = false
2. Wrap existing UpdateHDF_yf call in run_pytaaa.py behind this flag
3. Default flag to false after migration

### Phase 3: Operational rollout

1. Update run_pytaaa_daily.sh ordering:
   - run pytaaa_quotes_update.py first for each universe
   - then run pytaaa_main.py jobs
2. Verify logs show no UpdateHDF_yf execution path from run_pytaaa.py
3. Validate HDF freshness and key strategy outputs for several days

### Phase 4: Cleanup and hardening

1. Remove or deprecate integrated updater path from run_pytaaa.py once stable
2. Add tests:
   - host-gate policy in standalone updater
   - remote copy path
   - insert/replace semantics
   - removed-symbol NaN behavior
3. Document final architecture in DAILY_OPERATIONS_GUIDE.md


## Risk Notes

1. Current getLastDateFromHDF5() unconditional return should be treated as
   a high-priority correctness risk for window semantics.
2. Running both methods on the same day can produce redundant rewrites and
   more variability in cleaned series.
3. If remote copy is required operationally, disabling run_pytaaa.py updates
   before adding parity to pytaaa_quotes_update.py may break some nodes.


## Recommendation

Use pytaaa_quotes_update.py as the single quote/HDF maintenance entry point,
because it already includes the shared updater plus the stronger QC and
reconciliation layer. Keep run_pytaaa.py focused on analytics and portfolio
generation, not quote state mutation.
