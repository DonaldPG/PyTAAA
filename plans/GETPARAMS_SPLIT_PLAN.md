# GetParams.py Split Plan
## Decompose 918-LOC Monolith into Three Focused Modules

**Created:** 2026-03-02
**Target files:** `functions/GetParams.py` (742 LOC after Phase A–I)
**New files:** `functions/config_loader.py`, `functions/config_validators.py`,
`functions/config_accessors.py`
**Risk:** Low — `GetParams.py` becomes a backward-compat re-export shim;
no call sites change.

---

## Motivation

`functions/GetParams.py` does three unrelated things:

1. **Raw file I/O** — opens INI files, reads status `.params` files,
   writes status back to disk
2. **Validation** — checks that model file paths exist
3. **Typed config access** — extracts named values from the JSON config
   cache, applies type coercions, and constructs the large `params` dict

This makes it hard to test, hard to reason about, and hard to extend.
The `config_cache` (Phase F) already handles the actual JSON parsing; the
accessor layer on top of it should be thin and independently testable.

---

## Function Inventory

| Function | Lines | Category | Notes |
|---|---|---|---|
| `from_config_file()` | 38–51 | **loader** | Legacy INI reader |
| `get_symbols_file()` | 53–90 | **accessor** | Uses config_cache |
| `get_performance_store()` | 92–130 | **accessor** | Uses config_cache |
| `get_webpage_store()` | 132–164 | **accessor** | Uses config_cache |
| `get_web_output_dir()` | 166–181 | **accessor** | Uses config_cache |
| `get_central_std_values()` | 183–222 | **accessor** | Uses config_cache |
| `get_json_ftp_params()` | 224–252 | **accessor** | Uses config_cache |
| `get_holdings()` | 254–296 | **accessor** | Reads `.params` files |
| `get_json_params()` | 300–454 | **accessor** | Main params dict builder |
| `get_json_status()` | 456–475 | **accessor** | Reads status `.params` |
| `compute_long_hold_signal()` | 477–570 | **accessor** | Reads + computes from status |
| `get_status()` | 572–590 | **accessor** | Duplicate of `get_json_status` |
| `put_status()` | 592–649 | **loader** | Writes to status `.params` |
| `GetIP()` | 651–664 | **utility** | HTTP call for external IP |
| `GetEdition()` | 666–680 | **utility** | Platform detection |
| `parse_pytaaa_status()` | 682–714 | **loader** | Parses status file line-by-line |
| `validate_model_choices()` | 716–742 | **validator** | Checks model paths exist |

---

## Target Module Assignments

### `functions/config_loader.py`
Raw file I/O — opens files and parses formats without business logic.

```
from_config_file()       # INI → ConfigParser
parse_pytaaa_status()    # status .params → (dates, values) lists
put_status()             # write cumulative value back to status .params
```

**Dependencies:** `os`, `configparser`, `datetime`,
`functions.GetParams.get_performance_store` (for path resolution),
`functions.TAfunctions.dpgchannel`, `SMA` (used inside `put_status` via
`compute_long_hold_signal` — see note below).

> **Note on `put_status`:** it currently calls `compute_long_hold_signal()`
> (which itself reads `TAfunctions`). This signal computation should be kept
> in `config_accessors.py`; `put_status` should accept the pre-computed
> `(traded_values, last_signal)` as parameters, or call through the accessor.
> Split the function — let `config_loader.put_status` do only the file write,
> and have `config_accessors.put_status` be the public entry point that
> resolves the signal first and then delegates to the loader.

### `functions/config_validators.py`
Validation logic — checks config values are internally consistent and
that referenced files/paths exist.

```
validate_model_choices()   # dict[model_name, path] → dict[model_name, bool]
```

Future additions (not in current codebase but natural here):
- `validate_required_keys(config, required_keys)` — raise `KeyError` with
  a clear message listing which keys are missing
- `validate_performance_store(json_fn)` — assert the `performance_store`
  directory and expected `.params` files all exist before pipeline starts

**Dependencies:** `os`

### `functions/config_accessors.py`
Typed getters — all functions that accept a `json_fn` path and return
a typed value from the JSON config cache. No raw file parsing here.

```
get_json_params()           # main params dict (most-used function)
get_json_ftp_params()       # FTP connection params dict
get_symbols_file()          # str: path to symbols list file
get_performance_store()     # str: path to performance store directory
get_webpage_store()         # str: path to webpage output directory
get_web_output_dir()        # str: path to web output directory
get_central_std_values()    # dict: normalization central/std values
get_holdings()              # dict: current portfolio holdings
get_json_status()           # str: last cumulative portfolio value
get_status()                # str: same (kept as alias; deprecate later)
compute_long_hold_signal()  # tuple: signal computation from status history
put_status()                # public entry point: compute signal + write
GetIP()                     # str: external IP address (utility, move last)
GetEdition()                # str: platform edition string (utility)
```

**Dependencies:** `os`, `numpy`, `configparser`, `json`, `datetime`,
`functions.config_cache.config_cache`,
`functions.config_loader.{from_config_file, parse_pytaaa_status}`,
`functions.TAfunctions.{dpgchannel, SMA}`

### `functions/GetParams.py` (becomes a shim)
Keep the file but replace the implementation with explicit re-exports.
All 36+ call sites continue to work with **zero changes**.

```python
"""Backward-compatible re-exports from the split config modules.

All names previously in this module are still importable here.
New code should import from the specific submodule directly:
    from functions.config_accessors import get_json_params
    from functions.config_validators import validate_model_choices
    from functions.config_loader import parse_pytaaa_status
"""
from functions.config_loader import (
    from_config_file,
    parse_pytaaa_status,
)
from functions.config_validators import validate_model_choices
from functions.config_accessors import (
    get_json_params,
    get_json_ftp_params,
    get_symbols_file,
    get_performance_store,
    get_webpage_store,
    get_web_output_dir,
    get_central_std_values,
    get_holdings,
    get_json_status,
    get_status,
    compute_long_hold_signal,
    put_status,
    GetIP,
    GetEdition,
)

__all__ = [
    "from_config_file",
    "parse_pytaaa_status",
    "validate_model_choices",
    "get_json_params",
    "get_json_ftp_params",
    "get_symbols_file",
    "get_performance_store",
    "get_webpage_store",
    "get_web_output_dir",
    "get_central_std_values",
    "get_holdings",
    "get_json_status",
    "get_status",
    "compute_long_hold_signal",
    "put_status",
    "GetIP",
    "GetEdition",
]
```

---

## Call Site Impact

**Zero changes required.** All 36 import sites continue to work through
the shim:

```
functions/MakeValuePlot.py          get_json_params, get_webpage_store,
                                    get_web_output_dir, get_performance_store
functions/TAfunctions.py            get_webpage_store, get_performance_store,
                                    get_json_params, get_holdings, GetEdition
functions/WriteWebPage_pi.py        get_json_ftp_params, get_webpage_store,
                                    get_json_params, get_symbols_file,
                                    get_web_output_dir
functions/PortfolioPerformanceCalcs.py  get_webpage_store
functions/output_generators.py      get_json_params
functions/dailyBacktest.py          get_json_params, get_performance_store
functions/readSymbols.py            get_symbols_file (×5 lazy imports)
functions/quotes_for_list_adjClose.py   get_symbols_file
functions/CountNewHighsLows.py      get_json_params, get_symbols_file,
                                    get_webpage_store
functions/stock_cluster.py          get_json_params, get_symbols_file
functions/graph_plots.py            get_webpage_store
functions/make_stock_xcorr_network_plots.py  get_holdings, get_symbols_file
functions/backtesting/core_backtest.py   get_json_params, get_performance_store,
                                         get_webpage_store
functions/backtesting/monte_carlo_runner.py  get_json_params
pytaaa_backtest_montecarlo.py       (indirect)
run_normalized_score_history.py     get_json_params
scripts/extract_montecarlo_ranges.py    get_json_params
PyTAAA_backtest_sp500_pine_refactored.py  multiple
studies/lookahead_bias/*.py         get_json_params, get_symbols_file
tests/test_lookahead_bias.py        get_symbols_file
```

**No future-new code** should `from functions.GetParams import …`; the
shim docstring directs them to the specific submodule.

---

## Implementation Sequence

### Step 1 — Create `config_loader.py`

Copy `from_config_file`, `parse_pytaaa_status`, and the **file-write
half** of `put_status` (renaming it `_write_status_line`) into a new
file. Add Google-style docstrings; import only `os`, `configparser`,
`datetime`.

### Step 2 — Create `config_validators.py`

Copy `validate_model_choices` into a new file. Import only `os`.
Add stub for `validate_required_keys` (future-facing, not wired up yet).

### Step 3 — Create `config_accessors.py`

Move all remaining functions. The public `put_status` calls
`compute_long_hold_signal` (already in this file) and then calls
`config_loader._write_status_line` for the actual disk write.

Resolve the circular dependency risk: `config_accessors` imports from
`config_loader`, not vice versa. `config_loader` must not import from
`config_accessors`.

### Step 4 — Gut and rewrite `GetParams.py` as the shim

Replace the entire implementation with the re-export block shown above.
Keep the existing module docstring, updating it to document the split.

### Step 5 — Add tests

Create `tests/test_config_split.py`:

```
test_loader_from_config_file_returns_configparser
test_loader_parse_pytaaa_status_extracts_values
test_validator_validate_model_choices_existing_path
test_validator_validate_model_choices_missing_path
test_accessor_get_performance_store_reads_cache
test_accessor_get_symbols_file_reads_cache
test_getparams_shim_re_exports_all_names   # import each name from GetParams
test_getparams_shim_is_only_imports        # AST: GetParams.py has no def/class
```

The last test (`shim_is_only_imports`) is the key regression guard: it
parses `GetParams.py` with `ast` and asserts there are zero `FunctionDef`
or `ClassDef` nodes — only `ImportFrom` statements and the module
docstring.

---

## Acceptance Criteria

- [ ] `functions/config_loader.py` exists; contains exactly:
      `from_config_file`, `parse_pytaaa_status`, `_write_status_line`
- [ ] `functions/config_validators.py` exists; contains exactly:
      `validate_model_choices`
- [ ] `functions/config_accessors.py` exists; contains all remaining
      functions (see list above)
- [ ] `functions/GetParams.py` contains **no `def` or `class`
      statements** — only imports and `__all__`
- [ ] `from functions.GetParams import get_json_params` still works
      (shim re-export)
- [ ] All existing tests pass without modification
- [ ] 8 new tests in `tests/test_config_split.py` pass
- [ ] No circular import (`python -c "import functions.config_accessors"`)

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `put_status` depends on `compute_long_hold_signal` which depends on `TAfunctions` — potential import cycle | Keep `compute_long_hold_signal` in `config_accessors`; `config_loader._write_status_line` has no `TAfunctions` dependency |
| `get_holdings` reads `.params` files (raw I/O) but belongs semantically with accessors | Acceptable — it converts raw INI data into a typed dict; same pattern as `get_json_params` |
| `GetStatus` vs `get_json_status` duplicate | Keep both in shim for backward compat; add `# deprecated: use get_json_status` comment |
| `GetIP()` makes a live HTTP call — hard to unit-test | No test for `GetIP`; document as untestable utility; move to `config_accessors` and leave as-is |

---

## Commit Strategy

Single commit after all 4 files are correct and tests pass:

```
refactor(config): split GetParams.py monolith into three focused modules

- Create functions/config_loader.py (file I/O: from_config_file,
  parse_pytaaa_status, _write_status_line)
- Create functions/config_validators.py (validate_model_choices)
- Create functions/config_accessors.py (all 14 typed getters and
  compute functions)
- Reduce functions/GetParams.py to a backward-compat re-export shim
  with no function definitions
- Add tests/test_config_split.py (8 tests including AST guard)
- No call sites changed; all imports still work via shim

Closes #<issue>
```

---

## Recommended Mode

**VS Code Implement** — this is an architectural restructure requiring
simultaneous reasoning about import graphs and circular dependency
avoidance. The `put_status` / `compute_long_hold_signal` entanglement
makes it unsuitable for a GitHub Copilot agent without very precise
instructions.

If issuing as a GitHub Issue anyway, the spec must explicitly state:
- `config_loader.py` must NOT import from `config_accessors.py`
- `config_accessors.py` imports from `config_loader.py` and
  `config_cache.py` only
- `GetParams.py` must contain zero `def` or `class` statements
  (verified by AST test)
