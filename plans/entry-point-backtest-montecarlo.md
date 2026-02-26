# Entry-Point Monte Carlo Backtest Implementation Plan

**Feature**: JSON-Driven Monte Carlo Backtest CLI Tool  
**Branch**: `copilot/entry-point-backtest-montecarloV2`  
**Base Branch**: `main`  
**Created**: February 25, 2026  
**Status**: Planning

---

## Table of Contents

1. [GitHub Agent Implementation Guide](#github-agent-implementation-guide) ⭐
2. [Background & Context](#background--context)
3. [Problem Statement](#problem-statement)
4. [Proposed Solution](#proposed-solution)
5. [Success Criteria](#success-criteria)
6. [Technical Architecture](#technical-architecture)
7. [Phased Implementation](#phased-implementation)
8. [Testing Strategy](#testing-strategy)
9. [Code Review Checkpoints](#code-review-checkpoints)
10. [Risk Mitigation](#risk-mitigation)
11. [Decisions & Clarifications](#decisions--clarifications)

---

## GitHub Agent Implementation Guide

### 🤖 What the Agent Can Do (85% Complete Automation)

**Full Implementation:** GitHub Copilot agent can implement **all refactoring code**:
- ✅ Create feature branch and new file structure
- ✅ Extract and implement CLI interface using click
- ✅ Implement helper functions for path extraction
- ✅ Replace hardcoded paths with JSON lookups
- ✅ Refactor file output logic for dynamic naming
- ✅ Add comprehensive type annotations
- ✅ Write detailed docstrings for all functions
- ✅ Add inline comments explaining complex logic
- ✅ Update JSON configuration templates

**Testing:** Agent can implement and run **23 out of 28 tests (82%)**:
- ✅ All unit tests for new helper functions
- ✅ CLI argument parsing tests
- ✅ Path extraction and model ID tests
- ✅ Mock-based integration tests
- ✅ Configuration validation tests
- ✅ Regression tests with test fixtures

**Deliverables:** Agent creates production-ready PR including:
- ✅ New script: `pytaaa_backtest_montecarlo.py`
- ✅ New module: `functions/backtesting/` (5 files)
- ✅ Comprehensive test suite (23 tests passing)
- ✅ Updated README.md with new entry point documentation
- ✅ Full documentation with usage examples
- ✅ Updated JSON configuration templates
- ✅ Session summary in `docs/copilot_sessions/`
- ✅ Commit messages following conventions

### 🏠 What Requires Local Execution (5 Tests, 18%)

**Local-Only Tests** (requires production environment):
1. **E2E-01**: Full Monte Carlo backtest with `pytaaa_sp500_pine_montecarlo.json` (12+ trials)
2. **E2E-02**: Full backtest with different config (e.g., sp500_hma or naz100_hma)
3. **E2E-03**: Verify CSV output format matches original script
4. **E2E-04**: Verify PNG plots are identical to original
5. **E2E-05**: Verify exported optimized JSON parameters are correct

**Why Local-Only?**
- Requires access to production HDF5 quote files (`pyTAAA_data/*.h5`)
- Needs actual symbol files and full historical data
- Monte Carlo trials take 5-10 minutes with real data
- Output validation requires visual inspection of plots
- Performance benchmarks need realistic execution time

**Important:** Agent implements test stubs/fixtures that you execute locally after PR review.

### 🎯 Recommended Workflow

1. **Create GitHub Issue** with this plan (link to `plans/entry-point-backtest-montecarlo.md`)
2. **Assign to GitHub Copilot Agent**
3. **Agent Implements**:
   - All phases (0-5)
   - Complete test suite
   - Documentation
   - Opens PR with changes
4. **You Review PR**:
   - Code review checkpoints
   - Test coverage
   - Documentation accuracy
   - Type annotations present
5. **You Run Local Tests**:
   - Pull PR branch: `git checkout feature/entry-point-backtest-montecarlo`
   - Run: `uv run python pytaaa_backtest_montecarlo.py --json <config> --trials 3`
   - Verify outputs in `performance_store/pngs/`
   - Compare with original script outputs
6. **Merge if Passing**: All tests green, outputs validated

**Estimated Time:**
- Agent implementation: 6-8 hours (autonomous work)
- Your review: 1 hour
- Your local testing: 0.5 hour (mostly waiting for Monte Carlo)
- **Total human time: 1.5 hours**

### 📋 Test Distribution Summary

| Category | 🤖 Agent | 🏠 Local | Total |
|----------|----------|----------|-------|
| Unit Tests | 12 | 0 | 12 |
| Integration Tests | 6 | 2 | 8 |
| E2E Tests | 0 | 3 | 3 |
| Regression Tests | 5 | 0 | 5 |
| **TOTAL** | **23** | **5** | **28** |

---

## Background & Context

### Current State

The project has **PyTAAA_backtest_sp500_pine_refactored.py** with:
- **3100+ lines** of Monte Carlo backtest implementation
- **Hardcoded configuration** for sp500_pine model only
- **Hardcoded file paths** in `FilePathConfig` class
- **argparse CLI** with limited options (only `--trials`)
- **Platform-specific trial counts** (pi: 12, MacOS: 13, Windows64: 15, etc.)
- ** Comprehensive backtest logic** with:
  - Signal generation (percentile channels)
  - Monte Carlo parameter exploration
  - Portfolio performance calculations
  - Sharpe ratio, drawdown, CAGR metrics
  - PNG plot generation
  - CSV results export
  - Optimized parameter export

**Current Limitations:**
- Cannot be used for other models (naz100_hma, naz100_pine, sp500_hma, etc.)
- All paths hardcoded to `/Users/donaldpg/pyTAAA_data/sp500_pine/`
- JSON config file hardcoded in script
- Model identifier "sp500_pine" hardcoded in filenames
- Not reusable across different trading strategies

### Business Need

Create a **reusable Monte Carlo backtest tool** that:
1. Works with **any trading model** (sp500_pine, sp500_hma, naz100_hma, etc.)
2. Reads **all configuration from JSON** (no hardcoded paths)
3. Uses **click CLI** for consistency with other tools
4. Dynamically generates **model-specific output filenames with timestamps**
5. Supports **trial count override** via CLI
6. **Refactors logic into reusable functions** (new `functions/backtesting/` module)
7. **Maximizes reuse** of existing functions from `functions/` folder
8. Follows **project conventions** (type hints, docstrings, logging)

### Prior Art

The codebase has several JSON-driven CLI tools:
- `pytaaa_main.py` - Main entry point with click CLI
- `recommend_model.py` - Model recommendation tool with click CLI
- `daily_abacus_update.py` - Daily portfolio update with click CLI
- All use `get_json_params()`, `get_performance_store()`, `get_webpage_store()`
- All follow consistent click interface with `--json` option

**Existing Functions to Reuse:**
- `functions/ta/` - Technical analysis (SMA, HMA, channels, signals)
- `functions/PortfolioPerformanceCalcs.py` - Performance calculations
- `functions/PortfolioMetrics.py` - Sharpe, drawdown, CAGR metrics
- `functions/MakeValuePlot.py` - Chart generation
- `functions/data_loaders.py` - Quote loading
- `functions/GetParams.py` - Configuration management

**This implementation follows the same patterns** proven successful in existing tools.

---

## Problem Statement

**Primary Goal**: Refactor `PyTAAA_backtest_sp500_pine_refactored.py` into a reusable, JSON-driven CLI tool with modular architecture.

**Secondary Goals**:
1. **Extract logic into `functions/backtesting/` module** (new reusable functions)
2. **Maximize reuse** of existing functions from `functions/` folder
3. Create thin CLI entry point that orchestrates, doesn't implement
4. Replace all hardcoded paths with JSON configuration lookups
5. Extract model identifier from path (e.g., `.../sp500_pine/webpage` → `sp500_pine`)
6. Use click CLI for consistency with other project tools
7. Support Monte Carlo trial count from JSON with CLI override
8. Add timestamps to output filenames (prevent overwrites)
9. Add comprehensive type annotations and docstrings
10. Follow project code style and conventions

**Constraints**:
- **No changes to existing codebase** (only additions: new entry point + new module)
- Must preserve all existing backtest logic (no algorithmic changes)
- Must generate outputs matching `pyTAAAweb_backtestPortfolioValue.params` from `pytaaa_main.py`
- Must work in headless environment (Agg backend)
- Must handle missing configuration gracefully
- Platform-specific trial counts must be preserved
- Must maintain performance (no slowdown)
- Original script stays untouched (reference)

**Key Architectural Changes:**
- **Structure**: Monolithic script → Modular architecture (entry point + functions/backtesting/)
- **Code Reuse**: Inline logic → Reuse existing functions/ + new backtesting/ module
- **Configuration**: Hardcoded → JSON-driven
- **CLI**: argparse (trials only) → click (json + trials + standardized exit codes)
- **Reusability**: Single model → All models
- **Paths**: Static → Dynamic
- **File naming**: Hardcoded prefix → Model-specific prefix with timestamp

---

│ ~200 lines - Thin orchestration layer                           │
├─────────────────────────────────────────────────────────────────┤
│ 1. Parse CLI arguments (--json, --trials)                       │
│ 2. Load configuration from JSON                                 │
│ 3. Extract model identifier from webpage path                   │
│ 4. Set up dynamic output paths with timestamps                  │
│ 5. Call Monte Carlo runner from functions/backtesting/          │
│ 6. Exit with standardized code (0=success, 1=error, 2=config)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ delegates to
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ functions/backtesting/ (New Module - Extracted Logic)           │
├─────────────────────────────────────────────────────────────────┤
│ monte_carlo_runner.py:                                           │
│   ├─ run_monte_carlo_backtest(json_fn, trials, output_paths)   │
│   ├─ execute_single_trial(params, trial_num)                   │
│   └─ collect_trial_results(...)                                │
│                                                                  │
│ parameter_exploration.py:                                        │
│   ├─ generate_random_parameters(base_params, trial_num)        │
│   ├─ random_triangle(low, mid, high)                           │
│   └─ validate_parameter_ranges(params)                         │
│                                                                  │
│ signal_Entry Point** (`pytaaa_backtest_montecarlo.py` ~200 lines)
   - click-based interface
   - `--json`: Path to JSON configuration (required)
   - `--trials`: Number of Monte Carlo trials (optional, overrides JSON)
   - Orchestrates: config → setup → execute → output
   - Standardized exit codes (0=success, 1=error, 2=config error, 3=data not found)
   - Minimal logic, delegates to functions/backtesting/

2. **New Backtesting Module** (`functions/backtesting/` - extracted logic)
   - `monte_carlo_runner.py` - Main execution loop, trial management
   - `parameter_exploration.py` - Random parameter generation (random_triangle, etc.)
   - `signal_backtest.py` - Signal computation and portfolio value calculation
   - `output_writers.py` - CSV/JSON export with timestamped filenames
   - `config_helpers.py` - Model ID extraction, path setup, validation
   - All functions have type hints, docstrings, unit tests

3. **Reused Existing Functions** (maximize code reuse)
   - `functions/ta/` - SMA, HMA, percentile channels, signal generation
   - `functions/PortfolioMetrics.py` - Calculate Sharpe, drawdown, CAGR
   - `functions/MakeValuePlot.py` - Plot generation (if needed)
   - `functions/data_loaders.py` - load_quotes_for_analysis()
   - `functions/GetParams.py` - get_json_params(), get_performance_store(), etc.
   - `functions/TAfunctions.py` - Ranking, weight calculations

4. **Configuration Parameters** (JSON)
   - `backtest_monte_carlo_trials` (int, default: 250)
   - All existing valuation parameters preserved
   - `performance_store`: Output directory base
   - `webpage`: Used for model identifier extraction

5. **Output File Naming** (with timestamps)
   - CSV: `{model_id}_montecarlo_{date}_{timestamp}_{runnum}.csv`
   - PNG: `{model_id}_montecarlo_{date}_{timestamp}.png`
   - JSON: `{model_id}_optimized_{date}_{timestamp}.json`
   - Prevents overwrites, enables traceability             │
│    ├─ Create plots with model-specific filenames               │
│    └─ Export CSV results and optimized params                  │
│ 6. Exit with success/error code                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **CLI Interface** (click-based)
   - `--json`: Path to JSON configuration (required)
   - `--trials`: Number of Monte Carlo trials (optional, overrides JSON)
   - Help text explaining usage

2. **Helper Functions** (new utilities)
   - `extract_model_identifier(webpage_path: str) -> str`
     - Extract "sp500_pine" from ".../sp500_pine/webpage"
   - `setup_output_paths(json_fn: str) -> tuple[str, str, str]`
     - Returns: (model_id, output_dir, performance_store)
   - `validate_configuration(params: dict) -> dict`
     - Validate required JSON parameters, set defaults

3. **Refactored Core Logic** (preserve all existing)
   - Keep all backtest functions unchanged
   - Keep TradingConstants and BacktestConfig classes (reasonable defaults)
   - Remove FilePathConfig class (replace with dynamic lookups)
   - Update file output paths to use model_id
   - Update CSV headers/filenames to use model_id

4. **Configuration Parameters** (JSON)
   - `backtest_monte_carlo_trials` (int, default: 250)
   - All existing valuation parameters preserved
   - `performance_store`: Output directory base
   - `webpage`: Used for model identifier extraction

### Architecture Diagram

```
Input: JSON Config File
  ↓
extract_model_identifier(webpage path)
  ↓
model_id = "sp500_pine", "naz100_hma", etc.
  ↓
setup_output_paths(json_fn)
  ↓
output_dir = performance_store/pngs/
  ↓
Execute Monte Carlo Backtest
  ├─> Generate plots: {model_id}_montecarlo_*.png
  ├─> Export CSV: {model_id}_montecarlo_*.csv
  └─> Export optimized JSON: {model_id}_optimized_*.json
```

---

## Success Criteria

### Functional Requirements

✅ **FR1**: Script accepts `--json` and `--trials` CLI arguments  
✅ **FR2**: Model identifier extracted correctly from webpage path  
✅ **FR3**: All file paths derived dynamically from JSON configuration  
✅ **FR4**: Output files use model-specific prefixes (not hardcoded "sp500_pine")  
✅ **FR5**: Trial count from JSON with CLI override working  
✅ **FR6**: CSV and PNG outputs identical to original script  
✅ **FR7**: Optimized parameters exported to correct location  
✅ **FR8**: Works with multiple JSON configs (sp500_pine, sp500_hma, naz100_hma)  
✅ **FR9**: Error messages are clear and actionable  
✅ **FR10**: Script creates output directories if missing  

### Non-Functional Requirements

✅ **NFR1**: All functions have type annotations  
✅ **NFR2**: All functions have comprehensive docstrings  
✅ **NFR3**: Complex logic has inline comments  
✅ **NFR4**: Code follows PEP 8 style guidelines  
✅ **NFR5**: Logging statements for key operations  
✅ **NFR6**: Graceful error handling (file not found, missing keys, etc.)  
✅ **NFR7**: Performance identical to original script  
✅ **NFR8**: Matplotlib uses Agg backend (headless compatible)  

### Performance Targets

| Metric | Original | Target (Refactored) | Measurement |
|--------|----------|---------------------|-------------|
| Monte Carlo runtime (12 trials) | ~8-10 min | ~8-10 min | Wall clock time |
| Script startup time | < 5 sec | < 5 sec | Time to first backtest |
| Memory usage | ~2-4 GB | ~2-4 GB | Peak RSS |
| Output file size | Same | Same | Byte comparison |

**Note**: No performance regression expected - only structural chan200 lines)
├── PyTAAA_backtest_sp500_pine_refactored.py  # Original (kept untouched)
├── pytaaa_generic.json                     # Add backtest_monte_carlo_trials
├── pytaaa_model_switching_params.json      # Add backtest_monte_carlo_trials
├── pytaaa_sp500_pine_montecarlo.json       # Add backtest_monte_carlo_trials
├── functions/
│   ├── backtesting/                        # NEW MODULE (extracted logic)
│   │   ├── __init__.py                    # Module exports
│   │   ├── monte_carlo_runner.py          # Main execution loop (~400 lines)
│   │   ├── parameter_exploration.py       # Random param generation (~200 lines)
│   │   ├── signal_backtest.py             # Signal computation (~300 lines)
│   │   ├── output_writers.py              # File output logic (~150 lines)
│   │   └── config_helpers.py              # Path setup, validation (~100 lines)
│   ├── ta/                                 # EXISTING (reused)
│   ├── PortfolioMetrics.py                # EXISTING (reused)
│   ├── MakeValuePlot.py                   # EXISTING (reused)
│   ├── data_loaders.py                    # EXISTING (reused)
│   ├── GetParams.py                       # EXISTING (reused)
│   └── ...
├── tests/
│   ├── test_pytaaa_backtest_montecarlo.py      # Entry point tests
│   ├── test_backtesting_monte_carlo_runner.py  # Module tests
│   ├── test_backtesting_parameter_exploration.py
│   ├── test_backtesting_signal_backtest.py
│   ├── test_backtesting_output_writers.py
│   ├── test_backtesting_config_helpers.py

```
PyTAAA/
├── pytaaa_backtest_montecarlo.py          # New CLI entry point (~3200 lines)
├── PyTAAA_backtest_sp500_pine_refactored.py  # Original (kept for reference)
├── pytaaa_generic.json                     # Add backtest_monte_carlo_trials
├── pytaaa_model_switching_params.json      # Add backtest_monte_carlo_trials
├── pytaaa_sp500_pine_montecarlo.json       # Add backtest_monte_carlo_trials
├── tests/
│   ├── test_pytaaa_backtest_montecarlo.py  # New test suite
│   └── ...
├── docs/
│   ├── copilot_sessions/
│   │   └── 2026-02-25_entry-point-backtest-montecarlo.md  # Session summary
│   └── ...
└── plans/
    └── entry-point-backtest-montecarlo.md  # This file
```

### Data Flow

```python
# Step 1: CLI Parsing
json_fn, trials_override = parse_cli()

# Step 2: Configuration Loading
params = get_json_params(json_fn)
trials = trials_override or params.get('backtest_monte_carlo_trials', 250)

# Step 3: Path Setup
webpage_path = get_webpage_store(json_fn)  # → .../sp500_pine/webpage
model_id = extract_model_identifier(webpage_path)  # → "sp500_pine"
perf_store = get_performance_store(json_fn)  # → .../sp500_pine/data_store
output_dir = os.path.join(perf_store, 'pngs')  # → .../data_store/pngs

# Step 4: File Output
plot_fn = f"{model_id}_montecarlo_{date}.png"
csv_fn = f"{model_id}_montecarlo_{date}_{runnum}.csv"
optimized_json = f"{model_id}_optimized_{date}.json"

# Step 5: Execute Backtest
# ... all existing logic with dynamic paths ...
```

### Configuration Schema

Required JSON keys:
```json
{
  "Valuation": {
    "symbols_file": "/path/to/symbols.txt",
    "performance_store": "/path/to/data_store",
    "webpage": "/path/to/model_id/webpage",
    "stockList": "SP500",
    "uptrendSignalMethod": "percentileChannels",
    "numberStocksTraded": 5,
    "monthsToHold": 1,
    "LongPeriod": 396,
    // ... all other backtest parameters ...
    "backtest_monte_carlo_trials": 250  // NEW: optional, default 250
  }
}
```

---

## Phased Implementation

### Phase 0: Branch Setup and Planning (15 minutes)

**Goal**: Create feature branch and verify base state

**Checklist**:
- [ ] Checkout `main` branch
- [ ] Pull latest changes
- [ ] Create branch Module Structure and Config Helpers (2 hours)

**Goal**: Create new module structure and configuration helper functions

**Files to Create**:
- `functions/backtesting/__init__.py`
- `functions/backtesting/config_helpers.py` (~100 lines)
- `pytaaa_backtest_montecarlo.py` (new file, ~200 lines
git checkout main
git pulreate `functions/backtesting/` directory
- [ ] Create `functions/backtesting/__init__.py` with module docstring
- [ ] Create `functions/backtesting/config_helpers.py`:
  - [ ] Implement `extract_model_identifier()` function
    - [ ] Parse path, extract second-to-last component
    - [ ] Handle edge cases (short paths, missing slashes, etc.)
    - [ ] Add type annotations and docstring
  - [ ] Implement `setup_output_paths()` function
    - [ ] Call `get_performance_store(json_fn)`
    - [ ] Call `get_webpage_store(json_fn)`
    - [ ] Extract model_id
    - [ ] Calculate output_dir
    - [ ] Add type annotations and docstring
  - [ ] Implement `generate_output_filename()` function
    - [ ] Add timestamp to filename
    - [ ] Format: `{model_id}_{type}_{date}_{timestamp}_{optional_suffix}`
    - [ ] Add type annotations and docstring
  - [ ] Implement `validate_configuration()` function
    - [ ] Check required JSON keys exist
    - [ ] Set defaults for optional parameters
    - [ ] Raise clear errors for missing required keys
    - [ ] Add type annotations and docstring
- [ ] Create skeleton `pytaaa_backtest_montecarlo.py`:
  - [ ] Module-level docstring explaining purpose
  - [ ] Import click and config_helpers
  - [ ] Placeholder main() function
  - [ ] Standardized exit codes defined
- [ ] Add unit tests for all config_helpers function
- [ ] Copy `PyTAAA_backtest_sp500_pine_refactored.py` → `pytaaa_backtest_montecarlo.py`
- [ ] Add module-level docstring explaining purpose
- [ ] Add import for `click`
- [ ] Implement `extract_model_identifier()` function
  - [ ] Parse path, extract second-to-last component
  - [ ] Handle edge cases (short paths, missing slashes, etc.)
  - [ ] Add type annotations and docstring
  - [ ] Add unit tests
- [ ] Implement `setup_output_paths()` function
  - [ ] Call `get_performance_store(json_fn)`
  - [ ] Call `get_webpage_store(json_fn)`
  - [ ] Extract model_id
  - [ ] Calculate output_dir
  - [ ] Add type annotations and docstring
  - [ ] Add unit tests
- [ ] Implement `validate_configuration()` function
  - [ ] Check required JSON keys exist
  - [ ] Set defaults for optional parameters
  - [ ] Raise clear errors for missing required keys
  - [ ] Add type annotations and docstring
  - [ ] Add unit tests

**Example Helper Functions**:
```python
def extract_model_identifier(webpage_path: str) -> str:
    """Extract model identifier from webpage path.
    
    Args:
        webpage_path: Full path to webpage directory
            (e.g., "/Users/user/pyTAAA_data/sp500_pine/webpage")
    
    Returns:
        Model identifier (e.g., "sp500_pine")
    
    Raises:
        ValueError: If path is too short or invalid format
    
    Examples:
        >>> extract_model_identifier("/Users/user/pyTAAA_data/sp500_pine/webpage")
        'sp500_pine'
        >>> extract_model_identifier("/data/naz100_hma/webpage")
        'naz100_hma'
    """
    # Normalize path and split
    parts = os.path.normpath(webpage_path).split(os.sep)
    
    # Webpage should be last component, model_id second-to-last
    if len(parts) < 2:
        raise ValueError(f"Invalid webpage path (too short): {webpage_path}")
    
    if parts[-1] != "webpage":
        raise ValueError(f"Expected 'webpage' as last component: {webpage_path}")
    
    model_id = parts[-2]
    
    if not model_id or model_id == "":
        raise ValueError(f"Empty model identifier in path: {webpage_path}")
    
    return model_id


def setup_output_paths(json_fn: str) -> tuple[str, str, str]:
    """Set up output paths from JSON configuration.
    
    Args:
        json_fn: Path to JSON configuration file
    
    Returns:
        Tuple of (model_id, output_dir, performance_store)
    
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        KeyError: If required JSON keys are missing
    
    Examples:
        >>> model_id, output_dir, perf_store = setup_output_paths("config.json")
        >>> print(model_id)
        'sp500_pine'
    """
    # Get paths from JSON
    webpage_path = get_webpage_store(json_fn)
    perf_store = get_performance_store(json_fn)
    
    # Extract model identifier
    model_id = extract_model_identifier(webpage_path)
    
    # Calculate output directory
    output_dir = os.path.join(perf_store, 'pngs')
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    return model_id, output_dir, perf_store
```

**Testing Checklist**:

- [ ] **Unit Test**: `test_extract_model_identifier_valid_path()`
- [ ] **Unit Test**: `test_extract_model_identifier_short_path_raises()`
- [ ] **Unit Test**: `test_extract_model_identifier_no_webpage_suffix_raises()`
- [ ] **Unit Test**: `test_extract_model_identifier_empty_component_raises()`
- [ ] **Unit Test**: `test_setup_output_paths_creates_directory()`
- [ ] **Unit Test**: `test_setup_output_paths_returns_correct_values()`
- [ ] **Unit Test**: `test_setup_output_paths_missing_json_raises()`

**Verification Commands**:
```bash
# Syntax check
python -m py_compile pytaaa_backtest_montecarlo.py

# Import test
python -c "from pytaaa_backtest_montecarlo import extract_model_identifier"

# Unit tests
PYTHONPATH=$(pwd) uv run pytest tests/test_pytaaa_backtest_montecarlo.py::test_extract_model_identifier -v
```

**Commit**:
```bash
git add functions/backtesting/__init__.py
git add functions/backtesting/config_helpers.py
git add pytaaa_backtest_montecarlo.py
git add tests/test_backtesting_config_helpers.py
git commit -m "feat: add backtesting module structure and config helpers

- Create functions/backtesting/ module for extracted logic
- Add extract_model_identifier() helper function
- Add setup_output_paths() helper function
- Add generate_output_filename() with timestamp support
- Add validate_configuration() helper function
- Create thin CLI entry point skeleton
- Add comprehensive docstrings and type annotations
- Add unit tests for all config helpers
- Define standardized exit codes (0/1/2/3)
"
```

**Code Review Checkpoint #1**:
- [ ] Review by: _______________ Date: ___________
- [ ] Helper functions have clear purpose and interface
- [ ] Type annotations present and correct
- [ ] Docstrings comprehensive with examples
- [ ] Error handling with clear messages
- [ ] Edge cases handled (empty paths, missing keys, etc.)
- [ ] Tests cover happy path and error cases
- [ ] No hardcoded paths in helper functions

---

### Phase 2: Implement Parameter Exploration Module (2 hours)

**Goal**: Extract random parameter generation logic into reusable module

**Files to Create**:
- `functions/backtesting/parameter_exploration.py` (~200 lines)

**Implementation Checklist**:

- [ ] Remove argparse import and logic (lines ~1387-1402)
- [ ] Add click import at top of file
- [ ] Create `main()` function with click decorators
  - [ ] `@click.command()` decorator
  - [ ] `@click.option('--json', 'json_fn', required=True, ...)`
  - [ ] `@click.option('--trials', type=int, default=None, ...)`
  - [ ] Add comprehensive help text
  - [ ] Add function docstring
- [ ] Move script execution logic into `main()` function
  - [ ] Replace script-level code (lines ~1370-1380) with function
  - [ ] Load JSON configuration using `json_fn` parameter
  - [ ] Get trial count from JSON or CLI override
  - [ ] Call helper functions to set up paths
  - [ ] Execute backtest logic
  - [ ] Handle exceptions with try/except
  - [ ] Exit with appropriate code (0=success, 1=error)
- [ ] Add `if __name__ == "__main__": main()` at end
- [ ] Update all references to `json_fn` variable
- [ ] Update comments explaining CLI usage

**Example CLI Implementation**:
```python
@click.command()
@click.option(
    '--json', 'json_fn',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help='Path to JSON configuration file with Monte Carlo parameters'
)
@click.option(
    '--trials', 'trials_override',
    type=int,
    default=None,
    help='Number of Monte Carlo trials (overrides JSON config)'
)
def main(json_fn: str, trials_override: int | None) -> None:
    """Execute Monte Carlo backtest with JSON configuration.
    
    This tool runs Monte Carlo backtests for PyTAAA trading strategies.
    It supports multiple trading models (sp500_pine, naz100_hma, etc.)
    through JSON configuration files.
    
    Examples:
        # Run with default trials from JSON
        uv run python pytaaa_backtest_montecarlo.py --json config.json
        
        # Override trial count for quick test
        uv run python pytaaa_backtest_montecarlo.py --json config.json --trials 3
        
        # Run with different model
        uv run python pytaaa_backtest_montecarlo.py --json pytaaa_sp500_hma.json
    """
    try:
        print("="*80)
        print("PyTAAA Monte Carlo Backtest")
        print("="*80)
        print(f"Configuration: {json_fn}")
        
        # Load configuration
        params = get_json_params(json_fn, verbose=True)
        
        # Get trial count (CLI override or JSON or default)
        randomtrials = trials_override or params.get('backtest_monte_carlo_trials', 250)
        print(f"Monte Carlo trials: {randomtrials}")
        
        # Set up output paths
        model_id, output_dir, perf_store = setup_output_paths(json_fn)
        print(f"Model identifier: {model_id}")
        print(f"Output directory: {output_dir}")
        
        # ... rest of backtest logic ...
        
        print("="*80)
        print("Monte Carlo backtest completed successfully")
        print("="*80)
        
    except FileNotFoundError as e:
        click.echo(f"Error: File not found - {e}", err=True)
        sys.exit(1)
    except KeyError as e:
        click.echo(f"Error: Missing configuration key - {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
```

**Testing Checklist**:

- [ ] **Unit Test**: `test_cli_help_text()`
- [ ] **Unit Test**: `test_cli_json_required()`
- [ ] **Unit Test**: `test_cli_json_must_exist()`
- [ ] **Unit Test**: `test_cli_trials_optional()`
- [ ] **Unit Test**: `test_cli_trials_must_be_int()`
- [ ] **Integration Test**: `test_main_with_valid_config()`
- [ ] **Integration Test**: `test_main_with_trials_override()`
- [ ] **Integration Test**: `test_main_missing_json_exits()`

**Verification Commands**:
```bash
# Help text
python pytaaa_backtest_montecarlo.py --help

# Test with mock JSON (integration test)
cat > test_config.json << 'EOF'
{
  "Valuation": {
    "symbols_file": "/tmp/symbols.txt",
    "performance_store": "/tmp/data_store",
    "webpage": "/tmp/sp500_test/webpage",
    "backtest_monte_carlo_trials": 3
  }
}
EOF

python pytaaa_backtest_montecarlo.py --json test_config.json --trials 2
```

**Commit**:
```bash
git add pytaaa_backtest_montecarlo.py
git add tests/test_pytaaa_backtest_montecarlo.py
git commit -m "feat: implement click CLI interface for Monte Carlo backtest

- Replace argparse with click for consistency
- Add --json (required) and --trials (optional) options
- Move script logic into main() function
- Add comprehensive help text and examples
- Add error handling with clear messages
- Support trial count from JSON or CLI override
- Exit with appropriate status codes
"
```

**Code Review Checkpoint #2**:
- [ ] Review by: _______________ Date: ___________
- [ ] Click interface follows project conventions
- [ ] Help text is clear and includes examples
- [ ] Error messages are actionable
- [ ] CLI arguments validated properly
- [ ] JSON parameter override logic correct
- [ ] Exception handling comprehensive
- [ ] Tests cover all CLI edge cases

---

### Phase 3: Implement Signal Backtest Module (2.5 hours)

**Goal**: Extract signal computation and portfolio calculations, maximize reuse of existing functions

**Files to Create**:
- `functions/backtesting/signal_backtest.py` (~300 lines)

**Implementation Checklist**:

- [ ] Remove or comment out `FilePathConfig` class (lines ~203-250)
- [ ] Update hardcoded `json_fn` assignment (line ~1378)
  - [ ] Remove: `json_fn = '/Users/.../pytaaa_sp500_pine_montecarlo.json'`
  - [ ] Use `json_fn` parameter from CLI
- [ ] Update `symbols_file` loading (lines ~1414-1420)
  - [ ] Remove hardcoded path
  - [ ] Use `params['symbols_file']` from JSON
  - [ ] Already using `get_symbols_file(json_fn)` - verify correct
- [ ] Update `outfiledir` (line ~1571)
  - [ ] Replace: `outfiledir = "/Users/.../sp500_pine/data_store/pngs"`
  - [ ] Use: `outfiledir = output_dir` (from setup_output_paths)
- [ ] Update CSV filename prefix (line ~1574)
  - [ ] Replace: `"sp500_pine_montecarlo_"`
  - [ ] Use: `f"{model_id}_montecarlo_"`
- [ ] Update plot filename references (search for "SP500-percentileChannels")
  - [ ] Line ~223: `PLOT_FILENAME_PREFIX`
  - [ ] Replace with: `f"{model_id.upper()}-{params['uptrendSignalMethod']}_montecarlo_"`
- [ ] Update `FilePathConfig` usage (if any references remain)
  - [ ] Search for `FilePathConfig.` references
  - [ ] Replace with dynamic variables
- [ ] Verify `runnum` generation (check if it's derived or needs config)
  - [ ] Line ~1446: Uses symbol file basename logic
  - [ ] Consider simplifying or making configurable
- [ ] Add directory creation for output_dir
  - [ ] `os.makedirs(output_dir, exist_ok=True)`

**Example Replacements**:
```python
# BEFORE (hardcoded):
outfiledir = "/Users/donaldpg/pyTAAA_data/sp500_pine/data_store/pngs"
outfilename = os.path.join(
    outfiledir,
    "sp500_pine_montecarlo_"+str(dateForFilename)+"_"+str(runnum)+".csv"
)

# AFTER (dynamic):
outfiledir = output_dir  # From setup_output_paths()
os.makedirs(outfiledir, exist_ok=True)
outfilename = os.path.join(
    outfiledir,
    f"{model_id}_montecarlo_{dateForFilename}_{runnum}.csv"
)
```

**Testing Checklist**:

- [ ] **Unit Test**: `test_output_paths_no_hardcoded_values()`
- [ ] **Unit Test**: `test_csv_filename_uses_model_id()`
- [ ] **Unit Test**: `test_plot_filename_uses_model_id()`
- [ ] **Integration Test**: `test_different_configs_different_outputs()` (sp500 vs naz100)
- [ ] **Regression Test**: `test_outputs_match_original_format()` (with sp500_pine config)

**Verification Commands**:
```bash
# Search for remaining hardcoded paths
grep -n "sp500_pine" pytaaa_backtest_montecarlo.py
grep -n "/Users/donaldpg" pytaaa_backtest_montecarlo.py

# Test with different configs
python pytaaa_backtest_montecarlo.py --json pytaaa_sp500_pine_montecarlo.json --trials 2
python pytaaa_backtest_montecarlo.py --json pytaaa_naz100_hma.json --trials 2

# Verify output filenames
ls -l <output_dir>/*montecarlo*
```

**Commit**:
```bash
git add pytaaa_backtest_montecarlo.py
git commit -m "refactor: replace hardcoded paths with dynamic JSON lookups

- Remove FilePathConfig class (hardcoded paths)
- Use model_id from extract_model_identifier()
- Use output_dir from setup_output_paths()
- Dynamic CSV filename with model-specific prefix
- Dynamic plot filename with model-specific prefix
- Create output directories automatically
- Support multiple model configurations
- All paths now derived from JSON
"
```

**Code Review Checkpoint #3**:
- [ ] Review by: _______________ Date: ___________
- [ ] No hardcoded paths remain in script
- [ ] Model identifier used consistently
- [ ] Output directory creation handled
- [ ] Filenames include model-specific prefixes
- [ ] JSON configuration fully utilized
- [ ] Script works with different configs
- [ ] Regression test passes with original config

---

### Phase 4: Implement Monte Carlo Runner and Output Writers (2 hours)

**Goal**: Create main execution loop and file output logic

**Files to Create**:
- `functions/backtesting/monte_carlo_runner.py` (~400 lines)
- `functions/backtesting/output_writers.py` (~150 lines)

**Implementation Checklist**:

- [ ] Add `backtest_monte_carlo_trials` to `pytaaa_generic.json`
  - [ ] Default value: 250
  - [ ] Add comment explaining usage
  - [ ] Place in Valuation section
- [ ] Add `backtest_monte_carlo_trials` to `pytaaa_model_switching_params.json`
  - [ ] Default value: 250
  - [ ] Ensures production config has parameter
- [ ] Add `backtest_monte_carlo_trials` to `pytaaa_sp500_pine_montecarlo.json`
  - [ ] Default value: 250
  - [ ] Existing config gets new parameter
- [ ] Update `export_optimized_parameters()` function (line ~368-437)
  - [ ] Modify signature: add `output_dir` parameter
  - [ ] Change default output location to `output_dir`
  - [ ] Add model_id prefix to filename
  - [ ] Update docstring
- [ ] Update call to `export_optimized_parameters()` (line ~3167)
  - [ ] Pass `output_dir` and `model_id` arguments
  - [ ] Verify output path is correct

**Example JSON Update**:
```json
{
  "Valuation": {
    "symbols_file": "/path/to/symbols.txt",
    "performance_store": "/path/to/data_store",
    "webpage": "/path/to/model_id/webpage",
    ...existing params...,
    "backtest_monte_carlo_trials": 250,
    "_comment_backtest_monte_carlo_trials": "Number of Monte Carlo trials (default: 250, pi: 12, MacOS: 13, Windows64: 15)"
  }
}
```

**Example Export Function Update**:
```python
def export_optimized_parameters(
    base_json_fn: str, 
    optimized_params: Dict, 
    output_dir: str,
    model_id: str,
    output_fn: str = None
) -> str:
    """Export optimized parameters to a new JSON configuration file.
    
    Args:
        base_json_fn: Path to base JSON configuration file
        optimized_params: Dictionary of optimized parameter values
        output_dir: Directory where JSON should be exported
        model_id: Model identifier for filename prefix
        output_fn: Optional output filename (default: auto-generated)
        
    Returns:
        Path to the exported JSON file
        
    Raises:
        IOError: If unable to write the file
    """
    import json
    
    # ... existing load logic ...
    
    # Generate output filename
    if output_fn is None:
        from datetime import date
        today = date.today()
        output_fn = os.path.join(
            output_dir,
            f"{model_id}_optimized_{today.isoformat()}.json"
        )
    
    # ... rest of export logic ...
```

**Testing Checklist**:

- [ ] **Unit Test**: `test_json_configs_have_trials_parameter()`
- [ ] **Unit Test**: `test_export_uses_output_dir()`
- [ ] **Unit Test**: `test_export_filename_includes_model_id()`
- [ ] **Integration Test**: `test_exported_json_valid_format()`

**Verification Commands**:
```bash
# Verify JSON syntax
python -m json.tool pytaaa_generic.json > /dev/null
python -m json.tool pytaaa_model_switching_params.json > /dev/null

# Test parameter loading
python -c "
from functions.GetParams import get_json_params
params = get_json_params('pytaaa_generic.json')
print(f\"Trials: {params.get('backtest_monte_carlo_trials', 'NOT FOUND')}\")"
```

**Commit**:
```bash
git add pytaaa_generic.json
git add pytaaa_model_switching_params.json
git add pytaaa_sp500_pine_montecarlo.json
git add pytaaa_backtest_montecarlo.py
git commit -m "feat: add backtest_monte_carlo_trials parameter and update export

- Add backtest_monte_carlo_trials to all JSON templates (default: 250)
- Update export_optimized_parameters() to use output_dir
- Export filenames include model identifier prefix
- Optimized params saved to performance_store/pngs/
- Document parameter usage in JSON comments
"
```

**Code Review Checkpoint #4**:
- [ ] Review by: _______________ Date: ___________
- [ ] JSON configs validated (no syntax errors)
- [ ] New parameter documented in comments
- [ ] Export function updated correctly
- [ ] Output location appropriate
- [ ] Filename includes model identifier
- [ ] All JSON templates updated consistently

---

### Phase 5: Implement CLI Entry Point and Configuration Updates (1.5 hours)

**Goal**: Complete click CLI interface and update JSON templates

**Files to Modify**:
- `pytaaa_backtest_montecarlo.py` (complete implementation)
- `pytaaa_generic.json`
- `pytaaa_model_switching_params.json`
- `pytaaa_sp500_pine_montecarlo.json`

**Implementation Checklist**:

- [ ] Add module-level docstring to script
  - [ ] Purpose and usage
  - [ ] Examples
  - [ ] Related scripts
- [ ] Add/verify type annotations for all functions
  - [ ] Parameter types
  - [ ] Return types
  - [ ] Use `from typing import ...` as needed
- [ ] Add/verify docstrings for all functions
  - [ ] Purpose
  - [ ] Args with types
  - [ ] Returns with type
  - [ ] Raises with exception types
  - [ ] Examples (where helpful)
- [ ] Add inline comments for complex logic
  - [ ] Monte Carlo parameter exploration
  - [ ] Signal generation logic
  - [ ] Performance metric calculations
  - [ ] File output sections
- [ ] Update `README.md`
  - [ ] Add section on Monte Carlo backtest tool
  - [ ] Link to documentation
  - [ ] Include usage examples
- [ ] Create session summary document
  - [ ] What was implemented
  - [ ] Key decisions made
  - [ ] Testing performed
  - [ ] Follow-up items

**Module Docstring Example**:
```python
"""Monte Carlo backtest CLI tool for PyTAAA trading strategies.

This module provides a JSON-driven command-line interface for running
Monte Carlo backtests across different trading models (sp500_pine,
naz100_hma, etc.). It replaces hardcoded paths with dynamic configuration
and supports reusable backtest execution.

Usage:
    # Run with default trials from JSON
    uv run python pytaaa_backtest_montecarlo.py --json config.json
    
    # Override trial count
    uv run python pytaaa_backtest_montecarlo.py --json config.json --trials 50
    
    # Different model
    uv run python pytaaa_backtest_montecarlo.py --json pytaaa_naz100_hma.json

Configuration:
    Requires JSON file with 'Valuation' section containing:
    - symbols_file: Path to stock symbols
    - performance_store: Output directory base
    - webpage: Model identifier extraction
    - All backtest parameters (LongPeriod, MA1, etc.)
    - backtest_monte_carlo_trials: Number of trials (default: 250)

Outputs:
    - CSV: {model_id}_montecarlo_{date}_{runnum}.csv
    - Plots: {model_id}_montecarlo_{date}.png (2 files)
    - JSON: {model_id}_optimized_{date}.json

Related:
    - PyTAAA_backtest_sp500_pine_refactored.py: Original implementation
    - daily_abacus_update.py: Daily portfolio updates
    - recommend_model.py: Model recommendations

Author: PyTAAA Development Team
Date: February 25, 2026
"""
```

**Session Summary Structure**:
```markdown
# Entry-Point Monte Carlo Backtest Implementation

## Date and Context
- **Date**: February 25, 2026
- **Branch**: feature/entry-point-backtest-montecarlo
- **Author**: GitHub Copilot Agent
- **Base**: PyTAAA_backtest_sp500_pine_refactored.py

## Problem Statement
Convert hardcoded Monte Carlo backtest script into reusable JSON-driven CLI tool.

## Solution Overview
Created pytaaa_backtest_montecarlo.py with:
- Click CLI interface (--json, --trials)
- Dynamic path extraction from JSON
- Model identifier from webpage path
- Support for multiple trading models

## Key Changes
1. **New File**: pytaaa_backtest_montecarlo.py (~3200 lines)
2. **Helper Functions**: extract_model_identifier(), setup_output_paths()
3. **CLI**: Click interface replacing argparse
4. **Configuration**: Added backtest_monte_carlo_trials to JSON templates
5. **Paths**: All hardcoded paths replaced with JSON lookups
6. **Filenames**: Dynamic prefix using model_id

## Technical Details
- Model ID extraction: Parse webpage path, use second-to-last component
- Output directory: performance_store/pngs/
- Trial count: JSON default (250) with CLI override
- Export location: Same as CSV/PNG outputs

## Testing
- 23 automated tests (unit + integration)
- 5 local E2E tests (require HDF5 data)
- All regression tests passing
- Output format matches original

## Follow-up Items
- [ ] Local testing with production configs
- [ ] Performance validation (runtime unchanged)
- [ ] Visual comparison of plots
- [ ] Verify exports with different models
```

**Testing Checklist**:

- [ ] **Code Review**: Verify all functions have type annotations
- [ ] **Code Review**: Verify all functions have docstrings
- [ ] **Code Review**: Verify complex sections have comments
- [ ] **Docs Review**: README.md updated appropriately
- [ ] **Docs Review**: Session summary complete

**Verification Commands**:
```bash
# Check for missing type annotations
python -m mypy pytaaa_backtest_montecarlo.py --strict

# Check docstring coverage
python -c "
import pytaaa_backtest_montecarlo
import inspect
funcs = [f for f in dir(pytaaa_backtest_montecarlo) if callable(getattr(pytaaa_backtest_montecarlo, f))]
for f in funcs:
    obj = getattr(pytaaa_backtest_montecarlo, f)
    if not obj.__doc__:
        print(f'Missing docstring: {f}')
"

# Generate documentation
pydoc pytaaa_backtest_montecarlo > /tmp/pytaaa_backtest_montecarlo_docs.txt
head -50 /tmp/pytaaa_backtest_montecarlo_docs.txt
```

**Commit**:
```bash
git add pytaaa_backtest_montecarlo.py
git add README.md
git add docs/copilot_sessions/2026-02-25_entry-point-backtest-montecarlo.md
git commit -m "docs: add comprehensive documentation and type annotations

- Add module-level docstring with usage examples
- Add type annotations to all functions
- Add detailed docstrings with Args/Returns/Raises
- Add inline comments for complex logic sections
- Update README with Monte Carlo backtest tool section
- Create session summary document
- Document configuration requirements
- Include examples for different use cases
"
```

**Code Review Checkpoint #5**:
- [ ] Review by: _______________ Date: ___________
- [ ] All functions have type annotations
- [ ] All functions have docstrings
- [ ] Complex logic has inline comments
- [ ] Module docstring complete with examples
- [ ] README.md updated appropriately
- [ ] Session summary created
- [ ] Documentation is clear and accurate

---

### Phase 6: Documentation and Automated Testing (2 hours)

**Goal**: Complete documentation and comprehensive test suite

**Files to Create/Modify**:
- `tests/test_pytaaa_backtest_monte carlo.py`
- `tests/test_backtesting_monte_carlo_runner.py`
- `tests/test_backtesting_parameter_exploration.py`
- `tests/test_backtesting_signal_backtest.py`
- `tests/test_backtesting_output_writers.py`
- `README.md` (update)
- `docs/copilot_sessions/2026-02-25_entry-point-backtest-montecarlo.md`

**Implementation Checklist**:

- [ ] **Unit Tests** (12 tests)
  - [ ] `test_extract_model_identifier_valid()`
  - [ ] `test_extract_model_identifier_invalid_raises()`
  - [ ] `test_setup_output_paths_creates_dir()`
  - [ ] `test_setup_output_paths_returns_tuple()`
  - [ ] `test_cli_json_required()`
  - [ ] `test_cli_json_must_exist()`
  - [ ] `test_cli_trials_optional()`
  - [ ] `test_cli_trials_validates_int()`
  - [ ] `test_csv_filename_format()`
  - [ ] `test_plot_filename_format()`
  - [ ] `test_export_output_location()`
  - [ ] `test_json_parameter_loading()`

- [ ] **Integration Tests** (6 tests)
  - [ ] `test_main_with_test_config()` (mock data)
  - [ ] `test_main_trials_override()` (mock data)
  - [ ] `test_different_models_different_outputs()` (sp500 vs naz100)
  - [ ] `test_output_directory_created()` (filesystem check)
  - [ ] `test_error_handling_missing_json()` (error case)
  - [ ] `test_error_handling_invalid_config()` (error case)

- [ ] **Regression Tests** (5 tests)
  - [ ] `test_output_format_matches_original()` (compare with old script)
  - [ ] `test_csv_columns_unchanged()` (verify headers)
  - [ ] `test_all_existing_tests_pass()` (run existing suite)
  - [ ] `test_no_performance_regression()` (timing comparison)
  - [ ] `test_matplotlib_backend_agg()` (headless check)

- [ ] **Documentation Updates**
  - [ ] Update `README.md` to include new entry point
    - [ ] Add `pytaaa_backtest_montecarlo.py` to CLI tools section
    - [ ] Document `--json` and `--trials` options
    - [ ] Add usage examples for Monte Carlo backtesting
    - [ ] Explain relationship to `PyTAAA_backtest_sp500_pine_refactored.py`
    - [ ] Document new `functions/backtesting/` module
    - [ ] Add links to detailed documentation
  - [ ] Create session summary: `docs/copilot_sessions/2026-02-25_entry-point-backtest-montecarlo.md`
    - [ ] Document problem statement and solution
    - [ ] List all files created/modified
    - [ ] Explain key architectural decisions
    - [ ] Document testing approach
    - [ ] Add examples and usage patterns

- [ ] Run all automated tests
- [ ] Document test coverage

**Example Test Structure**:
```python
"""Test suite for pytaaa_backtest_montecarlo.py."""

import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from pytaaa_backtest_montecarlo import (
    extract_model_identifier,
    setup_output_paths
)


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_extract_model_identifier_valid(self):
        """Test model ID extraction from valid path."""
        path = "/Users/user/pyTAAA_data/sp500_pine/webpage"
        assert extract_model_identifier(path) == "sp500_pine"
    
    def test_extract_model_identifier_invalid_raises(self):
        """Test error handling for invalid paths."""
        with pytest.raises(ValueError, match="too short"):
            extract_model_identifier("/webpage")
    
    def test_setup_output_paths_creates_dir(self, tmp_path):
        """Test output directory creation."""
        # ... test implementation ...


class TestCLIInterface:
    """Test click CLI interface."""
    
    def test_cli_json_required(self):
        """Test --json argument is required."""
        # ... test implementation ...


class TestIntegration:
    """Integration tests with mock data."""
    
    @patch('pytaaa_backtest_montecarlo.load_quotes_for_analysis')
    def test_main_with_test_config(self, mock_load):
        """Test main() with mock configuration."""
        # ... test implementation ...


class TestRegression:
    """Regression tests against original script."""
    
    def test_output_format_matches_original(self):
        """Verify output format unchanged."""
        # ... test implementation ...
```

**Testing Checklist**:

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All regression tests pass
- [ ] Test coverage > 80%
- [ ] No warnings or errors in test output

**Verification Commands**:
```bash
# Run all tests
PYTHONPATH=$(pwd) uv run pytest tests/test_pytaaa_backtest_montecarlo.py -v

# Check coverage
PYTHONPATH=$(pwd) uv run pytest tests/test_pytaaa_backtest_montecarlo.py --cov=pytaaa_backtest_montecarlo --cov-report=html

# Test with different configs (integration)
PYTHONPATH=$(pwd) uv run pytest tests/test_pytaaa_backtest_montecarlo.py::TestIntegration -v

# Run regression suite
PYTHONPATH=$(pwd) uv run pytest tests/test_pytaaa_backtest_montecarlo.py::TestRegression -v
```

**Commit #1 (Tests)**:
```bash
git add tests/test_pytaaa_backtest_montecarlo.py
git add tests/test_backtesting_*.py
git commit -m "test: add comprehensive test suite for Monte Carlo backtest

- Add 12 unit tests for helper functions
- Add 6 integration tests with mock data
- Add 5 regression tests vs original script
- Test coverage > 80%
- All error cases handled
- CLI interface fully tested
- Output format validation
"
```

**Commit #2 (Documentation)**:
```bash
git add README.md
git add docs/copilot_sessions/2026-02-25_entry-point-backtest-montecarlo.md
git commit -m "docs: update README and add session summary

- Update README.md with pytaaa_backtest_montecarlo.py entry point
- Document --json and --trials CLI options
- Add usage examples for Monte Carlo backtesting
- Explain new functions/backtesting/ module
- Create comprehensive session summary document
- Document architectural decisions and testing approach
"
```

**Code Review Checkpoint #6**:
- [ ] Review by: _______________ Date: ___________
- [ ] Test coverage adequate (>80%)
- [ ] All edge cases tested
- [ ] Error handling tested
- [ ] Regression tests protect against changes
- [ ] Mock usage appropriate
- [ ] Tests are maintainable
- [ ] Test names descriptive
- [ ] README.md accurately documents new entry point
- [ ] README.md examples are correct and runnable
- [ ] Session summary is comprehensive
- [ ] All documentation follows project style

---

### Phase 7: Local E2E Testing (Human Testing Step)

**Goal**: Validate with production data and configurations

**Prerequisites**:
- Access to production HDF5 data files
- Production JSON configurations
- Production symbols files

**Testing Checklist**:

- [ ] **E2E-01**: Test with sp500_pine config
  ```bash
  uv run python pytaaa_backtest_montecarlo.py \
      --json pytaaa_sp500_pine_montecarlo.json \
      --trials 3
  ```
  - [ ] Script completes without errors
  - [ ] CSV file created in correct location
  - [ ] CSV filename has correct prefix (sp500_pine_montecarlo_)
  - [ ] PNG files created (2 files)
  - [ ] Optimized JSON exported
  - [ ] Output format matches original script

- [ ] **E2E-02**: Test with sp500_hma config (if available)
  ```bash
  uv run python pytaaa_backtest_montecarlo.py \
      --json /path/to/pytaaa_sp500_hma.json \
      --trials 3
  ```
  - [ ] Different model_id extracted correctly
  - [ ] Output files have sp500_hma prefix
  - [ ] Script completes successfully

- [ ] **E2E-03**: Test with naz100_hma config (if available)
  ```bash
  uv run python pytaaa_backtest_montecarlo.py \
      --json /path/to/pytaaa_naz100_hma.json \
      --trials 3
  ```
  - [ ] Different model_id extracted correctly
  - [ ] Output files have naz100_hma prefix
  - [ ] Script completes successfully

- [ ] **E2E-04**: Full production run (12-51 trials)
  ```bash
  uv run python pytaaa_backtest_montecarlo.py \
      --json pytaaa_sp500_pine_montecarlo.json
  ```
  - [ ] Runtime ~8-10 minutes (unchanged from original)
  - [ ] All outputs correct
  - [ ] No memory leaks
  - [ ] Plots visually identical to original

- [ ] **E2E-05**: Compare outputs with original script
  ```bash
  # Run original
  python PyTAAA_backtest_sp500_pine_refactored.py --trials 3
  mv output1.csv /tmp/original_output.csv
  
  # Run new
  python pytaaa_backtest_montecarlo.py \
      --json pytaaa_sp500_pine_montecarlo.json \
      --trials 3
  mv output2.csv /tmp/new_output.csv
  
  # Compare
  diff /tmp/original_output.csv /tmp/new_output.csv
  ```
  - [ ] CSV format identical
  - [ ] Values match (within numerical precision)
  - [ ] Plots visually identical

**Documentation**:
- [ ] Record test results in session summary
- [ ] Note any discrepancies or issues
- [ ] Document performance metrics
- [ ] Take screenshots of plots (if helpful)

**Verification Commands**:
```bash
# Check output files exist
ls -lh <performance_store>/pngs/*montecarlo*

# Verify CSV format
head -20 <performance_store>/pngs/sp500_pine_montecarlo_*.csv

# Check PNG files
open <performance_store>/pngs/sp500_pine_montecarlo_*.png

# Verify JSON export
cat <performance_store>/pngs/sp500_pine_optimized_*.json | python -m json.tool
```

**Commit**: N/A (testing only, results documented in session summary)

**Code Review Checkpoint #7**:
- [ ] Review by: _______________ Date: ___________
- [ ] All E2E tests passing
- [ ] Outputs validated against original
- [ ] Performance metrics acceptable
- [ ] No regressions found
- [ ] Multiple configs tested successfully

---

## Testing Strategy

### Automated Tests (23 tests)

**Unit Tests** (12 tests):
1. `test_extract_model_identifier_valid_path()`
2. `test_extract_model_identifier_short_path_raises()`
3. `test_extract_model_identifier_no_webpage_suffix_raises()`
4. `test_extract_model_identifier_empty_component_raises()`
5. `test_setup_output_paths_creates_directory()`
6. `test_setup_output_paths_returns_correct_tuple()`
7. `test_setup_output_paths_missing_json_raises()`
8. `test_cli_json_argument_required()`
9. `test_cli_json_file_must_exist()`
10. `test_cli_trials_optional_defaults_to_json()`
11. `test_cli_trials_validates_integer_type()`
12. `test_json_parameter_loading_with_override()`

**Integration Tests** (6 tests):
1. `test_main_with_mock_config_completes()`
2. `test_main_with_trials_override_respected()`
3. `test_different_models_produce_different_filenames()`
4. `test_output_directory_automatically_created()`
5. `test_error_handling_missing_required_json_key()`
6. `test_error_handling_invalid_configuration_format()`

**Regression Tests** (5 tests):
1. `test_csv_output_format_unchanged_from_original()`
2. `test_csv_column_headers_match_original()`
3. `test_matplotlib_backend_is_agg()`
4. `test_no_performance_degradation()`
5. `test_all_existing_project_tests_still_pass()`

### Local Tests (5 tests)

**E2E Tests** (3 tests):
1. **E2E-01**: Full run with pytaaa_sp500_pine_montecarlo.json (12 trials)
2. **E2E-02**: Full run with pytaaa_sp500_hma.json (12 trials)
3. **E2E-03**: Full run with pytaaa_naz100_hma.json (12 trials)

**Validation Tests** (2 tests):
1. **VAL-01**: Compare CSV outputs with original script
2. **VAL-02**: Visual comparison of PNG plots

### Test Execution Plan

```bash
# Phase 1-6: Automated testing (agent can run)
PYTHONPATH=$(pwd) uv run pytest tests/test_pytaaa_backtest_montecarlo.py -v

# Phase 7: Local testing (human runs on production system)
# E2E-01
uv run python pytaaa_backtest_montecarlo.py \
    --json pytaaa_sp500_pine_montecarlo.json --trials 3

# E2E-02
uv run python pytaaa_backtest_montecarlo.py \
    --json /Users/donaldpg/pytaaa_data/sp500_hma/pytaaa_sp500_hma.json --trials 3

# E2E-03
uv run python pytaaa_backtest_montecarlo.py \
    --json /Users/donaldpg/pytaaa_data/naz100_hma/pytaaa_naz100_hma.json --trials 3

# VAL-01: Compare outputs
python PyTAAA_backtest_sp500_pine_refactored.py --trials 3  # Original
python pytaaa_backtest_montecarlo.py --json pytaaa_sp500_pine_montecarlo.json --trials 3  # New
diff <orig_csv> <new_csv>
```

---

## Code Review Checkpoints

### Checkpoint #1: Helper Functions (Phase 1)
- [ ] Clear function purposes and interfaces
- [ ] Type annotations present and correct
- [ ] Doc strings comprehensive with examples
- [ ] Error handling with actionable messages
- [ ] Edge cases handled properly
- [ ] No hardcoded paths in helper functions
- [ ] Tests cover happy path and error cases

### Checkpoint #2: CLI Interface (Phase 2)
- [ ] Click interface follows project conventions
- [ ] Help text clear with examples
- [ ] Error messages actionable
- [ ] CLI arguments validated appropriately
- [ ] JSO parameter override logic correct
- [ ] Exception handling comprehensive
- [ ] Tests cover CLI edge cases

### Checkpoint #3: Dynamic Paths (Phase 3)
- [ ] No hardcoded paths remain
- [ ] Model identifier used consistently
- [ ] Output directory creation handled
- [ ] Filenames include model-specific prefixes
- [ ] JSON configuration fully utilized
- [ ] Works with different configs
- [ ] Regression test passes

### Checkpoint #4: Configuration (Phase 4)
- [ ] JSON configs validated (syntax)
- [ ] New parameter documented
- [ ] Export function updated correctly
- [ ] Output location appropriate
- [ ] Filename includes model identifier
- [ ] All JSON templates consistent

### Checkpoint #5: Documentation (Phase 5)
- [ ] All functions have type annotations
- [ ] All functions have docstrings
- [ ] Complex logic has comments
- [ ] Module docstring complete
- [ ] README.md updated
- [ ] Session summary created
- [ ] Documentation accurate

### Checkpoint #6: Testing (Phase 6)
- [ ] Test coverage >80%
- [ ] All edge cases tested
- [ ] Error handling tested
- [ ] Regression tests protect changes
- [ ] Mock usage appropriate
- [ ] Tests maintainable
- [ ] Test names descriptive

### Checkpoint #7: E2E Validation (Phase 7)
- [ ] All E2E tests passing
- [ ] Outputs validated vs original
- [ ] Performance metrics acceptable
- [ ] No regressions found
- [ ] Multiple configs successful

---

## Risk Mitigation

### Technical Risks

**Risk 1**: Path extraction logic fails with unexpected path formats
- **Mitigation**: Comprehensive unit tests with edge cases
- **Fallback**: Clear error messages guiding user to fix config

**Risk 2**: Output files overwrite important data
- **Mitigation**: Use date/runnum in filenames (existing pattern)
- **Fallback**: Document backup procedures in session summary

**Risk 3**: Performance regression from additional function calls
- **Mitigation**: Performance regression test comparing runtimes
- **Fallback**: Profile and optimize if needed

**Risk 4**: Numerical differences in outputs due to refactoring
- **Mitigation**: Regression test comparing CSV values
- **Fallback**: Investigate and fix any discrepancies

### Process Risks

**Risk 1**: Agent implementation deviates from plan
- **Mitigation**: Detailed phase checklists and code review checkpoints
- **Fallback**: Human review catches issues before merge

**Risk 2**: Local testing reveals production issues
- **Mitigation**: Comprehensive automated tests as foundation
- **Fallback**: Fix issues discovered in E2E testing phase

**Risk 3**: Documentation becomes outdated
- **Mitigation**: Documentation phase before PR
- **Fallback**: Update during code review if needed

---

## Rollback Plan

**No Rollback Needed** - Implementation is purely additive:

### What's Added (New Code Only)
1. **New CLI entry point**: `pytaaa_backtest_montecarlo.py` (~200 lines)
2. **New module**: `functions/backtesting/` (5 files, ~1150 lines total)
3. **JSON parameter**: `backtest_monte_carlo_trials` added to templates
4. **Tests**: 6 new test files for backtesting module + entry point
5. **Documentation**: Session summary, updated README

### What's Unchanged (No Modifications)
1. ✅ **Original script untouched**: `PyTAAA_backtest_sp500_pine_refactored.py` stays as-is
2. ✅ **Existing functions/ preserved**: No changes to existing modules
3. ✅ **Existing workflows intact**: pytaaa_main.py and all tools work unchanged
4. ✅ **Existing tests passing**: No regression in current test suite

### Rollback Strategy (If Needed)
1. **Delete new files**:
   ```bash
   rm pytaaa_backtest_montecarlo.py
   rm -rf functions/backtesting/
   rm tests/test_pytaaa_backtest_montecarlo.py
   rm tests/test_backtesting_*.py
   ```
2. **Revert JSON changes** (remove `backtest_monte_carlo_trials` lines)
3. **Keep git history** - branch preserved for future reference

**Risk Level**: **ZERO** - No existing code modified, only additions

---

## Decisions & Clarifications

### Decisions Made

1. **Script Name**: `pytaaa_backtest_montecarlo.py`
   - Follows existing naming pattern
   - Clear purpose from name

2. **CLI Style**: Click interface
   - Matches pytaaa_main.py and other tools
   - Consistent user experience

3. **Model ID Extraction**: From `webpage` path
   - Second-to-last path component
   - Example: `.../sp500_pine/webpage` → `sp500_pine`

4. **Trial Count Configuration**:
   - JSON parameter: `backtest_monte_carlo_trials` (default: 250)
   - CLI override: `--trials` option
   - Precedence: CLI > JSON > default

5. **Output Paths**:
   - Base: `performance_store/pngs/`
   - CSV: `{model_id}_montecarlo_{date}_{runnum}.csv`
   - PNG: `{model_id}_montecarlo_{date}.png` (2 files)
   - JSON: `{model_id}_optimized_{date}.json`

6. **Configuration Classes**:
   - Extract to: `functions/backtesting/` (shared constants)
   - No longer in entry point script (module owns them)

7. **Exit Codes** (standardized):
   ```python
   0 = Success
   1 = General error
   2 = Configuration error (missing key, invalid JSON)
   3 = Data file not found (HDF5, symbols file)
   ```

8. **File Overwrite Prevention**:
   - Add timestamp to all output filenames
   - Format: `{model_id}_{type}_{date}_HH-MM-SS_{suffix}`
   - Enables multiple runs without overwriting

9. **Logging**:
   - Stdout only (user controls redirection)
   - Consistent with existing tools
   - Can add file logging later if needed

### User Decisions Confirmed

**Q1**: Original Script Fate?
- ✅ **Decision**: Keep `PyTAAA_backtest_sp500_pine_refactored.py` as-is (reference)

**Q2**: Standardize Exit Codes?
- ✅ **Decision**: Yes (0=success, 1=error, 2=config, 3=data not found)

**Q3**: File Overwrite Behavior?
- ✅ **Decision**: Add timestamp to filename, include model name (prevents overwrites)

**Q4**: Logging Strategy?
- ✅ **Decision**: Stdout only (Option A, simple and user-controlled)

**Q5**: Commented Code Blocks?
- ✅ **Decision**: Remove outdated comments (Option A, cleaner refactoring)

**Q6**: Debug Functions?
- ✅ **Decision**: Keep as-is (Option A, useful for debugging)

**Q7**: Add `--version` Option?
- ✅ **Decision**: No (Option B, keep simple)

**Q8**: Runnum Logic?
- ✅ **Decision**: Keep existing logic (Option A, proven pattern)

### Implementation Notes

**Note 1**: **Maximize Code Reuse**
- Use existing functions from `functions/` wherever possible
- Extract unique Monte Carlo logic into `functions/backtesting/`
- Entry point is thin orchestration layer (~200 lines)
- No duplication of existing functionality

**Note 2**: **No Existing Code Modified**
- All changes are additions (new files only)
- Original script untouched (reference)
- Existing functions/ preserved
- Zero rollback risk

**Note 3**: **E2E Test Validation**
- Run without random perturbation matches pytaaa_main.py
- Compare `pyTAAAweb_backtestPortfolioValue.params` outputs
- Ensures algorithmic correctness preserved

**Note 4**: **Platform-specific trial counts preserved**
- Script adapts to platform in existing logic
- Behavior maintained in refactored version

**Note 5**: **Matplotlib backend handling**
- Already using `matplotlib.use("Agg")` pattern
- Preserved in backtesting module

**Note  6**: **Testing strategy**
- Agent can run 82% of tests (23/28)
- Human runs 18% locally (5/28) with production data
- This distribution is reasonable and efficient

---

## Summary

This plan creates a comprehensive, reusable Monte Carlo backtest CLI tool with modular architecture:

✅ **Refactors** monolithic script into thin CLI + reusable modules  
✅ **Maximizes code reuse** from existing functions/ folder  
✅ **Extracts logic** into new functions/backtesting/ module (~1150 lines)  
✅ **Creates thin entry point** (~200 lines orchestration)  
✅ **Supports** multiple trading models dynamically  
✅ **Follows** project conventions (click CLI, type hints, docstrings)  
✅ **Preserves** all existing logic and workflows (no modifications)  
✅ **Prevents overwrites** with timestamped filenames  
✅ **Standardizes** exit codes (0/1/2/3)  
✅ **Includes** comprehensive test suite (28 tests)  
✅ **Documents** all changes with session summary  
✅ **Enables** autonomous agent implementation (85%)  
✅ **Requires** minimal human involvement (1.5 hours)  
✅ **Zero rollback risk** (only additions, no modifications)  

**Estimated Implementation Time**:
- Agent autonomous work: 8-10 hours
- Code review: 1 hour
- Local E2E testing: 0.5 hour
- **Total**: ~9-11 hours (mostly automated)

**Success Criteria**:
- All automated tests pass (23/23)
- All local E2E tests pass (5/5)
- Outputs match `pyTAAAweb_backtestPortfolioValue.params` from pytaaa_main.py
- Works with multiple JSON configurations
- Timestamp filenames prevent overwrites
- No existing code modified
- No performance regression
- Code review checkpoints complete
- Documentation comprehensive

**Architecture Benefits**:
- **Modular**: Logic extracted into functions/backtesting/ (reusable)
- **Maintainable**: Thin entry point, well-tested modules
- **Extensible**: Easy to add features in backtesting/ module
- **Safe**: Zero risk (no existing code modified)
- **Validated**: E2E test ensures output parity with pytaaa_main.py

**Next Steps**:
1. ✅ Decisions confirmed (see User Decisions section)
2. Create GitHub issue with this plan
3. Assign to GitHub Copilot agent
4. Agent implements Phases 0-6
5. Human reviews PR and runs Phase 7 testing
6. Merge if all tests pass

---

## GitHub Issue Template

Use this template to create the GitHub issue for agent assignment:

### Title
```
Implement JSON-driven Monte Carlo backtest CLI tool (modular refactor)
```

### Body
```markdown
## Summary

Refactor `PyTAAA_backtest_sp500_pine_refactored.py` into a reusable, JSON-driven CLI tool with modular architecture. Extract logic into new `functions/backtesting/` module while maximizing reuse of existing `functions/` code.

## Plan

See comprehensive implementation plan: [plans/entry-point-backtest-montecarlo.md](./plans/entry-point-backtest-montecarlo.md)

## Objectives

1. **Create thin CLI entry point** (~200 lines): `pytaaa_backtest_montecarlo.py`
2. **Extract logic into modules** (~1150 lines): `functions/backtesting/` (5 files)
3. **Maximize code reuse**: Use existing functions from `functions/` folder
4. **Support all models**: Dynamic config from JSON (sp500_pine, naz100_hma, etc.)
5. **Prevent overwrites**: Timestamp in filenames
6. **Standardize exit codes**: 0=success, 1=error, 2=config, 3=data not found
7. **Zero risk**: No modifications to existing code (only additions)

## Architecture

```
pytaaa_backtest_montecarlo.py (~200 lines)
  ↓ delegates to
functions/backtesting/ (~1150 lines)
  ├─ monte_carlo_runner.py (main execution)
  ├─ parameter_exploration.py (random params)
  ├─ signal_backtest.py (signals & portfolio)
  ├─ output_writers.py (CSV/JSON export)
  └─ config_helpers.py (paths, validation)
  ↓ reuses
functions/ (existing modules - no changes)
  ├─ ta/ (technical analysis)
  ├─ PortfolioMetrics.py
  ├─ data_loaders.py
  └─ GetParams.py
```

## Acceptance Criteria

### Agent Implementation (Phases 0-6)
- [ ] All automated tests passing (23/23)
- [ ] Entry point created (~200 lines)
- [ ] functions/backtesting/ module created (5 files, ~1150 lines)
- [ ] JSON configs updated (backtest_monte_carlo_trials parameter)
- [ ] Code review checkpoints completed (all 7)
- [ ] Documentation complete (docstrings, session summary, README update)
- [ ] No modifications to existing code (only additions)
- [ ] Ready for local E2E testing

### Human Testing (Phase 7)
- [ ] E2E tests pass with production configs (5 tests)
- [ ] Outputs match `pyTAAAweb_backtestPortfolioValue.params` from pytaaa_main.py
- [ ] Multiple model configs work (sp500_pine, sp500_hma, naz100_hma)
- [ ] Timestamp filenames prevent overwrites
- [ ] Performance validated (~8-10 min runtime unchanged)

## Implementation Phases

1. **Phase 0**: Branch setup (15 min)
2. **Phase 1**: Module structure + config helpers (2 hr)
3. **Phase 2**: Parameter exploration module (2 hr)
4. **Phase 3**: Signal backtest module (2.5 hr)
5. **Phase 4**: Monte Carlo runner + output writers (2 hr)
6. **Phase 5**: CLI entry point + JSON configs (1.5 hr)
7. **Phase 6**: Documentation + automated tests (2 hr)
8. **Phase 7**: Local E2E testing (0.5 hr - human)

**Total**: ~12 hours (10.5 agent + 1.5 human)

## Key Constraints

- **No modifications**: Existing code untouched (only new files)
- **Code reuse**: Maximize use of existing functions/
- **Output parity**: Must match `pyTAAAweb_backtestPortfolioValue.params`
- **Timestamp filenames**: Format `{model_id}_{type}_{date}_HH-MM-SS`
- **Exit codes**: 0/1/2/3 (success/error/config/data)

## Testing Strategy

- **23 automated tests**: Unit + integration + regression (agent runs)
- **5 E2E tests**: Production data validation (human runs locally)
- **Output validation**: Compare with pytaaa_main.py results
- **Multi-model**: Test sp500_pine, sp500_hma, naz100_hma configs

## Related

- Original script: `PyTAAA_backtest_sp500_pine_refactored.py` (kept as reference)
- Similar tools: `pytaaa_main.py`, `recommend_model.py`, `daily_abacus_update.py`
- Async pattern: `plans/async-montecarlo-backtest.md`

## Labels

`enhancement`, `refactoring`, `autonomous-agent`, `cli`, `monte-carlo`
```

### Issue Creation Command
```bash
# Create issue via GitHub CLI
gh issue create \
  --title "Implement JSON-driven Monte Carlo backtest CLI tool (modular refactor)" \
  --body-file .github/ISSUE_TEMPLATE/montecarlo-backtest-refactor.md \
  --label "enhancement,refactoring,autonomous-agent,cli,monte-carlo" \
  --assignee "@me"
```
