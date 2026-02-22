# GitHub Copilot Custom Instructions for Python Project

## Project Summary

PyTAAA (Python Tactical Asset Allocation Advisor) is a Python 3.11+
trading system that recommends monthly portfolio allocations between
stock universes (Nasdaq 100, S&P 500) and cash using technical
analysis signals. It features:

- **Model Switching (Abacus)**: Dynamically selects between trading
  models (cash, naz100_pine, naz100_hma, naz100_pi, sp500_hma) based
  on normalized performance metrics (Sharpe, Sortino, drawdown).
- **Monte Carlo Optimization**: Thousands of backtesting iterations
  to find optimal lookback periods and normalization parameters.
- **Daily Portfolio Tracking**: Automated HTML dashboard generation.
- **Data Layer**: Stock quotes stored in HDF5 files; all configuration
  in JSON files (e.g., `pytaaa_model_switching_params.json`).

## Project Layout

```
PyTAAA/
├── pytaaa_main.py              # Modern CLI entry point (recommended)
├── PyTAAA.py                   # Legacy scheduler-based entry point
├── run_pytaaa.py               # Core execution pipeline
├── daily_abacus_update.py      # Daily portfolio tracking & web output
├── recommend_model.py          # Model recommendation engine
├── run_monte_carlo.py          # Monte Carlo parameter optimizer
├── run_normalized_score_history.py  # Score history analysis
├── update_json_from_csv.py     # Transfer Monte Carlo params to JSON
├── modify_saved_state.py       # Inspect/modify Monte Carlo state
├── scheduler.py                # Custom task scheduler (legacy)
├── pytaaa_generic.json         # Template JSON configuration
├── pytaaa_model_switching_params.json  # Abacus model config
├── pyproject.toml              # Project metadata and dependencies
├── functions/                  # Core library modules
│   ├── ta/                     # Technical analysis sub-package
│   │   ├── moving_averages.py  # SMA, HMA, MoveMax, MoveMin
│   │   ├── signal_generation.py  # computeSignal2D
│   │   ├── rolling_metrics.py  # Sharpe, Martin ratios
│   │   └── ...
│   ├── TAfunctions.py          # Backward-compat re-exports from ta/
│   ├── abacus_recommend.py     # Recommendation engine classes
│   ├── abacus_backtest.py      # Backtest data management
│   ├── MonteCarloBacktest.py   # Monte Carlo simulation engine
│   ├── PortfolioPerformanceCalcs.py  # Portfolio ranking pipeline
│   ├── PortfolioMetrics.py     # Performance metric calculations
│   ├── MakeValuePlot.py        # Chart generation (Matplotlib)
│   ├── GetParams.py            # JSON configuration loading
│   ├── UpdateSymbols_inHDF5.py # HDF5 quote management
│   ├── WriteWebPage_pi.py      # HTML generation and deployment
│   ├── logger_config.py        # Centralized logging configuration
│   └── ...
├── tests/                      # pytest test suite
├── docs/                       # Architecture and operations guides
│   ├── ARCHITECTURE.md         # Detailed architectural overview
│   └── copilot_sessions/       # Per-session summary documents
└── scripts/                    # Shell helper scripts
```

## Build, Test, and Run

**Always use `uv` to manage dependencies and run Python.**

```bash
# Install dependencies (first time or after pyproject.toml changes)
uv sync

# Run all tests
PYTHONPATH=$(pwd) uv run pytest

# Run a specific test file
PYTHONPATH=$(pwd) uv run pytest tests/test_abacus_recommend.py

# Run daily portfolio update
uv run python daily_abacus_update.py --json <path_to_config.json>

# Generate model recommendation
uv run python recommend_model.py --json pytaaa_model_switching_params.json

# Run Monte Carlo optimization
uv run python run_monte_carlo.py \
    --json pytaaa_model_switching_params.json --iterations 100
```

**Always set `PYTHONPATH=$(pwd)` when running tests** so that the
`functions/` package is importable. There are no separate compile or
build steps — this is a pure-Python project.

There are currently no CI/CD GitHub Actions workflows configured.

## Key Patterns and Conventions

### Matplotlib in Headless Environments
Always force the `Agg` backend before importing `pyplot` when running
in a headless (no display) environment:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

This pattern is used in `daily_abacus_update.py`,
`functions/MakeValuePlot.py`, and `docs/generate_method_charts.py`.

### Configuration via JSON
All runtime configuration is stored in JSON files. Use
`functions/GetParams.py` to load parameters. Prefer
`pytaaa_model_switching_params.json` as the main config for
Abacus/model-switching workflows.

### HDF5 Quote Storage
Historical stock quotes are stored in HDF5 files (via pandas
`HDFStore`). `functions/UpdateSymbols_inHDF5.py` manages downloads
and updates. Never hard-code quote file paths; always read them from
the JSON configuration.

### Logging
All modules use `functions/logger_config.py` via:
```python
from functions.logger_config import get_logger
logger = get_logger(__name__, log_file="module_name.log")
```

### Session Documentation
After significant Copilot sessions, create a summary in
`docs/copilot_sessions/` using the format
`YYYY-MM-DD_brief-description.md`.

## Dependency Management
- This Python project uses **uv** to manage dependencies.
- Use "uv run python" to run python scripts
- Dependencies are recorded in `pyproject.toml`, not `requirements.txt`.
- Always follow this when generating or modifying dependency code.

## Testing
- Use **pytest** as the testing framework.
- Write clear and maintainable test functions.
- Use fixtures for setup and teardown.

## Commit Message Conventions
- Use conventional commit prefixes:
  - `feat:` for features
  - `fix:` for bug fixes
  - `docs:` for documentation
  - `style:` for formatting and whitespace only
  - `refactor:` for code restructuring without behavior change
  - `test:` for tests
  - `chore:` for miscellaneous tasks
- Format: `<type>(optional scope): <description>`

## Python Code Style
- Follow PEP 8 guidelines.
- Use 4 spaces for indentation; no tabs.
- Use snake_case for functions and variables.
- Use CamelCase for classes.
- Constants in UPPER_SNAKE_CASE.
- Max line length: 79 characters for code, 72 for comments.
- Two blank lines before top-level functions/classes.
- Use double quotes for strings.
- Add meaningful docstrings following PEP 257.
- Use type annotations.
- Use descriptive names.
- Keep functions focused and simple.

## Logging
- Use the standard `logging` module for all logging purposes.
- Configure loggers with appropriate log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
- Use print statements for tracking progress and showing results as computations progress
- Include contextual information in log messages to aid debugging.
- Follow consistent log message formatting.
- Follow consistent print statement formatting.

## Comment Style
- Comments must be complete sentences starting with a capital letter.
- Limit comment lines to 72 characters.
- Use block comments indented to code level, starting with `#############`.
- Separate paragraphs in block comments with a single `#` line.
- Use inline comments for natural sub-tasks, separated by two spaces, starting with `# `.
- Use obvious comments sparingly; explain *why* not *what*.
- Write comments in English.
- Keep comments up to date.
- **Ensure comments are updated promptly to reflect any code changes.**

## Environment Setup
- Always set the `PYTHONPATH` to the root of the project directory for testing and development.
- Example: `export PYTHONPATH=$(pwd)` (for macOS/Linux) or `set PYTHONPATH=%cd%` (for Windows).
- This ensures that all modules in the project are correctly recognized.

## Copilot Session Documentation
- At the end of significant Copilot sessions, create a summary document in `docs/copilot_sessions/`.
- Use the naming format: `YYYY-MM-DD_brief-description.md` (e.g., `2025-10-04_daily-abacus-update-fix.md`).
- Include the following sections in session summaries:
  - **Date and Context**: When the session occurred and what prompted it
  - **Problem Statement**: What issue or task was being addressed
  - **Solution Overview**: High-level description of the solution implemented
  - **Key Changes**: List of files modified and what changed
  - **Technical Details**: Important implementation details for future reference
  - **Testing**: How the solution was verified
  - **Follow-up Items**: Any remaining tasks or considerations
- Keep summaries concise but informative for future reference.
- Session summaries help maintain project knowledge and facilitate onboarding.
