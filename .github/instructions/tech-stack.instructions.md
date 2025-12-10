# Tech Stack Instructions

> Version: 1.0.0
> Last Updated: 2025-12-10

## Context

This file defines the technology stack for the PyTAAA project (Tactical Asset Allocation Advisor). All paths are workspace-relative. Individual features may extend these choices in their specific implementation files.

## Core Technologies

### Application Framework
- **Framework:** CLI-based Python application with web page outputs
- **Version:** Python 3.11+ (required for modern type hints and performance improvements)
- **Execution standard:** **CRITICAL - ALWAYS use `uv run` for ALL Python operations. NEVER use bare `python`, `pytest`, etc.**
- **Entry Points:** `pytaaa_main.py` (CLI interface), `PyTAAA.py` (main scheduler)
- **Deployment:** pip install via uv package manager

### Core Processing Stack
- **NumPy:** >=1.24.0 (array operations, numerical computing)
- **Pandas:** >=2.2.3 (data manipulation and analysis)
- **Matplotlib:** >=3.10.0 (visualization and plotting)
- **Scikit-learn:** >=1.7.1 (machine learning for clustering and analysis)
- **SciPy:** >=1.15.1 (scientific computing and statistics)
- **Tables:** >=3.10.2 (HDF5 data storage and I/O)

### Specialized Libraries
- **Yfinance:** >=0.2.52 (Yahoo Finance API for stock quotes)
- **Finvizfinance:** >=1.1.0 (Financial data scraping from Finviz)
- **NetworkX:** >=3.5 (graph analysis for stock correlations)
- **Pandas-market-calendars:** >=4.6.1 (trading calendar handling)
- **Paramiko:** >=3.5.1 (SSH for remote quote updates)
- **Holidays:** >=0.66 (holiday data for market schedules)

### Data Formats & I/O
- **Input Formats:**
  - HDF5 (.hdf5) for historical stock quotes (via tables library)
  - JSON (.json) for configuration parameters
  - CSV (.csv) for portfolio and backtest data
- **Output Formats:**
  - HTML (.html) for web page reports and visualizations
  - JSON (.json) for status and holdings data
  - PNG (.png) for plots and charts (via matplotlib)
- **Temporary Storage:** `/tmp/` for temporary files during processing
- **Cross-Platform:** Linux (primary), Windows, macOS
- **Special Integrations:** Yahoo Finance API, Finviz web scraping, FTP for quote updates

## Environment & Packaging

### Environment Management
- **Primary Tool:** UV (fast Python package installer and resolver)
- **Standard:** ALL Python commands must use `uv run` prefix
- **Configuration:** pyproject.toml with uv.lock
- **Containerization:** None

### Command Standards
**ALWAYS use [TOOL] for Python commands:**
```bash
uv run python script.py        # NOT: python script.py
uv run pytest tests/          # NOT: pytest tests/
uv run [linter] check .       # NOT: [linter] check .
uv add package-name           # NOT: pip install package-name
uv sync                       # Update dependencies
```

### Dependency Management
- **File:** pyproject.toml (primary, with dependency-groups for dev)
- **Lock File:** uv.lock (ensures reproducible builds)
- **Development Dependencies:** Managed via [dependency-groups] in pyproject.toml

## Testing

### Test Framework
- **Framework:** pytest (for Python projects)
- **Test Categories:**
  - Unit tests (individual functions and classes in `functions/`)
  - Integration tests (component interactions for backtesting and data processing)
  - Data validation tests (accuracy of financial data processing)
  - Performance tests (for backtesting simulations)

### Testing Requirements
- **Test Patterns:** Standard pytest fixtures and parametrization for financial calculations
- **Coverage Target:** Not specified; focus on critical modules like `TAfunctions.py` and `calculateTrades.py`
- **Mocking Strategy:** Mock external APIs (yfinance, finviz) and file I/O for isolated testing

### Test Data Management (OPTIONAL)
**Add this section if your project needs test data files:**
- **Location:** [e.g., `tests/fixtures/`, `test_data/`, `tests/data/`]
- **Structure:** [e.g., `test_data/[category]/` with appropriate test files]
- **Exclusions:** [What's excluded from version control and why]
- **Setup:** [How test data is initialized or generated]
- **Formats:** [Real data formats used in testing]

## Project-Specific Considerations

### Financial Data Integrations
- **Yahoo Finance:** Via yfinance library for real-time and historical stock quotes
- **Finviz:** Via finvizfinance library for financial metrics and screening
- **Market Calendars:** Via pandas-market-calendars for trading day calculations
- **FTP/SSH:** Via paramiko for secure quote data transfers

### Data Format Support
- **Primary Data Source:** HDF5 files for Nasdaq 100/SP500 historical quotes
- **Processing Pipeline:** Daily quote updates → Backtesting simulations → Portfolio recommendations → HTML reports
- **Output Targets:** Web pages (HTML) for user viewing, JSON files for status tracking
- **Quality Assurance:** Validation of quote data integrity and sector/industry classifications

---

*Update this tech stack file as your project evolves. Add new dependencies, update versions, and document architectural decisions.*
