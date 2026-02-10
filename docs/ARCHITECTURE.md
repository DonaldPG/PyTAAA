# PyTAAA Architecture

**Last Updated:** February 9, 2026

---

## 1. Codebase Architectural Layout

### Directory Structure

```
PyTAAA/
├── pytaaa_main.py              # Modern CLI entry point
├── PyTAAA.py                   # Legacy scheduler-based entry point
├── run_pytaaa.py               # Core execution pipeline
├── daily_abacus_update.py      # Daily portfolio tracking
├── recommend_model.py          # Model recommendation engine
├── run_monte_carlo.py          # Monte Carlo parameter optimizer
├── run_normalized_score_history.py  # Score history analysis
├── update_json_from_csv.py     # Config update utility
├── modify_saved_state.py       # State management utility
├── compute_allocations.py      # Allocation computation
├── compute_new_allocations.py  # Alternative allocation computation
├── pytaaa_quotes_update.py     # Quote data cleaning
├── re-generateHDF5.py          # Legacy HDF5 regeneration
├── scheduler.py                # Custom task scheduler
├── pytaaa_generic.json         # Template JSON configuration
├── pytaaa_model_switching_params.json  # Abacus model config
├── pyproject.toml              # Project metadata and dependencies
├── requirements.txt            # Pip requirements
├── functions/                  # Core library modules
│   ├── TAfunctions.py          # Technical analysis functions - 4100+ lines
│   ├── PortfolioPerformanceCalcs.py  # Portfolio ranking pipeline
│   ├── MonteCarloBacktest.py   # Monte Carlo simulation engine
│   ├── dailyBacktest.py        # Daily backtesting logic
│   ├── dailyBacktest_pctLong.py  # Backtest visualization
│   ├── calculateTrades.py      # Trade recommendation logic
│   ├── UpdateSymbols_inHDF5.py # HDF5 quote management
│   ├── WriteWebPage_pi.py      # HTML generation and deployment
│   ├── MakeValuePlot.py        # Chart generation
│   ├── GetParams.py            # Configuration loading
│   ├── readSymbols.py          # Symbol list management
│   ├── quotes_adjClose.py      # Quote downloading
│   ├── quotes_for_list_adjClose.py  # Batch quote operations
│   ├── abacus_recommend.py     # Recommendation engine classes
│   ├── abacus_backtest.py      # Backtest data management
│   ├── PortfolioMetrics.py     # Performance metric calculations
│   ├── allstats.py             # Statistical functions
│   ├── CountNewHighsLows.py    # Market breadth indicators
│   ├── stock_cluster.py        # Stock clustering
│   ├── CheckMarketOpen.py      # Market status detection
│   ├── SendEmail.py            # Email notifications
│   ├── clean_quote_data.py     # Quote data cleaning
│   ├── ftp_quotes.py           # Remote quote file transfer
│   ├── GetYieldCurve.py        # Treasury yield curve data
│   ├── logger_config.py        # Centralized logging
│   └── yahooFinance.py         # Yahoo Finance API wrapper
├── docs/                       # Documentation
├── tests/                      # Test suite
├── scripts/                    # Utility scripts
├── log_files/                  # Log output
├── pyTAAA_data/                # Local data files
└── .github/                    # GitHub metadata and plans
```

### Architectural Layers

The codebase follows a loosely layered architecture, though boundaries are not strictly enforced:

```mermaid
graph TB
    subgraph Entry Points
        A[pytaaa_main.py]
        B[PyTAAA.py]
        C[daily_abacus_update.py]
        D[recommend_model.py]
        E[run_monte_carlo.py]
    end

    subgraph Orchestration
        F[run_pytaaa.py]
        G[scheduler.py]
    end

    subgraph Core Computation
        H[PortfolioPerformanceCalcs.py]
        I[MonteCarloBacktest.py]
        J[dailyBacktest.py]
        K[calculateTrades.py]
        L[TAfunctions.py]
    end

    subgraph Data Access
        M[UpdateSymbols_inHDF5.py]
        N[GetParams.py]
        O[readSymbols.py]
        P[quotes_adjClose.py]
    end

    subgraph Output
        Q[WriteWebPage_pi.py]
        R[MakeValuePlot.py]
        S[SendEmail.py]
    end

    subgraph External
        T[Yahoo Finance API]
        U[HDF5 Files]
        V[.params Files]
        W[JSON Config]
    end

    A --> F
    B --> G --> F
    C --> F
    D --> I
    E --> I

    F --> H
    F --> K
    F --> Q

    H --> J
    H --> L
    H --> M

    J --> L
    I --> L

    M --> P --> T
    M --> U
    N --> W
    N --> V
    O --> T

    Q --> R
    Q --> S
```

---

## 2. Data Flow in pytaaa_main.py and Other Entry Points

### 2.1 pytaaa_main.py / run_pytaaa.py Data Flow

This is the primary execution pipeline. The data flows through these stages:

```mermaid
graph TD
    A[pytaaa_main.py] -->|--json config.json| B[run_pytaaa - json_fn]

    B --> C[get_json_params - Load JSON config]
    B --> D[get_symbols_file - Resolve symbols path]
    B --> E[get_status - Load prior portfolio value]
    B --> F[get_holdings - Load current holdings]

    C --> G{Check hour of day}
    G -->|Before 5 PM| H[get_symbols_changes - Check index changes]
    G -->|After 5 PM| I[Skip symbol update]

    H --> J{Quote server match?}
    J -->|Local server| K[UpdateHDF_yf - Download quotes to HDF5]
    J -->|Remote server| L[copy_updated_quotes - SFTP from remote]

    K --> M[CheckMarketOpen - Get market status]
    L --> M

    M --> N[PortfolioPerformanceCalcs]
    N --> N1[loadQuotes_fromHDF - Read HDF5]
    N1 --> N2[interpolate + clean data]
    N2 --> N3[computeSignal2D - Generate signals]
    N3 --> N4[sharpeWeightedRank_2D - Rank stocks]
    N4 --> N5[computeDailyBacktest - Run backtest]
    N5 --> N6[Generate plots via MakeValuePlot]
    N6 --> N7[Return: lastdate, symbols, weights, prices]

    N7 --> O[get_holdings - Reload updated holdings]
    O --> P[LastQuotesForSymbolList_hdf - Get current prices]
    P --> Q[Calculate portfolio value and profit]
    Q --> R[calculateTrades - Generate trade recommendations]

    R --> S{Value changed?}
    S -->|Yes| T[SendEmail - Notify user]
    S -->|No| U[Skip email]

    T --> W[writeWebPage - Generate HTML dashboard]
    U --> W
    W --> X[put_status - Save current value]
```

### 2.2 PyTAAA.py Data Flow

The legacy entry point wraps the same logic in a scheduler loop:

```mermaid
graph TD
    A[PyTAAA.py] --> B[GetParams - Load .params config]
    B --> C[Create Scheduler]
    C --> D[Schedule IntervalTask every N seconds]
    D --> E[Scheduler.start - Begin loop]

    E --> F[IntervalTask]
    F --> G[Same pipeline as run_pytaaa]
    G --> H[Sleep until next interval]
    H --> F

    E --> I[Sleep for runtime duration]
    I --> J[Scheduler.halt]
```

**Key difference:** [`PyTAAA.py`](../PyTAAA.py:1) uses legacy [`GetParams()`](../functions/GetParams.py:551) reading from `.params` files, while [`run_pytaaa.py`](../run_pytaaa.py:32) uses [`get_json_params()`](../functions/GetParams.py:219) reading from JSON.

### 2.3 recommend_model.py Data Flow

```mermaid
graph TD
    A[recommend_model.py] -->|--json config.json| B[Load JSON config]
    B --> C[ConfigurationHelper.ensure_config_defaults]
    B --> D[DateHelper.get_recommendation_dates]
    B --> E[ConfigurationHelper.get_recommendation_lookbacks]

    E --> F[BacktestDataLoader.build_model_paths]
    F --> G[Load .params files for each model]
    G --> H[MonteCarloBacktest - Initialize with model data]

    H --> I[For each lookback period]
    I --> J[compute_daily_metrics - Sharpe, Sortino, drawdown]
    J --> K[rank_models - Rank by composite score]
    K --> L[Normalize scores using central_values and std_values]

    L --> M[ModelRecommender - Generate recommendation]
    M --> N[RecommendationDisplay - Format output]
    N --> O[Console output: model rankings and scores]
    N --> P[PlotGenerator - Create recommendation_plot.png]
```

### 2.4 run_monte_carlo.py Data Flow

```mermaid
graph TD
    A[run_monte_carlo.py] -->|--json config.json| B[Load JSON config]
    B --> C[BacktestDataLoader.build_model_paths]
    C --> D[Load all model portfolio value histories]

    D --> E[MonteCarloBacktest - Initialize]
    E --> F{Search strategy}
    F -->|Explore| G[Random lookback sampling]
    F -->|Exploit| H[Refine around best params]
    F -->|Explore-Exploit| I[Dynamic transition]

    G --> J[For each iteration]
    H --> J
    I --> J

    J --> K[Select 3 lookback periods]
    K --> L[Compute model-switching portfolio]
    L --> M[Calculate performance metrics]
    M --> N[Compute normalized score]
    N --> O{New best?}
    O -->|Yes| P[Update best params and display]
    O -->|No| Q[Continue]

    P --> R[Save to CSV: all parameters and metrics]
    Q --> R
    R --> S{More iterations?}
    S -->|Yes| J
    S -->|No| T[Save state to monte_carlo_state.pkl]
    T --> U[Output: CSV results + best config]
```

### 2.5 daily_abacus_update.py Data Flow

```mermaid
graph TD
    A[daily_abacus_update.py] -->|--json config.json| B[Load JSON config]
    B --> C[Read PyTAAA_holdings.params]
    C --> D[Detect active trading_model]

    D --> E{Which model?}
    E -->|naz100_*| F[Route to Naz100 symbols]
    E -->|sp500_*| G[Route to SP500 symbols]
    E -->|cash| H[Skip quote update]

    F --> I{Market open?}
    G --> I
    I -->|Yes| J[Check HDF5 freshness]
    I -->|No| K[Skip quote download]

    J -->|Stale| L[UpdateHDF_yf - Download new quotes]
    J -->|Fresh| K

    L --> M[run_pytaaa - Full pipeline]
    K --> M
    M --> N[Generate HTML + PNG charts]
    N --> O[Deploy to web output directory]
```

---

## 3. Data: Sources, Gathering, and Storage

### 3.1 Data Sources

| Data Type | Source | Method | Frequency |
|---|---|---|---|
| **Stock Quotes** | Yahoo Finance | `yfinance` Python package | Daily (after market close) |
| **Index Composition** | Wikipedia | Web scraping via `pandas.read_html()` | Daily check |
| **Market Status** | Yahoo Finance / NASDAQ | HTTP scraping | Each run cycle |
| **Sector/Industry** | Finviz | `finvizfinance` package | On demand |
| **Treasury Yields** | US Treasury | XML feed (legacy, non-functional) | Not currently active |
| **Configuration** | Local JSON files | File I/O | Manual updates |
| **Holdings** | Local `.params` files | File I/O | Manual updates after trades |

### 3.2 Data Gathering Process

#### Stock Quote Download Pipeline

```mermaid
graph LR
    A[Yahoo Finance API] -->|yfinance| B[downloadQuotes]
    B --> C[Raw DataFrame: OHLCV + Adj Close]
    C --> D[interpolate - Fill NaN gaps]
    D --> E[cleantobeginning - Fill leading NaNs]
    E --> F[cleanspikes - Remove outliers]
    F --> G[HDF5 File Storage]
```

The quote download is managed by [`UpdateHDF_yf()`](../functions/UpdateSymbols_inHDF5.py:679):
1. Reads the current HDF5 file to determine the last stored date
2. Downloads only new data from Yahoo Finance (incremental update)
3. Cleans the data (interpolation, spike removal, boundary filling)
4. Appends to the existing HDF5 store
5. Handles stock splits and adjusted close prices automatically via `yfinance`

#### Symbol List Management

Symbol lists are maintained in two ways:
1. **Local text files** (e.g., `Naz100_Symbols.txt`, `SP500_Symbols.txt`) — authoritative list
2. **Web scraping** from Wikipedia — used to detect additions/removals from the index

[`get_symbols_changes()`](../functions/readSymbols.py:724) compares the local list against the web-scraped list and reports added/removed tickers. The local file is updated to reflect changes.

### 3.3 Data Storage

#### HDF5 Files (Primary Quote Storage)

| File Pattern | Contents | Location |
|---|---|---|
| `Naz100_Symbols_.hdf5` | Nasdaq 100 adjusted close prices, all dates | `symbols/` directory under data store |
| `SP500_Symbols_.hdf5` | S&P 500 adjusted close prices, all dates | `symbols/` directory under data store |

**Structure:** Each HDF5 file contains a pandas DataFrame with:
- **Index:** Date strings (YYYY-MM-DD)
- **Columns:** Stock ticker symbols
- **Values:** Adjusted close prices (float)

Data is loaded via [`loadQuotes_fromHDF()`](../functions/UpdateSymbols_inHDF5.py:39) which returns:
- `adjClose` — 2D numpy array [symbols × dates]
- `symbols` — List of ticker strings
- `datearray` — Array of datetime objects
- Volume and other arrays (when available)

#### .params Files (State and Configuration)

| File | Format | Contents |
|---|---|---|
| `PyTAAA_status.params` | `cumu_value: YYYY-MM-DD HH:MM:SS.ffffff value signal traded_value` | Daily portfolio value history |
| `PyTAAA_holdings.params` | Key-value pairs | Current stocks, shares, buy prices, ranks, cumulative cash in, trading model |
| `PyTAAA_ranks.params` | Space-separated | Current stock rankings |
| `pyTAAAweb_backtestPortfolioValue.params` | Space-separated columns | Backtest results: date, buy-hold value, traded value, new highs, new lows, selected model |

#### JSON Configuration Files

| File | Purpose |
|---|---|
| `pytaaa_generic.json` | Template with all configuration sections |
| `pytaaa_model_switching_params.json` | Abacus model-switching parameters |
| `abacus_combined_PyTAAA_*.json` | Combined status for multi-model tracking |

#### Pickle Files

| File | Purpose |
|---|---|
| `monte_carlo_state.pkl` | Saved Monte Carlo optimization state for resume capability |

#### CSV/Excel Output

| File | Purpose |
|---|---|
| `monte_carlo_results.csv` | Full Monte Carlo iteration results |
| `abacus_best_performers.csv` | Top-performing parameter configurations |

### 3.4 Data Directory Layout

The primary data store is located at `/Users/donaldpg/pyTAAA_data/` with this structure:

```
pyTAAA_data/
├── naz100_pine/
│   ├── symbols/
│   │   ├── Naz100_Symbols.txt
│   │   └── Naz100_Symbols_.hdf5
│   └── data_store/
│       ├── PyTAAA_status.params
│       ├── PyTAAA_holdings.params
│       ├── PyTAAA_ranks.params
│       └── pyTAAAweb_backtestPortfolioValue.params
├── naz100_hma/
│   └── ... same structure ...
├── naz100_pi/
│   └── ... same structure ...
├── sp500_hma/
│   ├── symbols/
│   │   ├── SP500_Symbols.txt
│   │   └── SP500_Symbols_.hdf5
│   └── data_store/
│       └── ... same structure ...
├── sp500_pine/
│   └── ... same structure ...
└── naz100_sp500_abacus/
    └── data_store/
        ├── PyTAAA_status.params
        ├── PyTAAA_holdings.params
        └── pyTAAAweb_backtestPortfolioValue.params
```

---

## 4. Block Diagrams

### 4.1 Major Components Block Diagram

```mermaid
graph TB
    subgraph User Interface Layer
        CLI[CLI Entry Points<br/>pytaaa_main.py<br/>recommend_model.py<br/>run_monte_carlo.py<br/>daily_abacus_update.py]
        WEB[Web Dashboard<br/>pyTAAAweb.html<br/>Chart Pages]
        EMAIL[Email Notifications<br/>SendEmail.py]
    end

    subgraph Orchestration Layer
        SCHED[Task Scheduler<br/>scheduler.py]
        PIPELINE[Execution Pipeline<br/>run_pytaaa.py]
        ABACUS[Abacus Meta-Model<br/>abacus_recommend.py<br/>abacus_backtest.py]
    end

    subgraph Computation Engine
        PERF[Portfolio Performance<br/>PortfolioPerformanceCalcs.py]
        SIGNAL[Signal Generation<br/>TAfunctions.py<br/>computeSignal2D<br/>sharpeWeightedRank_2D]
        BACKTEST[Backtesting<br/>dailyBacktest.py<br/>MonteCarloBacktest.py]
        TRADE[Trade Calculator<br/>calculateTrades.py]
        METRICS[Portfolio Metrics<br/>PortfolioMetrics.py<br/>allstats.py]
        BREADTH[Market Breadth<br/>CountNewHighsLows.py]
    end

    subgraph Data Layer
        HDF5[HDF5 Quote Store<br/>UpdateSymbols_inHDF5.py]
        PARAMS[.params File I/O<br/>GetParams.py]
        SYMBOLS[Symbol Management<br/>readSymbols.py]
        QUOTES[Quote Download<br/>quotes_adjClose.py<br/>yfinance]
    end

    subgraph Output Layer
        HTML[HTML Generator<br/>WriteWebPage_pi.py]
        PLOTS[Chart Generator<br/>MakeValuePlot.py<br/>dailyBacktest_pctLong.py]
        FTP[File Transfer<br/>ftpMoveDirectory<br/>piMoveDirectory]
    end

    subgraph External Systems
        YAHOO[Yahoo Finance]
        WIKI[Wikipedia]
        FINVIZ[Finviz]
        SMTP[Gmail SMTP]
        REMOTE[Remote Web Server]
    end

    CLI --> PIPELINE
    CLI --> ABACUS
    SCHED --> PIPELINE

    PIPELINE --> PERF
    PIPELINE --> TRADE
    PIPELINE --> HTML

    ABACUS --> BACKTEST
    ABACUS --> METRICS

    PERF --> SIGNAL
    PERF --> BACKTEST
    PERF --> BREADTH
    SIGNAL --> HDF5
    BACKTEST --> SIGNAL

    TRADE --> PARAMS
    TRADE --> QUOTES

    HDF5 --> QUOTES
    QUOTES --> YAHOO
    SYMBOLS --> WIKI
    SYMBOLS --> FINVIZ

    HTML --> PLOTS
    HTML --> FTP
    FTP --> REMOTE
    EMAIL --> SMTP

    WEB -.->|reads| FTP
```

### 4.2 User Perspective Data Flow

This diagram shows the data flow from a user's perspective — what they do, what happens, and what they see:

```mermaid
graph TD
    subgraph User Actions
        U1[Configure JSON file<br/>with trading parameters]
        U2[Run daily update<br/>or scheduled task]
        U3[Run model recommendation<br/>on 1st weekday of month]
        U4[Run Monte Carlo<br/>optimization periodically]
        U5[Execute trades manually<br/>on brokerage platform]
        U6[Update holdings file<br/>with new positions]
    end

    subgraph System Processing
        S1[Download latest quotes<br/>from Yahoo Finance]
        S2[Clean and store quotes<br/>in HDF5]
        S3[Compute technical signals<br/>for all stocks]
        S4[Rank stocks by<br/>Sharpe-weighted performance]
        S5[Run daily backtest<br/>against buy-and-hold]
        S6[Calculate trade<br/>recommendations]
        S7[Evaluate all models<br/>with normalized scoring]
        S8[Optimize lookback periods<br/>via Monte Carlo]
    end

    subgraph User Sees
        V1[HTML Dashboard<br/>with portfolio status]
        V2[PNG Charts<br/>performance and signals]
        V3[Email Notification<br/>with trade suggestions]
        V4[Model Recommendation<br/>with rankings table]
        V5[Monte Carlo Results<br/>CSV with best params]
        V6[Recommendation Plot<br/>visual analysis]
    end

    U1 --> S1
    U2 --> S1
    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> S5
    S5 --> S6

    S6 --> V1
    S5 --> V2
    S6 --> V3

    U3 --> S7
    S7 --> V4
    S7 --> V6

    U4 --> S8
    S8 --> V5

    V4 --> U5
    U5 --> U6
    U6 --> U2
```

### 4.3 Backend Computation Data Flow

This diagram shows the internal data transformations from raw market data to final outputs:

```mermaid
graph TD
    subgraph Data Ingestion
        D1[Yahoo Finance API<br/>Adjusted Close Prices]
        D2[Wikipedia<br/>Index Composition]
        D3[Local HDF5<br/>Historical Quotes]
        D4[JSON Config<br/>Trading Parameters]
        D5[.params Files<br/>Holdings and Status]
    end

    subgraph Quote Processing
        Q1[Download incremental quotes<br/>UpdateHDF_yf]
        Q2[Interpolate missing values<br/>interpolate]
        Q3[Clean boundaries<br/>cleantobeginning + cleantoend]
        Q4[Remove spikes<br/>despike_2D + cleanspikes]
        Q5[Compute gain/loss matrix<br/>adjClose t / adjClose t-1]
    end

    subgraph Signal Computation
        SG1[Compute uptrend signals<br/>computeSignal2D]
        SG2a[HMA Method<br/>Hull Moving Averages]
        SG2b[MinMax Method<br/>Percentile Channels]
        SG2c[3MAs Method<br/>Triple MA Crossover]
        SG3[Count new highs/lows<br/>newHighsAndLows]
        SG4[Compute market breadth<br/>uptrending stock count]
    end

    subgraph Ranking Engine
        R1[Sharpe-weighted ranking<br/>sharpeWeightedRank_2D]
        R2[Multi-timeframe Sharpe<br/>move_sharpe_2D]
        R3[Recent trend analysis<br/>recentTrendComboGain]
        R4[Channel fit analysis<br/>recentChannelFit]
        R5[Final stock weights<br/>top N stocks selected]
    end

    subgraph Backtesting
        B1[Daily backtest simulation<br/>computeDailyBacktest]
        B2[Monte Carlo portfolio<br/>random stock selection]
        B3[Traded portfolio value<br/>signal-weighted returns]
        B4[Buy-and-hold baseline<br/>equal-weight returns]
    end

    subgraph Model Switching - Abacus
        MS1[Load all model histories<br/>BacktestDataLoader]
        MS2[Compute metrics per lookback<br/>Sharpe, Sortino, drawdown]
        MS3[Normalize scores<br/>central_values + std_values]
        MS4[Rank models<br/>composite normalized score]
        MS5[Select best model<br/>for current month]
    end

    subgraph Output Generation
        O1[Portfolio value calculation<br/>shares x current price]
        O2[Trade recommendations<br/>calculateTrades]
        O3[HTML dashboard<br/>writeWebPage]
        O4[PNG charts<br/>MakeValuePlot functions]
        O5[Email notification<br/>SendEmail]
        O6[Status file update<br/>put_status]
    end

    D1 --> Q1
    D2 --> Q1
    D3 --> Q1
    Q1 --> Q2 --> Q3 --> Q4 --> Q5

    D4 --> SG1
    Q5 --> SG1
    SG1 --> SG2a
    SG1 --> SG2b
    SG1 --> SG2c
    Q5 --> SG3
    SG1 --> SG4

    Q5 --> R1
    SG1 --> R1
    R1 --> R2
    R1 --> R3
    R1 --> R4
    R2 --> R5
    R3 --> R5
    R4 --> R5

    Q5 --> B1
    SG1 --> B1
    R5 --> B1
    B1 --> B2
    B1 --> B3
    B1 --> B4

    D5 --> MS1
    MS1 --> MS2
    MS2 --> MS3
    MS3 --> MS4
    MS4 --> MS5

    R5 --> O1
    D5 --> O1
    O1 --> O2
    O2 --> O3
    B1 --> O4
    SG3 --> O4
    SG4 --> O4
    O2 --> O5
    O1 --> O6
```

---

## 5. Key Architectural Patterns

### 5.1 Dual Configuration System

The codebase maintains two parallel configuration systems:

1. **Legacy `.params` files** — Used by [`PyTAAA.py`](../PyTAAA.py:1) and the original `GetParams()`, `GetHoldings()`, `GetStatus()`, `PutStatus()` functions
2. **Modern JSON files** — Used by [`pytaaa_main.py`](../pytaaa_main.py:1) and the newer `get_json_params()`, `get_holdings()`, `get_status()`, `put_status()` functions

Both systems coexist in [`GetParams.py`](../functions/GetParams.py:1), with the JSON-based functions accepting a `json_fn` parameter. This dual system exists because the codebase evolved from `.params` files to JSON without fully deprecating the legacy approach.

**Evidence:** [`GetParams.py`](../functions/GetParams.py:1) contains both [`GetParams()`](../functions/GetParams.py:551) (legacy, no arguments) and [`get_json_params()`](../functions/GetParams.py:219) (modern, takes `json_fn`). Similarly, [`GetHoldings()`](../functions/GetParams.py:694) vs [`get_holdings()`](../functions/GetParams.py:170).

### 5.2 Function-Based Architecture

The codebase is primarily organized around functions rather than classes. The few classes that exist are:

| Class | Module | Purpose |
|---|---|---|
| `allstats` | [`allstats.py`](../functions/allstats.py:4) | Statistical calculations on price arrays |
| `MonteCarloBacktest` | [`MonteCarloBacktest.py`](../functions/MonteCarloBacktest.py:1) | Monte Carlo simulation engine |
| `BacktestDataLoader` | [`abacus_backtest.py`](../functions/abacus_backtest.py:20) | Backtest data loading |
| `ConfigurationHelper` | [`abacus_recommend.py`](../functions/abacus_recommend.py:25) | Configuration management |
| `DateHelper` | [`abacus_recommend.py`](../functions/abacus_recommend.py:1) | Date calculations |
| `ModelRecommender` | [`abacus_recommend.py`](../functions/abacus_recommend.py:1) | Recommendation generation |
| `RecommendationDisplay` | [`abacus_recommend.py`](../functions/abacus_recommend.py:1) | Output formatting |
| `PlotGenerator` | [`abacus_recommend.py`](../functions/abacus_recommend.py:1) | Plot generation |
| `AllocationComputer` | [`compute_allocations.py`](../compute_allocations.py:36) | Allocation computation |
| `Task`, `Scheduler` | [`scheduler.py`](../scheduler.py:4) | Task scheduling |
| `PerformanceMetrics` | [`MonteCarloBacktest.py`](../functions/MonteCarloBacktest.py:50) | Named tuple for metrics |

The newer Abacus-related modules (`abacus_recommend.py`, `abacus_backtest.py`) use a more object-oriented design, reflecting a deliberate refactoring effort documented in [`.github/RECOMMEND_MODEL_REFACTORING_PLAN.md`](../.github/RECOMMEND_MODEL_REFACTORING_PLAN.md:1).

### 5.3 Data Flow Coupling

Functions communicate primarily through:
1. **Numpy arrays** — The dominant data structure. `adjClose` (2D: symbols × dates), `signal2D`, `gainloss` arrays flow through the computation pipeline.
2. **File I/O** — `.params` files serve as both persistent storage and inter-process communication.
3. **Function return values** — Tuples of arrays and lists (e.g., `PortfolioPerformanceCalcs` returns `lastdate, symbols, weights, prices`).
4. **Global state** — Some functions use `os.chdir()` to set working directory, and the legacy `PyTAAA.py` uses module-level variables.

---

## 6. References

1. **HDF5 Format:** The HDF Group. "Hierarchical Data Format, version 5." https://www.hdfgroup.org/solutions/hdf5/. Used via PyTables (`tables` package) for efficient storage of large numerical arrays.

2. **yfinance:** Ran Aroussi. "yfinance - Yahoo! Finance market data downloader." https://github.com/ranaroussi/yfinance. Primary data source for stock quotes.

3. **Click CLI Framework:** Pallets Projects. "Click - Python composable command line interface toolkit." https://click.palletsprojects.com/. Used in modern entry points for argument parsing.

4. **FastAPI:** Sebastián Ramírez. "FastAPI - modern, fast web framework for building APIs." https://fastapi.tiangolo.com/. Used in the complementary `pytaaa_web` codebase.

5. **Numba JIT:** Numba Development Team. "Numba: A High Performance Python Compiler." https://numba.pydata.org/. Used in [`MonteCarloBacktest.py`](../functions/MonteCarloBacktest.py:58) for performance-critical numerical computations.

6. **pandas-market-calendars:** Used in [`readSymbols.py`](../functions/readSymbols.py:9) for determining trading days and market holidays.

7. **Paramiko:** Used in [`WriteWebPage_pi.py`](../functions/WriteWebPage_pi.py:19) and [`ftp_quotes.py`](../functions/ftp_quotes.py:1) for SFTP file transfers.

8. **Hull Moving Average:** Hull, A. (2005). The HMA reduces lag compared to traditional SMAs by using weighted moving averages of different periods. Implementation in [`hma()`](../functions/TAfunctions.py:1252).

9. **Sharpe Ratio:** Sharpe, W.F. (1994). "The Sharpe Ratio." *Journal of Portfolio Management*, 21(1), 49-58. The annualized version used throughout PyTAAA follows the standard formula: `(geometric_mean_return^252 - 1) / (std_dev * sqrt(252))`.

10. **Monte Carlo Methods in Finance:** Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer. The explore-exploit search strategy in [`run_monte_carlo.py`](../run_monte_carlo.py:1) adapts concepts from this domain.
