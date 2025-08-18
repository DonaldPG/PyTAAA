# Abacus Portfolio Implementation Plan with Task Breakdown

## Context and General Instructions

### Project Overview
We are implementing a model switching portfolio tracker called "naz100_sp500_abacus" that combines NAZ100 and SP500 stock universes and uses monthly model switching decisions based on the existing recommend_model.py system.

### Prerequisites
- Working PyTAAA.master codebase on branch "copilot/model-wrapper"
- Existing NAZ100 data at `/Users/donaldpg/pyTAAA_data/Naz100`
- Existing SP500 data at `/Users/donaldpg/pyTAAA_data/SP500`
- Python environment with uv dependency management
- Access to existing HDF5 files and PyTAAA infrastructure

### Development Guidelines
- Follow existing PyTAAA code patterns and conventions
- Use `uv run python` for script execution
- Write tests for each component before moving to next task
- Keep modifications to existing files minimal
- Use environment variables for configuration where possible
- Follow PEP 8 and project coding standards

### Testing Strategy
Each major section must have all tests passing before proceeding. Use pytest for testing and include both unit tests and integration tests where appropriate.

---

## Task Breakdown with Time Estimates

### Phase 1: Data Infrastructure Setup (2 hours total)

**Objective**: Create combined data source and folder structure for abacus portfolio

- [ ] **Task 1.1**: Create abacus directory structure (30 min)
  - [ ] Create `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/` with subdirectories
  - [ ] Create `data_store/`, `quotes/`, `symbols/`, `pyTAAA_web/`, `config/` folders
  - [ ] Set up proper permissions and verify access

- [ ] **Task 1.2**: Implement HDF5 data merger (60 min)
  - [ ] Create `create_abacus_hdf5.py` script
  - [ ] Load NAZ100 and SP500 HDF5 files
  - [ ] Merge symbol lists (handle duplicates)
  - [ ] Combine quote data with date alignment
  - [ ] Save to `abacus_combined.h5`

- [ ] **Task 1.3**: Generate combined symbols file (15 min)
  - [ ] Extract unique symbols from combined data
  - [ ] Create `abacus_symbols.txt` file
  - [ ] Validate symbol count and format

- [ ] **Task 1.4**: Write tests for data infrastructure (15 min)
  - [ ] Test HDF5 file creation and data integrity
  - [ ] Test symbol file generation
  - [ ] Test directory structure creation

### Phase 2: Configuration System (1.5 hours total)

**Objective**: Set up configuration files and environment variable support

- [ ] **Task 2.1**: Create abacus configuration files (30 min)
  - [ ] Create `abacus_config.json` with universe settings
  - [ ] Create `switching_history.json` template
  - [ ] Define configuration schema and defaults

- [ ] **Task 2.2**: Extend PyTAAA.py for environment variables (45 min)
  - [ ] Add environment variable support for data paths
  - [ ] Add environment variable support for symbols file
  - [ ] Add environment variable support for HDF5 file
  - [ ] Maintain backward compatibility with existing usage

- [ ] **Task 2.3**: Update model switching parameters (15 min)
  - [ ] Add abacus model path to `pytaaa_model_switching_params.json`
  - [ ] Verify integration with existing model paths
  - [ ] Test configuration loading

### Phase 3: Daily Portfolio Management (2 hours total)

**Objective**: Create daily update system for abacus portfolio

- [ ] **Task 3.1**: Implement daily abacus update wrapper (60 min)
  - [ ] Create `daily_abacus_update.py` as PyTAAA.py wrapper
  - [ ] Set environment variables for abacus data
  - [ ] Call existing PyTAAA IntervalTask function
  - [ ] Add error handling and logging

- [ ] **Task 3.2**: Implement abacus web content generation (45 min)
  - [ ] Create `generate_abacus_web.py` script
  - [ ] Add functions for switching timeline plots
  - [ ] Add functions for universe comparison charts
  - [ ] Copy existing Monte Carlo plots to web directory

- [ ] **Task 3.3**: Write daily update tests (15 min)
  - [ ] Test environment variable setting
  - [ ] Test PyTAAA integration
  - [ ] Test web content generation
  - [ ] Test error handling scenarios

### Phase 4: Monthly Universe Evaluation (2 hours total)

**Objective**: Implement monthly universe switching logic

- [ ] **Task 4.1**: Create universe evaluation system (75 min)
  - [ ] Create `monthly_universe_evaluation.py` script
  - [ ] Implement NAZ100 vs SP500 analysis functions
  - [ ] Add universe selection logic based on performance metrics
  - [ ] Add switching decision tracking and logging

- [ ] **Task 4.2**: Extend recommend_model.py for universe comparison (30 min)
  - [ ] Add `--compare-universes` command line option
  - [ ] Implement universe comparison mode
  - [ ] Generate switching decision plots
  - [ ] Maintain backward compatibility

- [ ] **Task 4.3**: Write universe evaluation tests (15 min)
  - [ ] Test universe comparison logic
  - [ ] Test switching decision making
  - [ ] Test recommend_model.py integration
  - [ ] Test plot generation

### Phase 5: Integration and Automation (1.5 hours total)

**Objective**: Create setup and automation scripts

- [ ] **Task 5.1**: Implement setup script (45 min)
  - [ ] Create `setup_abacus_portfolio.py` script
  - [ ] Automate initial directory creation
  - [ ] Automate initial data preparation
  - [ ] Add validation and error checking

- [ ] **Task 5.2**: Create daily automation script (30 min)
  - [ ] Create `run_abacus_daily.py` main runner
  - [ ] Add monthly evaluation trigger logic
  - [ ] Integrate daily update calls
  - [ ] Add comprehensive logging

- [ ] **Task 5.3**: Write integration tests (15 min)
  - [ ] Test complete setup process
  - [ ] Test daily automation flow
  - [ ] Test monthly evaluation triggers
  - [ ] Test end-to-end integration

### Phase 6: Validation and Documentation (1 hour total)

**Objective**: Final testing and documentation

- [ ] **Task 6.1**: Comprehensive system testing (30 min)
  - [ ] Run complete setup and daily update cycle
  - [ ] Verify all generated files and outputs
  - [ ] Test web content generation
  - [ ] Validate against existing PyTAAA patterns

- [ ] **Task 6.2**: Documentation and usage instructions (30 min)
  - [ ] Create README for abacus system
  - [ ] Document configuration options
  - [ ] Create usage examples
  - [ ] Document maintenance procedures

---

## Success Criteria

### Technical Requirements
- [ ] All tests pass with 100% success rate
- [ ] Combined HDF5 file contains both universe data correctly
- [ ] Daily updates generate expected portfolio values
- [ ] Monthly evaluations make reasonable universe selections
- [ ] Web outputs render correctly with abacus-specific content
- [ ] Integration with existing PyTAAA systems works seamlessly

### Quality Assurance
- [ ] Code follows established PyTAAA patterns
- [ ] Error handling covers expected failure scenarios
- [ ] Logging provides adequate debugging information
- [ ] Configuration system is intuitive and maintainable
- [ ] Performance impact on existing systems is minimal

---

## Potential Risks and Guardrails

### Data Integrity Risks
- **Risk**: Data corruption during HDF5 merging
- **Guardrail**: Implement comprehensive data validation and backup procedures

### Integration Risks
- **Risk**: Breaking existing PyTAAA functionality
- **Guardrail**: Extensive testing of existing workflows after modifications

### Performance Risks
- **Risk**: Slower execution due to larger combined dataset
- **Guardrail**: Benchmark performance and optimize data access patterns

### Configuration Complexity
- **Risk**: Configuration becomes too complex to maintain
- **Guardrail**: Keep configuration simple with sensible defaults

---

## Additional Context for Implementation

### File Locations
- Primary codebase: `/Users/donaldpg/PyProjects/worktree/PyTAAA.master`
- Data storage: `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/`
- Source data: `/Users/donaldpg/pyTAAA_data/Naz100/` and `/Users/donaldpg/pyTAAA_data/SP500/`

### Key Dependencies
- Existing PyTAAA.py for daily portfolio management
- Existing recommend_model.py for model analysis
- Existing run_monte_carlo.py for backtesting
- HDF5 libraries for data management
- Matplotlib for plot generation

### Development Environment
- Use `uv run python` for all script execution
- Run `uv run pytest` for testing
- Set PYTHONPATH to project root: `export PYTHONPATH=$(pwd)`
- Use branch "copilot/model-wrapper" for all changes

---

## Detailed Implementation Specifications

### Step 1: Create Combined Data Source

**Create combined HDF5 file** (`create_abacus_hdf5.py`):
```python
#!/usr/bin/env python3
"""Create combined NAZ100 + SP500 HDF5 file for abacus portfolio."""

def merge_hdf5_files():
    """Merge NAZ100 and SP500 HDF5 files into abacus_combined.h5."""
    # Load both HDF5 files
    # Combine symbol lists (remove duplicates)
    # Merge quote data with consistent date ranges
    # Save to /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/quotes/
```

**Abacus data folder structure**:
```
/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/
├── data_store/
│   ├── PyTAAA_status.params
│   └── pyTAAAweb_backtestPortfolioValue.params
├── quotes/
│   └── abacus_combined.h5
├── symbols/
│   └── abacus_symbols.txt (combined symbol list)
├── pyTAAA_web/
└── config/
    ├── abacus_config.json (current universe selection)
    └── switching_history.json (track decisions)
```

### Step 2: Extend PyTAAA.py for Daily Updates

**Create** `daily_abacus_update.py`:
```python
#!/usr/bin/env python3
"""Daily update script for abacus portfolio - wrapper around PyTAAA.py."""

import os
import sys
import json
from datetime import datetime

def main():
    # Load current universe selection from config
    with open('/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/config/abacus_config.json') as f:
        config = json.load(f)
    
    current_universe = config.get('current_universe', 'combined')
    
    # Set environment for PyTAAA.py to use abacus data
    os.environ['PYTAAA_DATA_PATH'] = '/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus'
    os.environ['PYTAAA_SYMBOLS_FILE'] = 'symbols/abacus_symbols.txt'
    os.environ['PYTAAA_HDF5_FILE'] = 'quotes/abacus_combined.h5'
    
    # Import and run existing PyTAAA logic
    from PyTAAA import IntervalTask
    IntervalTask()  # Run daily portfolio update
    
    # Add abacus-specific web content generation
    generate_abacus_web_content()

def generate_abacus_web_content():
    """Generate abacus-specific web content."""
    # Add universe switching indicators to plots
    # Generate switching timeline
    # Create recommendation plots
```

**Minimal modifications to PyTAAA.py** (add environment variable support):
```python
# In PyTAAA.py, add support for environment-based configuration
def get_data_path():
    return os.environ.get('PYTAAA_DATA_PATH', default_path)

def get_symbols_file():
    return os.environ.get('PYTAAA_SYMBOLS_FILE', default_symbols_file)
```

### Step 3: Extend recommend_model.py for Monthly Evaluation

**Create** `monthly_universe_evaluation.py`:
```python
#!/usr/bin/env python3
"""Monthly universe switching evaluation - enhanced recommend_model.py."""

import json
from datetime import datetime, date
from recommend_model import main as recommend_main

def evaluate_universe_switch():
    """Evaluate which universe to use for next month."""
    
    # Run recommend_model.py analysis for NAZ100 vs SP500
    naz100_recommendation = run_recommend_analysis('naz100')
    sp500_recommendation = run_recommend_analysis('sp500')
    
    # Compare performance metrics
    best_universe = select_best_universe(naz100_recommendation, sp500_recommendation)
    
    # Update abacus config with decision
    update_universe_selection(best_universe)
    
    # Generate switching decision plot (like recommendation_plot.png)
    generate_switching_plot(best_universe, naz100_recommendation, sp500_recommendation)

def run_recommend_analysis(universe: str):
    """Run recommend_model.py analysis for specific universe."""
    # Temporarily modify model paths to use specific universe
    # Run existing recommend_model logic
    # Return recommendation results
    
def select_best_universe(naz100_rec, sp500_rec):
    """Select best universe based on recommendation analysis."""
    # Compare normalized scores, Sharpe ratios, etc.
    # Return 'naz100', 'sp500', or 'combined'
```

**Minor modifications to recommend_model.py**:
```python
# Add universe comparison mode
@click.option('--compare-universes', is_flag=True, 
              help='Compare NAZ100 vs SP500 for universe switching')
def main(date, lookbacks, compare_universes):
    if compare_universes:
        return run_universe_comparison(date, lookbacks)
    # ...existing code...
```

### Step 4: Enhanced Web Content Generation

**Create** `generate_abacus_web.py`:
```python
#!/usr/bin/env python3
"""Generate abacus-specific web content."""

def generate_enhanced_web_content():
    """Generate web content with abacus enhancements."""
    
    # Run standard PyTAAA web generation
    from functions.PortfolioPerformanceCalcs import PortfolioPerformanceCalcs
    # ...existing web generation...
    
    # Add abacus-specific content
    add_switching_timeline()
    add_universe_comparison_chart()
    copy_monte_carlo_plots()  # Copy existing plots from assets/
    add_switching_rationale()

def add_switching_timeline():
    """Add timeline showing universe switching decisions."""
    # Read switching_history.json
    # Generate timeline plot
    # Add to web output

def copy_monte_carlo_plots():
    """Copy existing Monte Carlo plots to web directory."""
    import shutil
    plots_to_copy = [
        "recommendation_plot.png",
        "assets/model_switching_portfolio_performance.png"
    ]
    for plot in plots_to_copy:
        if os.path.exists(plot):
            shutil.copy(plot, web_directory)
```

### Step 5: Integration Scripts

**Create** `setup_abacus_portfolio.py`:
```python
#!/usr/bin/env python3
"""One-time setup for abacus portfolio system."""

def setup():
    # Create directory structure
    # Create combined HDF5 file
    # Generate initial configuration files
    # Set up initial universe selection
```

**Create** `run_abacus_daily.py`:
```python
#!/usr/bin/env python3
"""Daily runner combining all abacus tasks."""

def main():
    # Check if first day of month (run universe evaluation)
    if is_first_trading_day_of_month():
        print("Running monthly universe evaluation...")
        from monthly_universe_evaluation import evaluate_universe_switch
        evaluate_universe_switch()
    
    # Run daily portfolio update
    print("Running daily portfolio update...")
    from daily_abacus_update import main as daily_update
    daily_update()
    
    print("Abacus portfolio update completed.")
```

### Step 6: Configuration Files

**Create** `abacus_config.json`:
```json
{
  "portfolio_name": "naz100_sp500_abacus",
  "current_universe": "combined",
  "last_evaluation_date": "2025-08-01",
  "switching_enabled": true,
  "lookback_periods": [47, 177, 178],
  "data_sources": {
    "naz100_path": "/Users/donaldpg/pyTAAA_data/Naz100",
    "sp500_path": "/Users/donaldpg/pyTAAA_data/SP500"
  }
}
```

**Update** `pytaaa_model_switching_params.json`:
```json
{
  // Add abacus model to existing model paths
  "model_paths": {
    "abacus_portfolio": "/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/data_store/PyTAAA_status.params"
  }
}
```

---

## Key Files to Create/Modify

### New Files (minimal set):
- `create_abacus_hdf5.py` (data preparation)
- `daily_abacus_update.py` (wrapper around PyTAAA.py)
- `monthly_universe_evaluation.py` (wrapper around recommend_model.py)
- `generate_abacus_web.py` (web content enhancement)
- `setup_abacus_portfolio.py` (one-time setup)
- `run_abacus_daily.py` (daily automation)

### Files to Modify (minimal changes):
- `PyTAAA.py` (add environment variable support)
- `recommend_model.py` (add universe comparison mode)
- `run_monte_carlo.py` (already supports abacus via model paths)

---

## Testing Strategy

1. **Data Integration**: Verify combined HDF5 contains both universes correctly
2. **Daily Updates**: Test daily_abacus_update.py generates correct portfolio values
3. **Monthly Switching**: Test monthly_universe_evaluation.py makes reasonable decisions
4. **Web Output**: Verify all plots and content generate correctly
5. **Backtest Validation**: Use existing run_monte_carlo.py to validate historical performance

---

## Implementation Advantages

1. **Minimal Code Changes**: Reuses 90% of existing codebase
2. **Proven Infrastructure**: Leverages battle-tested PyTAAA.py and recommend_model.py
3. **Existing Plots**: Uses existing Monte Carlo plotting from run_monte_carlo.py
4. **Simple Maintenance**: Follows established patterns and conventions
5. **Quick Implementation**: Can be completed in days rather than weeks

This plan provides a structured approach with clear checkboxes for tracking progress, realistic time estimates, and comprehensive guardrails to ensure successful implementation.