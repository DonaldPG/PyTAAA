# Simplified Implementation Plan: Model Switching Portfolio Tracker (naz100_sp500_abacus)

## Overview

Create a streamlined abacus portfolio system by:

- Extending existing PyTAAA.py to work with combined NAZ100+SP500 data
- Modifying recommend_model.py for monthly universe switching decisions
- Adding abacus-specific web content generation
- Leveraging existing Monte Carlo backtesting (run_monte_carlo.py)

## Requirements
- Combined HDF5 Data Source: Merge NAZ100 and SP500 stock histories into unified HDF5
- Daily Updates: Modified PyTAAA.py for abacus portfolio management
- Monthly Evaluation: Enhanced recommend_model.py for universe switching
- Web Integration: Extended web generation with abacus-specific content
- Minimal Code Duplication: Reuse existing infrastructure

## Implementation Steps

### Step 1: Create Combined Data Source
Create combined HDF5 file (create_abacus_hdf5.py):
```
#!/usr/bin/env python3
"""Create combined NAZ100 + SP500 HDF5 file for abacus portfolio."""

def merge_hdf5_files():
    """Merge NAZ100 and SP500 HDF5 files into abacus_combined.h5."""
    # Load both HDF5 files
    # Combine symbol lists (remove duplicates)
    # Merge quote data with consistent date ranges
    # Save to /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/quotes/
```

### Create abacus data folder structure:
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
Create daily_abacus_update.py:
```
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

Minimal modifications to PyTAAA.py (add environment variable support):
```
# In PyTAAA.py, add support for environment-based configuration
def get_data_path():
    return os.environ.get('PYTAAA_DATA_PATH', default_path)

def get_symbols_file():
    return os.environ.get('PYTAAA_SYMBOLS_FILE', default_symbols_file)
```

### Step 3: Extend recommend_model.py for Monthly Evaluation
Create monthly_universe_evaluation.py:
```
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

Minor modifications to recommend_model.py:
```
# Add universe comparison mode
@click.option('--compare-universes', is_flag=True, 
              help='Compare NAZ100 vs SP500 for universe switching')
def main(date, lookbacks, compare_universes):
    if compare_universes:
        return run_universe_comparison(date, lookbacks)
    # ...existing code...
```

### Step 4: Enhanced Web Content Generation
Create generate_abacus_web.py:
```
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
Create setup_abacus_portfolio.py:
```
#!/usr/bin/env python3
"""One-time setup for abacus portfolio system."""

def setup():
    # Create directory structure
    # Create combined HDF5 file
    # Generate initial configuration files
    # Set up initial universe selection
```

Create run_abacus_daily.py:
```
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
Create abacus_config.json:

Update pytaaa_model_switching_params.json:
```
{
  // Add abacus model to existing model paths
  "model_paths": {
    "abacus_portfolio": "/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/data_store/PyTAAA_status.params"
  }
}
```

### Key Files to Create
New Files (minimal set):
- create_abacus_hdf5.py (data preparation)
- daily_abacus_update.py (wrapper around PyTAAA.py)
- monthly_universe_evaluation.py (wrapper around recommend_model.py)
- generate_abacus_web.py (web content enhancement)
- setup_abacus_portfolio.py (one-time setup)
- run_abacus_daily.py (daily automation)

Files to Modify (minimal changes):
-  PyTAAA.py (add environment variable support)
- recommend_model.py (add universe comparison mode)
- run_monte_carlo.py (already supports abacus via model paths)

### Testing Strategy
**Data Integration:** Verify combined HDF5 contains both universes correctly
**Daily Updates:** Test daily_abacus_update.py generates correct portfolio values
**Monthly Switching:** Test monthly_universe_evaluation.py makes reasonable decisions
**Web Output:** Verify all plots and content generate correctly
**Backtest Validation:** Use existing run_monte_carlo.py to validate historical performance

### Implementation Advantages
**Minimal Code Changes:** Reuses 90% of existing codebase
**Proven Infrastructure:** Leverages battle-tested PyTAAA.py and recommend_model.py
**Existing Plots:** Uses existing Monte Carlo plotting from run_monte_carlo.py
#**Simple Maintenance:** Follows established patterns and conventions
**Quick Implementation:** Can be completed in days rather than weeks