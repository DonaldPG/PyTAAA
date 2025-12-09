# Clean Code Refactoring Plan - PyTAAA Backtesting System

## 🎯 Overview
Refactor `PyTAAA_backtest_sp500_pine.py` following Clean Code principles and
SOLID design patterns while maintaining full backward compatibility.

**File Preservation Strategy**:
- **ORIGINAL (DO NOT MODIFY)**: `PyTAAA_backtest_sp500_pine.py` - Preserved for comparison and rollback
- **REFACTORED (WORKING COPY)**: `PyTAAA_backtest_sp500_pine_refactored.py` - All changes go here
- **MODULAR CODE**: `src/backtest/` - Extracted modules imported by refactored file

**Project Structure**:
```
PyTAAA.master/
├── src/
│   ├── __init__.py
│   └── backtest/
│       ├── __init__.py           # Package exports
│       ├── config.py             # ✅ TradingConstants, BacktestConfig, FilePathConfig
│       ├── metrics.py            # Performance metrics (CAGR, Sharpe, drawdown)
│       ├── monte_carlo.py        # Monte Carlo simulation and parameter generation
│       ├── signals.py            # Signal generation (percentile channels, MAs)
│       ├── portfolio.py          # Portfolio value calculations and rebalancing
│       ├── plotting.py           # Visualization and plot generation
│       └── io.py                 # File I/O for CSV and JSON handling
├── functions/                    # Existing utility functions (unchanged)
├── PyTAAA_backtest_sp500_pine.py           # ORIGINAL - DO NOT MODIFY
├── PyTAAA_backtest_sp500_pine_refactored.py # WORKING COPY - imports from src/backtest
└── tests/
    └── test_backtest/            # Tests for backtest modules
```

## 📋 Refactoring Strategy
- **Modular approach**: Extract to `src/backtest/` package
- **Incremental migration**: One module at a time with testing
- **Zero regression**: Compare refactored output to original output
- **Recovery option**: Original file `PyTAAA_backtest_sp500_pine.py` remains untouched

## 🔄 Step-by-Step Implementation Plan

### **Step 1: Extract Constants and Configuration** ✅ *[COMPLETED]*
**Priority**: High | **Risk**: Low

#### 📝 Checklist:
- [x] Create `src/backtest/` package structure
- [x] Create `src/backtest/config.py` with `TradingConstants` class
- [x] Create `BacktestConfig` class for configuration values  
- [x] Create `FilePathConfig` class for hardcoded paths
- [ ] Update `PyTAAA_backtest_sp500_pine_refactored.py` to import from `src.backtest.config`
- [ ] Replace magic numbers with named constants in refactored file
- [ ] Fix plot filenames ("Naz100-fSMAs" → "SP500-percentileChannels")
- [ ] **Test**: Run both original and refactored, verify identical CSV output

---

### **Step 2: Extract Performance Metrics** → `src/backtest/metrics.py`
- [ ] Create `PerformanceMetrics` class
- [ ] Extract CAGR, Sharpe, drawdown, return calculations
- [ ] Remove duplicate calculations
- [ ] Update refactored file to import from metrics module

---

### **Step 3: Extract Parameter Generation** → `src/backtest/monte_carlo.py`
- [ ] Create `MonteCarloParameterGenerator` class
- [ ] Extract exploration/variant/linux parameter methods
- [ ] Simplify nested if-elif chains

---

### **Step 4: Extract Signal Generation** → `src/backtest/signals.py`
- [ ] Extract percentile channel signals
- [ ] Extract moving average fallback signals

---

### **Step 5: Data Classes** → `src/backtest/portfolio.py`
- [ ] Create `BacktestParameters`, `MarketData`, `BacktestResults` dataclasses
- [ ] Reduce function parameters from 17+ to 3-4

---

### **Step 6: Extract Plotting** → `src/backtest/plotting.py`
- [ ] Create `BacktestPlotter` class
- [ ] Extract histogram and performance plot generation

---

### **Step 7: Extract File I/O** → `src/backtest/io.py`
- [ ] Extract CSV, JSON, and results file handling

---

### **Step 8-10: Refactor Main Loop, Naming, Cleanup**
- [ ] Break down monolithic Monte Carlo loop
- [ ] Improve variable/function names
- [ ] Final cleanup and documentation

---

## 🧪 Testing Protocol

### Validation After Each Step:
```bash
# 1. Syntax check on config module
PYTHONPATH=$(pwd) uv run python -m py_compile src/backtest/config.py

# 2. Test imports work
PYTHONPATH=$(pwd) uv run python -c "
from src.backtest.config import TradingConstants, BacktestConfig, FilePathConfig
print('✓ Config imports successful')
"

# 3. Compare outputs (after completing Step 1)
# Run original:
uv run python PyTAAA_backtest_sp500_pine.py
# Run refactored:  
PYTHONPATH=$(pwd) uv run python PyTAAA_backtest_sp500_pine_refactored.py
# Compare CSV outputs for identical results
```

## 🚀 Current Status

**Step 1 Progress**: 
- ✅ Created `src/` and `src/backtest/` package structure
- ✅ Created `src/__init__.py` and `src/backtest/__init__.py`
- ✅ Created `src/backtest/config.py` with all configuration classes
- ✅ Verified imports work correctly
- ⏳ **Next**: Update refactored file to import from `src.backtest.config`