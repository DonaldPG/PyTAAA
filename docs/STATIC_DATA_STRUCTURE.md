# Static Data Structure

**Purpose:** Frozen copy of production data for deterministic refactoring tests  
**Location:** `/Users/donaldpg/pyTAAA_data_static/`  
**Created:** February 11, 2026  
**Source:** `/Users/donaldpg/pyTAAA_data/` (snapshot date: February 2026)

## Directory Structure

```
pyTAAA_data_static/
├── Naz100/
│   └── symbols/
│       ├── Naz100_Symbols.txt
│       └── Naz100_Symbols_.hdf5
├── SP500/
│   └── symbols/
│       ├── SP500_Symbols.txt
│       └── SP500_Symbols_.hdf5
├── naz100_pine/
│   ├── symbols/ -> ../Naz100/symbols/
│   ├── data_store/
│   │   └── *.params (output files)
│   └── pytaaa_naz100_pine.json (configured for static testing)
├── naz100_hma/
│   ├── symbols/ -> ../Naz100/symbols/
│   ├── data_store/
│   └── pytaaa_naz100_hma.json
├── naz100_pi/
│   ├── symbols/ -> ../Naz100/symbols/
│   ├── data_store/
│   └── pytaaa_naz100_pi.json
├── sp500_pine/
│   ├── symbols/ -> ../SP500/symbols/
│   ├── data_store/
│   ├── webpage/
│   └── pytaaa_sp500_pine.json
├── sp500_hma/
│   ├── symbols/ -> ../SP500/symbols/
│   ├── data_store/
│   └── pytaaa_sp500_hma.json
└── naz100_sp500_abacus/
    ├── data_store/
    ├── webpage/
    └── pytaaa_naz100_sp500_abacus.json
```

## Key Modifications from Live Data

1. **JSON configs updated:** All paths point to `/Users/donaldpg/pyTAAA_data_static/`
2. **Internet updates disabled:** `updateQuotes: false`, `downloadNewData: false` (if present)
3. **HDF5 files frozen:** No modifications to stock data files during testing
4. **Output isolation:** `.params` files written to static directory (not live)

## Example Configuration

From `sp500_pine/pytaaa_sp500_pine.json`:

```json
{
  "Valuation": {
    "symbols_file": "/Users/donaldpg/pyTAAA_data_static/SP500/symbols/SP500_Symbols.txt",
    "performance_store": "/Users/donaldpg/pyTAAA_data_static/sp500_pine/data_store",
    "webpage": "/Users/donaldpg/pyTAAA_data_static/sp500_pine/webpage",
    "stockList": "SP500"
  }
}
```

## Verification

The static data setup is verified by:

1. Running all 7 baseline commands successfully
2. Producing deterministic outputs (same inputs → same outputs)
3. No internet access required during execution
4. `.params` file checksums remain constant across runs

## Refresh Procedure

To update static data with newer market data:

```bash
# Backup current static data
mv /Users/donaldpg/pyTAAA_data_static /Users/donaldpg/pyTAAA_data_static.backup_$(date +%Y%m%d)

# Copy fresh data from live directory
rsync -av --exclude='*.log' --exclude='*.tmp' \
  /Users/donaldpg/pyTAAA_data/ \
  /Users/donaldpg/pyTAAA_data_static/

# Verify JSON configs still have correct paths
# (Re-run Phase 0 setup script if needed)
```

## Critical Notes

- **Never** modify HDF5 files in static directory manually
- **Never** run with internet updates enabled on static configs
- **Always** use static data for refactoring validation
- **Only** use live data (`/Users/donaldpg/pyTAAA_data/`) for production runs
- Static data is excluded from git (in `.gitignore`)

## Related Documentation

- [Refactoring Plan](../plans/REFACTORING_PLAN_final.md) - Phase 0
- [Refactoring Tools](../refactor_tools/README.md) - Validation scripts
