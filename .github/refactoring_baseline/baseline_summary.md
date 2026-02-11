# Baseline Summary - recommend_model.py

**Date Generated:** January 19, 2026  
**Branch:** feature/abacus_model_switching  
**Commit:** b7939fe

---

## Test Configuration

### Command Executed
```bash
uv run python recommend_model.py \
  --json /Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json
```

### Runtime Parameters

**Lookback Periods:** [55, 157, 174] days

**Target Dates:**
- Primary: 2026-01-19 (today)
- First weekday of month: 2026-01-01

**Data Format:** backtested  
**Data Source:** pyTAAAweb_backtestPortfolioValue.params

**Models Included:**
1. cash
2. naz100_pine
3. naz100_hma
4. naz100_pi
5. sp500_hma
6. sp500_pine

### Normalization Parameters

| Metric        | Central Value | Std Deviation |
|---------------|---------------|---------------|
| Annual Return | 43.5%         | 4.0%          |
| Sharpe Ratio  | 1.450         | 0.165         |
| Sortino Ratio | 1.300         | 0.140         |
| Max Drawdown  | -54.0%        | 7.5%          |
| Avg Drawdown  | -11.5%        | 7.2%          |

### Focus Period Blending
- **Enabled:** Yes
- **Focus Period:** 2005-01-01 to 2025-10-01
- **Full Period Weight:** 0.2
- **Focus Period Weight:** 0.8

---

## Output Summary

### Recommendations Generated

**For 2026-01-19:**
- Closest available date: 2026-01-17 (2 days difference)
- **Best model:** sp500_hma

**Model Rankings (Normalized Score):**
1. sp500_hma    12.739
2. naz100_hma   11.776
3. naz100_pine   9.599
4. sp500_pine    8.927
5. cash          0.000
6. naz100_pi    -3.133

**For 2026-01-01:**
- Closest available date: 2025-12-31 (1 days difference)
- **Best model:** naz100_hma

**Model Rankings (Normalized Score):**
1. sp500_pine   14.634
2. naz100_hma   12.739
3. sp500_hma    10.405
4. naz100_pine   9.575
5. cash          0.000
6. naz100_pi    -3.133

### Portfolio Performance

**Model-Switching Portfolio:**
- Initial Value: $10,000.00
- Final Value: $20,431,068,801.83
- Annualized Return: 51.4%
- Time Period: 35.1 years

**Effectiveness Metrics:**
- Outperformance Rate: 45.0%
- vs Equal-Weight Base: 0.0% excess annual return

### Data Points Loaded
- naz100_pine: 8826 data points
- naz100_hma: 8826 data points
- naz100_pi: 8826 data points
- sp500_hma: 8826 data points
- sp500_pine: 8826 data points

---

## Output Files

1. **Console Output:** `console_output.txt`
   - Complete stdout/stderr capture
   - 100+ lines of detailed output

2. **Recommendation Plot:** `recommendation_plot_baseline.png`
   - Saved to: `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pyTAAA_web/recommendation_plot.png`
   - Baseline copy: `.github/refactoring_baseline/recommendation_plot_baseline.png`

3. **Model-Switching Portfolio Data:** `model_switching_portfolio_baseline.csv`
   - Heavy black line data from plot
   - 8835 data points (1991-01-02 to 2026-01-16)
   - Columns: date, portfolio_value
   - Initial: $10,000.00, Final: $20,431,068,801.83
   - Used for numerical comparison of refactored version

4. **Log File:** Not generated (logger may write to different location)

---

## Validation Checklist

- [x] Command executed successfully
- [x] No Python exceptions raised
- [x] Console output captured
- [x] Recommendation plot generated
- [x] Both recommendation dates processed
- [x] All 6 models loaded successfully
- [x] Portfolio metrics calculated
- [x] Normalized scores computed
- [x] Parameters summary displayed

---

## Notes for Refactoring Comparison

When comparing refactored version:
- Console output should match line-by-line (except timestamps)
- Plot should be visually identical (compare data series, not pixel-perfect)
- **Portfolio values CSV should match exactly** (verify with diff or numerical comparison)
- Normalized scores should match to at least 3 decimal places
- Model rankings should be identical
- Portfolio final value should match exactly

**Acceptable differences:**
- Timestamps in console output
- Log file locations/timestamps
- Floating-point differences < 1e-10
- Plot rendering minor differences (font antialiasing, etc.)

**Unacceptable differences:**
- Different best model recommendations
- Different model rankings
- Different normalized scores (> 0.001 difference)
- Different portfolio values (> 1e-6 relative difference)
- Missing output files
- Python exceptions or errors
