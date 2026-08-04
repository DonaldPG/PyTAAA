# PyTAAA Margin Debt & Portfolio Visualization

## Overview
Create a script (and/or functions and classes) to create a **two‑subplot figure**:
1. **Top subplot:** Equal‑weight index average (red) and PyTAAA traded portfolio values (black).
2. **Bottom subplot:** Margin Debt / GDP ratio with long‑term trend and normalized ratio.
3. **Save png and include in webpage/pyTAAAweb.html:** Each method should include a version of this plot, updated at a frequency appropriate for new data entries in the margin debt and GDP data sources

---

## 🧩 Data Sources
- **FINRA Margin Debt:** https://www.finra.org/investors/margin-statistics  
- **FRED GDP:** Series ID `GDP`  
- **FRED Margin Debt (historical):** Series ID `BOGZ1FL663067003Q`  
- **PyTAAA HDF file:** Contains daily historical stock quotes and portfolio values.

---

## Steps / Example code -- use as a starting point and customize appropriately for inclusion in the PyTAAA repo. Note that this only guides collecting the margin debt and GDP data and plot generation. Also needed is incorporating this into ./run_pytaaa_daily.sh and showing the resulting plot in webpage/pyTAAAweb.html before the plot titled "Global Liquidity Proxy" 

```python
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

# --- Load FINRA margin debt ---
finra_url = "https://www.finra.org/sites/default/files/MarginDebt.csv"
finra = pd.read_csv(finra_url)
finra['Date'] = pd.to_datetime(finra['Date'])
finra = finra.sort_values('Date')

# --- Load FRED margin debt and GDP ---
fred_margin = pdr.DataReader('BOGZ1FL663067003Q', 'fred').resample('M').ffill()
fred_margin = fred_margin.rename(columns={'BOGZ1FL663067003Q': 'MarginDebt_FRED'})
gdp = pdr.DataReader('GDP', 'fred').resample('M').ffill().rename(columns={'GDP': 'GDP_USD'})

# --- Merge and compute ratio ---
merged = pd.merge(fred_margin, finra, left_index=True, right_on='Date', how='outer')
merged = pd.merge(merged, gdp, left_on='Date', right_index=True, how='outer').sort_values('Date')
merged['MarginDebt_USD'] = merged['Debit Balances in Customers’ Securities Margin Accounts'].combine_first(merged['MarginDebt_FRED'])
merged['MarginDebt_GDP'] = merged['MarginDebt_USD'] / merged['GDP_USD']

# --- Compute rolling lows and normalization ---
merged['Low5y']  = merged['MarginDebt_GDP'].rolling(window=60,  min_periods=1).min()
merged['Low10y'] = merged['MarginDebt_GDP'].rolling(window=120, min_periods=1).min()
merged['Normalized'] = merged['MarginDebt_GDP'] / merged['Low10y']

# --- Load PyTAAA portfolio data ---
hdf_path = 'path_to_pytaaa.h5'
with pd.HDFStore(hdf_path, 'r') as store:
    eq_index = store['equal_weight_index']
    portfolio = store['pytaaa_portfolio_value']

eq_index_m = eq_index.resample('M').mean()
portfolio_m = portfolio.resample('M').mean()

# --- Merge portfolio data ---
merged_all = merged.merge(eq_index_m, left_on='Date', right_index=True, how='left')
merged_all = merged_all.merge(portfolio_m, left_on='Date', right_index=True, how='left')

# --- Plot ---
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10,8), sharex=True)

# Top subplot
ax_top.plot(merged_all['Date'], eq_index_m, color='red', label='Equal-Weight Index')
ax_top.plot(merged_all['Date'], portfolio_m, color='black', label='PyTAAA Portfolio')
ax_top.set_title('PyTAAA Portfolio vs Equal-Weight Index')
ax_top.set_ylabel('Value')
ax_top.legend()
ax_top.grid(True)

# Bottom subplot
ax_bottom.plot(merged_all['Date'], merged_all['MarginDebt_GDP']*100, color='steelblue', label='Margin Debt / GDP (%)')
ax_bottom.plot(merged_all['Date'], merged_all['Low10y']*100, color='orange', linestyle='--', label='10-Year Rolling Low (%)')
ax_bottom.plot(merged_all['Date'], merged_all['Normalized'], color='darkred', alpha=0.6, label='Normalized Ratio')
ax_bottom.set_title('Margin Debt / GDP and Long-Term Trend')
ax_bottom.set_xlabel('Date')
ax_bottom.set_ylabel('Percent / Normalized')
ax_bottom.legend()
ax_bottom.grid(True)

plt.tight_layout()
plt.show()
```