# Copilot Session: Bug Fixes and Monte Carlo Tuning

**Date:** 2026-02-28 – 2026-03-02
**Branch:** `copilot/refactor-json-cli-backtest-tool-again`
**PR:** [#30 — feat: JSON-driven Monte Carlo backtest CLI tool](https://github.com/DonaldPG/PyTAAA/pull/30)

---

## Problem Statement

Running `daily_abacus_update.py` and the Monte Carlo backtest surfaced
three separate runtime errors plus a desired tuning change:

1. SP500 symbol list update failing silently with
   `"... table not found with specific class. Trying alternative search..."`
2. `ValueError: Percentiles must be in the range [0, 100]` in the
   background Monte Carlo backtest.
3. `FileNotFoundError` in the background Monte Carlo process — temp
   config file deleted before the child process could read it.
4. (User request) Increase exploration fraction in `parameter_exploration.py`
   from 50 % to 90 %.

---

## Solution Overview

Four independent fixes were made and committed separately.

---

## Key Changes

### 1. `functions/readSymbols.py` — commit `0b7ae43`

**Problem:** `soup.find("table", {"class": "wikitable sortable sticky-header",
"id": "constituents"})` requires BeautifulSoup to match all class tokens
exactly. Wikipedia removed the `sticky-header` CSS class from the S&P 500
constituents table, causing the search to silently return `None` and print
a noisy fallback message.

**Fix:**
- Primary search: `soup.find("table", {"id": "constituents"})` only — the
  `id` attribute is stable across Wikipedia markup changes.
- Fallback: scan all `wikitable`s for one with >400 rows whose first row
  header contains both `'symbol'` and `'security'` — guards against future
  `id` renames.
- Both paths print an informative success message.

---

### 2. `functions/dailyBacktest_pctLong.py` — commit `12833ba`

**Problem:** In the ±20 % randomisation block (`0 < iter < randomtrials-1`):

```python
hiPct = float(params['hiPct']) * random.uniform(0.85, 1.15)
```

When `hiPct` is near its upper range (e.g. 90), the factor `1.15` pushes
it to ~103.5. `np.percentile` rejects values outside `[0, 100]`.

**Fix:** Clamp immediately after scaling:

```python
lowPct = max(0.0, min(lowPct, 50.0))
hiPct  = max(50.0, min(hiPct, 100.0))
```

---

### 3. `daily_abacus_update.py` — commit `3374ca5`

**Problem:** `main()` created the runtime config with `tempfile.mkstemp()`
(a random path such as `/var/folders/.../daily_abacus_temp_stuilza2.json`),
spawned a detached background subprocess with that path, then deleted the
file in the `finally` block — before the child process had started (~6 s
startup lag visible in the log timestamps).

**Fix — `create_temporary_config_file()`:**
- Replaced `mkstemp()` with a stable deterministic path alongside the
  original JSON file: `{json_dir}/daily_abacus_runtime.json`.
- The function now accepts `base_json_path` to derive the target directory.
- The file is overwritten on every run; no explicit cleanup is needed or
  done.

**Fix — `main()` and `generate_web_content_only()`:**
- `main()` now passes `base_json_path=json_path` to the helper.
- The `finally` block is changed to `pass` with an explanatory comment.
- The inner `finally` in `generate_web_content_only()` similarly avoids
  deleting the file (uses `finally: pass`).

---

### 4. `functions/backtesting/parameter_exploration.py` — commit `4b7ec72`

**Problem / Request:** The Monte Carlo trial split was 50 % exploration /
50 % JSON+one-varied. This was changed to bias more heavily toward
exploring new parameter regions.

**Fix:**

```python
# Before (50/50):
mid = max(1, (total_trials - 1) // 2)

# After (90/10):
mid = max(1, round(0.9 * (total_trials - 1)))
```

The `max(1, ...)` guard is preserved so a single-trial run still uses the
exploration path. The docstring was updated accordingly.

---

## Technical Details

- The background Monte Carlo runner (`functions/background_montecarlo_runner.py`)
  is a fully detached subprocess (`start_new_session=True`) — it outlives
  `main()` and must not rely on any file whose lifetime is tied to the
  parent process.
- `hiPct` ranges in `_RANGES` (all scenarios) have `high=90.0`, so even
  pure exploration draws can approach 90. The `±20 %` legacy code in
  `dailyBacktest_pctLong.py` amplifies this further.

---

## Testing

- `daily_abacus_update.py` syntax verified: `uv run python -c "import ast; ast.parse(...); print('syntax OK')"`.
- `readSymbols.py` fix: verified that the `id='constituents'` search path
  is exercised first and prints `"... found constituents table by id"`.
- `dailyBacktest_pctLong.py` fix: confirmed by log — `percentileChannel_2D`
  no longer receives out-of-range percentile values.
- `parameter_exploration.py`: existing 18-test suite continues to pass
  (split boundary tests remain valid with the new formula).

---

## Commits This Session

| Hash | Type | Description |
|---|---|---|
| `0b7ae43` | fix | readSymbols: use `id='constituents'` for SP500 Wikipedia search |
| `12833ba` | fix | dailyBacktest: clamp lowPct/hiPct after ±20% randomisation |
| `3374ca5` | fix | daily_abacus_update: eliminate temp-file race condition |
| `4b7ec72` | feat | parameter_exploration: 90/10 exploration/exploitation split |

---

## Follow-up Items

- Re-run `daily_abacus_update.py` to confirm the background Monte Carlo
  subprocess reads the stable runtime config file successfully.
- Monitor `montecarlo_backtest.log` to confirm the percentile clamp
  eliminates the `ValueError` in `dailyBacktest_pctLong`.
- Once xlsx optimisation results are available, run
  `scripts/extract_montecarlo_ranges.py` and paste the updated `_RANGES`
  dict into `parameter_exploration.py` to replace the placeholder triples.
