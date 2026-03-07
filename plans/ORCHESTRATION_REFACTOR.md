# PyTAAA Codebase Refactoring Plan

**Date:** 2026-03-03
**Last reviewed:** 2026-03-07 (algorithm-drift findings; algorithm restoration added)
**Status:** Planning — Phases A, E, F, G, H, Config-split complete; Algorithm Restoration item added
**Branch:** main

---

## Background

Phases A–H have been completed:

| Phase | Description | Status |
|---|---|---|
| A | Remove debug prints from `GetParams.py` public functions | ✅ Done |
| E | Extract `makeMinimumSpanningTree` → `functions/graph_plots.py` | ✅ Done |
| F | `ConfigCache` singleton; replace 6 `json.load` calls | ✅ Done |
| G | Unit tests for `sharpeWeightedRank_2D`, `computeSignal2D`, `computeDailyBacktest` | ✅ Done |
| H | Data layer side effects (duplicate functions, module-level `matplotlib.use()`, SIGINT handler, lazy loggers) | ✅ Done |
| Config split | `GetParams.py` → `config_loader.py` + `config_validators.py` + `config_accessors.py` (shim preserved) | ✅ Done |
| **AR** | **Algorithm Restoration — `delta_rank_sharpe_weight_2D` + `stockWeightMethod` dispatch** | ⏳ AR-1 done; AR-2–5 planned |

All tests pass (count will grow as items are implemented; do not hard-code the
target number in commit messages). The items below are the next work to be done.

---

## Branch Setup

All work in this plan is done on a dedicated feature branch. Never
commit refactoring changes directly to `main`.

```bash
# From the repo root, ensure main is up to date first
git checkout main && git pull

# Create and switch to the feature branch
git checkout -b orchestration-refactor

# Verify
git branch
```

The branch persists for the duration of the plan. Each phase is committed
and pushed to `origin/orchestration-refactor`. A pull request to `main`
is opened only after all phases are complete and the full comparison
checklist (see below) passes.

---

## Pre-Refactor Baseline Capture

**Complete this entire section before touching any code.** The baseline
is the ground truth used to verify that each refactoring phase produces
identical results to the original.

### Key insight: two reference levels

| Reference | Location | Purpose |
|---|---|---|
| **Static frozen reference** | `/Users/donaldpg/pyTAAA_data_static/<method>/pyTAAA_web/` | Pre-existing frozen copy of known-good output; use this as the ultimate reference |
| **Current-code baseline** | `analysis_results2/baseline_pre_refactor/<method>/` | Output of the current (pre-refactor) code; must match static reference; used to catch regressions after each phase |

After refactoring, each phase's output must match the current-code
baseline stored here. The static reference independently confirms
that the current-code baseline is itself correct.

### Why runs must be per-method

Each of the 5 trading methods has its own JSON config outside the repo:

| Method | JSON | `Valuation.webpage` output dir |
|---|---|---|
| `naz100_hma` | `/Users/donaldpg/pyTAAA_data/naz100_hma/pytaaa_naz100_hma.json` | `.../naz100_hma/webpage/` |
| `naz100_pine` | `/Users/donaldpg/pyTAAA_data/naz100_pine/pytaaa_naz100_pine.json` | `.../naz100_pine/webpage/` |
| `naz100_pi` | `/Users/donaldpg/pyTAAA_data/naz100_pi/pytaaa_naz100_pi.json` | `.../naz100_pi/webpage/` |
| `sp500_hma` | `/Users/donaldpg/pyTAAA_data/sp500_hma/pytaaa_sp500_hma.json` | `.../sp500_hma/webpage/` |
| `sp500_pine` | `/Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json` (lowercase `pytaaa_data` root) | `/Users/donaldpg/pyTAAA_data/sp500_pine/webpage/` |
| `naz100_sp500_abacus` | `/Users/donaldpg/pyTAAA_data/naz100_sp500_abacus/pytaaa_naz100_sp500_abacus.json` | `.../naz100_sp500_abacus/pyTAAA_web/` |

HTML, params, CSV, and PNG outputs for each method go to **that method's
own subdirectory** — never to the repo root. Captures using
`pytaaa_model_switching_params.json` alone miss all 4 single-method
outputs.

---

### Step 0 — Create the baseline directory tree

```bash
BASELINE=analysis_results2/baseline_pre_refactor
for m in naz100_hma naz100_pine naz100_pi sp500_hma sp500_pine \
         naz100_sp500_abacus recommend montecarlo_backtest; do
    mkdir -p "$BASELINE/$m/logs" "$BASELINE/$m/html" \
             "$BASELINE/$m/csv" "$BASELINE/$m/params"
done
echo "Baseline captured: $(date)" > "$BASELINE/README.txt"
echo "Git commit: $(git rev-parse HEAD)" >> "$BASELINE/README.txt"
```

---

### ⚠ Protect the static reference before running A–E

Each per-method JSON has a `web_output_dir` key that points **into
`pyTAAA_data_static`** — the frozen reference you are comparing against.
`WriteWebPage_pi.py` writes deployed HTML and PNG files to that
directory, which would overwrite your reference data.

Write-protect the static reference tree before any baseline run:

```bash
chmod -R a-w /Users/donaldpg/pyTAAA_data_static
```

Any write attempt to `pyTAAA_data_static` during a run will fail
silently (the deploy step is non-fatal). All primary comparison
artifacts (`*.params`, `*.csv`, HTML under `Valuation.webpage`) go to
`pyTAAA_data/<method>/` and are unaffected.

Restore write access after all baseline runs are committed:

```bash
chmod -R u+w /Users/donaldpg/pyTAAA_data_static
```

---

### Run A — `naz100_hma` method (`pytaaa_main.py`)

```bash
METH=naz100_hma
JSON=/Users/donaldpg/pyTAAA_data/$METH/pytaaa_${METH}.json
OUTDIR=/Users/donaldpg/pyTAAA_data/$METH
WEBDIR=$OUTDIR/webpage

uv run python pytaaa_main.py --json "$JSON" \
    > "$BASELINE/$METH/logs/stdout.txt" 2>&1
```

> **Wait for async backtests to finish before copying outputs.**
> The main process exits quickly but spawns background Monte Carlo
> backtest processes that write PNG files. Allow 10–20 minutes and
> confirm no `pytaaa_main` or backtest child processes remain before
> running the copy commands below.

```bash
# Numerical params (most important comparison artifact)
find "$OUTDIR" -maxdepth 1 -name '*.params' \
    -exec cp {} "$BASELINE/$METH/params/" \;
find "$OUTDIR" -maxdepth 1 -name '*.csv' \
    -exec cp {} "$BASELINE/$METH/csv/" \;

# data_store params (backtest performance records)
DATADIR=$OUTDIR/data_store
find "$DATADIR" -maxdepth 1 -name '*.params' \
    -exec cp {} "$BASELINE/$METH/params/" \;

# HTML report (written to Valuation.webpage)
find "$WEBDIR" -maxdepth 1 -name 'pyTAAAweb*.html' \
    -exec cp {} "$BASELINE/$METH/html/" \; 2>/dev/null || true

# Backtest PNG plots (async-written; only copy after processes finish)
find "$OUTDIR/webpage" -maxdepth 2 -name '*acktest*.png' \
    -exec cp {} "$BASELINE/$METH/html/" \;
```

**Verify against static reference:**
```bash
diff "$BASELINE/$METH/params/PyTAAA_ranks.params" \
     /Users/donaldpg/pyTAAA_data_static/$METH/PyTAAA_ranks.params \
    && echo "MATCH" || echo "DIFF — investigate before continuing"
```

---

### Run B — `naz100_pine` method

```bash
METH=naz100_pine
JSON=/Users/donaldpg/pyTAAA_data/$METH/pytaaa_${METH}.json
OUTDIR=/Users/donaldpg/pyTAAA_data/$METH
WEBDIR=$OUTDIR/webpage

uv run python pytaaa_main.py --json "$JSON" \
    > "$BASELINE/$METH/logs/stdout.txt" 2>&1
```

> **Wait for async backtests to finish (10–20 min) before copying.**

```bash
find "$OUTDIR" -maxdepth 1 -name '*.params' \
    -exec cp {} "$BASELINE/$METH/params/" \;
find "$OUTDIR" -maxdepth 1 -name '*.csv' \
    -exec cp {} "$BASELINE/$METH/csv/" \;
DATADIR=$OUTDIR/data_store
find "$DATADIR" -maxdepth 1 -name '*.params' \
    -exec cp {} "$BASELINE/$METH/params/" \;
find "$WEBDIR" -maxdepth 1 -name 'pyTAAAweb*.html' \
    -exec cp {} "$BASELINE/$METH/html/" \; 2>/dev/null || true
find "$OUTDIR/webpage" -maxdepth 2 -name '*acktest*.png' \
    -exec cp {} "$BASELINE/$METH/html/" \;
```

---

### Run C — `naz100_pi` method

```bash
METH=naz100_pi
JSON=/Users/donaldpg/pyTAAA_data/$METH/pytaaa_${METH}.json
OUTDIR=/Users/donaldpg/pyTAAA_data/$METH
WEBDIR=$OUTDIR/webpage

uv run python pytaaa_main.py --json "$JSON" \
    > "$BASELINE/$METH/logs/stdout.txt" 2>&1
```

> **Wait for async backtests to finish (10–20 min) before copying.**

```bash
find "$OUTDIR" -maxdepth 1 -name '*.params' \
    -exec cp {} "$BASELINE/$METH/params/" \;
find "$OUTDIR" -maxdepth 1 -name '*.csv' \
    -exec cp {} "$BASELINE/$METH/csv/" \;
DATADIR=$OUTDIR/data_store
find "$DATADIR" -maxdepth 1 -name '*.params' \
    -exec cp {} "$BASELINE/$METH/params/" \;
find "$WEBDIR" -maxdepth 1 -name 'pyTAAAweb*.html' \
    -exec cp {} "$BASELINE/$METH/html/" \; 2>/dev/null || true
find "$OUTDIR/webpage" -maxdepth 2 -name '*acktest*.png' \
    -exec cp {} "$BASELINE/$METH/html/" \;
```

---

### Run D — `sp500_hma` method

```bash
METH=sp500_hma
JSON=/Users/donaldpg/pyTAAA_data/$METH/pytaaa_${METH}.json
OUTDIR=/Users/donaldpg/pyTAAA_data/$METH
WEBDIR=$OUTDIR/webpage

uv run python pytaaa_main.py --json "$JSON" \
    > "$BASELINE/$METH/logs/stdout.txt" 2>&1
```

> **Wait for async backtests to finish (10–20 min) before copying.**

```bash
find "$OUTDIR" -maxdepth 1 -name '*.params' \
    -exec cp {} "$BASELINE/$METH/params/" \;
find "$OUTDIR" -maxdepth 1 -name '*.csv' \
    -exec cp {} "$BASELINE/$METH/csv/" \;
DATADIR=$OUTDIR/data_store
find "$DATADIR" -maxdepth 1 -name '*.params' \
    -exec cp {} "$BASELINE/$METH/params/" \;
find "$WEBDIR" -maxdepth 1 -name 'pyTAAAweb*.html' \
    -exec cp {} "$BASELINE/$METH/html/" \; 2>/dev/null || true
find "$OUTDIR/webpage" -maxdepth 2 -name '*acktest*.png' \
    -exec cp {} "$BASELINE/$METH/html/" \;
```

---

### Run E — `sp500_pine` method

**Notes:**
- JSON lives under lowercase `pytaaa_data`; outputs are written to uppercase
  `pyTAAA_data` — two different root directories.
- All `.params` and `.csv` files are in `data_store/`, not at the `OUTDIR`
  root (unlike other methods). Do not search `OUTDIR` at maxdepth 1.
- `web_output_dir` and `Valuation.webpage` are the same path for this method,
  so the `chmod -R a-w pyTAAA_data_static` guard is not needed here, but
  apply it anyway for consistency.

```bash
METH=sp500_pine
JSON=/Users/donaldpg/pytaaa_data/$METH/pytaaa_${METH}.json
OUTDIR=/Users/donaldpg/pyTAAA_data/$METH
WEBDIR=$OUTDIR/webpage
DATADIR=$OUTDIR/data_store

uv run python pytaaa_main.py --json "$JSON" \
    > "$BASELINE/$METH/logs/stdout.txt" 2>&1
```

> **Wait for async backtests to finish (10–20 min) before copying.**

```bash
# All params and CSV are under data_store/ for this method
find "$DATADIR" -maxdepth 1 -name '*.params' \
    -exec cp {} "$BASELINE/$METH/params/" \;
find "$DATADIR" -maxdepth 1 -name '*.csv' \
    -exec cp {} "$BASELINE/$METH/csv/" \;

# HTML reports and symbol chart pages
find "$WEBDIR" -maxdepth 1 -name 'pyTAAAweb*.html' \
    -exec cp {} "$BASELINE/$METH/html/" \; 2>/dev/null || true

# Backtest PNG plots (async-written; only copy after processes finish)
find "$OUTDIR" -maxdepth 3 -name '*acktest*.png' \
    -exec cp {} "$BASELINE/$METH/html/" \;
```

---

### Run F — Abacus / model-switching (`daily_abacus_update.py`)

**Important:** `daily_abacus_update.py` calls
`write_abacus_backtest_portfolio_values()` which checks
`if "abacus" not in json_config_path.lower()` — a **filename** check,
not a content check. The JSON filename must contain "abacus".

```bash
METH=naz100_sp500_abacus
JSON=/Users/donaldpg/pyTAAA_data/$METH/pytaaa_naz100_sp500_abacus.json

uv run python daily_abacus_update.py --json "$JSON" \
    > "$BASELINE/$METH/logs/stdout.txt" 2>&1

cp abacus_best_performers.csv \
    "$BASELINE/$METH/csv/" 2>/dev/null || true
find "$BASELINE" -maxdepth 1 -name '*.params' \
    -exec cp {} "$BASELINE/$METH/params/" \; 2>/dev/null || true
```

**Expected non-fatal warnings:**
- "Could not place web files on server…" — FTP not needed here.
- "unable to write updates to pyTAAAweb html" — non-fatal; caused by
  `figure*_htmlText` being `None` when MakeValuePlot cannot read HDF5
  chart data. Run still completes. The `Valuation.webpage` path must
  be writable and HDF5 files must exist at the configured path.

**Key numbers to record (grep from stdout):**
- Active model selected (grep `active_model`)
- Abacus portfolio value line

---

### Run G — Model recommendation (`recommend_model.py`)

```bash
uv run python recommend_model.py \
    --json pytaaa_model_switching_params.json \
    > "$BASELINE/recommend/logs/stdout.txt" 2>&1
```

**Key output:** The final recommendation line (BUY / HOLD / SELL) and
normalized score table. These must be identical after refactoring.

---

### Run H — Short Monte Carlo backtest (`pytaaa_backtest_montecarlo.py`)

Use a small `--trials` count. The goal is to verify the backtest
machinery produces the same numerical outputs after Item 8, not to
find optimal parameters.

```bash
uv run python pytaaa_backtest_montecarlo.py \
    --json pytaaa_sp500_pine_montecarlo.json --trials 5 \
    > "$BASELINE/montecarlo_backtest/logs/stdout.txt" 2>&1

find pngs -maxdepth 1 -name '*.png' \
    -exec cp {} "$BASELINE/montecarlo_backtest/html/" \;
find abacus_spreadsheets -maxdepth 1 -name '*.csv' \
    -exec cp {} "$BASELINE/montecarlo_backtest/csv/" \;
```

**Key numbers to record:**
- Best-trial Sharpe ratio
- Best-trial CAGR
- Best-trial max drawdown

---

### Note on `run_monte_carlo.py` (formerly Run 5)

**Do not include `run_monte_carlo.py` in the numerical baseline.**
Monte Carlo optimization is non-deterministic — output varies across
runs regardless of code changes. Include it only as a **smoke test**
("runs to completion without exception") after major refactoring phases,
not as a diffable baseline:

```bash
# Smoke test only — do not diff output
uv run python run_monte_carlo.py \
    --json pytaaa_model_switching_params.json --iterations 3 \
    > /tmp/montecarlo_smoke.txt 2>&1
echo "Exit: $?"
```

---

### Commit the baseline

```bash
git add analysis_results2/baseline_pre_refactor/
git commit -m "chore(baseline): capture pre-refactor outputs for all methods (Runs A-G)

Per-method baseline directories under analysis_results2/baseline_pre_refactor/:
  naz100_hma, naz100_pine, naz100_pi, sp500_hma, sp500_pine,
  naz100_sp500_abacus, recommend, montecarlo_backtest
Each contains logs/, html/, csv/, params/ from the current code.
Static reference: /Users/donaldpg/pyTAAA_data_static/<method>/pyTAAA_web/"
```

---

### Comparison checklist (run after each phase)

After completing any phase that touches a code path used by the runs
above, re-run the affected entry point(s) and compare. Use the method
that exercises the changed code.

```bash
BASELINE=analysis_results2/baseline_pre_refactor

##############################################################
# Per-method params comparison (most reliable numerical check)
##############################################################
METH=naz100_hma
diff "$BASELINE/$METH/params/PyTAAA_ranks.params" \
     /Users/donaldpg/pyTAAA_data/$METH/PyTAAA_ranks.params \
    && echo "OK" || echo "REGRESSION"

diff "$BASELINE/$METH/params/PyTAAA_diagnostic.params" \
     /Users/donaldpg/pyTAAA_data/$METH/PyTAAA_diagnostic.params \
    && echo "OK" || echo "REGRESSION"

##############################################################
# Stdout log comparison (filter out timestamps)
##############################################################
diff \
    <(grep -v "elapsed\|time is\|finished at\|[0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}" \
          "$BASELINE/$METH/logs/stdout.txt") \
    <(grep -v "elapsed\|time is\|finished at\|[0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\}" \
          /tmp/${METH}_post.txt)

##############################################################
# HTML report — spot-check key lines
##############################################################
grep -c "pyTAAAweb" "$BASELINE/$METH/html/pyTAAAweb.html" 2>/dev/null
grep "last_symbols_text\|cumu_value\|Lifetime profit" \
     "$BASELINE/$METH/logs/stdout.txt"
```

**Phases requiring re-run of all per-method entry points:**
- After Phase IV items (8, 1, 7, 15) — any changes to the backtest
  or orchestration pipeline

**Phases requiring only the affected method:**
- Phase I and II items — re-run only the entry point(s) that call the
  changed code

---

## Per-Phase Completion Workflow

After completing **every item or logical group of items**, perform all
five steps below before starting the next item. Do not batch these across
multiple items.

### Step 1 — Run the full test suite

```bash
PYTHONPATH=$(pwd) uv run pytest --tb=short -q
```

All tests must pass. If any fail, fix them before proceeding.

### Step 2 — Run the affected entry points and compare to baseline

Refer to the Comparison Checklist in the Baseline Capture section.
Run only the method(s) whose code paths were touched by the phase.
For Phase IV items (8, 1, 7, 15) run all per-method entry points.

Save the post-phase stdout to a temp file for diffing:

```bash
# Example for naz100_hma after a change to run_pytaaa.py
METH=naz100_hma
JSON=/Users/donaldpg/pyTAAA_data/$METH/pytaaa_${METH}.json
BASELINE=analysis_results2/baseline_pre_refactor

uv run python pytaaa_main.py --json "$JSON" \
    > /tmp/${METH}_post_phaseN.txt 2>&1

# Compare params (most reliable)
diff "$BASELINE/$METH/params/PyTAAA_ranks.params" \
     /Users/donaldpg/pyTAAA_data/$METH/PyTAAA_ranks.params \
    && echo "OK" || echo "REGRESSION"

# Compare stdout (filter timestamps)
diff \
    <(grep -v "elapsed\|time is\|finished at\|[0-9]\{2\}:[0-9]\{2\}" \
          "$BASELINE/$METH/logs/stdout.txt") \
    <(grep -v "elapsed\|time is\|finished at\|[0-9]\{2\}:[0-9]\{2\}" \
          /tmp/${METH}_post_phaseN.txt)
```

The params diff must be empty. The stdout diff must contain only
expected timestamp/elapsed-time lines. Any unexpected difference is
a regression — fix before committing.

### Step 3 — Code review

Review the diff of all staged changes:

```bash
git diff --staged
```

Check for:
- No unintended behaviour changes
- All new functions have Google-style docstrings and type annotations
- No new inline `import` statements inside functions
- No new hard-coded paths
- PEP 8 compliance (line length ≤ 79 chars)

### Step 4 — Write a session summary

Create a summary file in `docs/copilot_sessions/` using the naming
format `YYYY-MM-DD_orchestration-refactor-phaseN-itemM.md`.

Required sections (keep it concise — 1–3 sentences each):

```markdown
## Date and Context
## Problem Statement
## Solution Overview
## Key Changes   ← list of files modified and what changed
## Technical Details
## Testing
## Follow-up Items
```

### Step 5 — Commit and push

```bash
git add -p   # Stage changes interactively (avoids accidental staging)
git commit -m "<type>(scope): <description>

- File 1: what changed
- File 2: what changed
- Tests: what was added or updated"
git push origin orchestration-refactor
```

Commit format follows conventional commits:
- `refactor(scope):` — code restructuring without behaviour change
- `fix(scope):` — bug fix
- `perf(scope):` — performance improvement
- `test(scope):` — tests only
- `chore(scope):` — housekeeping (dead code, rename)

---

---

## Algorithm Restoration (Behavioral Priority)

### Background — Why the algorithm drifted

Three distinct implementations of the core portfolio weighting function
now exist across the codebase.  Their differences are documented in full
in `docs/STOCK_SELECTION_ALGORITHM_COMPARISON.md`.  In summary:

| ID | Name in code | Algorithm | Status |
|---|---|---|---|
| A | Master `sharpeWeightedRank_2D` | deltaRank momentum-of-momentum + soft signal suppression + inverse-Sharpe weights + `despike_2D` + symbol-list filter + global `rankdata` | Production in master branch only |
| B | `UnWeightedRank_2D` | deltaRank + equal weights; signal2D not applied to ranking (bug) | In worktree2; not called in production |
| C | `sharpeWeightedRank_2D` in worktree2 | Absolute Sharpe rank + hard binary signal gate + proportional-Sharpe weights + rolling window | Current worktree2 production |

**Objective:** Restore Method A as the default algorithm; retain Methods B
and C as selectable alternatives; introduce a `stockWeightMethod` JSON key
to choose between them at runtime.

---

### Item AR-1 — Port `despike_2D` and `move_sharpe_2D` into worktree2  ✅ ALREADY DONE

**Status: Pre-resolved — no code change required.**

Both helper functions already exist in worktree2:

- `despike_2D` — present in `functions/TAfunctions.py` re-export block
  (line 102); implemented in the package and called in
  `functions/output_generators.py` and
  `functions/background_plot_generator.py`.
- `move_sharpe_2D` — present in `functions/ta/rolling_metrics.py` and
  re-exported from `functions/TAfunctions.py` (line 129).

`functions/allPairsRank.py` already imports `move_sharpe_2D` from
`functions.TAfunctions` and will continue to resolve correctly.

**Files changed:** *(none)*

---

### Item AR-2 — Create `delta_rank_sharpe_weight_2D`  *(Medium Risk)*

**Problem:** No worktree2 function implements the full Method A algorithm.
The existing `UnWeightedRank_2D` is closest but omits signal masking of
`monthgainloss`, `despike_2D`, `move_sharpe_2D` weighting, and the
index-membership delta penalty.

**Solution:** Create `delta_rank_sharpe_weight_2D` in
`functions/TAfunctions.py`.  The function is a modernised port of the
master's `sharpeWeightedRank_2D` with the following design decisions:

#### Signature (proposed)

```python
def delta_rank_sharpe_weight_2D(
    json_fn: str,
    datearray: np.ndarray,
    symbols: list[str],
    adjClose: np.ndarray,
    signal2D: np.ndarray,
    signal2D_daily: np.ndarray,
    LongPeriod: int,
    numberStocksTraded: int,
    riskDownside_min: float,
    riskDownside_max: float,
    rankThresholdPct: float,
    stddevThreshold: float = 4.0,
    is_backtest: bool = False,
    makeQCPlots: bool = False,
    stockList: str = "SP500",
) -> np.ndarray:
```

Returns `monthgainlossweight: np.ndarray` of shape `(n_stocks, n_dates)`.

#### Algorithm steps (in order)

1. **Spike removal** — `adjClose_despike = despike_2D(adjClose, LongPeriod,
   stddevThreshold)`.  All subsequent ratio and rank computations use
   `adjClose_despike`, not the raw `adjClose`.

2. **Signal normalisation** — Work on a local copy so the caller's array
   is not mutated.  Normalise to [0, 1]:
   ```python
   signal2D = signal2D.copy()
   signal2D -= signal2D.min()
   denom = signal2D.max()
   if denom > 0:
       signal2D /= denom
   ```

3. **Daily gain/loss with soft signal mask** — Compute `gainloss` from
   `adjClose_despike` daily ratios; multiply by `signal2D` so downtrending
   stocks contribute `gainloss = 1.0` (neutral, no gain/loss).  This is
   the soft-suppression mechanism that makes Method A different from Method C:
   downtrending stocks are not excluded, they merely compete with a
   neutral return.  `gainloss` drives the portfolio value tracking line.

4. **LongPeriod gain/loss with soft signal mask** —
   ```python
   monthgainloss = np.ones_like(adjClose_despike)
   monthgainloss[:, LP:] = adjClose_despike[:, LP:] / adjClose_despike[:, :-LP]
   monthgainloss *= signal2D          # downtrending -> neutral ratio
   monthgainloss[monthgainloss == 0] = 1.0
   ```
   This is the key difference from Method B: `monthgainloss` for a
   downtrending stock is forced to 1.0, pushing it toward the middle of
   the rank distribution rather than letting it compete on raw price momentum.

5. **Cross-sectional rank — no look-ahead bias**

   `scipy.stats.rankdata(array, axis=0)` ranks stocks against each other
   within each date column independently.  There is no look-ahead: rank at
   date `j` depends only on the values of `monthgainloss[:, j]`, not on
   any future column.  A single vectorised call is sufficient:

   ```python
   from scipy.stats import rankdata

   monthgainlossRank = rankdata(monthgainloss, axis=0).astype(float)
   # Shift previous-period gainloss back by LongPeriod and rank:
   monthgainlossPrevious = np.ones_like(monthgainloss)
   monthgainlossPrevious[:, LP:] = monthgainloss[:, :-LP]
   monthgainlossPreviousRank = rankdata(monthgainlossPrevious, axis=0).astype(float)
   ```

   Both rank arrays have shape `(n_stocks, n_dates)`.  Higher rank = higher
   LongPeriod gain.  No reversal step is needed if `delta` is defined so
   that a *rising* rank (improving momentum) produces a *higher* selection
   score (see step 6).

6. **Delta and rank-threshold penalty** —
   ```python
   delta = (monthgainlossRank - monthgainlossPreviousRank) \
           / (monthgainlossRank + rankthreshold)
   # Penalise stocks outside the acceptable rank band:
   rank_ceiling = (1.0 - rankThresholdPct) \
                  * (monthgainlossRank.max() - monthgainlossRank.min())
   delta[monthgainloss > rank_ceiling] = -n_stocks / 2.0
   ```
   A positive `delta` means the stock's rank is *improving* faster than
   its current rank position, which is the momentum-of-momentum signal.

7. **Index-membership penalty** — Read `currentSymbolList` via
   `read_symbols_list_local(json_fn)`.  Penalise ex-index stocks so they
   sink to the bottom of `deltaRank`:
   ```python
   for i, sym in enumerate(symbols):
       if sym not in currentSymbolList and sym != "CASH":
           delta[i, :] = -n_stocks / 2.0
   ```
   When `active_mask` is provided (from `detect_infilled_from_df` via the
   data pipeline), additionally zero-out delta for infilled dates without
   iterating over symbols:
   ```python
   if active_mask is not None:
       delta[~active_mask] = -n_stocks / 2.0
   ```

8. **`deltaRank` and month carry-forward** —
   ```python
   deltaRank = rankdata(delta, axis=0)   # cross-sectional, no look-ahead
   # Freeze selections at month-start: carry forward within each calendar month
   for j in range(1, n_dates):
       if datearray[j].month == datearray[j - 1].month:
           monthgainloss[:, j] = monthgainloss[:, j - 1]
           deltaRank[:, j]     = deltaRank[:, j - 1]
   ```

9. **Selection mask — top-N by deltaRank** —
   ```python
   # Higher deltaRank = faster-improving momentum = selected
   selected = deltaRank >= (deltaRank.max(axis=0) - numberStocksTraded - 0.5)
   # selected shape: (n_stocks, n_dates), dtype bool
   ```
   This mirrors the pattern in the study code (`rank >= rank.max() - N - 0.5`)
   applied column-wise.  No per-date loop is required.

10. **Inverse-Sharpe weighting and normalisation** —
    Compute `riskDownside = 1 / move_sharpe_2D(adjClose, gainloss, LongPeriod)`,
    clip to `[riskDownside_min, riskDownside_max]`, fill NaN.  Then build
    the weight array in three vectorised lines:
    ```python
    raw_weights = selected.astype(float) / riskDownside  # high Sharpe -> large weight
    col_sums = raw_weights.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0                        # avoid divide-by-zero
    monthgainlossweight = raw_weights / col_sums          # columns sum to 1.0
    ```
    No per-date loop is needed.  The `/col_sums` normalisation guarantees
    every date's weights sum to exactly 1.0 regardless of how many stocks
    are selected.  No post-loop clipping or fallback loop is required —
    the only edge case is `col_sums == 0` (no stocks selected on a date),
    which is handled by the guard above; those dates get zero weight for
    all stocks (the pipeline caller handles allocation to CASH in that case).

#### Scope and constraints
- The function must NOT write any files when called from a backtest
  (`is_backtest=True`).  The uptrending-symbols log file write and any
  QC plot generation must be gated on `not is_backtest` and `makeQCPlots`.
- The function must NOT mutate the caller's `signal2D` array.  Operate on
  a local copy (`signal2D = signal2D.copy()` at entry).
- All `print()` calls in master must be replaced with `logger.debug()` via
  `get_logger(__name__)` (project convention).
- The `activeCount` / `rankthresholdpercentequiv` loop from the master
  is not required by the simplified implementation: selection is done
  by `deltaRank >= deltaRank.max(axis=0) - numberStocksTraded - 0.5` and
  normalisation is a single `/ col_sums` divide.  Omit `activeCount`.

**Tests required:**
- Synthetic 3-stock × 60-date test: verify the stock with the highest
  persistent `monthgainloss` improvement is selected; verify output weights
  sum to ≈1.0 for every date column.
- Verify that `weights.sum(axis=0)` is exactly 1.0 (not just approximately)
  for all dates except those where no stock is selected (col_sums == 0).
- Verify that a stock with constant price (simulating infill) falls out
  of selection after `LongPeriod` dates.
- Verify that setting `signal2D` to all-zero forces all stocks to neutral
  `monthgainloss = 1.0`; the selection should then fall back to pure
  deltaRank ordering (all stocks compete equally).
- Verify `is_backtest=True` suppresses any file writes.

**Files changed:**
- `functions/TAfunctions.py` — add `delta_rank_sharpe_weight_2D`
- `tests/test_delta_rank_sharpe_weight.py` — new test file

---

### Item AR-3 — `stockWeightMethod` JSON key + validation  *(Low Risk)*

**Problem:** No mechanism exists to choose among the three weighting
algorithms at runtime.  The current worktree2 hard-codes Method C.

**Solution:** Add `stockWeightMethod` to the validated JSON parameter set.

#### Valid values

| Value | Algorithm | Function called |
|---|---|---|
| `"delta_rank_sharpe_weight"` | Method A — restored master | `delta_rank_sharpe_weight_2D()` |
| `"equal_weight"` | Method B — equal weight | `UnWeightedRank_2D()` |
| `"abs_sharpe_weight"` | Method C — absolute Sharpe | `sharpeWeightedRank_2D()` |

**Default:** `"delta_rank_sharpe_weight"` when the key is absent from the
JSON file.  This is the intended behavior for all existing configs that
have not yet been updated.

**Where to add validation:**
- `functions/config_validators.py` — add `stockWeightMethod` to the
  allowed-keys set and validate its value against the three strings above.
- `functions/config_accessors.py` — add a `get_stock_weight_method(params)`
  accessor that returns the value with the default applied.
- `pytaaa_generic.json` — add `"stockWeightMethod": "delta_rank_sharpe_weight"`
  inside the `Valuation` sub-object (alongside `uptrendSignalMethod` which
  already lives at `Valuation.uptrendSignalMethod`).  The top-level JSON
  object has five keys (`Email`, `FTP`, `Setup`, `Valuation`, `stock_server`);
  all trading-method parameters are under `Valuation`.

**Note on `uptrendSignalMethod`:** This existing key controls how
`computeSignal2D` determines whether a stock is uptrending.  It is
orthogonal to `stockWeightMethod`.  Both keys must be present (or
default) for the system to operate.

**Files changed:**
- `functions/config_validators.py`
- `functions/config_accessors.py`
- `pytaaa_generic.json`

---

### Item AR-4 — Wire 3-way dispatch in call sites  *(Medium Risk)*

**Problem:** `sharpeWeightedRank_2D` (Method C) is called directly in
four locations without any `stockWeightMethod` awareness:

1. `functions/output_generators.py` line 1209 — live portfolio computation
2. `functions/backtesting/core_backtest.py` line 578 — backtest computation
3. `functions/dailyBacktest.py` line 360 — legacy daily backtest path
4. `functions/dailyBacktest_pctLong.py` line 2114 — legacy long-form backtest

A fifth location, `functions/backtesting/parameter_exploration.py`, builds
parameter dicts that include `uptrendSignalMethod` but not `stockWeightMethod`;
these dicts must also carry the new key.

Note: `functions/allPairsRank.py` defines its own `allPairs_sharpeWeightedRank_2D`
which is a different function — it does not call the dispatch path and
does not need to be updated.

**Solution:** Replace the direct `sharpeWeightedRank_2D(...)` calls
with a 3-way conditional as specified by the user:

```python
stock_weight_method = get_stock_weight_method(params)  # from config_accessors

if stock_weight_method == "delta_rank_sharpe_weight":
    monthgainlossweight = delta_rank_sharpe_weight_2D(
        json_fn, datearray, symbols, adjClose_despike,
        signal2D, signal2D_daily, LongPeriod, numberStocksTraded,
        riskDownside_min, riskDownside_max, rankThresholdPct,
        stddevThreshold=stddevThreshold,
        is_backtest=is_backtest, makeQCPlots=makeQCPlots,
        stockList=params.get("stockList", "SP500"),
    )
elif stock_weight_method == "equal_weight":
    monthgainlossweight = UnWeightedRank_2D(
        datearray, adjClose_despike,
        signal2D, LongPeriod, numberStocksTraded,
        riskDownside_min, riskDownside_max, rankThresholdPct,
    )
elif stock_weight_method == "abs_sharpe_weight":
    monthgainlossweight = sharpeWeightedRank_2D(
        json_fn, datearray, symbols, adjClose_despike,
        signal2D, signal2D_daily, LongPeriod, numberStocksTraded,
        riskDownside_min, riskDownside_max, rankThresholdPct,
        stddevThreshold=stddevThreshold,
    )
else:
    raise ValueError(
        f"Unknown stockWeightMethod: {stock_weight_method!r}.  "
        "Valid values: 'delta_rank_sharpe_weight', 'equal_weight', "
        "'abs_sharpe_weight'."
    )
```

**Note on `adjClose_despike`:** All three branches should receive
`adjClose_despike` (output of `despike_2D`) rather than raw `adjClose`,
because `delta_rank_sharpe_weight_2D` calls `despike_2D` internally for
Method A and the other methods benefit from the same clean prices.  For
Methods B and C the `despike_2D` step in the caller is a new pre-processing
step that was not present before; validate that this does not significantly
alter outputs for existing production configs before deploying.

**Also update `backtesting/parameter_exploration.py`:** Any dict-building
function that sets `uptrendSignalMethod` must also set `stockWeightMethod`
(defaulting to `"delta_rank_sharpe_weight"` when absent from the source
params dict).  This ensures Monte Carlo trials carry the correct method
through the entire backtest pipeline.

**Also update `backtesting/output_writers.py`:** The CSV column list
already includes `uptrendSignalMethod`; add `stockWeightMethod` alongside
it so trial results are fully attributable.

**Files changed:**
- `functions/output_generators.py`
- `functions/backtesting/core_backtest.py`
- `functions/backtesting/parameter_exploration.py`
- `functions/backtesting/output_writers.py`

---

### Item AR-5 — Update JSON config files  *(No Risk)*

Add `"stockWeightMethod": "delta_rank_sharpe_weight"` to each per-method
JSON config that specifies an `uptrendSignalMethod`.  The default behaviour
will apply if the key is absent, but explicit values in the config files
make the choice visible and auditable.

**Files to update** (all outside the repo; update in-place):
- `/Users/donaldpg/pyTAAA_data/naz100_hma/pytaaa_naz100_hma.json`
- `/Users/donaldpg/pyTAAA_data/naz100_pine/pytaaa_naz100_pine.json`
- `/Users/donaldpg/pyTAAA_data/naz100_pi/pytaaa_naz100_pi.json`
- `/Users/donaldpg/pyTAAA_data/sp500_hma/pytaaa_sp500_hma.json`
- `/Users/donaldpg/pytaaa_data/sp500_pine/pytaaa_sp500_pine.json`
- `pytaaa_model_switching_params.json` (repo root — controls Abacus)
- `pytaaa_generic.json` (template — already covered by AR-3)

Also update `pytaaa_sp500_pine_montecarlo.json` and any other Monte Carlo
config files if they specify `uptrendSignalMethod` directly.

---

### AR Sequencing and Baseline Impact

The Algorithm Restoration items are **behavioral changes**, not pure
refactors.  They change the default algorithm from Method C (current
production in worktree2) to Method A (master branch original).

This has implications for the Pre-Refactor Baseline Capture:

- The existing baseline instructions capture Method C (current code).
- After AR is implemented, Method A becomes the default and produces
  **different numbers** than the Method-C baseline.
- **Do not use Method-C baselines to validate AR changes.**
  Instead, compare AR results against Method A outputs from the master
  branch (`/Users/donaldpg/PyProjects/PyTAAA.master`).

**Recommended order:**
```
1. AR-1 — ALREADY DONE (despike_2D and move_sharpe_2D confirmed present)
2. Complete AR-3 (JSON key validation) — no behaviour change with Method-C default
3. Complete AR-4 (wire dispatch) — no behaviour change while default is still
   Method C; temporarily use "abs_sharpe_weight" as the effective default
   during development by keeping existing call paths unchanged
4. Complete AR-2 (implement delta_rank_sharpe_weight_2D)
5. Change default to "delta_rank_sharpe_weight" and run end-to-end against
   master outputs to validate
6. Complete AR-5 (update JSON configs)
```

This order allows each step to be validated independently before switching
the default.

---

## Critical Bugs to Fix First

Before any architectural work, three bugs produce silently incorrect
behaviour in every run:

1. **Broken `in locals()` state persistence** (`run_pytaaa.py` lines 128 and 159):
   `daily_update_done in locals()` evaluates to `True` or `False` and the
   result is immediately discarded. Because `run_pytaaa()` gets a fresh
   frame on every call, state never persists. Both variables are always
   reinitialised from scratch.
   **Severity is higher than it appears:** `UpdateHDF_yf` (which is slow
   and rate-limited) runs on every single scheduler invocation regardless
   of `hourOfDay`, because `daily_update_done` is always reset to `False`.
   The hour-of-day guard is completely non-functional.

2. **`import logging` inside `except` blocks** (`run_pytaaa.py`):
   Six occurrences. `logging` is imported at the top of the module by
   convention; inline imports inside `except` blocks are an antipattern and
   mask missing top-level imports.

3. **`builtins.print` permanently replaced** (`daily_abacus_update.py`
   `suppress_matplotlib_output()`):
   The function monkey-patches `builtins.print` for the lifetime of the
   process. Every subsequent `print()` call anywhere — including in
   `run_pytaaa()` — goes through a filter that may silently drop output.

---

## The 15-Item Prioritised Plan

### Item 1 — `QuoteCache` singleton  *(Medium Risk — revised from Low)*

**Problem:** `loadQuotes_fromHDF` is called 5–6 times per `run_pytaaa()`
execution from independent call sites, reading the same HDF5 file into
RAM each time.

**Known call sites per run:**
1. `UpdateHDF_yf` → staleness check
2. `PortfolioPerformanceCalcs` → `load_quotes_for_analysis`
3. `CountNewHighsLows.newHighsAndLows`
4. `MakeValuePlot.makeValuePlot` (line 758)
5. `MakeValuePlot.makeDailyChannelOffsetSignal` (line 875)
6. `stock_cluster.getClusterForSymbolsList`

**Additional call sites found in review (not in original list):**
7. `clean_quote_data.py`
8. `quotes_for_list_adjClose.py`
9. `allPairsRank.py` at module scope (see Item 16)
10. `GetYieldCurve.py` with outdated call signature (see Item 18)

**Solution:**
1. First route ALL callers through `functions/data_loaders.py` —
   `load_quotes_for_analysis()` is the intended choke point but is
   bypassed by 7+ modules that still import `loadQuotes_fromHDF`
   directly. Consolidate here before adding the cache.
2. Create `functions/quote_cache.py` mirroring `functions/config_cache.py`
   — thread-safe singleton keyed by `(symbols_file, json_fn)` tuple.
3. Call `invalidate()` inside `UpdateHDF_yf` after the HDF5 write.

**Risk note (revised to Medium):** Correctness depends on exhaustively
identifying every write site. One missed `invalidate()` call is a silent
data-freshness bug in production. Do Item 8 (backtest consolidation)
**before** this item — backtesting reorganisation creates new call sites
that must also be audited.

**Files changed:** `functions/quote_cache.py` (new),
`functions/data_loaders.py`, `functions/CountNewHighsLows.py`,
`functions/MakeValuePlot.py` (×2), `functions/stock_cluster.py`,
`functions/UpdateSymbols_inHDF5.py` (invalidate call), plus any other
direct callers identified in the full audit

---

### Item 2 — Replace `builtins.print` monkey-patch  *(Low Risk)*

**Problem:** `suppress_matplotlib_output()` in `daily_abacus_update.py`
replaces `builtins.print` with a filtered version for the whole process
lifetime. This is a global side effect that is impossible to reason about.

**Solution:** Replace the monkey-patch with a `logging.Filter` or a
`contextlib.redirect_stdout` context manager scoped to the specific
`subprocess` or function call that produces the unwanted output.

**Files changed:** `daily_abacus_update.py`

---

### Item 3 — Fix `in locals()` and inline `import logging`  *(Low Risk)*

**Problem:** See Critical Bugs section above. The operational consequence
is that `UpdateHDF_yf` runs on every scheduler invocation — `hourOfDay`
never limits it because `daily_update_done` is reset to `False` at the
start of every call with no memory of the previous call's value.

**Solution:**
- Replace the two broken `in locals()` patterns with module-level
  sentinel variables (`_daily_update_done: bool = False`,
  `_calcs_update_count: int = 0`). These will persist correctly across
  repeated scheduler calls because they live at module scope.
- Move `import logging` to the top of `run_pytaaa.py` (currently
  imported 6 times inside `except` blocks).

**Tests required:** Add a behavioral regression test confirming that
`UpdateHDF_yf` is not called on a second run when `hourOfDay > 15` and
the first run has already set the sentinel. This is the only item in the
plan where a specific test for fix correctness is mandatory.

**Files changed:** `run_pytaaa.py`

---

### Item 4 — Delete dead commented-out code  *(No Risk)*

**Problem:** `daily_abacus_update.py` contains ~80 lines of commented-out
`if update_needed:` block and several other dead code sections.
`run_pytaaa.py` contains `if 0 == 0:` as a dead guard condition.

**Caution on `if 0 == 0:`:** This guard wraps the `calculateTrades()`
call. Before deleting it, confirm that `calculateTrades()` is always
intended to run unconditionally — the guard may have been a deliberate
temporary disable. If it was intentional, replace with a proper feature
flag in the JSON config rather than dead code. If it was accidental, just
delete the `if` line and dedent the body.

**Solution:** Delete all confirmed dead code. Version control preserves
history.

**Files changed:** `daily_abacus_update.py`, `run_pytaaa.py`

---

### Item 5 — Remove hard-coded developer paths  *(Low Risk)*

**Problem:** `run_monte_carlo.py` and `recommend_model.py` both have a
legacy `else` branch containing hard-coded `/Users/donaldpg/pyTAAA_data`
paths activated when `--json` is not supplied.

**Solution:** Make `--json` required (or at minimum validate that the
fallback path is not a developer home directory). Remove the `else`
branches entirely. Update any scripts or docs that call these without
`--json`.

**Files changed:** `run_monte_carlo.py`, `recommend_model.py`

---

### Item 6 — Route all config reads through `config_cache`  *(Low Risk)*

**Problem:** `daily_abacus_update.py` defines its own `load_config_file()`
function that calls `json.load()` directly, duplicating the work already
done in `config_cache.py` (Phase F).

**Solution:** Delete `load_config_file()` from `daily_abacus_update.py`.
Replace all call sites with `config_cache.get(path)`.

**Known JSON write sites that each require a following `config_cache.invalidate(path)` call:**
1. `GetParams.py` → `put_status()` — already done (Phase F)
2. `update_json_from_csv.py` → `json.dump()` — already done (Phase F)
3. `daily_abacus_update.py` line 611 → `json.dump()` — **NOT YET DONE**,
   will silently serve stale cached config on the next read

Audit all `json.dump` call sites across the codebase before closing
this item, not just the three listed above.

**Files changed:** `daily_abacus_update.py`

---

### Item 7 — Class-based pipeline orchestration  *(High Risk — revised from Medium)*

**Problem:** `run_pytaaa.py` is a 402-line procedural function with broken
state management, doubled data loads, and no natural seams for testing.
`daily_abacus_update.py` passes a temporary config dict to `run_pytaaa()`
via a temp file, creating a fragile handoff.

**Solution:** Two-layer class architecture:

```
AbacusDailyUpdate          (daily_abacus_update.py)
  - Detects active model
  - Merges model config into a runtime config dict (in memory — no temp file)
  - Composes (not subclasses) PyTaaaDailyUpdate
  - Calls self._runner.run(runtime_config_dict)

PyTaaaDailyUpdate          (run_pytaaa.py)
  - Owns shared pipeline state as typed attributes:
      self.adj_close: pd.DataFrame
      self.signal_2d: pd.DataFrame
      self.holdings: dict
      self.last_symbols_text: list[str]
      self.last_symbols_weight: list[float]
      self.last_symbols_price: list[float]
      self._daily_update_done: bool = False
      self._calcs_update_count: int = 0
  - Methods call library functions; no algorithms in the class
  - _update_quotes()
  - _run_portfolio_calcs()
  - _build_report() → returns HTML string
  - _send_email()
  - _update_webpage()
  - run(json_fn) orchestrates all of the above
```

**Critical instance lifecycle requirement:** `PyTaaaDailyUpdate` must be
instantiated **once** and `run()` called in a loop — not re-instantiated
per iteration. If the scheduler re-creates the object each call, all
sentinel attributes reset and the problem from Item 3 is reproduced at
the class level. The instance must live at the scope of the scheduler
loop, not inside it.

**Reject temp-file IPC:** `AbacusDailyUpdate` must pass the merged config
as a dict to `PyTaaaDailyUpdate.run()`, not write a temp JSON file and
pass a path. File-path coupling between the two classes recreates the
fragile handoff the class design is meant to eliminate.

**Risk note (revised to High):** This item changes the call contract for
the live daily production pipeline and the scheduler in
`daily_abacus_update.py` simultaneously. Requires Item 3 (sentinels) and
Item 6 (config cache) to be complete first.

**Files changed:** `run_pytaaa.py`, `daily_abacus_update.py`

---

### Item 8 — Consolidate three backtest implementations  *(Medium Risk)*

**Problem:** Three separate implementations exist:
- `functions/dailyBacktest_pctLong.py` (3,120 lines) — original, still
  called by `MakeValuePlot.makeDailyMonteCarloBacktest`
- `functions/backtesting/core_backtest.py` (968 lines) — newer refactor
- `functions/backtesting/monte_carlo_runner.py` (613 lines) — newer refactor

Sharpe/CAGR math is duplicated independently across all three. The
`backtesting/` sub-package is a refactor of `dailyBacktest_pctLong.py`
but both are active.

**Solution:**
1. **Before touching any code:** Create golden-file regression tests that
   run both implementations on identical synthetic input (identical HDF5
   fixture, identical parameters) and assert outputs match within a numeric
   tolerance (e.g. `np.allclose(old_result, new_result, rtol=1e-6)`).
   This is mandatory — without it the consolidation is a black-box swap.
2. Consolidate the three separate Sharpe/CAGR implementations into a
   single `functions/metrics.py` (or extend
   `functions/ta/rolling_metrics.py`). Do this first so the other two
   implementations can reference it.
3. Update `MakeValuePlot.makeDailyMonteCarloBacktest` to import from
   `functions/backtesting/` instead of `dailyBacktest_pctLong.py`.
4. Move `dailyBacktest_pctLong.py` to `purgatory/` (do not delete yet).

**Sequencing note:** Do this item **before Item 1** (QuoteCache). The
backtest reorganisation creates new `loadQuotes_fromHDF` call sites that
must be included in the QuoteCache invalidation audit. Doing Item 1 first
means the audit must be repeated.

**Files changed:** `functions/MakeValuePlot.py`, `functions/backtesting/`,
`functions/metrics.py` (new), `functions/dailyBacktest_pctLong.py` (moved)

---

### Item 9 — Extract HTML report builder  *(Medium Risk)*  ✅ DONE (36276db)

**Problem:** `run_pytaaa.py` contains ~100 lines of raw HTML string
concatenation (`message_text = message_text + "<tr><td>..."`) mixed
directly into orchestration logic. This cannot be unit-tested.

**Solution:** Extract to `build_holdings_html_report()` in
`functions/report_builders.py`. Input: structured data
(holdings dicts, computed values). Output: HTML string.

**Strongly recommended: migrate to Jinja2 during the extraction.**
Extracting to a new file while keeping string concatenation perpetuates
the antipattern and makes a later migration to Jinja2 harder. Since the
function is being rewritten anyway, replace the concatenation with a
Jinja2 template (`functions/templates/holdings_report.html.j2`). Jinja2
is already a de-facto standard with zero operational overhead, and the
result is a pure function that is trivially testable with template
rendering assertions.

**Files changed:** `functions/report_builders.py` (new),
`functions/templates/holdings_report.html.j2` (new), `run_pytaaa.py`

---

### Item 10 — Extract deployment functions from `WriteWebPage_pi.py`  *(Low Risk)*

**Problem:** `WriteWebPage_pi.py` mixes HTML templating with FTP/rsync
deployment (`ftpMoveDirectory`, `piMoveDirectory`). These are two
unrelated concerns in one file.

**Solution:** Move deployment functions to `functions/deploy.py`. Update
import in `WriteWebPage_pi.py`.

**Files changed:** `functions/deploy.py` (new), `functions/WriteWebPage_pi.py`

---

### Item 11 — Move process management out of `MakeValuePlot.py`  *(Low Risk)*

**Problem:** `MakeValuePlot.py` manages background subprocess lifecycle
for Monte Carlo runs. `functions/background_montecarlo_runner.py` already
exists for this purpose.

**Solution:** Move process management code from `MakeValuePlot.py` into
`functions/background_montecarlo_runner.py`. Update call sites.

**Files changed:** `functions/MakeValuePlot.py`,
`functions/background_montecarlo_runner.py`

---

### Item 12 — Delete `start_pytaaa()` wrapper in `pytaaa_main.py`  *(No Risk)*

**Problem:** `pytaaa_main.py` defines a `start_pytaaa()` function that
does nothing except call `run_pytaaa(json_fn)`. The indirection adds no
value.

**Pre-condition:** Verify that `start_pytaaa()` is not the Click/argparse
`callback` entry point before deleting it. If it is registered as the
CLI entry point, the deletion must preserve the decorator and move the
`run_pytaaa(json_fn)` call inline into the decorated function.

**Solution:** Call `run_pytaaa(json_fn)` directly from the CLI handler.
Delete `start_pytaaa()`.

**Files changed:** `pytaaa_main.py`

---

### Item 13 — Rename `PortfolioPerformanceCalcs` function  *(Low Risk — revised from Medium)*

**Problem:** `functions/PortfolioPerformanceCalcs.py` exports a function
named `PortfolioPerformanceCalcs` — a CamelCase name violating PEP 8 for
functions. Three call sites must be updated.

**Solution:** Add `run_portfolio_analysis()` as the canonical name.
Provide a deprecated alias for one release cycle, then remove it.
With exactly 3 grepped call sites and a deprecated alias in place,
this is safe to execute without risk of silent breakage.

**Call sites to update:**
- `run_pytaaa.py`
- `daily_abacus_update.py`
- Any test files referencing the old name

**Files changed:** `functions/PortfolioPerformanceCalcs.py`, `run_pytaaa.py`,
`daily_abacus_update.py`

---

### Item 14 — `MonteCarloConfig` dataclass  *(Medium Risk)*  ✅ DONE (12f0480)

**Problem:** `MonteCarloBacktest.__init__` accepts 15+ positional and
keyword parameters. Call sites are fragile — parameter order mistakes
produce silent misbehaviour.

**Solution:** Define a `MonteCarloConfig` `@dataclasses.dataclass` that
groups all constructor parameters with defaults and type annotations.
`__init__` accepts a single `config` argument. The dataclass can be
instantiated from a JSON dict via a `from_dict()` classmethod.

**Required: `__post_init__` validation.**
Several constructor parameters have interdependencies
(`focus_period_start`/`focus_period_end`, `n_lookbacks` vs array
lengths, etc.). All cross-field validation logic must be in
`__post_init__`. If this logic is not migrated from the existing
`__init__`, the dataclass will silently accept invalid configurations.

**Files changed:** `functions/MonteCarloBacktest.py`,
`functions/backtesting/monte_carlo_runner.py`, `run_monte_carlo.py`

---

### Item 15 — Decompose `MonteCarloBacktest` God Class  *(High Risk)*

**Problem:** `functions/MonteCarloBacktest.py` (2,650 lines, 40+ methods,
15+ constructor parameters) violates SRP. It mixes:
- Data loading and preparation
- Simulation / trial loop
- Performance metric calculation
- Plot generation
- State persistence (`save_state()` / `load_state()`)
- CSV logging

**Proposed decomposition (revised after review):**

| New Class | Responsibility |
|---|---|
| `MonteCarloSimulator` | Owns the trial loop and parameter sweep |
| `PerformanceEvaluator` | Delegates to `functions/metrics.py` (from Item 8) — does NOT reimplement Sharpe/CAGR |
| `MonteCarloPlotter` | All `matplotlib` plot generation |
| `MonteCarloCheckpointer` | `save_state()` / `load_state()` — pickle I/O durability only |
| `MonteCarloReporter` | CSV logging and run observability (separate lifecycle from persistence) |
| `ModelSwitchingBacktest` | `_calculate_model_switching_portfolio()` |

**Key design decisions from review:**
- `PerformanceEvaluator` must call `functions/metrics.py` (created in
  Item 8), not re-contain its own metric logic. A new independent
  implementation would create a fourth Sharpe computation — the opposite
  of the plan's goal.
- Pickle state and CSV logging have fundamentally different lifecycles
  (durability vs. observability) and belong in separate classes
  (`MonteCarloCheckpointer` and `MonteCarloReporter` respectively).
- This item **depends on Item 8** being complete so `functions/metrics.py`
  exists before `PerformanceEvaluator` is written.

**Note:** The simulation loop is tightly stateful. Decomposition requires
deep understanding of implicit coupling between `_select_best_model()`,
`_calculate_model_switching_portfolio()`, and the metrics used in model
selection. Human review required at each class split.

**Files changed:** `functions/MonteCarloBacktest.py` (major rewrite),
all callers

---

### Item 16 — Fix `allPairsRank.py` module-level HDF5 call  *(Low Risk)*

**Status: PRE-RESOLVED — no code change required.**

**Investigation result:** `functions/allPairsRank.py` lines 9-10 appear to
call `loadQuotes_fromHDF()` at module scope, but closer inspection of the
source confirmed that these lines reside inside a `'''...'''` triple-quoted
string (a module-level docstring / block comment). They are never executed
at import time. No live module-scope HDF5 call exists in this file.

**Original concern:** `functions/allPairsRank.py` calls `loadQuotes_fromHDF()`
at module scope — a real disk read that fires on any import of the module.

**Files changed:** *(none)*

---

### Item 17 — Fix `os.chdir()` at module scope in `run_pytaaa.py`  *(Low Risk)*

**Problem:** Line 24 of `run_pytaaa.py`:
```python
os.chdir(os.path.abspath(os.path.dirname(__file__)))
```
executes when the module is imported. This mutates the process working
directory globally and silently breaks any caller that has a different CWD
assumption. It is invisible to callers of `run_pytaaa()`.

**Solution:** Move the `os.chdir` into the `run_pytaaa()` function body
and document explicitly why CWD change is needed. Better still: resolve
the underlying relative-path dependency that makes the `chdir` necessary
and remove it entirely.

**Files changed:** `run_pytaaa.py`

---

### Item 18 — Replace synchronous `get_SectorAndIndustry_google()` in hot loop  *(Moderate Risk)*  ✅ DONE (f7e5c71)

**Problem:** `run_pytaaa.py` calls `get_SectorAndIndustry_google(symbol)`
inside a `for i in range(len(holdings_ranks)):` loop. This is N
synchronous web requests per daily run, in the critical path of the email
and webpage generation. A slow or unavailable Google response stalls the
entire pipeline.

**Solution:** Route through `cache/stock_fundamentals_cache.json` (the
cache infrastructure already exists). Pre-fetch sector/industry data for
all holding symbols in a single batch call before entering the loop, using
cached values where available. Fall back to empty strings on
network failure without stalling.

**Files changed:** `run_pytaaa.py`, `functions/stock_fundamentals_cache.py`

---

## Implementation Order (revised)

Items are ordered by risk and dependency. Items 8 and 1 have been
resequenced (8 before 1). Item 7 has been moved to Phase IV (risk
revised to High). Items 2 and 3 were listed in Phase II but the
Critical Bugs section states they must be fixed before any architectural
work — promoted to Phase 0 (prerequisite fixes).

```
Phase 0   (critical bugs):   3, 2              ✅ DONE (dc5ceb3, f904cdb)
Phase I   (no risk):         12, 4, 17         ✅ DONE (1e7828b)
Phase II  (low risk):        5, 6, 10+11*, 13, 16  ✅ DONE (e949c16, afbfcce)
Phase III (medium risk):     18, 9, 14         ✅ DONE (f7e5c71, 36276db, 12f0480)
Phase AR  (algorithm restoration, behavioral):
                             AR-1 ✅ done; AR-3, AR-4, AR-2, AR-5
                             (execute in that order; see AR Sequencing note)
Phase IV  (high risk):       8, 1, 7, 15
```

*Items 10 and 11 (deploy.py extraction, process mgmt extraction) are the
same type of extraction — consider combining into one commit.

**Starting point:** Phase 0 (Items 3 then 2) to remove silent bugs that
corrupt every run, then Phase I items (no-risk / no-behaviour-change),
then Phase II, in parallel where independent. Items 8 → 1 must stay in
that order within Phase IV. Item 7 depends on Items 3 and 6 being
complete. Item 15 depends on Item 8 being complete.

**Phase 0 commit order:** Item 3 first (`run_pytaaa.py` sentinel fix +
`import logging` + regression test), then Item 2 (`daily_abacus_update.py`
builtins.print removal) as a second independent commit.

---

## Agentic AI Suitability (revised)

| Category | Items | Guidance |
|---|---|---|
| Mechanical / bounded (agentic-ready) | 2–6, 9–14, 16–18, AR-1, AR-3, AR-5 | Agentic execution with human review before each commit |
| New algorithm implementation | AR-2 | Agentic draft; human review of deltaRank logic and rolling-window ranking before commit |
| Call-site wiring (behavioral) | AR-4 | Agentic execution; human must validate that Method-C results are unchanged transitionally |
| Exhaustive audit required | 1 | Agentic execution but human must validate full write-site list |
| Behavioural equivalence verification | 8 | Human review of golden-file results before routing live traffic |
| Production pipeline call-contract change | 7 | Human review at each method extraction point |
| Tightly-coupled stateful loop | 15 | Human domain judgment throughout |

---

## Constraints

- **Always use `uv run python`** — never bare `python3`
- **Always set `PYTHONPATH=$(pwd)`** when running tests
- **All new code must have Google-style docstrings and type annotations**
- **Max line length:** 79 characters (code), 72 (comments)
- **Commit format:** `<type>(scope): <description>` (conventional commits)
- **All tests must pass after every commit** (do not hard-code a count;
  the number will grow as items are implemented)

---

## Review Findings

The plan was reviewed by a sub-agent architectural review on 2026-03-03.
The findings below have been incorporated into the items above; this
section documents the review for traceability.

### Missing items (not in original plan)

| Finding | Severity | Action taken |
|---|---|---|
| `allPairsRank.py` calls `loadQuotes_fromHDF()` at module scope — same Phase H miss | Moderate | Added as Item 16 |
| `os.chdir()` at module scope in `run_pytaaa.py` mutates CWD on import | Moderate | Added as Item 17 |
| `get_SectorAndIndustry_google()` called N times in hot loop — N sync web requests per run | Moderate | Added as Item 18 |
| `holdings` dict accessed by raw string keys throughout; no `TypedDict` / dataclass | Minor | Noted; can be implemented as part of Item 7 class refactor |
| `daily_abacus_update.py` line 611 `json.dump()` is missing `config_cache.invalidate()` — stale cache bug | Moderate | Added to Item 6 with full write-site audit requirement |
| `GetYieldCurve.py` calls `loadQuotes_fromHDF()` with outdated call signature | Minor | To be caught and fixed during Item 1 write-site audit |
| f-strings throughout `run_pytaaa.py` — old-style `format()` calls | Minor | Added as Phase I candidate alongside Item 17 |
| `data_loaders.py` not consistently used — 7+ modules bypass it | Moderate | Added to Item 1 as prerequisite step |

### Risk assessment corrections

| Item | Original | Revised | Reason |
|---|---|---|---|
| Item 1 (QuoteCache) | Low | Medium | Exhaustive write-site audit required; one miss = silent production bug |
| Item 7 (class-based pipeline) | Medium | High | Changes live daily pipeline call contract + scheduler simultaneously |
| Item 13 (rename function) | Medium | Low | Exactly 3 call sites + deprecated alias makes it safe |

### Weaknesses in original items (now corrected)

- **Item 3:** The bug is more severe than described — `UpdateHDF_yf`
  runs on every scheduler call regardless of `hourOfDay`. The hour guard
  is completely non-functional, not just ineffective at state persistence.
  Item 3 now includes a mandatory behavioral regression test.

- **Item 6:** Original plan said "call `invalidate()` wherever the JSON
  file is written" without enumerating write sites. `daily_abacus_update.py`
  line 611 was missing. Now explicitly listed per site.

- **Item 7:** The original plan did not specify where `PyTaaaDailyUpdate`
  is instantiated. If re-instantiated per scheduler loop iteration, the
  class reproduces the exact Item 3 bug at class level. Now specifies
  the instance must live at scheduler scope.

- **Item 8:** "Verify numerical equivalence" had no methodology. Now
  requires golden-file regression tests with synthetic HDF5 fixtures
  before any route-switching. Sequencing updated: Item 8 precedes Item 1.

- **Item 9:** Extracting string concatenation to a new file perpetuates
  the antipattern. Now recommends Jinja2 template migration during
  extraction.

- **Item 12:** Added pre-condition check that `start_pytaaa()` is not
  the Click entry point before deleting.

- **Item 14:** Added `__post_init__` validation requirement for
  cross-field constraints in the `MonteCarloConfig` dataclass.

- **Item 15:** `MonteCarloStateManager` split into `MonteCarloCheckpointer`
  (pickle durability) and `MonteCarloReporter` (CSV observability) because
  they have fundamentally different lifecycles. `PerformanceEvaluator`
  must delegate to `functions/metrics.py` from Item 8 to avoid creating
  a fourth independent Sharpe implementation.

### Sequencing corrections

- Item 8 (backtest consolidation) must precede Item 1 (QuoteCache)
  because the backtest reorganisation creates new HDF5 call sites that
  must be in the write-site audit.
- Item 7 depends on Items 3 and 6 being complete first.
- Item 15 depends on Item 8 being complete (needs `functions/metrics.py`).

### Test strategy gaps identified

- No integration test for the full daily pipeline end-to-end with a
  synthetic HDF5 fixture (would catch Item 7 regressions)
- No golden-file regression test before Item 8 consolidation
- No behavioral test for the Item 3 bug fix
  (`UpdateHDF_yf` not called twice in one day)
- Test count in constraints was hard-coded at "32" — revised to "all
  tests must pass" since the count grows with each phase
