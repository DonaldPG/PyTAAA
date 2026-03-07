# Stock Selection Algorithm Comparison

Three implementations of the core portfolio weighting function exist across
the codebase. This document compares them side-by-side so the design intent,
actual production behaviour, and the refactored replacement can be evaluated
together.

---

## The Three Implementations

| | **A — Master (original)** | **B — worktree2 `equalWeightedRank_2D`** | **C — worktree2 `sharpeWeightedRank_2D`** |
|---|---|---|---|
| **File** | `PyTAAA.master/functions/TAfunctions.py:1784` | `worktree2/functions/TAfunctions.py:1693` | `worktree2/functions/TAfunctions.py:881` |
| **Function name** | `sharpeWeightedRank_2D` | `equalWeightedRank_2D` | `sharpeWeightedRank_2D` |
| **Called in production?** | Yes (master branch) | No | Yes (worktree2 / main) |

---

## The Role of signal2D

`signal2D` is a 2D binary array of shape `(n_stocks, n_dates)` produced by
`computeSignal2D()` before the weighting function is called. A value of 1
means the stock is in an uptrend on that date; 0 means downtrend.

### How signal2D is computed — model differences

`computeSignal2D()` is called with `uptrendSignalMethod` from each model's
JSON config. This is **the primary reason naz100_pine, naz100_hma, and
naz100_pi produce different stock selections and weights even on the same
date with the same price data**.

| Model | `uptrendSignalMethod` | Uptrend criterion |
|---|---|---|
| `naz100_hma` | `'HMAs'` | Price > `MA2factor * HMA(MA1)` OR (price > min(HMA(MA2), HMA(MA2+offset)) AND shortest HMA rising) |
| `naz100_pine` | `'minmaxChannels'` | `mediumSignal + wideSignal > 0`, where each signal is the mid-channel's position within the min/max channel band |
| `naz100_pi` | `'SMAs'` or percentile channels | Different MA periods or percentile-based channel crossovers |

Each method uses different period parameters (MA1, MA2, MA2offset,
narrowDays, mediumDays, wideDays, lowPct, hiPct) from the JSON config. Any
given stock may be signalled "uptrending" by `naz100_hma` but "downtrending"
by `naz100_pine` on the same date, producing entirely different eligible
pools and therefore different results in backtests and live operation.

All methods share these edge-case rules:

- Stock with NaN price on the last date: signal forced to 0 (partial guard
  for stocks removed from the index; fires only if `adjClose[:, -1]` is NaN)
- Stock with constant price at start of series: signal forced to 0 (detects
  infill / warm-up period before the stock started trading)

### How signal2D is used — differences across the three implementations

**A — Master `sharpeWeightedRank_2D` (full design):**

signal2D controls which stocks have a neutral `monthgainloss` entering the
ranking. It is applied to **both** the daily gain/loss and the LongPeriod
gain/loss before `deltaRank` is computed:

```python
# Normalise to 0/1 (mutates input array in-place):
signal2D -= signal2D.min()
signal2D *= signal2D.max()

gainloss = gainloss * signal2D          # daily returns for portfolio value
gainloss[gainloss == 0] = 1.0

monthgainloss = monthgainloss * signal2D   # LongPeriod gain — feeds deltaRank
monthgainloss[monthgainloss == 0] = 1.0
```

A downtrending stock (signal = 0) gets `monthgainloss = 1.0` — the neutral
ratio, meaning no gain and no loss. This pushes the stock toward the middle
of `monthgainlossRank` and consequently toward the middle of `deltaRank`,
making it very unlikely to rank in the top N. The effect is **soft and
graded**: a recently-downtrending stock that has just turned uptrending may
still carry lag in `delta` for up to LongPeriod days. It is not a hard
exclusion.

This is why different uptrend methods produce different results. Different
stocks are forced to `monthgainloss = 1.0` each month, which changes the
rank ordering, which changes `delta`, which changes who is selected.

**B — `equalWeightedRank_2D` (partial copy, contains a bug):**

signal2D is applied to **daily `gainloss` only** (used for portfolio value
tracking and the active-stock counter). `monthgainloss`, which directly
feeds `deltaRank`, is computed from raw `adjClose` **without the signal
mask**:

```python
signal_mask = (signal2D > 0.5).astype(float)
gainloss = gainloss * signal_mask          # signal applied to daily returns only
gainloss[gainloss == 0] = 1.0

# monthgainloss: no signal mask applied — raw prices compete
monthgainloss[:,LongPeriod:] = adjClose[:,LongPeriod:] / adjClose[:,:-LongPeriod]
```

This is a bug relative to A: downtrending stocks compete for selection using
their actual LongPeriod price gains regardless of their uptrend signal.
signal2D has no effect on which stocks are selected.

**C — `sharpeWeightedRank_2D` (current worktree2 production):**

signal2D is a **hard binary gate**. A stock with signal = 0 is completely
excluded from the eligible pool before any Sharpe-based ranking:

```python
signal_mask = (signal2D > 0).astype(float)
valid_signals = signal_mask[:, j] > 0
valid_sharpe  = ~np.isnan(sharpe_2d[:, j])
eligible = valid_signals & valid_sharpe
```

Rolling Sharpe is computed for all stocks, but only those with signal = 1
on day `j` can be selected. There is no grace period or decay — the
exclusion is immediate and complete.

This is a **harder** gate than A. Different uptrend methods produce different
eligible pools: a stock excluded by `naz100_hma` is invisible to the Sharpe
ranking entirely, whereas in A it merely competes with a neutral
`monthgainloss`.

### signal2D — role summary

| Aspect | A — Master | B — `equalWeightedRank_2D` | C — `sharpeWeightedRank_2D` |
|---|---|---|---|
| Applied to `monthgainloss`? | Yes — soft suppression toward neutral rank | No (bug — raw prices rank) | N/A — no LongPeriod ranking |
| Applied to daily `gainloss`? | Yes | Yes | Yes (via `signal_mask`) |
| Effect on selection | Graded suppression decaying over LongPeriod | None (bug) | Hard binary exclusion |
| Mutates input array? | Yes (in-place normalisation) | No | No |
| Why models differ | Different neutralised stocks → different deltaRank orderings | No effect on selection | Different eligible pools → different Sharpe rankings |

---

## Algorithm Steps

### Step 0 — Price pre-processing

| | A — Master | B — `equalWeightedRank_2D` | C — `sharpeWeightedRank_2D` |
|---|---|---|---|
| Spike removal | `despike_2D(adjClose, LongPeriod, stddevThreshold)` applied before any calculation | None | None |
| Signal normalisation | `signal2D -= signal2D.min(); signal2D *= signal2D.max()` — mutates input in-place | Binary mask `(signal2D > 0.5)` — no mutation | Binary mask `(signal2D > 0)` — no mutation |

---

### Step 1 — Period gain/loss

All three compute `monthgainloss[:, LP:] = adjClose[:, LP:] / adjClose[:, :-LP]`.

- **A**: Uses de-spiked prices. signal2D multiplied in so non-uptrending
  stocks have `monthgainloss = 1.0` (neutral).
- **B**: Raw `adjClose`. signal2D is **not** applied to `monthgainloss` (bug).
- **C**: Does not compute a LongPeriod `monthgainloss`. Uses rolling daily
  returns for Sharpe ratio instead.

---

### Step 2 — Selection criterion

**A (master) and B — deltaRank (momentum-of-momentum)**

```
monthgainlossRank        = rankdata(monthgainloss, axis=0)
monthgainlossPreviousRank= rankdata(monthgainloss shifted back LongPeriod, axis=0)

delta = -(monthgainlossRank - monthgainlossPreviousRank)
        / (monthgainlossRank + rankthreshold)

# Penalise stocks outside rankThresholdPct range:
delta[outside_range] = -n_stocks / 2

# A only: also penalise ex-index members:
delta[not_in_currentSymbolList] = -n_stocks / 2

deltaRank = rankdata(delta, axis=0)   # rank 1 = fastest-improving rank

selected = deltaRank <= rankthresholdpercentequiv
```

In A, signal2D has already forced non-uptrending stocks to
`monthgainloss = 1.0`, so their `delta` stays near zero and they rarely
reach the top of `deltaRank`. In B, this suppression is absent — downtrending
stocks may reach the top on raw price momentum.

**C — pure Sharpe selection**

```
sharpe_2d[i, j] = mean(returns[j-LP:j]) / std(returns[j-LP:j]) * sqrt(252)

eligible = (signal2D[:, j] > 0) & ~np.isnan(sharpe_2d[:, j])
sorted_indices = argsort(-sharpe_2d[eligible, j])
selected = sorted_indices[:numberStocksTraded]
```

No change-in-rank step. Picks the highest absolute Sharpe among uptrending
stocks on day j.

---

### Step 3 — Weighting selected stocks

| | A — Master | B — `equalWeightedRank_2D` | C — `sharpeWeightedRank_2D` |
|---|---|---|---|
| Method | Inverse-Sharpe (risk-adjusted) | Equal weight | Proportional-to-Sharpe |
| Formula | `w = (1/rankthresholdpercentequiv) / riskDownside` | `w = 1 / rankthresholdpercentequiv` | `w = sharpe / sum(sharpe[selected])` |
| `riskDownside` | `1 / move_sharpe_2D(...)`, clipped and column-normalised | Accepted but not used | Accepted but not used |
| Direction | High Sharpe -> high weight | Equal | High Sharpe -> high weight |

A and C both favour higher-Sharpe stocks but with different formulae and
clipping. B ignores Sharpe for weighting entirely.

---

### Step 4 — Index membership filter

| | A — Master | B — `equalWeightedRank_2D` | C — `sharpeWeightedRank_2D` |
|---|---|---|---|
| Ex-index stocks | `delta[not_in_currentSymbolList] = -n_stocks / 2` | None — all HDF5 symbols compete | None — signal2D gate is the only filter |

The master reads `currentSymbolList` from the symbols file at runtime and
penalises ex-index stocks explicitly. B and C rely entirely on the data
layer (constant-price detection in `computeSignal2D`) to suppress them.

---

### Step 5 — Month holding

| | A — Master | B — `equalWeightedRank_2D` | C — `sharpeWeightedRank_2D` |
|---|---|---|---|
| Mechanism | Carry-forward of `monthgainloss` and `deltaRank` within each calendar month | Same | Forward-fill zero-weight columns only |
| Effective behaviour | Weights frozen at month-start selection | Weights frozen at month-start selection | Weights recompute every trading day |

---

### Step 6 — Look-ahead bias

| | A — Master | B — `equalWeightedRank_2D` | C — `sharpeWeightedRank_2D` |
|---|---|---|---|
| Ranking scope | Global — `rankdata` across all dates simultaneously | Global — same | Point-in-time — rolling window ending at day j |
| Backtest validity | Look-ahead bias invalidates backtests | Look-ahead bias invalidates backtests | No look-ahead bias |

For live operation (only the last column used), look-ahead bias has no
effect. For backtests it inflates historical performance in A and B.
C was written specifically to eliminate this.

---

## Summary Table

| Aspect | A — Master | B — `equalWeightedRank_2D` | C — `sharpeWeightedRank_2D` (production) |
|---|---|---|---|
| Selection algorithm | deltaRank (momentum-of-momentum) | deltaRank (momentum-of-momentum) | Absolute Sharpe rank — no deltaRank |
| signal2D role in selection | Soft suppression via neutral `monthgainloss` | Not applied to ranking (bug) | Hard binary gate |
| Why models differ | Different neutralised stocks -> different deltaRank | signal2D has no ranking effect (bug) | Different eligible pools |
| Weighting | Inverse-Sharpe | Equal weight | Proportional-to-Sharpe |
| `riskDownside_min/max` used | Yes | No (API compat only) | No (API compat only) |
| Spike removal | `despike_2D` | None | None |
| Index membership filter | Runtime symbol list + delta penalty | None | None (signal2D NaN guard only) |
| Month holding | Explicit carry-forward | Explicit carry-forward | Forward-fill zeros only |
| Look-ahead bias | Yes (global rankdata) | Yes (global rankdata) | No (rolling window) |
| Called in production | master branch | Not called | worktree2 / main |

---

## Key Design Intent vs Actual Behaviour

The documented algorithm (`docs/SMAs_method.md`, `docs/PINE_method.md`)
describes **A** precisely: deltaRank selection, signal2D applied to
`monthgainloss` before ranking, inverse-Sharpe weighting, `despike_2D`,
and a runtime index-membership guard. This is what master runs in
production. Its multi-year operational divergence between models is
explained by signal2D: different uptrend methods mark different stocks as
neutral in `monthgainloss`, driving entirely different deltaRank orderings
and therefore different selections.

**B** (`equalWeightedRank_2D`) correctly implements deltaRank and month
holding, but fails to apply signal2D to `monthgainloss` (removing the
mechanism that causes models to diverge), and drops Sharpe weighting and
`despike_2D`.

**C** (`sharpeWeightedRank_2D` in worktree2) eliminates look-ahead bias but
is a fundamentally different algorithm. signal2D still causes models to
diverge (different eligible pools), but through a hard gate rather than the
designed soft suppression, and the selection criterion has no
momentum-of-momentum component.

---

## Assessment: NaN in HDF5 for Index Membership

### The current mechanism and its limitations

The current approach to ignoring non-constituent stocks relies on pattern
detection inside `computeSignal2D`:

```python
# Constant price at start of series -> signal = 0 (pre-trading infill)
index = np.argmax(np.clip(np.abs(gainloss[ii, :]-1), 0, 1e-8)) - 1
signal2D[ii, 0:index] = 0

# NaN on last date -> signal = 0 (most recent removal from index)
if jj == adjClose.shape[1]-1 and isnan(adjClose[ii, -1]):
    signal2D[ii, jj] = 0
```

This only partially works:

- Catches stocks not yet trading (constant-price fill at the start of history).
- Catches stocks removed from the index if `UpdateSymbols_inHDF5` writes
  NaN on the most recent date.
- Does **not** catch stocks dropped mid-history if the HDF5 forward-fills
  the last known price as a constant (current default on delisting).
- Does **not** catch stocks re-added to the index after a gap.
- Master compensates with a runtime `currentSymbolList` check (delta
  penalty for non-members), but this uses only the **current** symbol list
  and has no knowledge of historical membership periods.

### The NaN-in-HDF5 approach

Store NaN in `adjClose` for every date on which a stock was not a
constituent of the index. Membership becomes a data fact rather than an
inferred heuristic.

**How each implementation would respond:**

| Where NaN appears | A — Master | B — `equalWeightedRank_2D` | C — `sharpeWeightedRank_2D` |
|---|---|---|---|
| `monthgainloss` computation | NaN ratio -> set to 1.0 (neutral rank) | Same if bug is fixed; same raw race if not | Not used |
| `sharpe_2d` computation | Not used | Not used | NaN Sharpe -> `valid_sharpe = False` -> excluded |
| `computeSignal2D` | NaN guard fires if extended to all j, not just last | Same | Same |
| Effective selection | Stock neutralised, unlikely to enter top deltaRank | Same (if bug fixed) | Stock excluded via `valid_sharpe` |

**Advantages:**

- Single source of truth: index membership is encoded in the price data,
  not inferred from price patterns.
- Works identically for all signal methods without method-specific logic.
- Removes the need for master's runtime `currentSymbolList` lookup inside
  the weighting function.
- The B&H benchmark `active_mask` (commit `95de402`) becomes derivable
  directly from `~np.isnan(adjClose)` — no separate parameter needed.
- Historical backtests would correctly exclude ex-constituent stocks for
  the periods they were not in the index.

**Obstacles:**

1. **Historical constituent data**: Requires a reliable record of exactly
   which dates each stock was in the index. Nasdaq 100 rebalances
   quarterly; S&P 500 changes are announced. This must be sourced (e.g.,
   Polygon.io, Tiingo, Wikipedia revision history) or reconstructed. Without
   it the migration cannot be completed without creating silent gaps.

2. **HDF5 migration**: All existing infill periods (constant or linearly
   interpolated prices) must be replaced with NaN — a one-time bulk
   conversion of all price files.

3. **NaN propagation audit**: `adjClose_despike`, cumulative portfolio value
   (`np.cumprod`), rolling Sharpe, and `monthgainloss` all operate on
   `adjClose`. Each must handle interior NaN correctly. Most are already
   guarded (`isnan -> 1.0`), but the full computation chain needs
   verification, particularly for channel-based signal methods.

4. **`computeSignal2D` inner-loop guard**: The NaN guard only fires at
   `jj == adjClose.shape[1]-1`. To suppress mid-history non-membership dates,
   the guard must be extended to all j.

5. **`UpdateSymbols_inHDF5` update logic**: Currently forward-fills the last
   known price on delisting. It would need to write NaN once delisting or
   index removal is confirmed, and maintain a constituent list to distinguish
   stale-but-active prices from NaN-required non-membership dates.

### Recommendation

The NaN-in-HDF5 approach is architecturally correct and is the right
long-term target. The precondition is reliable historical constituent data.
Without that, the migration cannot be implemented correctly.

Suggested migration path:

1. Source historical Nasdaq 100 and S&P 500 constituent tables with
   effective dates for all additions and removals.
2. Write a one-time conversion script that reads each HDF5 symbol, applies
   the constituent table, and overwrites non-membership dates with NaN.
3. Extend the NaN guard in `computeSignal2D` from last-date-only to all
   dates (remove the `jj == adjClose.shape[1]-1` condition).
4. Remove the `currentSymbolList` runtime lookup from the master weighting
   function.
5. Derive `active_mask` from `~np.isnan(adjClose)`.

Until constituent data is available, the existing constant-price detection
remains the best available approximation. Extending it to detect linearly
interpolated (not just constant) fill segments would improve coverage for
the mid-history delisting case.

---

## Open Design Questions

1. **deltaRank with point-in-time ranking**: Should C restore deltaRank
   using a rolling window (rank `monthgainloss` for each j over the trailing
   LongPeriod window) to combine momentum-of-momentum selection with
   backtest validity?

2. **`despike_2D`**: Present in A, absent from B and C. Prevents a single
   bad price print from distorting an entire month's selection.

3. **signal2D applied to `monthgainloss` in B**: This is a bug. B should
   apply the signal mask to `monthgainloss` as A does, or be retired.

4. **`riskDownside_min/max` parameters**: Both B and C accept these but
   discard them. If inverse-Sharpe weighting is restored, these bounds
   become meaningful.

5. **NaN-in-HDF5**: Adopt once historical constituent data is available.
