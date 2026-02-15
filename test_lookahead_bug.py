#!/usr/bin/env python3
"""Demonstrate the look-ahead bias bug in ranking calculation."""

import numpy as np
import bottleneck as bn

# Simulate the bug: 3 stocks, 5 dates
monthgainloss = np.array([
    [1.0, 1.1, 1.2, 1.3, 1.4],     # Stock A - steady growth
    [1.0, 1.05, 1.15, 1.25, 1.35], # Stock B - slower growth
    [1.0, 1.08, 1.18, 1.28, 1.38]  # Stock C - medium growth
])

print('='*70)
print('DEMONSTRATING LOOK-AHEAD BIAS BUG')
print('='*70)
print(f'\nmonthgainloss shape: {monthgainloss.shape}')
print('monthgainloss (3 stocks x 5 dates):')
print(monthgainloss)
print()

# BUGGY WAY (current code): Rank across axis=0 using ALL time data
print('BUGGY APPROACH (current code):')
print('  bn.rankdata(monthgainloss, axis=0) ranks across ALL dates at once')
rank_buggy = bn.rankdata(monthgainloss, axis=0)
print('  Result:')
print(rank_buggy)
maxrank_buggy = np.max(rank_buggy)
print(f'  maxrank from ALL time: {maxrank_buggy}')
print()

# CORRECT WAY: Rank each time column independently
print('CORRECT APPROACH (point-in-time):')
print('  Rank stocks independently for each date')
rank_correct = np.zeros_like(monthgainloss, dtype=int)
for t in range(monthgainloss.shape[1]):
    rank_correct[:, t] = bn.rankdata(monthgainloss[:, t])
print('  Result:')
print(rank_correct)
print()

print('='*70)
print('COMPARISON AT DATE INDEX 2 (the rebalancing decision date):')
print('='*70)
print(f'  Stock values at date 2: {monthgainloss[:, 2]}')
print(f'  Buggy ranks:   {rank_buggy[:, 2]}')
print(f'  Correct ranks: {rank_correct[:, 2]}')
print()
print('  ❌ Ranks are IDENTICAL here, but the NORMALIZATION differs!')
print()

# Show the normalization bug
print('='*70)
print('NORMALIZATION BUG (affects weight calculation):')
print('='*70)
print('After rank reversal: rank_reversed = -(rank - maxrank - 1) + 2')
print()

rank_buggy_reversed = -(rank_buggy - maxrank_buggy - 1) + 2
print(f'  Buggy (maxrank={maxrank_buggy} from ALL time):')
print(rank_buggy_reversed)
print()

for t in range(monthgainloss.shape[1]):
    maxrank_t = np.max(rank_correct[:, t])
    rank_correct[:, t] = -(rank_correct[:, t] - maxrank_t - 1) + 2
print('  Correct (maxrank per time period):')
print(rank_correct)
print()

print('='*70)
print('IMPACT:')
print('='*70)
print('  At date 2, portfolio decisions use:')
print(f'    Buggy:   {rank_buggy_reversed[:, 2]} (contaminated by future data)')
print(f'    Correct: {rank_correct[:, 2]} (only past data)')
print()
print('  ⚠️  This causes DIFFERENT stock selections when more data is added!')
print('='*70)
