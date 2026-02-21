#!/usr/bin/env python
"""Quick test to check import issues."""

import sys
print("Step 1: About to import ta.moving_averages...")
sys.stdout.flush()

from functions.ta.moving_averages import SMA
print("Step 2: Imported SMA successfully")
sys.stdout.flush()

print("Step 3: About to import ta.signal_generation...")
sys.stdout.flush()

from functions.ta.signal_generation import computeSignal2D
print("Step 4: Imported computeSignal2D successfully")
sys.stdout.flush()

print("Step 5: About to import TAfunctions...")
sys.stdout.flush()

from functions import TAfunctions
print("Step 6: Imported TAfunctions successfully")
sys.stdout.flush()

print("SUCCESS: All imports worked!")
