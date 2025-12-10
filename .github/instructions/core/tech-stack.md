# Technology Stack

## Core Technologies
- **Python**: 3.11+ (required for modern type hints and performance improvements)
- **PySide6**: Latest (GUI framework for interactive controls and visualization)
- **NumPy**: Latest (numerical computing for Monte Carlo simulations)

## Dependency Management

- **Tool**: UV (fast Python package installer and resolver)
- **Files**: pyproject.toml + uv.lock
- **Standard**: ALL Python commands must use `uv run` prefix

## Key Libraries

```python
# Core processing
import numpy as np           # Monte Carlo simulations and numerical computing

# GUI and visualization
from PySide6 import QtWidgets, QtCore, QtGui  # Interactive GUI framework
import matplotlib.pyplot as plt              # Chart generation and plotting

# Domain-specific libraries (to be added)
# import scipy.stats          # Statistical distributions for risk modeling
# import pandas as pd         # Data manipulation for analysis results
```

## Testing Framework

- **Framework**: pytest
- **Structure**: `tests/test_*.py`
- **Test Data**: `test_data/scenarios/` for sample investment scenarios

## Deployment

- **Primary Method**: pip install via uv package manager
- **CI/CD**: None
- **Target Environment**: Cross-platform desktop application
- **Distribution**: Direct installation via uv