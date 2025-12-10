# Project Context

## What This Project Is
Demonstrates creating an example codebase using agentic copilot assistance for a tail-risk hedging exercise.

**Key capabilities**:
- Do Monte Carlo modeling of 2 investment and income streams with different risk
- Compare different risk profiles and combinations of the 2 investments
- Create charts and include some interactive controls to change parameters in bullet 2

## Architecture
- **Main Modules**: `src/monte_carlo/`, `src/risk_analysis/`, `src/visualization/`
- **Interface**: GUI (PySide6) with interactive controls
- **Deployment**: pip install via uv package manager

## Key Differentiators
- **Monte Carlo Risk Modeling**: Sophisticated modeling of investment and income stream risks
- **Interactive Parameter Control**: Real-time adjustment of risk parameters and visualization
- **Comparative Analysis**: Side-by-side comparison of different investment risk profiles

## Development Focus
- Financial risk modeling accuracy
- Interactive visualization workflows
- Real-time parameter adjustment capabilities

## Data Formats
- **Input**: Configuration files (.json), parameter sets (.csv)
- **Output**: Risk analysis charts (.png), comparative reports (.json), interactive dashboards
- **Test Data**: Sample investment scenarios in `test_data/scenarios/`