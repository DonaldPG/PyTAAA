# Tail Hedge Simulation Framework

This project models asymmetric risk and tail-hedging strategies for portfolio construction using Monte Carlo simulation and behavioral overlays. Inspired by Deutsche Bank's asymmetric strategy and Morgan Stanley's alternative risk premia.

## Modules

- `risk_premia.py`: Simulates alternative risk premia (momentum, carry, value, volatility).
- `asymmetric_sim.py`: Monte Carlo engine with volatility spikes and downside asymmetry.
- `optimizer.py`: Portfolio optimizer using semi-variance and CVaR.
- `hedge_overlay.py`: Evaluates tail-risk hedges (e.g., puts, VIX calls).
- `reporting.py`: Generates performance reports and visualizations.

## Getting Started

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run simulations from `notebooks/exploratory_analysis.ipynb`

## References

- Deutsche Bank CIO Special: Managing Investment Uncertainty
- Morgan Stanley: Introduction to Alternative Risk Premia
- Kahneman & Tversky: Prospect Theory



File Path	          Purpose
src/risk_premia.py	  Simulates alternative risk premia with asymmetric behavior
src/optimizer.py	  Optimizes portfolio using semi-variance or CVaR
src/hedge_overlay.py  Models protective put strategy as portfolio insurance
notebooks/exploratory_analysis.ipynb	Visualizes portfolio returns and hedge effects
README.md	          Documents the framework and usage
requirements.txt	  Lists Python dependencies (numpy, pandas, etc.)