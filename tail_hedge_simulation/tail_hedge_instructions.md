These are  **structured prompts** for VS Code with Copilot to build a custom portfolio framework inspired by the Deutsche Bank asymmetric strategy and Morgan Stanley’s alternative risk premia concepts. Each prompt will include:

- **Objective / Requirement**
- **Suggested Methods / Modules**
- **Expected Output / Reporting**

---

## 🧩 1. Portfolio Construction with Asymmetric Risk Premia

**Prompt:**
> Build a Python module that constructs a multi-asset portfolio using alternative risk premia (momentum, carry, value, volatility). Incorporate asymmetric downside risk preferences into the optimization.

**Requirements:**
- Use synthetic or historical data for asset classes (equities, bonds, commodities, FX).
- Simulate risk premia returns with non-normal distributions.
- Penalize downside deviations more than upside.

**Methods:**
- Use `numpy`, `pandas`, and `scipy.optimize` for portfolio weights.
- Implement semi-variance or CVaR as the risk metric.
- Include behavioral weighting (e.g., loss aversion factor λ > 1).

**Reporting:**
- Portfolio weights and expected return
- Downside risk metrics (CVaR, semi-variance)
- Comparison to mean-variance optimized portfolio
- Visuals: efficient frontier under asymmetric risk

---

## 🎲 2. Monte Carlo Simulation with Asymmetric Volatility

**Prompt:**
> Create a Monte Carlo engine that simulates asset paths with volatility spikes and asymmetric downside shocks. Use it to evaluate option payoffs and portfolio drawdowns.

**Requirements:**
- Simulate geometric Brownian motion with conditional volatility amplification on negative returns.
- Include jump-diffusion or skewed noise (e.g., skew-normal distribution).
- Track option payoffs and portfolio value over time.

**Methods:**
- Use `numpy` for path generation
- Model volatility as:
  \[
  \sigma_t = \sigma \cdot (1 + \lambda \cdot \mathbb{1}_{\Delta S_t < 0})
  \]
- Option pricing via Black-Scholes and path-dependent payoff tracking

**Reporting:**
- Distribution of final portfolio values
- Option payoff histograms
- Drawdown statistics
- Tail risk metrics (Expected Shortfall, max drawdown)

---

## 🛡️ 3. Hedging Strategy Evaluation

**Prompt:**
> Build a module to evaluate the effectiveness of tail-risk hedges (e.g., put options, VIX calls) under asymmetric market stress scenarios.

**Requirements:**
- Simulate stressed market conditions with fast downside moves
- Include hedging instruments with defined payoff structures
- Compare unhedged vs hedged portfolio performance

**Methods:**
- Use Monte Carlo paths from previous module
- Model hedge payoffs using option Greeks and payoff formulas
- Evaluate hedge cost vs protection

**Reporting:**
- Hedge effectiveness score (e.g., reduction in CVaR)
- Cost-benefit analysis
- Visuals: portfolio value with and without hedge

---

## 📊 4. Factor Attribution and Behavioral Overlay

**Prompt:**
> Create a factor attribution engine that decomposes portfolio returns into traditional beta, alternative risk premia, and behavioral overlays.

**Requirements:**
- Use synthetic or real return data
- Attribute returns to CAPM beta, momentum, carry, value, volatility
- Include behavioral overlays (e.g., loss aversion, skew preference)

**Methods:**
- Use regression or PCA for factor decomposition
- Behavioral overlay via utility-weighted returns:
  \[
  U(x) = x \cdot \mathbb{1}_{x \geq 0} + \lambda x \cdot \mathbb{1}_{x < 0}
  \]

**Reporting:**
- Factor contribution table
- Behavioral-adjusted Sharpe ratio
- Visuals: return attribution bar chart

---

## 🧠 5. Strategy Backtest and Regime Sensitivity

**Prompt:**
> Backtest the custom portfolio across different market regimes (bull, bear, high volatility) and evaluate performance stability.

**Requirements:**
- Use historical data segmented by regime
- Apply portfolio weights from asymmetric optimizer
- Track performance, drawdowns, and regime-specific metrics

**Methods:**
- Use `yfinance` or `pandas-datareader` for data
- Regime classification via volatility or drawdown thresholds
- Performance metrics: CAGR, Sharpe, Sortino, max drawdown

**Reporting:**
- Regime-wise performance table
- Stability score (e.g., variance of Sharpe across regimes)
- Visuals: rolling performance metrics

