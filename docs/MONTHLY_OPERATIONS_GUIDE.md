# Monthly Operations Guide

## Overview

This guide covers the monthly workflow for the PyTAAA Model-Switching Trading System. Monthly operations focus on parameter optimization, model recommendations, and portfolio rebalancing decisions.

## Monthly Workflow Timeline

### First Weekday of the Month

This is the primary monthly trading decision point.

#### 1. Parameter Optimization (Monthly)
```bash
# Run comprehensive parameter optimization
# This typically takes 30-60 minutes depending on iterations
uv run python run_monte_carlo.py --search explore-exploit --verbose

# For faster optimization (if time is limited)
uv run python run_monte_carlo.py --search exploit --verbose

# Extended parameter discovery (quarterly recommended)
uv run python run_monte_carlo.py --search explore --verbose
```

#### 2. Generate Trading Recommendations
```bash
# Generate recommendations using optimized parameters
uv run python recommend_model.py --lookbacks use-saved

# Generate recommendations for specific date (if needed)
uv run python recommend_model.py --date 2025-08-01 --lookbacks use-saved

# Generate recommendations with custom lookback periods
uv run python recommend_model.py --lookbacks "50,150,250"
```

#### 3. Analyze Recommendation Results

The system will output detailed analysis including:
- **Recommended Model**: Which trading strategy to follow
- **Model Rankings**: Performance comparison of all available models  
- **Analysis Parameters**: Lookback periods and performance metrics used
- **Supporting Data**: Sharpe ratios, normalized scores, and historical performance

#### 4. Portfolio Rebalancing Decision

Based on recommendations, decide whether to:
- **Switch Models**: Change to the recommended trading strategy
- **Stay Current**: Continue with existing model if performance is acceptable
- **Go to Cash**: Use cash model during uncertain market conditions

## Interpreting Recommendations

### Sample Recommendation Output
```
============================================================
MODEL RECOMMENDATION RESULTS
============================================================
Recommendation Parameters:
  Lookback periods: [67, 143, 289] days (from saved state)
  Target date: 2025-08-01 (Thursday)
  First weekday of month: 2025-08-01 (Thursday)

----------------------------------------
Model ranks on 2025-08-01:
----------------------------------------
1. sp500_hma      1.456
2. naz100_hma     1.234  
3. naz100_pi      0.876
4. naz100_pine    0.654
5. cash          0.000

Recommended model: sp500_hma
```

### Decision Framework

**When to Switch Models:**
- Recommended model has significantly higher score (>0.2 difference)
- Current model ranks 3rd or lower consistently
- Market conditions favor the recommended strategy

**When to Stay with Current Model:**
- Current model ranks 1st or 2nd
- Score differences are minimal (<0.1)
- Recent switching costs outweigh potential benefits

**When to Consider Cash Model:**
- All equity models show poor recent performance
- Market volatility is extremely high
- Economic uncertainty warrants defensive positioning

## Portfolio Rebalancing Process

### Step 1: Determine Current Holdings

```bash
# Detect your current active model
current_model=$(uv run python monthly_update.py detect-model)
echo "Current model: $current_model"

# View current holdings
cat /Users/donaldpg/pyTAAA_data/$current_model/data_store/PyTAAA_holdings.params
```

### Step 2: Get New Model Holdings

```bash
# View recommended model holdings (replace 'sp500_hma' with recommendation)
recommended_model="sp500_hma"
cat /Users/donaldpg/pyTAAA_data/$recommended_model/data_store/PyTAAA_holdings.params

# Check recommended model's current stock selections
cat /Users/donaldpg/pyTAAA_data/$recommended_model/data_store/current_holdings.json
```

### Step 3: Calculate Required Trades

The system can help calculate what trades are needed:

```bash
# Generate trade recommendations (if switching models)
uv run python monthly_update.py generate-trades --from-model naz100_hma --to-model sp500_hma

# Review calculated trades
cat monthly_trade_recommendations.json
```

### Step 4: Execute Trades

**Manual Execution Process:**
1. **Sell positions** not in the new model
2. **Buy new positions** recommended by the target model
3. **Rebalance weights** to match target allocations
4. **Update holdings file** to reflect new positions

```bash
# After executing trades, update the holdings file
# This tells the system which model you're now following
cp /Users/donaldpg/pyTAAA_data/$recommended_model/data_store/PyTAAA_holdings.params \
   /Users/donaldpg/pyTAAA_data/current_model/data_store/PyTAAA_holdings.params

# Update the daily tracking system
uv run python monthly_update.py update-active-model --model $recommended_model
```

## Risk Management Considerations

### Portfolio Diversification
- **Geographic Diversification**: Consider mix of NASDAQ-100 vs S&P 500 models
- **Strategy Diversification**: Balance between momentum (Pine) and trend-following (HMA) approaches
- **Temporal Diversification**: Avoid excessive model switching (limit to monthly decisions)

### Drawdown Management
- Monitor maximum drawdown limits (typically 15-25%)
- Consider reducing position sizes during high volatility periods
- Use cash model as safe harbor during extreme market stress

### Transaction Cost Analysis
- **Switching Costs**: Account for bid-ask spreads and commissions
- **Tax Implications**: Consider tax consequences of frequent trading
- **Market Impact**: Large position changes may affect execution prices

## Advanced Monthly Analysis

### Model Performance Deep Dive

```bash
# Generate comprehensive performance analysis
uv run python comprehensive_portfolio_diagnostics.py --model-comparison

# View detailed backtest results
uv run python run_monte_carlo.py --iterations 1 --verbose

# Analyze model switching history
grep "Best model:" recommend_model.log | tail -12
```

### Parameter Sensitivity Analysis

```bash
# Test sensitivity to different lookback periods
uv run python recommend_model.py --lookbacks "25,75,150"
uv run python recommend_model.py --lookbacks "100,200,400" 
uv run python recommend_model.py --lookbacks "50,150,250"

# Compare results to understand parameter robustness
```

### Market Condition Analysis

Consider these factors when making monthly decisions:

**Technical Indicators:**
- Market volatility (VIX levels)
- Sector rotation patterns
- Interest rate environment
- Economic calendar events

**Fundamental Factors:**
- Earnings season impacts
- Federal Reserve policy changes
- Geopolitical events
- Seasonal market patterns

## Monthly Maintenance Tasks

### System Health Review

```bash
# Review monthly system performance
tail -100 monte_carlo_run.log | grep -E "BEST PERFORMANCE|ERROR|WARNING"

# Check data quality across all models
for model in naz100_hma naz100_pi naz100_pine sp500_hma cash; do
    echo "=== $model ==="
    tail -5 /Users/donaldpg/pyTAAA_data/$model/data_store/PyTAAA_status.params
done
```

### State File Management

```bash
# Backup current Monte Carlo state
cp monte_carlo_state.pkl "monte_carlo_state.monthly_backup_$(date +%Y%m).pkl"

# Clean up old backup files (keep last 6 months)
ls -t monte_carlo_state.pkl.backup_* | tail -n +7 | xargs rm -f

# Verify state file integrity
uv run python modify_saved_state.py inspect | head -10
```

### Configuration Updates

```bash
# Review and update configuration if needed
cp pytaaa_model_switching_params.json pytaaa_model_switching_params.json.backup
nano pytaaa_model_switching_params.json

# Validate configuration changes
uv run python -c "import json; json.load(open('pytaaa_model_switching_params.json'))"
```

## Performance Tracking and Documentation

### Monthly Performance Report

Create a monthly summary:

```bash
# Generate monthly performance report
cat << EOF > monthly_report_$(date +%Y%m).txt
Monthly Performance Report - $(date +"%B %Y")
============================================

Current Model: $(uv run python monthly_update.py detect-model)

Recommendation: $(grep "Recommended model:" recommend_model.log | tail -1)

Portfolio Performance:
$(tail -5 /Users/donaldpg/pyTAAA_data/$(uv run python monthly_update.py detect-model)/data_store/PyTAAA_status.params)

Model Rankings:
$(grep -A 10 "Model ranks" recommend_model.log | tail -10)

Parameter Optimization:
Lookback Periods: $(grep "Lookback periods:" recommend_model.log | tail -1)
Best Performance: $(grep "NEW BEST PERFORMANCE" monte_carlo_run.log | tail -1)

Action Taken: [TO BE FILLED]
Notes: [TO BE FILLED]
EOF
```

### Trading Journal

Maintain a trading journal for decision documentation:

```bash
# Add entry to trading journal
echo "$(date +%Y-%m-%d): Monthly review completed" >> trading_journal.txt
echo "Recommendation: [FILL IN]" >> trading_journal.txt  
echo "Decision: [FILL IN]" >> trading_journal.txt
echo "Rationale: [FILL IN]" >> trading_journal.txt
echo "---" >> trading_journal.txt
```

## Troubleshooting Monthly Operations

### Issue: Parameter Optimization Fails

**Symptoms:** Monte Carlo search exits with errors or poor results

**Solutions:**
```bash
# Check data integrity
uv run python comprehensive_portfolio_diagnostics.py

# Reset and rebuild state if corrupted
mv monte_carlo_state.pkl monte_carlo_state.pkl.corrupted
uv run python run_monte_carlo.py --search explore --reset

# Run with reduced iterations if memory issues
uv run python run_monte_carlo.py --iterations 50 --search exploit
```

### Issue: Conflicting Recommendations

**Symptoms:** Different lookback periods suggest different models

**Solutions:**
```bash
# Run sensitivity analysis with multiple parameter sets
uv run python recommend_model.py --lookbacks "30,90,180"
uv run python recommend_model.py --lookbacks "60,120,300" 
uv run python recommend_model.py --lookbacks "100,200,400"

# Use majority consensus or stay with current model if unclear
```

### Issue: Poor Recent Performance

**Symptoms:** All models showing negative returns

**Solutions:**
- Consider increasing cash allocation
- Review market conditions for systemic issues
- Reduce position sizes during high volatility
- Consider temporary pause in model switching

## Integration with Daily Operations

### Coordination Points

**After Monthly Decisions:**
- Update daily tracking to reflect new model choice
- Modify daily update scripts if model changed
- Set alerts for new model's performance monitoring

**Mid-Month Reviews:**
```bash
# Optional mid-month recommendation check
uv run python recommend_model.py --lookbacks use-saved

# Compare with month-start recommendation
diff <(grep "Recommended model" recommend_model.log | tail -2)
```

### Documentation Handoff

After completing monthly operations:

1. **Update Documentation**: Record decisions and rationale
2. **Configure Daily Scripts**: Ensure daily operations reflect new model
3. **Set Calendar Reminders**: Schedule next monthly review
4. **Backup Critical Files**: Ensure all changes are backed up

## Best Practices Summary

### Decision Making
- **Use Data-Driven Approach**: Rely on quantitative analysis over intuition
- **Consider Multiple Timeframes**: Don't optimize for short-term performance only
- **Document Decisions**: Keep detailed records of rationale and outcomes
- **Maintain Discipline**: Stick to systematic approach, avoid emotional decisions

### Risk Management
- **Diversify Across Strategies**: Don't put all capital in single model
- **Monitor Drawdowns**: Set clear limits for acceptable losses
- **Consider Transaction Costs**: Factor in real-world trading costs
- **Plan for Black Swans**: Have contingency plans for extreme events

### System Maintenance
- **Regular Optimization**: Run parameter optimization monthly
- **Data Quality Checks**: Verify data integrity before major decisions
- **Backup Procedures**: Maintain robust backup and recovery systems
- **Performance Review**: Regular analysis of system and decision quality

The monthly operations are the core of the model-switching system. Take time to understand the recommendations thoroughly and consider all risk factors before making portfolio changes.