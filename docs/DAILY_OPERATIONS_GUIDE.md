# Daily Operations Guide

## Overview

This guide covers the daily workflow and maintenance tasks for the PyTAAA Model-Switching Trading System. Daily operations focus on data updates, portfolio value tracking, and system health monitoring.

## Daily Workflow

### Morning Routine (Before Market Open)

#### 1. System Health Check
```bash
# Check for any system errors from overnight processing
tail -50 daily_abacus_update.log
grep "ERROR\|WARNING" *.log | tail -10

# Verify data file integrity
ls -la /Users/donaldpg/pyTAAA_data/*/data_store/PyTAAA_status.params
```

#### 2. Update Market Data
```bash
# Update stock quotes and market data
uv run python daily_abacus_update.py

# Verify updates completed successfully
echo "Update completed at: $(date)"
```

#### 3. Review Current Portfolio Status
```bash
# Check current portfolio values for your active model
# Replace 'naz100_hma' with your current model
cat /Users/donaldpg/pyTAAA_data/naz100_hma/data_store/PyTAAA_status.params | tail -5

# View holdings summary
cat /Users/donaldpg/pyTAAA_data/naz100_hma/data_store/PyTAAA_holdings.params
```

### Evening Routine (After Market Close)

#### 1. Final Data Update
```bash
# Final update with closing prices
uv run python daily_abacus_update.py

# Log the update
echo "$(date): Daily update completed" >> daily_operations.log
```

#### 2. Performance Review
```bash
# Check portfolio performance
uv run python comprehensive_portfolio_diagnostics.py

# Review any diagnostic alerts
cat diagnostic_results.json | grep -E '"alert"|"warning"'
```

#### 3. System Maintenance
```bash
# Clean up old log files (keep last 30 days)
find . -name "*.log" -mtime +30 -delete

# Backup important state files
cp monte_carlo_state.pkl "monte_carlo_state.backup_$(date +%Y%m%d).pkl"
```

## Key Files to Monitor

### Portfolio Status Files
- `/Users/donaldpg/pyTAAA_data/{model}/data_store/PyTAAA_status.params`
  - Contains daily portfolio values and performance metrics
  - Format: `cumu_value: YYYY-MM-DD HH:MM:SS.SSSSSS VALUE1 VALUE2 VALUE3`

### Holdings Files
- `/Users/donaldpg/pyTAAA_data/{model}/data_store/PyTAAA_holdings.params`
  - Contains current stock positions and quantities
  - Updated when following model recommendations

### Log Files
- `daily_abacus_update.log` - Daily update operations
- `comprehensive_portfolio_diagnostics.log` - Portfolio analysis
- `monthly_update.log` - Monthly trading operations

## Daily Checks and Alerts

### Portfolio Performance Alerts

Monitor for these conditions in your daily checks:

```bash
# Check for significant portfolio drops (>5% in one day)
python -c "
import json
with open('diagnostic_results.json', 'r') as f:
    data = json.load(f)
    daily_return = data.get('daily_return', 0)
    if daily_return < -0.05:
        print(f'ALERT: Large daily loss: {daily_return:.2%}')
    elif daily_return > 0.05:
        print(f'INFO: Large daily gain: {daily_return:.2%}')
    else:
        print(f'OK: Daily return: {daily_return:.2%}')
"
```

### Data Quality Checks

```bash
# Verify quote updates are current (within last 24 hours)
find /Users/donaldpg/pyTAAA_data/*/data_store/PyTAAA_status.params -mtime -1 -ls

# Check for missing data files
for model in naz100_hma naz100_pi naz100_pine sp500_hma; do
    if [ ! -f "/Users/donaldpg/pyTAAA_data/$model/data_store/PyTAAA_status.params" ]; then
        echo "WARNING: Missing status file for $model"
    fi
done
```

### System Resource Monitoring

```bash
# Check disk space usage
df -h | grep -E "pyTAAA_data|PyProjects"

# Monitor memory usage of any running Python processes
ps aux | grep python | grep -E "PyTAAA|monte_carlo|daily_update"
```

## Troubleshooting Common Issues

### Issue: Daily Update Fails

**Symptoms**: `daily_abacus_update.py` exits with errors

**Solutions**:
```bash
# Check network connectivity
ping finance.yahoo.com

# Verify data file permissions
ls -la /Users/donaldpg/pyTAAA_data/*/data_store/

# Run update with verbose logging
uv run python daily_abacus_update.py --verbose
```

### Issue: Portfolio Values Not Updating

**Symptoms**: Status files show old dates

**Solutions**:
```bash
# Check if the correct model is being updated
uv run python monthly_update.py detect-model

# Manually specify the model to update
uv run python daily_abacus_update.py --model naz100_hma

# Verify holdings file exists and is valid
cat /Users/donaldpg/pyTAAA_data/your_model/data_store/PyTAAA_holdings.params
```

### Issue: Log Files Growing Too Large

**Symptoms**: Disk space warnings, slow system performance

**Solutions**:
```bash
# Rotate log files
for log in *.log; do
    if [ -f "$log" ] && [ $(stat -f%z "$log") -gt 10485760 ]; then
        mv "$log" "${log}.old"
        touch "$log"
        echo "Rotated large log file: $log"
    fi
done

# Clean up old backup files (keep last 10)
ls -t monte_carlo_state.pkl.backup_* | tail -n +11 | xargs rm -f
```

## Daily Maintenance Scripts

### Create Daily Health Check Script

Save as `daily_health_check.sh`:
```bash
#!/bin/bash
# Daily system health check

echo "=== PyTAAA Daily Health Check - $(date) ==="

# Check system status
echo "1. Checking system logs..."
if grep -q "ERROR" *.log; then
    echo "⚠️  Errors found in logs"
    grep "ERROR" *.log | tail -5
else
    echo "✅ No errors in recent logs"
fi

# Check data freshness
echo "2. Checking data freshness..."
current_date=$(date +%Y-%m-%d)
if ls /Users/donaldpg/pyTAAA_data/*/data_store/PyTAAA_status.params | xargs grep -l "$current_date" > /dev/null; then
    echo "✅ Portfolio data is current"
else
    echo "⚠️  Portfolio data may be stale"
fi

# Check disk space
echo "3. Checking disk space..."
disk_usage=$(df -h . | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$disk_usage" -gt 90 ]; then
    echo "⚠️  High disk usage: ${disk_usage}%"
else
    echo "✅ Disk usage OK: ${disk_usage}%"
fi

echo "=== Health check completed ==="
```

### Create Portfolio Summary Script

Save as `daily_portfolio_summary.sh`:
```bash
#!/bin/bash
# Generate daily portfolio summary

current_model=$(uv run python monthly_update.py detect-model)
echo "=== Daily Portfolio Summary - $(date) ==="
echo "Active Model: $current_model"

status_file="/Users/donaldpg/pyTAAA_data/$current_model/data_store/PyTAAA_status.params"
if [ -f "$status_file" ]; then
    echo "Latest Portfolio Value:"
    tail -1 "$status_file"
    
    echo "Recent Performance (last 5 days):"
    tail -5 "$status_file"
else
    echo "⚠️  Status file not found for model: $current_model"
fi
```

## Automation Setup

### Cron Job Configuration

Add to your crontab (`crontab -e`):
```bash
# PyTAAA Daily Operations
# Update portfolio values twice daily (market open and close)
30 9 * * 1-5 cd /Users/donaldpg/PyProjects/worktree/PyTAAA.master && uv run python daily_abacus_update.py >> daily_operations.log 2>&1
30 16 * * 1-5 cd /Users/donaldpg/PyProjects/worktree/PyTAAA.master && uv run python daily_abacus_update.py >> daily_operations.log 2>&1

# Health check every morning
0 8 * * 1-5 cd /Users/donaldpg/PyProjects/worktree/PyTAAA.master && ./daily_health_check.sh >> health_check.log 2>&1

# Clean up old files weekly
0 2 * * 0 cd /Users/donaldpg/PyProjects/worktree/PyTAAA.master && find . -name "*.log.old" -mtime +7 -delete
```

## Best Practices

### Data Management
1. **Regular Backups**: Keep daily backups of key configuration and state files
2. **Log Rotation**: Implement automatic log file rotation to prevent disk space issues
3. **Data Validation**: Always verify data quality before making trading decisions

### Performance Monitoring
1. **Track Key Metrics**: Monitor daily returns, drawdowns, and portfolio values
2. **Set Alerts**: Configure alerts for significant portfolio changes
3. **Regular Reviews**: Conduct weekly performance reviews

### System Maintenance
1. **Keep Dependencies Updated**: Regular `uv` updates for security and performance
2. **Monitor Resource Usage**: Track CPU, memory, and disk usage
3. **Test Backup Procedures**: Regularly test your backup and recovery procedures

## Emergency Procedures

### System Recovery
If the system becomes unresponsive or corrupted:

1. **Stop all processes**:
   ```bash
   pkill -f "python.*PyTAAA"
   ```

2. **Restore from backup**:
   ```bash
   cp monte_carlo_state.pkl.backup_$(date +%Y%m%d) monte_carlo_state.pkl
   ```

3. **Verify data integrity**:
   ```bash
   uv run python comprehensive_portfolio_diagnostics.py
   ```

4. **Resume operations**:
   ```bash
   uv run python daily_abacus_update.py
   ```

### Market Holiday Handling
The system automatically handles market holidays, but you should:

1. Verify no updates are attempted on holidays
2. Check for any accumulated data gaps after holiday periods
3. Run manual updates if needed after extended closures

For additional support and troubleshooting, refer to the comprehensive system documentation and implementation progress files.