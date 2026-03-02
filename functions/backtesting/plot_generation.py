"""
Monte Carlo backtest plot generation for PyTAAA trading strategies.

This module generates PNG charts showing portfolio performance over time
with performance metrics annotations. Copied from PyTAAA_backtest_sp500_pine_refactored.py.
"""

import os
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gmean
from matplotlib import rc

from functions.GetParams import get_webpage_store, get_performance_store


# Plot display configuration
show_cagr_in_plot = True  # True = show CAGR in plots, False = show AvgProfit

def calculate_cagr(end_value, start_value, days):
    """
    Calculate Compound Annual Growth Rate (CAGR) with proper error handling.
    
    Args:
        end_value: Portfolio value at end of period
        start_value: Portfolio value at start of period  
        days: Number of trading days in period
        
    Returns:
        CAGR as decimal (e.g., 0.125 for 12.5% annual growth)
    """
    if start_value <= 0 or end_value <= 0 or days <= 0:
        return 0.0
    
    try:
        # Standard CAGR formula: (End/Start)^(trading_days/days) - 1
        trading_days = 252
        cagr = (end_value / start_value) ** (trading_days / days) - 1.0
        
        # Validate reasonable CAGR range (-50% to +100%)
        if cagr < -0.5 or cagr > 1.0:
            print(f" ... Warning: CAGR {cagr:.3f} outside reasonable range")
            
        return cagr
        
    except (ZeroDivisionError, ValueError, OverflowError) as e:
        print(f" ... Error calculating CAGR: {e}")
        return 0.0


def generate_backtest_plot(
    datearray,
    monthvalue,
    monthvalueVariablePctInvest,
    BuyHoldPortfolioValue,
    symbols_file,
    results_dict,
    base_json_fn,
    iter_num,
    QminusSMA,
    monthPctInvested,
    last_symbols_text="",
    output_dir=None,
    runnum=None
):
    """
    Generate a Monte Carlo backtest performance plot (full original implementation).
    
    Creates a 2-panel plot showing:
    - Upper panel: Portfolio value over time with performance metrics
    - Lower panel: Percentage invested or QminusSMA over time
    
    Args:
        datearray: Array of dates
        monthvalue: 2D array of portfolio values [stocks x dates]
        monthvalueVariablePctInvest: 2D array of variable percent invested values
        BuyHoldPortfolioValue: Array of buy-and-hold portfolio values
        symbols_file: Path to symbols file
        results_dict: Dictionary containing backtest results and parameters
        base_json_fn: Base JSON configuration file path
        iter_num: Iteration number for filename
        QminusSMA: Array of quote minus SMA values
        monthPctInvested: Array of percentage invested over time
        last_symbols_text: Text string of last selected symbols
        output_dir: Optional output directory (if None, determined from JSON)
        runnum: Run identifier string (if None, auto-generated from symbols file)
        
    Returns:
        Path to generated PNG file
    """
    # Extract parameters from results
    params = results_dict.get('parameters', {})
    monthsToHold = params.get('monthsToHold', 1)
    numberStocksTraded = params.get('numberStocksTraded', 5)
    LongPeriod = params.get('LongPeriod', 252)
    MA1 = params.get('MA1', 200)
    MA2 = params.get('MA2', 50)
    MA2offset = params.get('MA2offset', 0)
    lowPct = params.get('lowPct', 20.0)
    hiPct = params.get('hiPct', 80.0)
    sma2factor = params.get('sma2factor', 0.91)
    rankThresholdPct = params.get('rankThresholdPct', 0.5)
    sma_filt_val = params.get('sma_filt_val', 0.02)
    riskDownside_min = params.get('riskDownside_min', 0.5)
    riskDownside_max = params.get('riskDownside_max', 5.0)
    
    finalValue = results_dict.get('finalValue', 10000.0)
    sharpeRatio = results_dict.get('sharpeRatio', 0.0)
    
    # Get variable percent invest values
    QminusSMADays = results_dict.get('QminusSMADays', 355)
    QminusSMAFactor = results_dict.get('QminusSMAFactor', 0.90)
    PctInvestSlope = results_dict.get('PctInvestSlope', 5.45)
    PctInvestIntercept = results_dict.get('PctInvestIntercept', -0.01)
    maxPctInvested = results_dict.get('maxPctInvested', 1.25)
    
    # Calculate average portfolio values
    avg_monthvalue = np.average(monthvalue, axis=0)
    avg_varpct_monthvalue = np.average(monthvalueVariablePctInvest, axis=0)
    
    # Calculate performance metrics for different time periods
    n_days = len(avg_monthvalue)
    PortfolioValue = avg_monthvalue
    
    # Calculate Sharpe ratios
    PortfolioDailyGains = PortfolioValue[1:] / PortfolioValue[:-1]
    index = 3780 if n_days >= 3780 else n_days
    
    Sharpe15Yr = ( gmean(PortfolioDailyGains[-index:])**252 -1. ) / ( np.std(PortfolioDailyGains[-index:])*np.sqrt(252) ) if n_days >= index else np.nan
    Sharpe10Yr = ( gmean(PortfolioDailyGains[-2520:])**252 -1. ) / ( np.std(PortfolioDailyGains[-2520:])*np.sqrt(252) ) if n_days >= 2520 else np.nan
    Sharpe5Yr = ( gmean(PortfolioDailyGains[-1260:])**252 -1. ) / ( np.std(PortfolioDailyGains[-1260:])*np.sqrt(252) ) if n_days >= 1260 else np.nan
    Sharpe3Yr = ( gmean(PortfolioDailyGains[-756:])**252 -1. ) / ( np.std(PortfolioDailyGains[-756:])*np.sqrt(252) ) if n_days >= 756 else np.nan
    Sharpe2Yr = ( gmean(PortfolioDailyGains[-504:])**252 -1. ) / ( np.std(PortfolioDailyGains[-504:])*np.sqrt(252) ) if n_days >= 504 else np.nan
    Sharpe1Yr = ( gmean(PortfolioDailyGains[-252:])**252 -1. ) / ( np.std(PortfolioDailyGains[-252:])*np.sqrt(252) ) if n_days >= 252 else np.nan
    
    # Calculate returns
    Return15Yr = (PortfolioValue[-1] / PortfolioValue[-index])**(252./index) if n_days >= index else np.nan
    Return10Yr = (PortfolioValue[-1] / PortfolioValue[-2520])**(1/10.) if n_days >= 2520 else np.nan
    Return5Yr = (PortfolioValue[-1] / PortfolioValue[-1260])**(1/5.) if n_days >= 1260 else np.nan
    Return3Yr = (PortfolioValue[-1] / PortfolioValue[-756])**(1/3.) if n_days >= 756 else np.nan
    Return2Yr = (PortfolioValue[-1] / PortfolioValue[-504])**(1/2.) if n_days >= 504 else np.nan
    Return1Yr = (PortfolioValue[-1] / PortfolioValue[-252]) if n_days >= 252 else np.nan
    
    # Calculate drawdowns
    MaxPortfolioValue = np.zeros(len(PortfolioValue))
    for jj in range(len(PortfolioValue)):
        MaxPortfolioValue[jj] = max(MaxPortfolioValue[jj-1] if jj > 0 else 0, PortfolioValue[jj])
    PortfolioDrawdown = PortfolioValue / MaxPortfolioValue - 1.
    Drawdown15Yr = np.mean(PortfolioDrawdown[-index:]) if n_days >= index else np.nan
    Drawdown10Yr = np.mean(PortfolioDrawdown[-2520:]) if n_days >= 2520 else np.nan
    Drawdown5Yr = np.mean(PortfolioDrawdown[-1260:]) if n_days >= 1260 else np.nan
    Drawdown3Yr = np.mean(PortfolioDrawdown[-756:]) if n_days >= 756 else np.nan
    Drawdown2Yr = np.mean(PortfolioDrawdown[-504:]) if n_days >= 504 else np.nan
    Drawdown1Yr = np.mean(PortfolioDrawdown[-252:]) if n_days >= 252 else np.nan
    
    # Calculate CAGR values
    CAGR15Yr = calculate_cagr(PortfolioValue[-1], PortfolioValue[-index], index) if n_days >= index else np.nan
    CAGR10Yr = calculate_cagr(PortfolioValue[-1], PortfolioValue[-2520], 2520) if n_days >= 2520 else np.nan
    CAGR5Yr = calculate_cagr(PortfolioValue[-1], PortfolioValue[-1260], 1260) if n_days >= 1260 else np.nan
    CAGR3Yr = calculate_cagr(PortfolioValue[-1], PortfolioValue[-756], 756) if n_days >= 756 else np.nan
    CAGR2Yr = calculate_cagr(PortfolioValue[-1], PortfolioValue[-504], 504) if n_days >= 504 else np.nan
    CAGR1Yr = calculate_cagr(PortfolioValue[-1], PortfolioValue[-252], 252) if n_days >= 252 else np.nan
    
    # Variable percent invest metrics
    VarPctPortfolioValue = avg_varpct_monthvalue
    VarPctDailyGains = VarPctPortfolioValue[1:] / VarPctPortfolioValue[:-1]
    
    VarPctSharpe15Yr = ( gmean(VarPctDailyGains[-index:])**252 -1. ) / ( np.std(VarPctDailyGains[-index:])*np.sqrt(252) ) if n_days >= index else np.nan
    VarPctSharpe10Yr = ( gmean(VarPctDailyGains[-2520:])**252 -1. ) / ( np.std(VarPctDailyGains[-2520:])*np.sqrt(252) ) if n_days >= 2520 else np.nan
    VarPctSharpe5Yr = ( gmean(VarPctDailyGains[-1260:])**252 -1. ) / ( np.std(VarPctDailyGains[-1260:])*np.sqrt(252) ) if n_days >= 1260 else np.nan
    VarPctSharpe3Yr = ( gmean(VarPctDailyGains[-756:])**252 -1. ) / ( np.std(VarPctDailyGains[-756:])*np.sqrt(252) ) if n_days >= 756 else np.nan
    VarPctSharpe2Yr = ( gmean(VarPctDailyGains[-504:])**252 -1. ) / ( np.std(VarPctDailyGains[-504:])*np.sqrt(252) ) if n_days >= 504 else np.nan
    VarPctSharpe1Yr = ( gmean(VarPctDailyGains[-252:])**252 -1. ) / ( np.std(VarPctDailyGains[-252:])*np.sqrt(252) ) if n_days >= 252 else np.nan
    
    VarPctReturn15Yr = (VarPctPortfolioValue[-1] / VarPctPortfolioValue[-index])**(252./index) if n_days >= index else np.nan
    VarPctReturn10Yr = (VarPctPortfolioValue[-1] / VarPctPortfolioValue[-2520])**(1/10.) if n_days >= 2520 else np.nan
    VarPctReturn5Yr = (VarPctPortfolioValue[-1] / VarPctPortfolioValue[-1260])**(1/5.) if n_days >= 1260 else np.nan
    VarPctReturn3Yr = (VarPctPortfolioValue[-1] / VarPctPortfolioValue[-756])**(1/3.) if n_days >= 756 else np.nan
    VarPctReturn2Yr = (VarPctPortfolioValue[-1] / VarPctPortfolioValue[-504])**(1/2.) if n_days >= 504 else np.nan
    VarPctReturn1Yr = (VarPctPortfolioValue[-1] / VarPctPortfolioValue[-252]) if n_days >= 252 else np.nan
    
    MaxVarPctPortfolioValue = np.zeros(len(VarPctPortfolioValue))
    for jj in range(len(VarPctPortfolioValue)):
        MaxVarPctPortfolioValue[jj] = max(MaxVarPctPortfolioValue[jj-1] if jj > 0 else 0, VarPctPortfolioValue[jj])
    VarPctPortfolioDrawdown = VarPctPortfolioValue / MaxVarPctPortfolioValue - 1.
    VarPctDrawdown15Yr = np.mean(VarPctPortfolioDrawdown[-index:]) if n_days >= index else np.nan
    VarPctDrawdown10Yr = np.mean(VarPctPortfolioDrawdown[-2520:]) if n_days >= 2520 else np.nan
    VarPctDrawdown5Yr = np.mean(VarPctPortfolioDrawdown[-1260:]) if n_days >= 1260 else np.nan
    VarPctDrawdown3Yr = np.mean(VarPctPortfolioDrawdown[-756:]) if n_days >= 756 else np.nan
    VarPctDrawdown2Yr = np.mean(VarPctPortfolioDrawdown[-504:]) if n_days >= 504 else np.nan
    VarPctDrawdown1Yr = np.mean(VarPctPortfolioDrawdown[-252:]) if n_days >= 252 else np.nan
    
    # Calculate beat-buy-hold metrics (simplified - set to 0 for now)
    beatBuyHoldTest2 = 0.0
    beatBuyHoldTest2VarPct = 0.0
    
    # Determine output directory if not provided
    if output_dir is None:
        perf_store = get_performance_store(base_json_fn)
        model_base = os.path.dirname(perf_store)
        output_dir = os.path.join(model_base, "pytaaa_backtest")
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate standardized filename
    webpage_path = get_webpage_store(base_json_fn)
    model_id = webpage_path.rstrip("/").split("/")[-2]
    
    today = datetime.date.today()
    date_str = f"{today.year}-{today.month}-{today.day}"
    
    # Determine runnum from parameter or symbols_file
    if runnum is None:
        basename = os.path.basename(symbols_file)
        runnum_map = {
            "symbols.txt": "2501a",
            "Naz100_Symbols.txt": "250b",
            "biglist.txt": "2503",
            "ProvidentFundSymbols.txt": "2504",
            "sp500_symbols.txt": "2505",
            "cmg_symbols.txt": "2507",
            "SP500_Symbols.txt": "2506",
        }
        runnum = runnum_map.get(basename, "2508d")
    
    plot_filename = f"{model_id}_backtest_montecarlo_{date_str}_{runnum}_{iter_num:03d}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    
    # Set up figure (copied from original)
    plt.rcParams['figure.edgecolor'] = 'grey'
    plt.rc('savefig', edgecolor='grey')
    plt.close(1)
    fig = plt.figure(1, figsize=(10, 10 * 1080/1920))
    plt.clf()
    
    # Create subplots
    subplotsize = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
    
    # Upper panel: Portfolio value
    plt.subplot(subplotsize[0])
    plt.grid()
    
    # Set y-axis limits based on model type (fixed max for consistency across plots)
    if 'naz100' in model_id.lower():
        plotmax = 1e12  # Naz100 models
    else:
        plotmax = 1e10  # SP500 and other models
    plt.ylim([7000, plotmax])
    plt.yscale('log')
    
    # Plot portfolio values
    plt.plot(BuyHoldPortfolioValue, lw=3, c='r')
    plt.plot(avg_monthvalue, lw=4, c='k')
    plt.plot(avg_varpct_monthvalue, lw=2, c='b')
    
    # Add title with parameters
    title_text = str(iter_num)+":  "+ \
                  str(int(numberStocksTraded))+"__"+   \
                  str(int(monthsToHold))+"__"+   \
                  str(int(LongPeriod))+"-"+   \
                  str(int(MA1))+"-"+   \
                  str(int(MA2))+"-"+   \
                  str(int(MA2+MA2offset))+"-"+   \
                  format(sma2factor,'5.3f')+"_"+   \
                  format(rankThresholdPct,'.1%')+"_"+   \
                  format(sma_filt_val,'6.5f')+"__"+   \
                  format(riskDownside_min,'6.3f')+"-"+  \
                  format(riskDownside_max,'6.3f')+"__"+   \
                  "{:,}".format(int(finalValue))+'__'+   \
                  format(sharpeRatio,'5.2f')+'\n{'+   \
                  str(QminusSMADays)+"-"+   \
                  format(QminusSMAFactor,'6.2f')+"_-"+   \
                  format(PctInvestSlope,'6.2f')+"_"+   \
                  format(PctInvestIntercept,'6.2f')+"_"+   \
                  format(maxPctInvested,'4.2f')+"}__"+   \
                  "{:,}".format(int(avg_varpct_monthvalue[-1]))+'__'+   \
                  format(VarPctSharpe1Yr,'5.2f')
    
    plt.title(title_text, fontsize = 9)
    
    # Calculate plot range for text positioning
    plotrange = np.log10(plotmax) - np.log10(7000.)
    
    # Add text annotations
    plt.text(50, 10.**(np.log10(7000) + (.47 * plotrange)), 
            os.path.basename(symbols_file), fontsize=8)
    plt.text(50, 10.**(np.log10(7000) + (.05 * plotrange)),
            "Backtested on " + datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"),
            fontsize=7.5)
    
    # Format metrics for display
    fSharpe15Yr = format(Sharpe15Yr,'5.2f')
    fSharpe10Yr = format(Sharpe10Yr,'5.2f')
    fSharpe5Yr = format(Sharpe5Yr,'5.2f')
    fSharpe3Yr = format(Sharpe3Yr,'5.2f')
    fSharpe2Yr = format(Sharpe2Yr,'5.2f')
    fSharpe1Yr = format(Sharpe1Yr,'5.2f')
    fReturn15Yr = format(Return15Yr,'5.2f')
    fReturn10Yr = format(Return10Yr,'5.2f')
    fReturn5Yr = format(Return5Yr,'5.2f')
    fReturn3Yr = format(Return3Yr,'5.2f')
    fReturn2Yr = format(Return2Yr,'5.2f')
    fReturn1Yr = format(Return1Yr,'5.2f')
    fDrawdown15Yr = format(Drawdown15Yr,'.1%')
    fDrawdown10Yr = format(Drawdown10Yr,'.1%')
    fDrawdown5Yr = format(Drawdown5Yr,'.1%')
    fDrawdown3Yr = format(Drawdown3Yr,'.1%')
    fDrawdown2Yr = format(Drawdown2Yr,'.1%')
    fDrawdown1Yr = format(Drawdown1Yr,'.1%')
    
    fCAGR15Yr = format(CAGR15Yr, '.1%')
    fCAGR10Yr = format(CAGR10Yr, '.1%')
    fCAGR5Yr = format(CAGR5Yr, '.1%')
    fCAGR3Yr = format(CAGR3Yr, '.1%')
    fCAGR2Yr = format(CAGR2Yr, '.1%')
    fCAGR1Yr = format(CAGR1Yr, '.1%')
    
    fVSharpe15Yr = format(VarPctSharpe15Yr,'5.2f')
    fVSharpe10Yr = format(VarPctSharpe10Yr,'5.2f')
    fVSharpe5Yr = format(VarPctSharpe5Yr,'5.2f')
    fVSharpe3Yr = format(VarPctSharpe3Yr,'5.2f')
    fVSharpe2Yr = format(VarPctSharpe2Yr,'5.2f')
    fVSharpe1Yr = format(VarPctSharpe1Yr,'5.2f')
    fVReturn15Yr = format(VarPctReturn15Yr,'5.2f')
    fVReturn10Yr = format(VarPctReturn10Yr,'5.2f')
    fVReturn5Yr = format(VarPctReturn5Yr,'5.2f')
    fVReturn3Yr = format(VarPctReturn3Yr,'5.2f')
    fVReturn2Yr = format(VarPctReturn2Yr,'5.2f')
    fVReturn1Yr = format(VarPctReturn1Yr,'5.2f')
    fVDrawdown15Yr = format(VarPctDrawdown15Yr,'.1%')
    fVDrawdown10Yr = format(VarPctDrawdown10Yr,'.1%')
    fVDrawdown5Yr = format(VarPctDrawdown5Yr,'.1%')
    fVDrawdown3Yr = format(VarPctDrawdown3Yr,'.1%')
    fVDrawdown2Yr = format(VarPctDrawdown2Yr,'.1%')
    fVDrawdown1Yr = format(VarPctDrawdown1Yr,'.1%')
    
    # Conditional plot display logic for CAGR vs AvgProfit toggle
    if show_cagr_in_plot:
        header_text = 'Period Sharpe CAGR      Avg DD'
        display_15yr = fCAGR15Yr
        display_10yr = fCAGR10Yr 
        display_5yr = fCAGR5Yr
        display_3yr = fCAGR3Yr
        display_2yr = fCAGR2Yr
        display_1yr = fCAGR1Yr
        
        vdisplay_15yr = format(calculate_cagr(VarPctPortfolioValue[-1], VarPctPortfolioValue[-index], index), '.1%') if n_days >= index else 'N/A'
        vdisplay_10yr = format(calculate_cagr(VarPctPortfolioValue[-1], VarPctPortfolioValue[-2520], 2520), '.1%') if n_days >= 2520 else 'N/A'
        vdisplay_5yr = format(calculate_cagr(VarPctPortfolioValue[-1], VarPctPortfolioValue[-1260], 1260), '.1%') if n_days >= 1260 else 'N/A'
        vdisplay_3yr = format(calculate_cagr(VarPctPortfolioValue[-1], VarPctPortfolioValue[-756], 756), '.1%') if n_days >= 756 else 'N/A'
        vdisplay_2yr = format(calculate_cagr(VarPctPortfolioValue[-1], VarPctPortfolioValue[-504], 504), '.1%') if n_days >= 504 else 'N/A'
        vdisplay_1yr = format(calculate_cagr(VarPctPortfolioValue[-1], VarPctPortfolioValue[-252], 252), '.1%') if n_days >= 252 else 'N/A'
    else:
        header_text = 'Period Sharpe AvgProfit  Avg DD'
        display_15yr = fReturn15Yr
        display_10yr = fReturn10Yr
        display_5yr = fReturn5Yr
        display_3yr = fReturn3Yr
        display_2yr = fReturn2Yr
        display_1yr = fReturn1Yr
        
        vdisplay_15yr = fVReturn15Yr
        vdisplay_10yr = fVReturn10Yr
        vdisplay_5yr = fVReturn5Yr
        vdisplay_3yr = fVReturn3Yr
        vdisplay_2yr = fVReturn2Yr
        vdisplay_1yr = fVReturn1Yr
    
    # Performance tables
    plt.text(50,10.**(np.log10(7000)+(.95*plotrange)),header_text,fontsize=7.5)
    plt.text(50,10.**(np.log10(7000)+(.91*plotrange)),'15 Yr '+fSharpe15Yr+'  '+display_15yr+'  '+fDrawdown15Yr,fontsize=8)
    plt.text(50,10.**(np.log10(7000)+(.87*plotrange)),'10 Yr '+fSharpe10Yr+'  '+display_10yr+'  '+fDrawdown10Yr,fontsize=8)
    plt.text(50,10.**(np.log10(7000)+(.83*plotrange)),' 5 Yr  '+fSharpe5Yr+'  '+display_5yr+'  '+fDrawdown5Yr,fontsize=8)
    plt.text(50,10.**(np.log10(7000)+(.79*plotrange)),' 3 Yr  '+fSharpe3Yr+'  '+display_3yr+'  '+fDrawdown3Yr,fontsize=8)
    plt.text(50,10.**(np.log10(7000)+(.75*plotrange)),' 2 Yr  '+fSharpe2Yr+'  '+display_2yr+'  '+fDrawdown2Yr,fontsize=8)
    plt.text(50,10.**(np.log10(7000)+(.71*plotrange)),' 1 Yr  '+fSharpe1Yr+'  '+display_1yr+'  '+fDrawdown1Yr,fontsize=8)
    
    plt.text(2250,10.**(np.log10(7000)+(.95*plotrange)),header_text,fontsize=7.5,color='b')
    plt.text(2250,10.**(np.log10(7000)+(.91*plotrange)),'15 Yr '+fVSharpe15Yr+'  '+vdisplay_15yr+'  '+fVDrawdown15Yr,fontsize=8,color='b')
    plt.text(2250,10.**(np.log10(7000)+(.87*plotrange)),'10 Yr '+fVSharpe10Yr+'  '+vdisplay_10yr+'  '+fVDrawdown10Yr,fontsize=8,color='b')
    plt.text(2250,10.**(np.log10(7000)+(.83*plotrange)),' 5 Yr  '+fVSharpe5Yr+'  '+vdisplay_5yr+'  '+fVDrawdown5Yr,fontsize=8,color='b')
    plt.text(2250,10.**(np.log10(7000)+(.79*plotrange)),' 3 Yr  '+fVSharpe3Yr+'  '+vdisplay_3yr+'  '+fVDrawdown3Yr,fontsize=8,color='b')
    plt.text(2250,10.**(np.log10(7000)+(.75*plotrange)),' 2 Yr  '+fVSharpe2Yr+'  '+vdisplay_2yr+'  '+fVDrawdown2Yr,fontsize=8,color='b')
    plt.text(2250,10.**(np.log10(7000)+(.71*plotrange)),' 1 Yr  '+fVSharpe1Yr+'  '+vdisplay_1yr+'  '+fVDrawdown1Yr,fontsize=8,color='b')
    
    if beatBuyHoldTest2 > 0.:
        plt.text(50,10.**(np.log10(7000)+(.65*plotrange)),format(beatBuyHoldTest2,'.2%')+'  beats BuyHold...')
    else:
        plt.text(50,10.**(np.log10(7000)+(.65*plotrange)),format(beatBuyHoldTest2,'.2%'))
    
    if beatBuyHoldTest2VarPct > 0.:
        plt.text(50,10.**(np.log10(7000)+(.59*plotrange)),format(beatBuyHoldTest2VarPct,'.2%')+'  beats BuyHold...',color='b')
    else:
        plt.text(50,10.**(np.log10(7000)+(.59*plotrange)),format(beatBuyHoldTest2VarPct,'.2%'),color='b')
    
    plt.text(50,10.**(np.log10(7000)+(.54*plotrange)),last_symbols_text,fontsize=8)
    
    # Set up x-axis with year labels
    xlocs = []
    xlabels = []
    for i in range(1, len(datearray)):
        if datearray[i].year != datearray[i-1].year:
            xlocs.append(i)
            xlabels.append(str(datearray[i].year))
    
    if len(xlocs) < 12:
        plt.xticks(xlocs, xlabels)
    else:
        plt.xticks(xlocs[::2], xlabels[::2])
    
    # Lower panel: QminusSMA and monthPctInvested
    plt.subplot(subplotsize[1])
    plt.grid()
    
    # Plot Q-SMA and percent invested
    plt.plot(QminusSMA, 'm-', lw=0.8)
    plt.plot(monthPctInvested, 'r-', lw=0.8)
    
    # Set up x-axis
    if len(xlocs) < 12:
        plt.xticks(xlocs, xlabels)
    else:
        plt.xticks(xlocs[::2], xlabels[::2])
    
    plt.draw()
    
    # Save figure
    plt.savefig(plot_path, format='png', edgecolor='gray')
    plt.close(fig)
    
    print(f" ... Saved plot: {plot_path}")
    
    return plot_path
