
# import time, threading

import numpy as np
import os
import datetime
# from scipy.stats import rankdata
#import nose
from scipy.stats import gmean
from math import sqrt

## local imports
from functions.quotes_for_list_adjClose import *
from functions.TAfunctions import *
from functions.GetParams import get_json_params, get_performance_store
from functions.CountNewHighsLows import newHighsAndLows
# from functions.UpdateSymbols_inHDF5 import UpdateHDF5


def print_even_year_selections(
    datearray: list,
    symbols: list,
    monthgainlossweight: np.ndarray
) -> None:
    """
    Print stocks selected at the beginning of every even-numbered year,
    and monthly selections for years 2000-2003.

    Displays one line per date showing the date, selected stock list,
    and each stock's fraction of total value. Fractions sum to 1.0.

    Args:
        datearray: Array of dates corresponding to each column.
        symbols: List of stock symbols.
        monthgainlossweight: 2D array of portfolio weights (stocks x dates).
        iter_num: Current iteration number for header display.
    """
    # Print for every random trial (removed iter_num != 0 check)
    print("")
    print("=" * 80)
    print(f"STOCK SELECTIONS AT BEGINNING OF SELECTED YEARS ")
    print("=" * 80)

    # Track years already printed to avoid duplicates
    year_inc = 1  # Print every year (change to 2 for every even year)
    printed_years = set()

    for ii in range(1, len(datearray)):
        current_date = datearray[ii]
        prev_date = datearray[ii - 1]

        # Check if this is the first trading day of an even-numbered year
        if (current_date.year != prev_date.year and
                current_date.year % year_inc == 0):

            # Skip if already printed this year
            if current_date.year in printed_years:
                continue
            printed_years.add(current_date.year)

            # Get weights for this date
            weights = monthgainlossweight[:, ii]

            # Find stocks with non-zero weights
            selected_indices = np.where(weights > 0)[0]

            if len(selected_indices) == 0:
                print(f"{current_date}: No stocks selected")
                continue

            # Build list of (symbol, weight) tuples
            selected_stocks = []
            for idx in selected_indices:
                selected_stocks.append((symbols[idx], weights[idx]))

            # Sort by weight descending
            selected_stocks.sort(key=lambda x: x[1], reverse=True)

            # Calculate sum of weights for verification
            weight_sum = sum(w for _, w in selected_stocks)

            # Format stock list with weights
            stock_list = ", ".join(
                [f"{sym}:{wt:.4f}" for sym, wt in selected_stocks]
            )

            # Print the line
            print(f"\n{current_date}: [{stock_list}] Sum={weight_sum:.4f}")

    print("=" * 80)


def computeDailyBacktest(
        json_fn,
        datearray,
        symbols,
        adjClose,
        numberStocksTraded=7,
        trade_cost=7.95,
        monthsToHold=4,
        LongPeriod=104,
        MA1=207,
        MA2=26,
        MA2offset=3,
        sma2factor=.911,
        rankThresholdPct=.02,
        riskDownside_min=.272,
        riskDownside_max=4.386,
        narrowDays=[6.,40.2],
        mediumDays=[25.2,38.3],
        wideDays=[75.2,512.3],
        stddevThreshold=4.0,
        lowPct=17,
        hiPct=84,
        uptrendSignalMethod='uptrendSignalMethod',
        verbose=False
):

    # put params in a dictionary
    params = {}
    params['numberStocksTraded'] = numberStocksTraded
    params['monthsToHold'] = monthsToHold
    params['LongPeriod'] = LongPeriod
    params['stddevThreshold'] = stddevThreshold
    params['MA1'] = MA1
    params['MA2'] = MA2
    params['MA2offset'] = MA2offset
    params['sma2factor'] = sma2factor
    params['MA2factor'] = sma2factor
    params['rankThresholdPct'] = rankThresholdPct
    params['riskDownside_min'] = riskDownside_min
    params['riskDownside_max'] = riskDownside_max
    params['narrowDays'] = narrowDays
    params['mediumDays'] = mediumDays
    params['wideDays'] = wideDays
    params['uptrendSignalMethod'] = uptrendSignalMethod
    params['lowPct'] = lowPct
    params['hiPct'] = hiPct
    _params = get_json_params(json_fn)
    params['stockList'] = _params['stockList']

    print("\n\n\n ... inside dailyBacktest.py/computeDailyBacktest ...")
    print("\n   . params = " + str(params))
    print("\n   . _params = " + str(_params))
    print("\n   . params.get('stockList') = " + str(params.get('stockList')))

    print("\n\n ... inside dailyBacktest.py")
    print("   . params = " + str(params))

    MaxPortfolioValue = np.zeros(adjClose.shape[1],dtype=float)
    MaxBuyHoldPortfolioValue = np.zeros(adjClose.shape[1],dtype=float)
    numberStocksUpTrendingNearHigh = np.zeros( adjClose.shape[1], dtype=float)
    numberStocksUpTrendingBeatBuyHold = np.zeros( adjClose.shape[1], dtype=float)

    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    activeCount = np.zeros(adjClose.shape[1],dtype=float)

    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    activeCount = np.zeros(adjClose.shape[1],dtype=float)

    numberSharesCalc = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
    gainloss[isnan(gainloss)]=1.
    value = 10000. * np.cumprod(gainloss,axis=1)
    BuyHoldFinalValue = np.average(value,axis=0)[-1]

    print(" gainloss check: ",gainloss[isnan(gainloss)].shape)
    print(" value check: ",value[isnan(value)].shape)
    lastEmptyPriceIndex = np.zeros(adjClose.shape[0],dtype=int)


    for ii in range(adjClose.shape[0]):
        # take care of special case where constant share price is inserted at beginning of series
        index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1
        if verbose:
            print(
                "first valid price and date = ",
                symbols[ii]," ",index," ",datearray[index]
            )
        lastEmptyPriceIndex[ii] = index
        activeCount[lastEmptyPriceIndex[ii]+1:] += 1

    monthgainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    monthgainloss[:,LongPeriod:] = adjClose[:,LongPeriod:] / adjClose[:,:-LongPeriod]
    monthgainloss[isnan(monthgainloss)]=1.

    ####################################################################
    ###
    ### calculate signal for uptrending stocks (in signal2D)
    ### - method depends on params uptrendSignalMethod
    ###
    ####################################################################
    ###
    ### Use either 3 SMA's or channels to comput uptrending signal


    if params['uptrendSignalMethod'] == 'percentileChannels':
        signal2D, lowChannel, hiChannel = computeSignal2D( adjClose, gainloss, params )
    else:
        signal2D = computeSignal2D( adjClose, gainloss, params )

    signal2D_1 = signal2D.copy()

    # Apply rolling window data quality filter if enabled (read from
    # the JSON params object `_params` so CLI/json overrides work)
    if _params.get('enable_rolling_filter', False):  # Default disabled for performance
        from functions.rolling_window_filter import apply_rolling_window_filter
        signal2D = apply_rolling_window_filter(
            adjClose, signal2D, _params.get('window_size', 50),
            symbols=symbols, datearray=datearray
        )

    # copy to daily signal
    signal2D_daily = signal2D.copy()
    # hold signal constant for each month based on filtered daily signals
    # Use `signal2D_daily` (which contains the output of the rolling filter)
    # to decide selections at rebalance dates, then forward-fill those
    # monthly selections so zeros produced by the filter are preserved.
    n_days = adjClose.shape[1]
    # Initialize with the first day's filtered signals
    last_month_signals = signal2D_daily[:, 0].copy()
    for jj in range(1, n_days):
        is_rebalance = (
            (datearray[jj].month != datearray[jj-1].month) and
            ((datearray[jj].month - 1) % monthsToHold == 0)
        )
        if is_rebalance:
            # At rebalance, use the filtered daily signals for that date
            last_month_signals = signal2D_daily[:, jj].copy()
        # Set the month's signal to the last rebalance selection
        signal2D[:, jj] = last_month_signals

    numberStocks = np.sum(signal2D,axis = 0)

    print(" signal2D check: ",signal2D[isnan(signal2D)].shape)

    ########################################################################
    ### compute weights for each stock based on:
    ### 1. uptrending signal in "signal2D"
    ### 1. delta-rank computed from gain/loss over "LongPeriod_random"
    ### 2. sharpe ratio computed from daily gains over "LongPeriod"
    ########################################################################

    # Index-consistency assertions: ensure symbol ordering and
    # filtered daily signals are used at rebalance dates.
    try:
        assert signal2D.shape[0] == len(symbols), (
            f"Signal rows ({signal2D.shape[0]}) != symbols ({len(symbols)})"
        )
        assert signal2D_daily.shape[0] == len(symbols), (
            f"Daily-signal rows ({signal2D_daily.shape[0]}) != symbols ({len(symbols)})"
        )
        assert not np.isnan(signal2D).any(), "NaN present in signal2D"
        assert not np.isnan(signal2D_daily).any(), "NaN present in signal2D_daily"

        # Recompute rebalance indices and ensure that at each rebalance
        # the monthly signal was taken from the filtered daily signal
        rebalance_indices = []
        for jj in range(1, adjClose.shape[1]):
            is_rebalance = (
                (datearray[jj].month != datearray[jj-1].month) and
                ((datearray[jj].month - 1) % monthsToHold == 0)
            )
            if is_rebalance:
                rebalance_indices.append(jj)

        mismatches = []
        for jj in rebalance_indices:
            # Expect that the monthly-held signal at the rebalance
            # equals the filtered daily signal for that same date.
            if not np.array_equal(signal2D[:, jj], signal2D_daily[:, jj]):
                mismatches.append((jj,
                                   int(np.sum(signal2D_daily[:, jj] > 0)),
                                   int(np.sum(signal2D[:, jj] > 0))))

        if mismatches:
            print("\nINDEX CONSISTENCY ASSERTION FAIL: mismatches at rebalance dates")
            print("(date_idx, daily_nonzero_count, month_nonzero_count)")
            for m in mismatches[:20]:
                print(m)
            raise AssertionError(
                "Index-consistency check failed: monthly signals differ from "
                "filtered daily signals at rebalance dates."
            )
    except AssertionError:
        # Re-raise after printing to make failure visible in logs
        raise

    monthgainlossweight = sharpeWeightedRank_2D(
        json_fn, datearray, symbols, adjClose, signal2D ,signal2D_daily,
        LongPeriod, numberStocksTraded, riskDownside_min, riskDownside_max,
        rankThresholdPct, stddevThreshold=stddevThreshold,
        stockList=params.get('stockList', 'SP500')
    )

    print_even_year_selections(
        datearray=datearray,
        symbols=symbols,
        monthgainlossweight=monthgainlossweight
    )

    print("\n ... completed signal2D and monthgainlossweight calculations. Now computing portfolio values ... \n")

    ########################################################################
    ### compute traded value of stock for each month
    ########################################################################

    # initialize monthvalue (assume B&H)
    monthvalue = value.copy()

    starting_valid_symbols = value[:,0]
    starting_valid_symbols_count = starting_valid_symbols[starting_valid_symbols > 1.e-4].shape[0]

    print("  \n\n\n")
    for ii in np.arange(1,monthgainloss.shape[1]):

        if (datearray[ii].month != datearray[ii-1].month) and \
                ( (datearray[ii].month - 1)%monthsToHold == 0) and \
                np.max(np.abs(monthgainlossweight[:,ii-1] - monthgainlossweight[:,ii])) > 1.e-4 :
            commission = 0
            symbol_changed_count = 0
            valuemean=np.mean(monthvalue[:,ii-1])
            valuesum=np.sum(monthvalue[:,ii-1])

            # compute yesterday's holdings value
            yesterdayValue = np.sum( monthgainlossweight[:,ii-1] * monthvalue[:,ii-1] )
            todayValue = np.sum( gainloss[:,ii] * yesterdayValue * monthgainlossweight[:,ii] )
            # reduce CASH by commission amount
            weight_changes = np.abs(monthgainlossweight[:,ii-1]-monthgainlossweight[:,ii])
            symbol_changed_count = weight_changes[weight_changes > 1.e-4].shape[0]
            # handle special case for buying index
            if symbol_changed_count > 2 * numberStocksTraded:
                # handle special case for buying index
                commission = trade_cost
            else:
                commission = symbol_changed_count * trade_cost
            commission_factor = (valuesum-commission*monthvalue.shape[0])/valuesum

            # print(
            #     "date,changed#, commission,valuemean,yesterdayValue,todayValue,commissionFactor(%)= ", \
            #     datearray[ii], symbol_changed_count, commission, valuemean,
            #     yesterdayValue,todayValue,format(commission_factor-1.,'5.2%')
            # )

            ### Note: this is only approximate to what I really want to do in trades. This models all percentages changing
            ###       which implies trading all stocks. But I really want just to adjust CASH balance if monthgainlossweight is constant.
            for jj in range(value.shape[0]):
                monthvalue[jj,ii] = gainloss[jj,ii]*valuesum*monthgainlossweight[jj,ii]*commission_factor   # re-balance using weights (that sum to 1.0 less commissions)

        else:
            for jj in range(value.shape[0]):
                monthvalue[jj,ii] = monthvalue[jj,ii-1]*gainloss[jj,ii]

    numberSharesCalc = monthvalue / adjClose    # for info only

    print("  \n\n\n")

    ########################################################################
    ### gather statistics on number of uptrending stocks
    ########################################################################

    index = 3780
    if monthvalue.shape[1] < 3780: index = monthvalue.shape[1]

    PortfolioValue = np.average(monthvalue,axis=0)
    PortfolioDailyGains = PortfolioValue[1:] / PortfolioValue[:-1]
    Sharpe15Yr = ( gmean(PortfolioDailyGains[-index:])**252 -1. ) / ( np.std(PortfolioDailyGains[-index:])*sqrt(252) )
    Sharpe10Yr = ( gmean(PortfolioDailyGains[-2520:])**252 -1. ) / ( np.std(PortfolioDailyGains[-2520:])*sqrt(252) )
    Sharpe5Yr = ( gmean(PortfolioDailyGains[-1260:])**252 -1. ) / ( np.std(PortfolioDailyGains[-1260:])*sqrt(252) )
    Sharpe3Yr = ( gmean(PortfolioDailyGains[-756:])**252 -1. ) / ( np.std(PortfolioDailyGains[-756:])*sqrt(252) )
    Sharpe2Yr = ( gmean(PortfolioDailyGains[-504:])**252 -1. ) / ( np.std(PortfolioDailyGains[-504:])*sqrt(252) )
    Sharpe1Yr = ( gmean(PortfolioDailyGains[-252:])**252 -1. ) / ( np.std(PortfolioDailyGains[-252:])*sqrt(252) )

    print("15 year : ",index,PortfolioValue[-1], PortfolioValue[-index],datearray[-index])

    Return15Yr = (PortfolioValue[-1] / PortfolioValue[-index])**(252./index)
    Return10Yr = (PortfolioValue[-1] / PortfolioValue[-2520])**(1/10.)
    Return5Yr = (PortfolioValue[-1] / PortfolioValue[-1260])**(1/5.)
    Return3Yr = (PortfolioValue[-1] / PortfolioValue[-756])**(1/3.)
    Return2Yr = (PortfolioValue[-1] / PortfolioValue[-504])**(1/2.)
    Return1Yr = (PortfolioValue[-1] / PortfolioValue[-252])

    MaxPortfolioValue *= 0.
    for jj in range(PortfolioValue.shape[0]):
        MaxPortfolioValue[jj] = max(MaxPortfolioValue[jj-1],PortfolioValue[jj])
    PortfolioDrawdown = PortfolioValue / MaxPortfolioValue - 1.
    Drawdown15Yr = np.mean(PortfolioDrawdown[-index:])
    Drawdown10Yr = np.mean(PortfolioDrawdown[-2520:])
    Drawdown5Yr = np.mean(PortfolioDrawdown[-1260:])
    Drawdown3Yr = np.mean(PortfolioDrawdown[-756:])
    Drawdown2Yr = np.mean(PortfolioDrawdown[-504:])
    Drawdown1Yr = np.mean(PortfolioDrawdown[-252:])

    BuyHoldPortfolioValue = np.mean(value,axis=0)
    BuyHoldDailyGains = BuyHoldPortfolioValue[1:] / BuyHoldPortfolioValue[:-1]
    BuyHoldSharpe15Yr = ( gmean(BuyHoldDailyGains[-index:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-index:])*sqrt(252) )
    BuyHoldSharpe10Yr = ( gmean(BuyHoldDailyGains[-2520:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-2520:])*sqrt(252) )
    BuyHoldSharpe5Yr  = ( gmean(BuyHoldDailyGains[-1126:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-1260:])*sqrt(252) )
    BuyHoldSharpe3Yr  = ( gmean(BuyHoldDailyGains[-756:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-756:])*sqrt(252) )
    BuyHoldSharpe2Yr  = ( gmean(BuyHoldDailyGains[-504:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-504:])*sqrt(252) )
    BuyHoldSharpe1Yr  = ( gmean(BuyHoldDailyGains[-252:])**252 -1. ) / ( np.std(BuyHoldDailyGains[-252:])*sqrt(252) )
    BuyHoldReturn15Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-index])**(252./index)
    BuyHoldReturn10Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-2520])**(1/10.)
    BuyHoldReturn5Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-1260])**(1/5.)
    BuyHoldReturn3Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-756])**(1/3.)
    BuyHoldReturn2Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-504])**(1/2.)
    BuyHoldReturn1Yr = (BuyHoldPortfolioValue[-1] / BuyHoldPortfolioValue[-252])
    for jj in range(BuyHoldPortfolioValue.shape[0]):
        MaxBuyHoldPortfolioValue[jj] = max(MaxBuyHoldPortfolioValue[jj-1],BuyHoldPortfolioValue[jj])

    BuyHoldPortfolioDrawdown = BuyHoldPortfolioValue / MaxBuyHoldPortfolioValue - 1.
    BuyHoldDrawdown15Yr = np.mean(BuyHoldPortfolioDrawdown[-index:])
    BuyHoldDrawdown10Yr = np.mean(BuyHoldPortfolioDrawdown[-2520:])
    BuyHoldDrawdown5Yr = np.mean(BuyHoldPortfolioDrawdown[-1260:])
    BuyHoldDrawdown3Yr = np.mean(BuyHoldPortfolioDrawdown[-756:])
    BuyHoldDrawdown2Yr = np.mean(BuyHoldPortfolioDrawdown[-504:])
    BuyHoldDrawdown1Yr = np.mean(BuyHoldPortfolioDrawdown[-252:])


    print("")
    print("")
    print("Sharpe15Yr, BuyHoldSharpe15Yr = ", Sharpe15Yr, BuyHoldSharpe15Yr)
    print("Sharpe10Yr, BuyHoldSharpe10Yr = ", Sharpe10Yr, BuyHoldSharpe10Yr)
    print("Sharpe5Yr, BuyHoldSharpe5Yr =   ", Sharpe5Yr, BuyHoldSharpe5Yr)
    print("Sharpe3Yr, BuyHoldSharpe3Yr =   ", Sharpe3Yr, BuyHoldSharpe3Yr)
    print("Sharpe2Yr, BuyHoldSharpe2Yr =   ", Sharpe2Yr, BuyHoldSharpe2Yr)
    print("Sharpe1Yr, BuyHoldSharpe1Yr =   ", Sharpe1Yr, BuyHoldSharpe1Yr)
    print("Return15Yr, BuyHoldReturn15Yr = ", Return15Yr, BuyHoldReturn15Yr)
    print("Return10Yr, BuyHoldReturn10Yr = ", Return10Yr, BuyHoldReturn10Yr)
    print("Return5Yr, BuyHoldReturn5Yr =   ", Return5Yr, BuyHoldReturn5Yr)
    print("Return3Yr, BuyHoldReturn3Yr =   ", Return3Yr, BuyHoldReturn3Yr)
    print("Return2Yr, BuyHoldReturn2Yr =   ", Return2Yr, BuyHoldReturn2Yr)
    print("Return1Yr, BuyHoldReturn1Yr =   ", Return1Yr, BuyHoldReturn1Yr)
    print("Drawdown15Yr, BuyHoldDrawdown15Yr = ", Drawdown15Yr, BuyHoldDrawdown15Yr)
    print("Drawdown10Yr, BuyHoldDrawdown10Yr = ", Drawdown10Yr, BuyHoldDrawdown10Yr)
    print("Drawdown5Yr, BuyHoldDrawdown5Yr =   ", Drawdown5Yr, BuyHoldDrawdown5Yr)
    print("Drawdown3Yr, BuyHoldDrawdown3Yr =   ", Drawdown3Yr, BuyHoldDrawdown3Yr)
    print("Drawdown2Yr, BuyHoldDrawdown2Yr =   ", Drawdown2Yr, BuyHoldDrawdown2Yr)
    print("Drawdown1Yr, BuyHoldDrawdown1Yr =   ", Drawdown1Yr, BuyHoldDrawdown1Yr)


    beatBuyHoldTest = ( (Sharpe15Yr-BuyHoldSharpe15Yr)/15. + \
                        (Sharpe10Yr-BuyHoldSharpe10Yr)/10. + \
                        (Sharpe5Yr-BuyHoldSharpe5Yr)/5. + \
                        (Sharpe3Yr-BuyHoldSharpe3Yr)/3. + \
                        (Sharpe2Yr-BuyHoldSharpe2Yr)/2. + \
                        (Sharpe1Yr-BuyHoldSharpe1Yr)/1. ) / (1/15. + 1/10.+1/5.+1/3.+1/2.+1)


    beatBuyHoldTest2 = 0
    if Return15Yr > BuyHoldReturn15Yr: beatBuyHoldTest2 += 1
    if Return10Yr > BuyHoldReturn10Yr: beatBuyHoldTest2 += 1
    if Return5Yr  > BuyHoldReturn5Yr:  beatBuyHoldTest2 += 1
    if Return3Yr  > BuyHoldReturn3Yr:  beatBuyHoldTest2 += 1.5
    if Return2Yr  > BuyHoldReturn2Yr:  beatBuyHoldTest2 += 2
    if Return1Yr  > BuyHoldReturn1Yr:  beatBuyHoldTest2 += 2.5
    if Return15Yr > 0: beatBuyHoldTest2 += 1
    if Return10Yr > 0: beatBuyHoldTest2 += 1
    if Return5Yr  > 0: beatBuyHoldTest2 += 1
    if Return3Yr  > 0: beatBuyHoldTest2 += 1.5
    if Return2Yr  > 0: beatBuyHoldTest2 += 2
    if Return1Yr  > 0: beatBuyHoldTest2 += 2.5
    if Drawdown15Yr > BuyHoldDrawdown15Yr: beatBuyHoldTest2 += 1
    if Drawdown10Yr > BuyHoldDrawdown10Yr: beatBuyHoldTest2 += 1
    if Drawdown5Yr  > BuyHoldDrawdown5Yr:  beatBuyHoldTest2 += 1
    if Drawdown3Yr  > BuyHoldDrawdown3Yr:  beatBuyHoldTest2 += 1.5
    if Drawdown2Yr  > BuyHoldDrawdown2Yr:  beatBuyHoldTest2 += 2
    if Drawdown1Yr  > BuyHoldDrawdown1Yr:  beatBuyHoldTest2 += 2.5
    # make it a ratio ranging from 0 to 1
    beatBuyHoldTest2 /= 27

    if beatBuyHoldTest2 > .60 :
        print("found monte carlo trial that beats BuyHold (test2)...")
        print("shape of numberStocksUpTrendingBeatBuyHold = ",numberStocksUpTrendingBeatBuyHold.shape)
        print("mean of numberStocksUpTrendingBeatBuyHold values = ",np.mean(numberStocksUpTrendingBeatBuyHold))

    print("")
    print(" ")
    print("The B&H portfolio final value is: ","{:,}".format(int(BuyHoldFinalValue)))
    print(" ")
    print("Monthly re-balance based on ",LongPeriod, "days of recent performance.")
    print("The portfolio final value is: ","{:,}".format(int(np.average(monthvalue,axis=0)[-1])))
    print(" ")
    print("Today's top ranking choices are: ")
    last_symbols_text = []
    for ii in range(len(symbols)):
        if monthgainlossweight[ii,-1] > 0:
            # print symbols[ii]
            print(datearray[-1], symbols[ii],monthgainlossweight[ii,-1])
            last_symbols_text.append( symbols[ii] )



    ###
    ### save backtest portfolio values ( B&H and system )
    ###
    try:
        json_dir = os.path.split(json_fn)
        p_store = get_performance_store(json_fn)
        filepath = os.path.join(p_store, "pyTAAAweb_backtestPortfolioValue.params" )
        
        # Compute new highs and lows for each date
        print("\n ... Computing new highs and lows for backtest output...")
        sumNewHighs, sumNewLows, _ = newHighsAndLows(
            json_fn, num_days_highlow=(73,293),
            num_days_cumu=(50,159),
            HighLowRatio=(1.654,2.019),
            HighPctile=(8.499,8.952),
            HGamma=(1.,1.),
            LGamma=(1.176,1.223),
            makeQCPlots=False,
            outputStats=False
        )
        # Sum across stocks (axis=0) and across parameter sets (axis=1 if multiple)
        # sumNewHighs = np.sum(newHighs_2D, axis=0)
        # sumNewLows = np.sum(newLows_2D, axis=0)
        # If multiple parameter sets were used, sum them to get 1D array
        if sumNewHighs.ndim > 1:
            sumNewHighs = np.sum(sumNewHighs, axis=-1)
            sumNewLows = np.sum(sumNewLows, axis=-1)
        
        textmessage = ""
        filepath = os.path.join(p_store, "pyTAAAweb_backtestPortfolioValue.params" )
        for idate in range(len(BuyHoldPortfolioValue)):
            textmessage = textmessage + \
                        str(datearray[idate]) + " " + \
                        str(BuyHoldPortfolioValue[idate]) + " " + \
                        str(np.average(monthvalue[:,idate]))  + " " + \
                        f"{sumNewHighs[idate]:.1f}" + " " + \
                        f"{sumNewLows[idate]:.1f}" + "\n"
        with open( filepath, "w" ) as f:
            f.write(textmessage)
        # for idate in range(len(BuyHoldPortfolioValue)):
        #     textmessage = textmessage + str(datearray[idate])+"  "+str(BuyHoldPortfolioValue[idate])+"  "+str(np.average(monthvalue[:,idate]))+"  "+str(int(sumNewHighs[idate]))+"  "+str(int(sumNewLows[idate]))+"\n"
        # with open( filepath, "w" ) as f:
        #     f.write(textmessage)
    except Exception as e:
        print(f"\n ... Error computing new highs/lows: {e}")
        print(" ... Writing 3-column output as fallback...")
        textmessage = ""
        for idate in range(len(BuyHoldPortfolioValue)):
            textmessage = textmessage + str(datearray[idate])+"  "+str(BuyHoldPortfolioValue[idate])+"  "+str(np.average(monthvalue[:,idate]))+"\n"
        with open( filepath, "w" ) as f:
            f.write(textmessage)

    ########################################################################
    ### compute some portfolio performance statistics and print summary
    ########################################################################

    print("final value for portfolio ", "{:,}".format(np.average(monthvalue[:,-1])))


    print("portfolio annualized gains : ", ( gmean(PortfolioDailyGains)**252 ))
    print("portfolio annualized StdDev : ", ( np.std(PortfolioDailyGains)*sqrt(252) ))
    print("portfolio sharpe ratio : ", ( gmean(PortfolioDailyGains)**252 ) / ( np.std(PortfolioDailyGains)*sqrt(252) ))

    # Compute trading days back to target start date
    targetdate = datetime.date(2008,1,1)
    lag = int((datearray[-1] - targetdate).days*252/365.25)

    # Print some stats for B&H and trading from target date to end_date
    print("")
    print("")
    BHValue = np.average(value,axis=0)
    BHdailygains = np.concatenate( (np.array([0.]), BHValue[1:]/BHValue[:-1]), axis = 0 )
    BHsharpefromtargetdate = ( gmean(BHdailygains[-lag:])**252 -1. ) / ( np.std(BHdailygains[-lag:])*sqrt(252) )
    BHannualgainfromtargetdate = ( gmean(BHdailygains[-lag:])**252 )
    print("start date for recent performance measures: ",targetdate)
    print("BuyHold annualized gains & sharpe from target date:   ", BHannualgainfromtargetdate,BHsharpefromtargetdate)

    Portfoliosharpefromtargetdate = ( gmean(PortfolioDailyGains[-lag:])**252 -1. ) / ( np.std(PortfolioDailyGains[-lag:])*sqrt(252) )
    Portfolioannualgainfromtargetdate = ( gmean(PortfolioDailyGains[-lag:])**252 )
    print("portfolio annualized gains & sharpe from target date: ", Portfolioannualgainfromtargetdate,Portfoliosharpefromtargetdate)

    return

