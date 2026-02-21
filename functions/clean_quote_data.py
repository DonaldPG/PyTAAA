# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:25:30 2020

@author: Don
"""


import yfinance as yf
import os
import datetime
import numpy as np
import pandas as pd
from typing import Any, Tuple
from matplotlib import pylab as plt


'''
from functions.TAfunctions import _is_odd, \
                                  #generateExamples, \
                                  #generatePredictionInput, \
                                  #generateExamples3layer, \
                                  #generateExamples3layerGen, \
                                  #generateExamples3layerForDate, \
                                  #generatePredictionInput3layer, \
                                  get_params, \
                                  #interpolate, \
                                  #cleanspikes, \
                                  #cleantobeginning, \
                                  #cleantoend,\
                                  #clean_signal,\
                                  #build_model, \
                                  #get_predictions_input, \
                                  #one_model_prediction, \
                                  #ensemble_prediction
'''

from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF

## local imports
# try:
#     os.chdir(os.path.abspath(os.path.dir(__file__)))
#     print(" ...change to folder for this file = "+os.getcwd())
# except:
#     os.chdir('C:\\Users\\don\\raspberrypi\\Py3TAAA-analyzestocksSP500')
#     print(" ...cannot change to folder for this file = "+os.getcwd())


# _cwd = os.getcwd()
# os.chdir(os.path.join(os.getcwd()))
# print(" ...working directory = "+os.getcwd())


def interpolate(self: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Interpolate missing values (after the first valid value)
    Parameters
    ----------
    method : {'linear'}
    Interpolation method.
    Time interpolation works on daily and higher resolution
    data to interpolate given length of interval
    Returns
    -------
    interpolated : Series
    from-- https://github.com/wesm/pandas/blob/master/pandas/core/series.py
    edited to keep only 'linear' method
    Usage: infill NaN values with linear interpolated values
    """
    inds = np.arange(len(self))
    values = np.array(self.copy())
    valid = np.ones((len(self)),'int')
    valid[np.isnan(values)] = 0
    invalid = 1 - valid
    firstIndex = valid.argmax()
    lastIndex = valid.shape[0]-valid[::-1].argmax()

    valid[range(len(valid)) < firstIndex] = 0
    valid[range(len(valid)) > lastIndex] = 0
    invalid[range(len(valid)) < firstIndex] = 1
    invalid[range(len(valid)) > lastIndex] = 1

    result = values.copy()
    if len(invalid[invalid==1]) > 0:
        result[invalid==1] = np.interp(inds[invalid==1], inds[valid==1],values[valid==1])

    return result


#----------------------------------------------
def cleantobeginning(self: np.ndarray) -> np.ndarray:
    """
    Copy missing values (to all dates prior the first valid value)

    Usage: infill NaN values at beginning with copy of first valid value
    """
    values = self.copy()
    invalid_bool = np.isnan(values)
    valid = np.ones((len(self)),'int')
    valid[ invalid_bool==True ] = 0
    firstIndex = valid.argmax()
    for i in range(firstIndex):
        values[i]=values[firstIndex]
    return values


#----------------------------------------------

def cleantoend(self: np.ndarray) -> np.ndarray:
    """
    Copy missing values (to all dates after the last valid value)

    Usage: infill NaN values at end with copy of last valid value
    """
    # reverse input 1D array and use cleantobeginning
    reverse = self[::-1]
    reverse = cleantobeginning(reverse)
    return reverse[::-1]


def cleanspikes(x: np.ndarray, periods: int = 20, stddevThreshold: float = 5.0) -> np.ndarray:
    # remove outliers from gradient of x (in 2 directions)
    x_clean = np.array(x).copy()
    test = np.zeros(x.shape[0],'float')
    gainloss_f = x[1:] / x[:-1]
    gainloss_r = x[:-1] / x[1:]
    valid_f = gainloss_f[gainloss_f != 1.]
    valid_f = valid_f[~np.isnan(valid_f)]
    if len(valid_f) > 0:
        Stddev_f = np.std(valid_f) + 1.e-5
    else:
        Stddev_f = 1.e-5
    valid_r = gainloss_r[gainloss_r != 1.]
    valid_r = valid_r[~np.isnan(valid_r)]
    if len(valid_r) > 0:
        Stddev_r = np.std(valid_r) + 1.e-5
    else:
        Stddev_r = 1.e-5
    forward_test = gainloss_f/Stddev_f - np.median(gainloss_f/Stddev_f)
    reverse_test = gainloss_r/Stddev_r - np.median(gainloss_r/Stddev_r)
    test[:-1] += reverse_test
    test[1:] += forward_test
    test[np.isnan(test)] = 1.e-10
    x_clean[ test > stddevThreshold ] = np.nan
    return x_clean


#----------------------------------------------
def nans_at_beginning(self: np.ndarray) -> np.ndarray:
    """
    replace repeated values at beginning with NaN's
    (for all dates prior the first valid value)

    Usage: infill NaN values at beginning with copy of first valid value
    """
    gains = self[1:] / self[:-1]
    first_nonnan_index = np.where((~np.isnan(gains)) & (gains != 1.))[0][0]
    first_value = self[first_nonnan_index]
    first_valid_index = np.where(self != first_value)[0][0]
    if first_valid_index > 0:
        values = self.copy()
        values[:first_valid_index-1] = np.nan
    else:
        values = self.copy()
    return values


#----------------------------------------------

def clean_signal(array1D: np.ndarray, symbol_name: str, verbose: bool = False) -> np.ndarray:
    ### clean input signals (again)
    quotes_before_cleaning = array1D.copy()
    adjClose = interpolate( array1D )
    adjClose = nans_at_beginning(adjClose)
    adjClose = cleantobeginning( adjClose )
    adjClose = cleantoend( adjClose )
    adjClose_changed = False in (adjClose==quotes_before_cleaning)
    if verbose:
        print("   ... inside PortfolioPerformanceCalcs ... symbol, did cleaning change adjClose? ", symbol_name, adjClose_changed)
    return adjClose


def get_hdf_name(symbols_file: str) -> Tuple[str, str]:
    (directory_name, file_name) = os.path.split(symbols_file)
    (shortname, extension) = os.path.splitext( file_name )

    print("file name for symbols = ","_"+shortname+"_")
    print("file type for symbols = ",extension)

    # set up to write quotes to disk.

    if shortname == "symbols" :
        listname = "TAA-Symbols"
    elif shortname == "cmg_symbols" :
        listname = "CMG-Symbols"
    elif shortname == "Naz100_Symbols" :
        listname = "Naz100_Symbols"
    elif shortname == "biglist" :
        listname = "biglist-Symbols"
    elif shortname == "ETF_symbols" :
        listname = "ETF-Symbols"
    elif shortname == "ProvidentFundSymbols" :
        listname = "ProvidentFund-Symbols"
    elif shortname == "SP500_Symbols" :
        listname = "SP500_Symbols"
    elif shortname == "SP1000_Symbols" or shortname == "RU1000_Symbols":
        listname = "RU1000_Symbols"
    else :
        listname = shortname

    hdf5_directory = os.path.join( os.getcwd(), "symbols" )
    hdf5filename = os.path.join(hdf5_directory, listname + "_.hdf5")

    print("")
    print("")
    print("symbol_directory = ", directory_name)
    print("symbols_file = ", symbols_file)
    print("shortname, extension = ",shortname, extension)
    print("hdf5filename = ",hdf5filename)

    return hdf5filename, listname


def remove_extra_dates_at_beginning(df: Any, ref_dates: Any) -> Any:
    # remove dates in df that precede dates in 'ref_dates'
    # assumes that index for df is dates in timedate format
    # assumes that ref_dates is in timedate format

    dates = df.index
    dates_str = [str(val) for val in list(dates)]

    ref_dates_str = [str(val) for val in ref_dates]
    extra_dates = list()
    for idate, date in enumerate(dates_str):
        if date not in ref_dates_str:
            print("extra date = "+str(date))
            extra_dates.append(date)
    extra_dates_str = [str(val) for val in extra_dates]
    extra_dates_datetime = []
    for idate in extra_dates_str:
        extra_dates_datetime.append(datetime.date(*[int(val) for val in idate.split('-')]))

    df_subset = df.drop(extra_dates_datetime)
    return df_subset


def get_stored_quotes(json_fn: str, stockList: str) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    import os

    # try:
    #     if stockList == "SP500":
    #         os.chdir(os.path.abspath(os.path.dirname(__file__)))
    #     elif stockList == "Naz100":
    #         os.chdir(os.path.abspath(os.path.dirname(__file__)))
    #         os.chdir(os.path.join("..", "Py3TAAADL_tracker"))
    #     elif stockList.lower() == "index":
    #         os.chdir(os.path.join(os.path.abspath(os.path.dirname(__file__))))
    #         os.chdir(os.path.join("..", "Py3TAAA_index"))
    # except:
    #     os.chdir('C:\\Users\\don\\tf\\tf\\PyTAAADLgit')
    #     os.chdir('C:\\Users\\don\\Desktop\\temp')
    #     if stockList == "SP500":
    #         os.chdir('C:\\Users\\don\\raspberrypi\\Py3TAAA-analyzestocksSP500')
    #     elif stockList == "Naz100":
    #         os.chdir('C:\\Users\\don\\raspberrypi\\Py3TAAADL_tracker')


    # if stockList == "SP500":
    #     os.chdir(os.path.abspath(os.path.dirname(__file__)))
    # elif stockList == "Naz100":
    #     os.chdir(os.path.abspath(os.path.dirname(__file__)))
    #     os.chdir(os.path.join("..", "Py3TAAADL_tracker"))
    # elif stockList.lower() == "index":
    #     os.chdir(os.path.join(os.path.abspath(os.path.dirname(__file__))))
    #     os.chdir(os.path.join("..", "Py3TAAA_index"))

    print("\n ... inside clean_SP500_data.py/get_stored_quotes")
    print(" ... stockList = " + stockList)
    # print(" ... cwd = "+os.getcwd())

    ## local imports
    # _cwd = os.getcwd()
    # os.chdir(os.path.join(os.getcwd()))
    # _data_path = os.getcwd()
    '''
    from functions.quotes_for_list_adjClose import get_Naz100List, \
                                                   arrayFromQuotesForList
    '''
    '''
    from functions.TAfunctions import (
                                      _is_odd, \
                                      generateExamples, \
                                      generatePredictionInput, \
                                      generateExamples3layer, \
                                      generateExamples3layerGen, \
                                      generateExamples3layerForDate, \
                                      generatePredictionInput3layer, \
                                      get_params, \
                                      interpolate, \
                                      cleantobeginning, \
                                      cleantoend,\
                                      build_model, \
                                      get_predictions_input, \
                                      one_model_prediction, \
                                      ensemble_prediction
                                      )
    '''

    from functions.GetParams import get_json_params

    from functions.UpdateSymbols_inHDF5 import (
        UpdateHDF_yf, loadQuotes_fromHDF
    )

    # os.chdir(_cwd)

    # --------------------------------------------------
    # Get program parameters.
    # --------------------------------------------------

    run_params = get_json_params(json_fn)

    # --------------------------------------------------
    # set filename for datafram containing model persistence input data.
    # --------------------------------------------------

    # --------------------------------------------------
    # Import list of symbols to process.
    # --------------------------------------------------

    # read list of symbols from disk.
    #stockList = 'Naz100'
    stockList = run_params['stockList']
    filename = os.path.abspath(run_params['symbols_file'])
    # filename = os.path.join(_data_path, 'symbols', stockList+'_Symbols.txt')                   # plotmax = 1.e10, runnum = 902

    # --------------------------------------------------
    # Get quotes for each symbol in list
    # process dates.
    # Clean up quotes.
    # Make a plot showing all symbols in lis
    # --------------------------------------------------

    ## update quotes from list of symbols
    (symbols_directory, symbols_file) = os.path.split(filename)
    basename, extension = os.path.splitext(symbols_file)
    print((" symbols_directory = ", symbols_directory))
    print(" symbols_file = ", symbols_file)
    print("symbols_directory, symbols.file = ", symbols_directory, symbols_file)
    ###############################################################################################
    do_update = True
    if do_update == True:
        UpdateHDF_yf(symbols_directory, symbols_file, json_fn)  ### assume hdf is already up to date
    adjClose, symbols, datearray, Close, _ = loadQuotes_fromHDF(
        filename, json_fn
    )
    Close = Close.to_numpy().swapaxes(0,1)
    return adjClose, Close, symbols, datearray


def get_quote_corrections(symbols_text: str, start_date: str = '1900-01-01') -> Tuple[list, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("\n\nBegin download of adjusted Close value")
    adj_data = yf.download(  # or pdr.get_data_yahoo(...
        # tickers list or string as well
        tickers = symbols_text,

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')

        # use start date
        start = start_date,

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval = "1d",

        # group by ticker (to access via data['SPY'])
        # (optional, default is 'column')
        group_by = 'column',

        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust = True,

        # download pre/post regular market hours data
        # (optional, default is False)
        prepost = False,

        # use threads for mass downloading? (True/False/Integer)
        # (optional, default is True)
        threads = True,

        # proxy URL scheme use use when downloading?
        # (optional, default is None)
        proxy = None
    )
    print("Begin download of Close value")
    data = yf.download(  # or pdr.get_data_yahoo(...
        # tickers list or string as well
        tickers = symbols_text,

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        # use start date
        start = start_date,

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval = "1d",

        # group by ticker (to access via data['SPY'])
        # (optional, default is 'column')
        group_by = 'column',

        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust = False,

        # download pre/post regular market hours data
        # (optional, default is False)
        prepost = False,

        # use threads for mass downloading? (True/False/Integer)
        # (optional, default is True)
        threads = True,

        # proxy URL scheme use use when downloading?
        # (optional, default is None)
        proxy = None
    )
    print(" ... days in data: "+str(len(data['Close'].index)))
    print(" ... days in adj_data: "+str(len(adj_data['Close'].index)))

    # drop dates that are only in one of the dataframes
    if len(data['Close'].index) > len(adj_data['Close'].index):
        extra_data_dates = [x for x in data['Close'].index \
                            if x not in adj_data['Close'].index]
        data_dates = data['Close'].index
        drop_dates_indices = []
        for i in range(len(extra_data_dates)):
            indx_to_drop = [ii for ii in range(len(data_dates)) if \
                            extra_data_dates[i] == data_dates[ii]][0]
            drop_dates_indices.append(indx_to_drop)
        for i in drop_dates_indices[::-1]:
            data.drop(labels=data_dates[i], axis=0, inplace=True)
    if len(data['Close'].index) < len(adj_data['Close'].index):
        extra_adj_data_dates = [x for x in adj_data['Close'].index \
                                if x not in data['Close'].index]
        adj_data_dates = adj_data['Close'].index
        drop_adj_dates_indices = []
        for i in range(len(extra_adj_data_dates)):
            indx_to_drop = [ii for ii in range(len(adj_data_dates)) if \
                            extra_adj_data_dates[i] == adj_data_dates[ii]][0]
            drop_adj_dates_indices.append(indx_to_drop)
        for i in drop_adj_dates_indices[::-1]:
            adj_data.drop(labels=adj_data_dates[i], axis=0, inplace=True)

    adj_close_factor = (adj_data['Close'] / data['Close']).to_numpy().swapaxes(0,1)
    adj_volume_factor = (adj_data['Volume'] / data['Volume']).to_numpy().swapaxes(0,1)
    quotes = adj_data['Close'].to_numpy().swapaxes(0,1)
    volume = adj_data['Volume'].to_numpy().swapaxes(0,1)
    symbols = data['Close'].columns
    dates = data['Close'].index
    dates_str = [str(val).split(" ")[0] for val in dates]
    dates_datetime = []
    for idate in dates_str:
        dates_datetime.append(datetime.date(*[int(val) for val in idate.split('-')]))
    return dates_datetime, symbols, quotes, volume,\
           adj_close_factor, adj_volume_factor


def fix_quotes(json_fn: str, _data_path: str, stockList: str = 'Naz100') -> None:
    # --------------------------------------------------
    # Import list of symbols to process.
    # --------------------------------------------------

    # read list of symbols from disk.
    symbols_fn = stockList + "_Symbols.txt"
    filename = os.path.join(_data_path, 'symbols', symbols_fn)                   # plotmax = 1.e10, runnum = 902

    # -------------------------------------------------
    # Get quotes for each symbol in list
    # process dates.
    # Clean up quotes.
    # Make a plot showing all symbols in list
    # --------------------------------------------------

    ## update quotes from list of symbols
    (symbols_directory, symbols_file) = os.path.split(filename)
    basename, extension = os.path.splitext(symbols_file)
    print("\n\n\n **************** ")
    print(" Beginning QC of stored quotes ")

    print((" stockList = ", stockList))
    print((" symbols_directory = ", symbols_directory))
    print(" symbols_file = ", symbols_file)
    print("symbols_directory, symbols_file = ", symbols_directory, symbols_file)

    with open(filename, 'r') as f:
        lines = f.read()
    symbols_text = lines.replace('\n',' ').replace(".","-")

    data = yf.download(  # or pdr.get_data_yahoo(...
            # tickers list or string as well
            tickers = symbols_text,

            # use "period" instead of start/end
            # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            # (optional, default is '1mo')
            period = "max",

            # fetch data by interval (including intraday if period < 60 days)
            # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            # (optional, default is '1d')
            interval = "1d",

            # group by ticker (to access via data['SPY'])
            # (optional, default is 'column')
            group_by = 'column',

            # adjust all OHLC automatically
            # (optional, default is False)
            auto_adjust = True,

            # download pre/post regular market hours data
            # (optional, default is False)
            prepost = False,

            # use threads for mass downloading? (True/False/Integer)
            # (optional, default is True)
            threads = True,

            # proxy URL scheme use use when downloading?
            # (optional, default is None)
            proxy = None
        )

    quotes = data['Close'].to_numpy()
    volume = data['Volume'].to_numpy()

    symbols = data['Close'].columns
    dates = data['Close'].index
    print(" ... first date = "+str(dates[0]))

    plt.figure(1, figsize=(12,9), dpi=150)
    plt.clf()
    plt.grid()
    plt.yscale('log')
    for isymbol in range(quotes.shape[1]):
        try:
            first_val = quotes[:,isymbol][~np.isnan(quotes[:,isymbol])][0]
            plt.plot(dates, quotes[:,isymbol] / first_val)
        except:
            pass

    num_days = 50
    gainloss_stddevs = quotes * 0.
    for isymbol in range(quotes.shape[1]):
        if isymbol%20 == 0:
            print(' ... starting '+str(isymbol)+' of '+str(quotes.shape[1]))
        for idate in range(quotes.shape[0]):
            if idate > num_days:
                gainloss_std = (quotes[idate-num_days:idate, isymbol] / \
                          quotes[idate-num_days-1:idate-1, isymbol]-1).std()
                gainloss = (quotes[idate, isymbol] / quotes[idate-1, isymbol]-1)
                if gainloss_std != 0.:
                    gainloss_stddevs[idate, isymbol] = gainloss / gainloss_std
                else:
                    gainloss_stddevs[idate, isymbol] = np.nan

    # plot value for equal amount invested in each stock
    gainloss = quotes * 0.
    gainloss[1:,:] = quotes[1:,:] / quotes[:-1,:]
    gainloss[np.isnan(gainloss)] = 1.
    avg_gainloss = gainloss.mean(axis=-1)
    avg_gainloss[0] = 10000.
    value = avg_gainloss.cumprod()
    plt.figure(2, figsize=(12,9), dpi=150)
    plt.clf()
    plt.grid()
    plt.yscale('log')
    plt.plot(dates, value)



    plt.figure(3, figsize=(12,9), dpi=150)
    plt.clf()
    plt.grid()
    for isymbol in range(quotes.shape[1]):
        plt.plot(dates, gainloss_stddevs[:,isymbol])


    ### ----------------------------------------------------
    ### get quotes from hdf
    ### ----------------------------------------------------
    print("\n\n ... load quotes from hdf.  "+filename)
    (
        __adjClose,
        __symbols,
        __dates,
        __df_quote_hdf,
        __listname
    ) = loadQuotes_fromHDF(filename, json_fn)


    ### ----------------------------------------------------
    ### get quote correction factors
    ### ----------------------------------------------------
    print("\n\n ... get corrections to quotes")
    if symbols_text[-1] == ' ':
        symbols_text = symbols_text[:-1]
    first_date = __dates[0]
    print("   . first_date = " + str(first_date))
    if type(first_date) == str:
        _fdate = first_date.split("-")
        print("   . _fdate = " + str(_fdate))
        first_date = datetime.date(_fdate)
    elif type(first_date) == datetime.date:
        _fdate = first_date
    fresh_dates, fresh_symbols,\
           fresh_quotes,\
           fresh_volume,\
           adj_close_factor,\
           adj_volume_factor = get_quote_corrections(
               symbols_text,
               start_date=first_date - datetime.timedelta(days=366)
           )
    print(" ... first fresh_dates = "+str(fresh_dates[0]))


    ### ----------------------------------------------------
    ### get quotes from hdf
    ### ----------------------------------------------------
    print("\n\n ... get stored quotes")
    quote_data = get_stored_quotes(json_fn, stockList)
    stored_adjClose, stored_Close, stored_symbols, stored_dates = quote_data
    print(" ... first stored_dates = "+str(stored_dates[0]))

    # remove symbols that don't have valid quotes
    valid_indices = []
    for i in range(stored_adjClose.shape[0]):
        _data = stored_adjClose[i,:]*1.
        if _data[~np.isnan(_data)].min() == _data[~np.isnan(_data)].max():
            print("symbol has no valid quotes "+str((i,stored_symbols[i])))
            continue
        valid_indices.append(i)
    stored_adjClose = stored_adjClose[valid_indices,:] * 1.
    stored_Close = stored_Close[valid_indices,:] * 1.
    stored_symbols = list(np.array(stored_symbols)[valid_indices])

    # remove quotes that have later values on same date
    valid_date_indices = []
    for j in range(stored_adjClose.shape[1]-1):
        if stored_dates[j] != stored_dates[j+1]:
            valid_date_indices.append(j)
        else:
            print("duplicate quote on date "+str((j,stored_dates[j])))
    valid_date_indices.append(stored_adjClose.shape[1]-1)
    stored_adjClose = stored_adjClose[:,valid_date_indices] * 1.
    stored_Close = stored_Close[:,valid_date_indices] * 1.
    stored_dates = list(np.array(stored_dates)[valid_date_indices])

    adjClose, Close, symbols, datearray = stored_adjClose, stored_Close, stored_symbols, stored_dates


    ### ----------------------------------------------------
    ### combine stored and fresh quotes in dataframe so they are
    ### aligned
    ### ----------------------------------------------------
    stored_data_dict = {'dates':np.array(stored_dates)}
    df_labels = ['dates']
    for i,_s in enumerate(stored_symbols):
        df_labels.append(_s)
        stored_data_dict[stored_symbols[i]] = stored_adjClose[i,:]
    df_stored = pd.DataFrame(stored_data_dict)

    fresh_data_dict = {'dates':np.array(fresh_dates)}
    df_labels = ['dates']
    for i,_s in enumerate(fresh_symbols):
        df_labels.append(_s)
        fresh_data_dict["_fresh_"+fresh_symbols[i]] = fresh_quotes[i,:]
    for i,_s in enumerate(fresh_symbols):
        df_labels.append(_s)
        fresh_data_dict["_fresh_vol_"+fresh_symbols[i]] = fresh_volume[i,:]
    for i,_s in enumerate(fresh_symbols):
        df_labels.append(_s)
        fresh_data_dict["_fresh_close_factor_"+fresh_symbols[i]] = adj_close_factor[i,:]
    for i,_s in enumerate(fresh_symbols):
        df_labels.append(_s)
        fresh_data_dict["_fresh_vol_factor_"+fresh_symbols[i]] = adj_volume_factor[i,:]
    df_fresh = pd.DataFrame(fresh_data_dict)

    # merge
    df_stored_and_fresh = pd.merge(df_stored, df_fresh, on='dates', how='outer')

    df_stored_and_fresh = df_stored_and_fresh.set_index('dates')
    df_stored_and_fresh = df_stored_and_fresh.sort_index()
    df_stored_and_fresh = df_stored_and_fresh[~df_stored_and_fresh.index.duplicated(keep='last')]
    print(" ... df_stored_and_fresh = "+str(df_stored_and_fresh))
    print(" ... first date in (merged) df_stored_and_fresh = "+str(df_stored_and_fresh.index[0]))


    ### ----------------------------------------------------
    ### synchronize dates to match 'datearray'
    ### ----------------------------------------------------

    dates = df_stored_and_fresh.index
    print(" ... first date in (merged) dates = "+str(dates[0]))

    # synchronize dates to match '_datearray'
    datearray_as_list = list(datearray)
    dates_as_list = list(dates)
    quotes_synchronized = list()
    missing_dates = list()
    common_dates = list()
    keep_date_indices_list = list()
    for idate, date in enumerate(datearray):
        if date not in dates_as_list:
            print("missing date = "+str(date))
            missing_dates.append(date)
            continue
        if date == datearray[-1]:
            common_dates.append(date)
            keep_date_indices_list.append(idate)
            i = dates_as_list.index(date)
            quotes_synchronized.append(quotes[i,:])
        elif date != datearray[idate+1]:
            common_dates.append(date)
            keep_date_indices_list.append(idate)
            i = dates_as_list.index(date)
            quotes_synchronized.append(quotes[i,:])
    quotes_synchronized = np.array(quotes_synchronized).swapaxes(0,1)

    extra_dates = list()
    for idate, date in enumerate(dates_as_list):
        if date not in datearray_as_list:
            print("extra date = "+str(date))
            extra_dates.append(date)
    extra_dates_str = [str(val) for val in extra_dates]
    missing_dates_str = [str(val) for val in missing_dates]


    adjClose = adjClose[:,keep_date_indices_list]
    Close = Close[:,keep_date_indices_list]
    #stored_symbols
    datearray = np.array(datearray)[keep_date_indices_list]

    extra_dates_dt = []
    for i,d in enumerate(extra_dates_str):
        d_tuple = tuple(np.array(d.split('-')).astype(int))
        extra_dates_dt.append(datetime.date(d_tuple[0],d_tuple[1],d_tuple[2]))
    if extra_dates_dt != []:
        df_stored_and_fresh = df_stored_and_fresh.drop(extra_dates_dt)

    ### ----------------------------------------------------
    ### combine stored and fresh quotes in dataframe so they are
    ### aligned
    ### ----------------------------------------------------

    _datearray = df_stored_and_fresh.index
    print(" ... first date in (merged) _datearray = "+str(_datearray[0]))
    _symbol_list = [x for x in df_stored_and_fresh.columns if 'fresh' not in x and 'dates' not in x]
    updated_adjClose = adjClose.copy()
    updated_Close = adjClose.copy()
    datearray_as_list = list(datearray)
    adj_factor_previous = 1.
    symbols = list(symbols)

    current_symbol_count = 0
    common_symbols = []
    excluded_symbols = []

    debug = False
    for icompany, symbol in enumerate(_symbol_list):

        if icompany%1 == 0:
            print("\n ... icompany, current_symbol_count, symbol = "+\
                  str(icompany)+", "+str(current_symbol_count)+", "+symbol)

        # only compare if symbols is in both stored and fresh quotes
        if symbol in symbols:
            iicompany = symbols.index(symbol)
        else:
            # quotes for this symbol only in stored
            excluded_symbols.append(symbol)
            print("   . iicompany not assigned "+excluded_symbols[-1])
            continue

        # only compare if symbols is in both stored and fresh quotes
        if symbol not in fresh_symbols:
            # quotes for this symbol only in stored
            excluded_symbols.append(symbol)
            print("   . symbol not in fresh_symbols "+excluded_symbols[-1])
            continue

        # quotes for this symbol in both stored and fresh
        # get index for this symbol. might not be same position in stored and fresh
        stored_close = df_stored_and_fresh[symbol].values
        stored_close = cleanspikes(stored_close, periods=20, stddevThreshold=5.0)
        stored_close = interpolate(stored_close)
        stored_close = nans_at_beginning(stored_close)
        stored_close = nans_at_beginning(stored_close[::-1])[::-1]

        try:
            fresh_close = df_stored_and_fresh["_fresh_"+symbol].values
            fresh_close = cleanspikes(fresh_close, periods=20, stddevThreshold=5.0)
            fresh_close = interpolate(fresh_close)
            fresh_close = nans_at_beginning(fresh_close)
            fresh_close = nans_at_beginning(fresh_close[::-1])[::-1]
            fresh_close_factor = df_stored_and_fresh["_fresh_close_factor_"+symbol].values
            common_symbols.append(symbol)
            current_symbol_count += 1
        except:
            # likely a symbol for a compnay that is not longer publicly traded
            excluded_symbols.append(symbol)
            print("   . symbol is not currently in index "+excluded_symbols[-1])
            continue

        if icompany%1 == 0:
            print("   . icompany, current_symbol_count, symbol = "+\
                  str(icompany)+", "+str(current_symbol_count)+", "+symbol)

        # process each date starting with current date
        print("\n\nlooping on dates in _datearray: ")
        print("   . _datearray[0] = "+str(_datearray[0]))
        print("   . _datearray[-1] = "+str(_datearray[-1]))
        for idate, date in enumerate(_datearray[::-1]):

            # get matching dates index
            try:
                if date == datearray_as_list[datearray_as_list.index(date)]:
                    idate_stored = datearray_as_list.index(date)
            except:
                if symbol == 'FOX' and datetime.date(2018,1,1) < date < datetime.date(2020,1,1):
                    print(str(date)+" case 0 "+ str((quotes[iicompany,idate_stored], adjClose[icompany,idate])))
                if symbol == 'AAPL' and datetime.date(2018,1,1) < date < datetime.date(2020,1,1):
                    print(str(date)+" case 0 "+ str((quotes[iicompany,idate_stored], adjClose[icompany,idate])))
                continue

            # case 1: neither fresh quote nor adjClose are NaN --> keep fresh
            if not np.isnan(stored_close[::-1][idate]) and not np.isnan(fresh_close[::-1][idate])  :
                # case 1: neither fresh quote nor adjClose are NaN --> keep fresh
                if date.year >= 2018 and symbol=='AAPL':
                    print(symbol+", "+str(date)+" case 1 "+ str((fresh_close[::-1][idate], stored_close[::-1][idate])))
                updated_adjClose[icompany,idate_stored] = fresh_close[::-1][idate]
                factor_today = fresh_close_factor[::-1][idate]
                updated_Close[icompany,idate_stored] = fresh_close[::-1][idate] / factor_today
                adj_factor_previous = updated_adjClose[icompany,idate] / stored_close[::-1][idate]
                adj_factor_previous = factor_today
                if symbol == 'FOX' and datetime.date(2018,1,1) < date < datetime.date(2020,1,1):
                    print(symbol+", "+str(date)+" case 1 "+ str((fresh_close[::-1][idate], stored_close[::-1][idate],factor_today)))
                if 2018 <= date.year <= 2019 and symbol=='AAPL':
                    print(symbol+", "+str(date)+" case 1 "+ str((fresh_close[::-1][idate], stored_close[::-1][idate])))
                if debug:
                    print(" . 1 keep fresh: " + str((_datearray[::-1][idate], fresh_close[::-1][idate],
                                                 stored_close[::-1][idate],updated_adjClose[icompany,:][::-1][idate],
                                                 np.round(np.array((fresh_close_factor[::-1][idate],factor_today,adj_factor_previous)),3) )))

            # case 2: fresh quote is not NaN but stored is NaN --> keep fresh_quote
            elif not np.isnan(fresh_close[::-1][idate]) and np.isnan(stored_close[::-1][idate]):
                # case 2: fresh quote is not NaN but stored is NaN --> keep NaN
                if date.year >= 2018 and symbol=='AAPL':
                    print(symbol+", "+str(date)+" case 2 "+ str((fresh_close[::-1][idate], stored_close[::-1][idate])))
                updated_adjClose[icompany,idate_stored] = fresh_close[::-1][idate]
                factor_today = fresh_close_factor[::-1][idate]
                updated_Close[icompany,idate_stored] = fresh_close[::-1][idate] / factor_today
                if symbol == 'FOX' and datetime.date(2018,1,1) < date < datetime.date(2020,1,1):
                    print(symbol+", "+str(date)+" case 2 "+ str((fresh_close[::-1][idate], stored_close[::-1][idate],factor_today)))
                if 2018 <= date.year <= 2019 and symbol=='AAPL':
                    print(symbol+", "+str(date)+" case 2 "+ str((fresh_close[::-1][idate], stored_close[::-1][idate])))
                if debug:
                    print(" . 2 keep fresh: " + str((_datearray[::-1][idate], fresh_close[::-1][idate],
                                                 stored_close[::-1][idate],updated_adjClose[icompany,:][::-1][idate],
                                                 np.round(np.array((fresh_close_factor[::-1][idate],factor_today,adj_factor_previous)),3) )))

            # case 3: fresh quote is NaN but adjClose is not NaN--> keep stored
            elif np.isnan(fresh_close[::-1][idate]) and not np.isnan(stored_close[::-1][idate]):
                # case 3: fresh quote is NaN but adjClose is not NaN--> keep stored
                if fresh_close_factor[::-1][idate]:
                    yesterday = stored_close[::-1][idate-1]
                    today = stored_close[::-1][idate]
                    yesterday_updated = updated_adjClose[icompany,:][::-1][idate-1]
                    today_updated = yesterday_updated * today / yesterday
                    adj_factor_previous = today_updated / today

                updated_adjClose[icompany,idate_stored] = stored_close[::-1][idate] * adj_factor_previous
                updated_Close[icompany,idate_stored] = stored_close[::-1][idate] * adj_factor_previous
                if symbol == 'FOX' and datetime.date(2018,1,1) < date < datetime.date(2020,1,1):
                    print(symbol+", "+str(date)+" case 3 "+ str((fresh_close[::-1][idate], stored_close[::-1][idate],factor_today)))
                if 2018 <= date.year <= 2019 and symbol=='AAPL':
                    print(symbol+", "+str(date)+" case 3 "+ str((fresh_close[::-1][idate], stored_close[::-1][idate])))
                if debug:
                    print(" . 3 keep stored: " + str((_datearray[::-1][idate], fresh_close[::-1][idate],
                                                 stored_close[::-1][idate],updated_adjClose[icompany,:][::-1][idate],
                                                 np.round(np.array((fresh_close_factor[::-1][idate],factor_today,adj_factor_previous)),3) )))

            # case 4:  fresh_quote is NaN --> keep stored
            elif np.isnan(fresh_close[::-1][idate]):
                # case 4:  quote is NaN --> keep stored
                updated_adjClose[icompany,idate_stored] = stored_close[::-1][idate] * adj_factor_previous
                if symbol == 'FOX' and datetime.date(2018,1,1) < date < datetime.date(2020,1,1):
                    print(symbol+", "+str(date)+" case 4 "+ str((fresh_close[::-1][idate], stored_close[::-1][idate],factor_today)))
                if 2018 <= date.year <= 2019 and symbol=='AAPL':
                    print(symbol+", "+str(date)+" case 4 "+ str((fresh_close[::-1][idate], stored_close[::-1][idate])))
                if debug:
                    print(" . 4 keep stored: " + str((_datearray[::-1][idate], fresh_close[::-1][idate],
                                                 stored_close[::-1][idate],updated_adjClose[icompany,:][::-1][idate],
                                                 np.round(np.array((fresh_close_factor[::-1][idate],factor_today,adj_factor_previous)),3) )))

            # case -1: fresh quote is not NaN but stored is first value --> keep NaN
            #          indicates that symbol was not yet in Nasdaq index
            elif not np.isnan(quotes[iicompany,idate_stored]) and adjClose[icompany,idate]==adjClose[icompany,0]:
                # case -1: fresh quote is not NaN but stored is first value --> keep NaN
                #         indicates that symbol was not yet in Nasdaq index
                updated_adjClose[icompany,idate_stored] = np.nan
                updated_Close[icompany,idate_stored] = np.nan
                if symbol == 'FOX' and datetime.date(2018,1,1) < date < datetime.date(2020,1,1):
                    print(symbol+", "+str(date)+" case -1 "+ str((fresh_close[::-1][idate], stored_close[::-1][idate],factor_today)))
                if 2018 <= date.year <= 2019 and symbol=='AAPL':
                    print(symbol+", "+str(date)+" case -1 "+ str((fresh_close[::-1][idate], stored_close[::-1][idate])))
                if debug:
                    print(" . -1 keep NaN: " + str((_datearray[::-1][idate], fresh_close[::-1][idate],
                                                 stored_close[::-1][idate],updated_adjClose[icompany,:][::-1][idate],
                                                 np.round(np.array((fresh_close_factor[::-1][idate],factor_today,adj_factor_previous)),3) )))

        plt.close(4)
        plt.figure(4, figsize=(12,9), dpi=150)
        plt.clf()
        plt.grid()
        plt.plot(_datearray,stored_close, 'k-', lw=2, label='stored '+symbol)
        plt.plot(_datearray,fresh_close, 'r-', label='quotes '+symbol)
        plt.plot(datearray,updated_adjClose[icompany,:], 'y-', lw=.35, label='fixed quotes '+symbol)
        plt.yscale('log')
        plt.legend()
        # plt_file = os.path.join(os.getcwd(), 'pngs', 'fix_quotes', symbol+'.png')
        plt_file = os.path.join(
            symbols_directory, "..", 'pngs', symbol+'.png'
        )
        plt.savefig(plt_file, format='png', dpi=300)


    ### ***********************************************************************
    ### Check for large single-day price changes that might signal an error
    ### ***********************************************************************
    print("\n\n\n")
    print("Checking for large single-day price changes that might signal an error")
    for icompany, symbol in enumerate(_symbol_list):

        # only compare if symbols is in both stored and fresh quotes
        if symbol in symbols:
            iicompany = symbols.index(symbol)
        else:
            # quotes for this symbol only in stored
            excluded_symbols.append(symbol)
            continue

        # only compare if symbols is in both stored and fresh quotes
        if symbol not in fresh_symbols:
            # quotes for this symbol only in stored
            excluded_symbols.append(symbol)
            #print("   . symbol not in fresh_symbols "+excluded_symbols[-1])
            continue

        # quotes for this symbol in both stored and fresh
        # get index for this symbol. might not be same position in stored and fresh
        column =_symbol_list.index(symbol)
        _symbol_updated_adjClose = updated_adjClose[column,:].copy()
        _symbol_adjClose = adjClose[column,:].copy()
        stored_adjdaily_change = _symbol_adjClose[1:] / _symbol_adjClose[:-1] - 1.
        fresh_adjdaily_change = _symbol_updated_adjClose[1:] / _symbol_updated_adjClose[:-1] - 1.

        stored_close = df_stored_and_fresh[symbol].values
        fresh_close = df_stored_and_fresh["_fresh_"+symbol].values
        stored_daily_change = stored_close[1:] / stored_close[:-1] - 1.
        fresh_daily_change = fresh_close[1:] / fresh_close[:-1] - 1.
        indices_big_change = np.where(np.abs(stored_daily_change - fresh_daily_change) > .009)[0]

        if len(indices_big_change) > 0:
            dates_big_change = np.array(_datearray)[indices_big_change + 1]

            # print dates of big daily price changes

            for ii, idates_big_change in enumerate(list(indices_big_change)):
                before_after_stored_close = np.array((stored_close[idates_big_change-2:idates_big_change+3]))
                before_after_fresh_close = np.array((fresh_close[idates_big_change-2:idates_big_change+3]))
                ratios = np.round(before_after_fresh_close / before_after_stored_close, 2)
                before_after_stored_close = np.round(before_after_stored_close, 2)
                before_after_fresh_close = np.round(before_after_fresh_close, 2)

                before_after_stored_adjclose = np.array((_symbol_adjClose[idates_big_change-2:idates_big_change+3]))
                before_after_fresh_adjclose = np.array((_symbol_updated_adjClose[idates_big_change-2:idates_big_change+3]))
                adj_ratios = np.round(before_after_fresh_adjclose / before_after_stored_adjclose, 2)
                before_after_stored_adjclose = np.round(before_after_stored_adjclose, 2)
                before_after_fresh_adjclose = np.round(before_after_fresh_adjclose, 2)

                if np.all(adj_ratios != ratios):

                    print("\n   . icompany, symbol, date, stored, fresh change " + " = " + \
                          str(icompany)+", " + symbol +", " + \
                          str(_datearray[idates_big_change]) +", " + \
                          format(stored_daily_change[idates_big_change], "5.2%") +", " + \
                          format(fresh_daily_change[idates_big_change], "5.2%") +", " + \
                          str(before_after_stored_close) +", " + \
                          str(before_after_fresh_close) +", " + \
                          str(ratios)
                          )
                    print("   . icompany, symbol, date, stored, fresh change " + " = " + \
                          str(icompany)+", " + symbol +", " + \
                          str(_datearray[idates_big_change]) +", " + \
                          format(stored_adjdaily_change[idates_big_change], "5.2%") +", " + \
                          format(fresh_adjdaily_change[idates_big_change], "5.2%") +", " + \
                          str(before_after_stored_adjclose) +", " + \
                          str(before_after_fresh_adjclose) +", " + \
                          str(adj_ratios)+"\n"
                          )
        fresh_close = cleanspikes(fresh_close, periods=20, stddevThreshold=5.0)
        fresh_close = interpolate(fresh_close)
        fresh_close = nans_at_beginning(fresh_close)
        fresh_close = nans_at_beginning(fresh_close[::-1])[::-1]
        fresh_close = cleantobeginning(fresh_close)
        fresh_close = cleantoend(fresh_close)
        updated_adjClose[column,:] = fresh_close * 1.

    print(" ... Finished checking for large single-day price changes that might signal an error")
    print("\n\n\n")


    print("\n\nlooping again on dates in _datearray: ")
    print("   . _datearray[0] = "+str(_datearray[0]))
    print("   . _datearray[-1] = "+str(_datearray[-1]))
    _datearray_txt = []
    for date in _datearray:
        _datearray_txt.append(date.strftime('%Y-%m-%d'))

    if updated_adjClose.shape[1] == len(_datearray):
        quotes_UpdatedSymbols = pd.DataFrame(updated_adjClose.swapaxes(0,1), index=_datearray_txt, columns=_symbol_list)
    else:
        quotes_UpdatedSymbols = pd.DataFrame(updated_adjClose, index=_datearray_txt, columns=_symbol_list)

    CASHadjClose = np.ones( (len(quotes_UpdatedSymbols.index)), float ) * 100000.
    for i in range(CASHadjClose.shape[0]):
        if i%10 == 0:
            CASHadjClose[i] = CASHadjClose[i-1] + .01
        else:
            CASHadjClose[i] = CASHadjClose[i-1]

    quotes_UpdatedSymbols['CASH'] = CASHadjClose / 100000.

    # get copy from hdf as dataframe prior to update
    # dirname = os.path.join( os.getcwd(), "symbols" )
    dirname = symbols_directory
    listname = stockList + "_Symbols"
    hdf5filename = os.path.join( dirname, listname + "_.hdf5" )
    print("hdf5 filename = ",hdf5filename)
    notupdated_data = pd.read_hdf(hdf5filename, listname)

    # reformat the dates and column labels as strings
    x = quotes_UpdatedSymbols.to_numpy()
    if 'SP500' in stockList:
        __datearray = [str(x) for x  in quotes_UpdatedSymbols.index]
    elif 'Naz100' in stockList:
        __datearray = [str(x) for x  in quotes_UpdatedSymbols.index]
        #__datearray = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in __datearray]
        if 'datetime' in str(type(__datearray[0])):
            __datearray = [d.strftime("%Y-%m-%d") for d in __datearray]
        if ' ' in __datearray[0]:
            __datearray = [d.split(' ')[0] for d in __datearray]
    elif 'Index' in stockList:
        __datearray = [str(x) for x  in quotes_UpdatedSymbols.index]
        if 'datetime' in str(type(__datearray[0])):
            __datearray = [d.strftime("%Y-%m-%d") for d in __datearray]
        if ' ' in __datearray[0]:
            __datearray = [d.split(' ')[0] for d in __datearray]


    __symbols = [str(x) for x in quotes_UpdatedSymbols.columns]
    df_updated = pd.DataFrame(x, index=__datearray, columns=__symbols)

    print("\n\nready to write update dataframe to hdf: ")
    print("   . __datearray[0] = "+str(__datearray[0]))
    print("   . __datearray[-1] = "+str(__datearray[-1]))
    print("\n\n ... type(df_updated.index[0]) = "+str(type(df_updated.index[0])))
    print("   . type(notupdated_data.index[0]) = "+str(type(notupdated_data.index[0])))
    assert type(df_updated.index[0]) == type(notupdated_data.index[0])


    # write updated, cleaned quotes to disk.
    #quotes_UpdatedSymbols.to_hdf( hdf5filename, key=listname, data_columns=quotes_UpdatedSymbols.columns, mode='a',format='table',append=False,complevel=5,complib='blosc')
    updated_symbols_list = [str(s) for s in quotes_UpdatedSymbols.columns]
    df_updated.to_hdf(hdf5filename, key=listname,
                      data_columns=quotes_UpdatedSymbols.columns,
                      mode='a', format='f', append=False,
                      complevel=5, complib='blosc')
    return

if __name__ == "__main__":

    ### --------------------------------------
    ### clean index quotes stored locally
    ### --------------------------------------
    _data_path = os.path.abspath(os.path.join(os.path.abspath(os.getcwd()),
                                              '..', 'Py3TAAA_index'))
    stock_list = 'Index'
    fix_quotes(_data_path, stockList=stock_list)


    ### --------------------------------------
    ### clean naz100 quotes stored locally
    ### --------------------------------------
    _data_path = os.path.abspath(os.path.join(os.path.abspath(os.getcwd()),
                                              '..', 'Py3TAAADL_tracker'))
    stock_list = 'Naz100'
    fix_quotes(_data_path, stockList=stock_list)


    ### --------------------------------------
    ### clean SP500 quotes stored locally
    ### --------------------------------------
    os.chdir(_cwd)
    _data_path = os.path.abspath(os.getcwd())
    stock_list = 'SP500'
    fix_quotes(_data_path, stockList=stock_list)
