import numpy as np
import datetime
from matplotlib.pylab import *

import datetime
from scipy import random
from scipy.stats import rankdata

import nose
import bottleneck as bn
import la

from functions.quotes_adjCloseVol import *
from functions.TAfunctions import *
from functions.readSymbols import *

def arrayFromQuotesForList(symbolsFile, beginDate, endDate):
    '''
    read in quotes and process to 'clean' ndarray plus date array
    - prices in array with dimensions [num stocks : num days ]
    - process stock quotes to show closing prices adjusted for splits, dividends
    - single ndarray with dates common to all stocks [num days]
    - clean up stocks by:
       - infilling empty values with linear interpolated value
       - repeat first quote to beginning of series
    '''

    # read symbols list
    symbols = readSymbolList(symbolsFile,verbose=True)

    # get quotes for each symbol in list (adjusted close)
    quote = downloadQuotes(symbols,date1=beginDate,date2=endDate,adjust=True,Verbose=True)

    # clean up quotes for missing values and varying starting date
    x=quote.copyx()
    #print "larry getlabel 0 = ", quote.getlabel(0) ### dpg diagnostic
    #print "larry getlabel 1 = ", quote.getlabel(1) ### dpg diagnostic
    #print "larry getlabel 2 = ", quote.getlabel(2) ### dpg diagnostic
    date = quote.getlabel(2)
    datearray=array(date)

    print " x check: ",x[:,0,:][isnan(x[:,0,:])].shape

    # Clean up input quotes
    #  - infill interior NaN values using nearest good values to linearly interpolate
    #  - copy first valid quote to from valid date to all earlier positions
    for ii in range(x.shape[0]):
        x[ii,0,:] = interpolate(x[ii,0,:])
        x[ii,0,:] = cleantobeginning(x[ii,0,:])

    #return x[:,0,:], symbols, datearray
    return x[:,0,:], quote.getlabel(0), datearray

def LastQuotesForList(symbolList, endDate):
    from matplotlib.finance import quotes_historical_yahoo
    '''
    read in quotes and process to 'clean' ndarray plus date array
    - prices in array with dimensions [num stocks : num days ]
    - process stock quotes to show closing prices adjusted for splits, dividends
    - single ndarray with dates common to all stocks [num days]
    - clean up stocks by:
       - infilling empty values with linear interpolated value
       - repeat first quote to beginning of series
    '''
    year = datetime.datetime.now().year
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    date2 = (int(year), int(month), int(day))
    date2 = datetime.date.today() + datetime.timedelta(-1)
    # get quotes for each symbol in list (adjusted close)
    quotelist = []
    for i in range(len(symbolList)):
        ticker = symbolList[i]
        #print "ticker = ", ticker,date2
        if ticker == 'CASH':
            quotelist.append(1.0)
        else:
            data = quotes_historical_yahoo(ticker, date2, endDate, asobject ="None")
            #print "data = ", data
            quotelist.append(data[0][6])
    #quote = downloadQuotes(symbolList,date1=(year,month,day),date2=(year,month,day),adjust=False,Verbose=True)

    """
    # clean up quotes for missing values and varying starting date
    x=quote.copyx()

    # Clean up input quotes
    #  - infill interior NaN values using nearest good values to linearly interpolate
    #  - copy first valid quote to from valid date to all earlier positions
    quotelist = []
    for ii in range(x.shape[0]):
        quotelist.append(x[ii,0,-1])
    """
    #print "quotelist =",quotelist
    #return quotelist
    return quotelist



