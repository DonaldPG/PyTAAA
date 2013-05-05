import numpy as np
import datetime
from matplotlib.pylab import *

import datetime
from scipy import random
from scipy.stats import rankdata

import nose
import bottleneck as bn
import la

import functions.quotes_adjCloseVol
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

    #print " x check: ",x[:,0,:][isnan(x[:,0,:])].shape

    # Clean up input quotes
    #  - infill interior NaN values using nearest good values to linearly interpolate
    #  - copy first valid quote to from valid date to all earlier positions
    for ii in range(x.shape[0]):
        x[ii,0,:] = interpolate(x[ii,0,:])
        x[ii,0,:] = cleantobeginning(x[ii,0,:])

    #return x[:,0,:], symbols, datearray
    return x[:,0,:], quote.getlabel(0), datearray

'''
def LastQuotesForSymbolList(symbolList):
    from matplotlib.finance import quotes_historical_yahoo
    """
    read in quotes and process to 'clean' ndarray plus date array
    - prices in array with dimensions [num stocks : num days ]
    - process stock quotes to show closing prices adjusted for splits, dividends
    - single ndarray with dates common to all stocks [num days]
    - clean up stocks by:
       - infilling empty values with linear interpolated value
       - repeat first quote to beginning of series
    """
    year = datetime.datetime.now().year
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    date2 = (int(year), int(month), int(day))
    date2 = datetime.date.today() + datetime.timedelta(0)
    # get quotes for each symbol in list (adjusted close)
    quotelist = []
    for i in range(len(symbolList)):
        ticker = symbolList[i]
        #print "ticker = ", ticker,date2
        if ticker == 'CASH':
            quotelist.append(1.0)
        else:
            data = quotes_historical_yahoo(ticker, date2, date2, asobject ="None")
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
'''


def get_quote_google( symbol ):
    import urllib
    import re
    #print " get_quote_google.....  symbol = ", symbol
    base_url = 'http://finance.google.com/finance?q='
    content = urllib.urlopen(base_url + symbol).read()
    #ref_34649_l
    #m = re.search('id="ref_34649_l".*?>(.*?)<', content)
    #n = re.search('id="ref_694653_l".*?>(.*?)<', content)
    m = re.search('class="pr".*?>*(?s)(.*?)<.*?>.*?<', content).group(0).split(">")[-1].split("<")[0]
    #print "content = ", content
    #print "m : ",content
    #print "m = ", m
    #from time import sleep
    #sleep(20)
    if m :
        quote = m
        #print "quote for symbol ", symbol, " = ", m
    else:
        quote = 'no quote available for: ' + symbol
    return quote


def LastQuotesForSymbolList( symbolList ):
    """
    read in latest (15-minute delayed) quote for each symbol in list.
    Use google for each symbol's quote.
    """
    from time import sleep
    quotelist = []
    for itick, ticker in enumerate( symbolList ):
        if ticker == 'CASH':
            print "ticker, quote = CASH 1.0"
            quotelist.append(1.0)
        else:
            try:
                data = get_quote_google( ticker )
                print "ticker, quote = ", ticker, data
                quotelist.append( data )
            except:
                print "could not get quote for ", ticker, "         will try again and again."
                sleep(3)
                symbolList[itick+1:itick+1] = [ticker]
    return quotelist


def LastQuotesForList( symbols_list ):

    from time import sleep
    from functions.StockRetriever import *

    stocks = StockRetriever()

    print "inside LastQuotesForList... location  0"

    # remove 'CASH' from symbols_list, if present. Keep track of position in list to re-insert
    cash_index = None
    try:
        cash_index = symbols_list.index('CASH')
        if cash_index >= 0 and cash_index <= len(symbols_list)-1 :
            symbols_list.remove('CASH')
    except:
        pass

    print "inside LastQuotesForList... location  1"

    attempt = 1
    NeedQuotes = True
    while NeedQuotes:
        try:
            a=stocks.get_current_info( symbols_list )
            print "inside LastQuotesForList 1, symbols_list = ", symbols_list
            print "inside LastQuotesForList 1, attempt = ", attempt
            #print "inside LastQuotesForList 1, len(a) = ", len(a)
            # convert from strings to numbers and put in a list
            quotelist = []
            for i in range(len(a)):
                singlequote = np.float((a[i]['LastTradePriceOnly']).encode('ascii','ignore'))
                quotelist.append(singlequote)
            #quotelist= np.array(quotelist).tolist()
            print symbols_list, quotelist
            NeedQuotes = False
        except:
            attempt += 1
            sleep(attempt)

    print "inside LastQuotesForList... location  2"

    # re-insert CASH in original position and also add curent price of 1.0 to quotelist
    if cash_index != None:
        #print "inside LastQuotesForList... re-inserting...", cash_index, symbols_list, quotelist
        if cash_index < len(symbols_list):
            symbols_list[cash_index:cash_index] = 'CASH'
            quotelist[cash_index:cash_index] = 1.0
        else:
            symbols_list.append('CASH')
            quotelist.append(1.0)

    print "attempts, sysmbols_list,quotelist =", attempt, symbols_list, quotelist
    return quotelist
