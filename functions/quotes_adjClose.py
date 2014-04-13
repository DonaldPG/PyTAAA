'''
Created on May 12, 202

@author: donaldpg
'''

def downloadQuotes(tickers, date1=None, date2=None, adjust=True, Verbose=False):
    """
    Given a ticker sequence, return historical Yahoo! quotes as a pandas DataFrame.

    Parameters
    ----------
    tickers : sequence
        A sequence (such as a list) of string tickers. For example:
        ['aapl', 'msft']
    date1 : {datetime.date, tuple}, optional
        The first date to grab historical quotes on. For example:
        datetime.date(2010, 1, 1) or (2010, 1, 1). By default the first
        date is (1900, 1, 1).
    date2 : {datetime.date, tuple}, optional
        The last date to grab historical quotes on. For example:
        datetime.date(2010, 12, 31) or (2010, 12, 31). By default the last
        date is 10 days beyond today's date.
    adjust : bool, optional
        Adjust (default) the open, close, high, and low prices. The
        adjustment takes splits and dividends into account such that the
        corresponding returns are correct. Volume is already split adjusted
        by Yahoo so it is not changed by the value of `adjust`.
    Verbose : bool, optional
        Print the ticker currently being loaded. By default the tickers are
        not printed.

    Returns
    -------
    quotes_df : DataFrame
        A pandas dataframe is returned. In order, the  axes contain: dates,
        quotes (adjusted close). The elements along the item axis depend on the value
        of `adjust`. When `adjust` is False, the items are

        ['open', 'close', 'high', 'low', 'volume', 'adjclose']

        When adjust is true (default), the adjusted close ('adjclose') is
        not included. The dates are datetime.date objects.

    Examples
    --------
    items = ['Adj Close']
    date1 = '2012-01-01'
    date2 = datetime.date.today()
    ticker = 'GOOGL'
    data = get_data_yahoo(ticker, start = date1, end = date2)[items]
    dates = data.index
    data.columns = [ticker]

    ticker = 'AMZN'
    data2 = get_data_yahoo(ticker, start = date1, end = date2)[items]
    dates2 = data2.index
    data2.columns = [ticker]

    data = data.join(data2, how='outer')
    data.sort_index( axis=0, inplace=True )

    data.tail()

                 GOOGL    AMZN
    Date
    2014-04-07  540.63  317.76
    2014-04-08  557.51  327.07
    2014-04-09  567.04  331.81
    2014-04-10  546.69  317.11
    2014-04-11  537.76  311.73


    """

    from time import sleep
    from matplotlib.finance import *
    #from la.external.matplotlib import quotes_historical_yahoo
    import pandas as pd
    from pandas.io.data import DataReader
    from pandas.io.data import get_data_yahoo
    #import la

    if date1 is None:
        date1 = datetime.date(1900, 1, 1)
    if date2 is None:
        date2 = datetime.date.today() + datetime.timedelta(+10)
    #quotes_df = None
    #lar = None
    items = ['Adj Close']
    if Verbose:
        print "Load data"

    i=0
    for itick, ticker in enumerate(tickers):
        if Verbose:
            print "\t" + ticker + "  ",

        data = []
        dates = []

        number_tries = 0
        try:
            # read in dataframe containing adjusted close quotes for a ticker in the list
            data = get_data_yahoo(ticker, start = date1, end = date2)[items]
            dates = data.index
            dates = [d.to_datetime() for d in dates]
            data.columns = [ticker]
            if Verbose:
                print i," of ",len(tickers)," ticker ",ticker," has ",data.shape[0]," quotes"

            if itick == 0:
                quotes_df = data
            else:
                quotes_df = quotes_df.join( data, how='outer' )
            i += 1
        except:
            print "could not get quotes for ", ticker, "         will try again and again."
            sleep(3)
            number_tries += 1
            if number_tries < 11:
                tickers[itick+1:itick+1] = [ticker]

    print "number of tickers successfully processed = ", i
    if i > 0 :
        quotes_df.sort_index( axis=0, inplace=True )
        return quotes_df

    else :
        # return empty DataFrame
        quotes_df = pd.DataFrame( [0,0], ['Dates',date2])
        quotes_df.columns = ['None']

    return quotes_df


