'''
Created on May 12, 202

@author: donaldpg
'''

def downloadQuotes(tickers, date1=None, date2=None, adjust=True, Verbose=False):
    """
    Given a ticker sequence, return historical Yahoo! quotes as a 3d larry.

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
    lar : larry
        A 3d larry is returned. In order, the three axes contain: tickers,
        item, and dates. The elements along the item axis depend on the value
        of `adjust`. When `adjust` is False, the items are

        ['open', 'close', 'high', 'low', 'volume', 'adjclose']

        When adjust is true (default), the adjusted close ('adjclose') is
        not included. The dates are datetime.date objects.

    Examples
    --------
    >>> from la.data.yahoo import quotes
    >>> lar = quotes(['aapl', 'msft'], (2010,10,1), (2010,10,5))
    >>> lar
    label_0
        aapl
        msft
    label_1
        open
        close
        high
        low
        volume
    label_2
        2010-10-01
        2010-10-04
        2010-10-05
    x
    array([[[  2.86150000e+02,   2.81600000e+02,   2.82000000e+02],
            [  2.82520000e+02,   2.78640000e+02,   2.88940000e+02],
            [  2.86580000e+02,   2.82900000e+02,   2.89450000e+02],
            [  2.81350000e+02,   2.77770000e+02,   2.81820000e+02],
            [  1.60051000e+07,   1.55256000e+07,   1.78743000e+07]],
            .
           [[  2.47700000e+01,   2.39600000e+01,   2.40600000e+01],
            [  2.43800000e+01,   2.39100000e+01,   2.43500000e+01],
            [  2.48200000e+01,   2.39900000e+01,   2.44500000e+01],
            [  2.43000000e+01,   2.37800000e+01,   2.39100000e+01],
            [  6.26236000e+07,   9.80868000e+07,   7.80329000e+07]]])

    >>> close = lar.lix[:,['close']]
    >>> close
    label_0
        aapl
        msft
    label_1
        2010-10-01
        2010-10-04
        2010-10-05
    x
    array([[ 282.52,  278.64,  288.94],
           [  24.38,   23.91,   24.35]])

    Calculate the log return from the close prices:

    >>> ret = close / close.lag(1, axis=-1)
    >>> ret = ret.log()
    >>> ret
    label_0
        aapl
        msft
    label_1
        2010-10-04
        2010-10-05
    x
    array([[-0.01382872,  0.03629843],
           [-0.01946634,  0.01823507]])

    """

    from time import sleep
    from matplotlib.finance import *
    from la.external.matplotlib import quotes_historical_yahoo
    import la

    if date1 is None:
        date1 = datetime.date(1900, 1, 1)
    if date2 is None:
        date2 = datetime.date.today() + datetime.timedelta(+10)
    lar = None
    items = ['close', 'volume']
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
            data, dates = quotes_historical_yahoo(ticker, date1, date2)
            data = np.array(data).T
            if Verbose:
                print i," of ",len(tickers)," ticker ",ticker," has ",data.shape[1]," quotes"
            qlar = la.larry(data[4:,:], [items, dates])
            qlar = qlar.insertaxis(0, ticker)
            if lar is None:
                lar = qlar
            else:
                lar = lar.merge(qlar)
            i += 1
        except:
            print "could not get quotes for ", ticker, "         will try again and again."
            sleep(3)
            number_tries += 1
            if number_tries < 11:
                tickers[itick+1:itick+1] = [ticker]

    print "number of tickers successfully processed = ", i
    if i > 0 :
        lar = lar.sortaxis(-1)
        lar = lar[:,::-1]   # reverse order to return adjClose and volume
        return lar
    else :
        # return empty labeled array (larry)
        lar = la.larry([0, 0],[items,date2])
        return lar

    return lar


