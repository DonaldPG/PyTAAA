import os
import numpy as np
from numpy import isnan
import datetime
#from yahooFinance import getQuote
from functions.quotes_adjClose import get_pe
# from functions.readSymbols import readSymbolList
from functions.readSymbols import read_symbols_list_local
from functions.GetParams import get_webpage_store, get_performance_store

import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')


def strip_accents(text):

    import unicodedata
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)


def normcorrcoef(a,b):
    return np.correlate(a,b)/np.sqrt(np.correlate(a,a)*np.correlate(b,b))[0]

"""
def interpolate(self, method='linear'):
    '''
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
    '''

    inds = np.arange(len(self))
    values = np.array(self.copy())
    invalid_bool = np.isnan(values)
    valid = np.ones((len(self)),'int')
    valid[ invalid_bool==True ] = 0
    invalid = 1 - valid
    firstIndex = valid.argmax()
    lastIndex = valid.shape[0]-valid[::-1].argmax()

    #valid = valid[firstIndex:lastIndex]
    #invalid = invalid[firstIndex:lastIndex]
    valid = valid[valid >= firstIndex]
    valid = valid[valid <= lastIndex]
    invalid = invalid[invalid >= firstIndex]
    invalid = invalid[invalid <= lastIndex]

    #inds = inds[firstIndex:]
    result = values.copy()
    #result[firstIndex:lastIndex][invalid[firstIndex:lastIndex]==1] = np.interp(inds[firstIndex:lastIndex][invalid[firstIndex:lastIndex]==1], inds[firstIndex:lastIndex][valid[firstIndex:lastIndex]==1],values[firstIndex:lastIndex][valid[firstIndex:lastIndex]==1])
    if len(invalid[invalid==1]) > 0:
        result[invalid==1] = np.interp(inds[invalid==1], inds[valid==1],values[valid==1])

    return result

#----------------------------------------------
def cleantobeginning(self):
    '''
    Copy missing values (to all dates prior the first valid value)

    Usage: infill NaN values at beginning with copy of first valid value
    '''
    inds = np.arange(len(self))
    values = self.copy()
    invalid_bool = np.isnan(values)
    valid = np.ones((len(self)),'int')
    valid[ invalid_bool==True ] = 0
    invalid = 1 - valid
    firstIndex = valid.argmax()
    for i in range(firstIndex):
        values[i]=values[firstIndex]
    return values
"""

'''
def interpolate(self, method='linear'):
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

    #print " ... inside interpolate .... len(self) = ", len(self)

    inds = np.arange(len(self))
    values = np.array(self.copy())
    #print " values = ", values
    #print " values.dtype = ", values.dtype
    #print " type(values) = ", type(values)
    invalid = isnan(values)
    valid = -invalid
    firstIndex = valid.argmax()

    #print " ... inside interpolate .... firstIndex = ", firstIndex


    valid = valid[firstIndex:]
    invalid = invalid[firstIndex:]

    #print " ... inside interpolate .... len(valid) = ", len(valid)
    #print " ... inside interpolate .... len(invalid) = ", len(invalid)

    inds = inds[firstIndex:]
    result = values.copy()
    result[firstIndex:][invalid] = np.interp(inds[invalid], inds[valid],values[firstIndex:][valid])
    return result
'''


# def interpolate(self, method='linear'):
#     '''
#     Interpolate missing values (after the first valid value)
#     Parameters
#     ----------
#     method : {'linear'}
#     Interpolation method.
#     Time interpolation works on daily and higher resolution
#     data to interpolate given length of interval
#     Returns
#     -------
#     interpolated : Series
#     from-- https://github.com/wesm/pandas/blob/master/pandas/core/series.py
#     edited to keep only 'linear' method
#     Usage: infill NaN values with linear interpolated values
#     '''

#     inds = np.arange(len(self))
#     values = np.array(self.copy())
#     invalid_bool = np.isnan(values)
#     valid = np.ones((len(self)),'int')
#     valid[ invalid_bool==True ] = 0
#     invalid = 1 - valid
#     firstIndex = valid.argmax()
#     lastIndex = valid.shape[0]-valid[::-1].argmax()

#     #valid = valid[firstIndex:lastIndex]
#     #invalid = invalid[firstIndex:lastIndex]
#     valid = valid[valid >= firstIndex]
#     valid = valid[valid <= lastIndex]
#     invalid = invalid[invalid >= firstIndex]
#     invalid = invalid[invalid <= lastIndex]

#     #inds = inds[firstIndex:]
#     result = values.copy()
#     #result[firstIndex:lastIndex][invalid[firstIndex:lastIndex]==1] = np.interp(inds[firstIndex:lastIndex][invalid[firstIndex:lastIndex]==1], inds[firstIndex:lastIndex][valid[firstIndex:lastIndex]==1],values[firstIndex:lastIndex][valid[firstIndex:lastIndex]==1])
#     if len(invalid[invalid==1]) > 0:
#         result[invalid==1] = np.interp(inds[invalid==1], inds[valid==1],values[valid==1])

#     return result


#----------------------------------------------
def cleantobeginning(self):
    """
    Copy missing values (to all dates prior the first valid value)

    Usage: infill NaN values at beginning with copy of first valid value
    """
    values = self.copy()
    #print " type(values) = ", type(values)
    invalid = isnan(values)
    valid = -invalid
    firstIndex = valid.argmax()
    for i in range(firstIndex):
        values[i]=values[firstIndex]
    return values


def interpolate(self, method='linear', verbose=False):
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

    import sys

    if sys.version_info < (2,7,12):
        if verbose:
            print(" ... inside interpolate (old).... len(self) = ", len(self))


        inds = np.arange(len(self))
        values = np.array(self.copy())
        if verbose:
            print(" ... values = ", values)
            print(" ... values.dtype = ", values.dtype)
            print(" ... type(values) = ", type(values))
        invalid = np.isnan(values)
        valid = -1 * invalid
        firstIndex = valid.argmax()

        if verbose:
            print(" ... inside interpolate .... firstIndex = ", firstIndex)


        valid = valid[firstIndex:]
        invalid = invalid[firstIndex:]

        if verbose:
            print(" ... inside interpolate .... len(valid) = ", len(valid))
            print(" ... inside interpolate .... len(invalid) = ", len(invalid))

        inds = inds[firstIndex:]
        result = values.copy()
        result[firstIndex:][invalid] = np.interp(inds[invalid], inds[valid==0],values[firstIndex:][valid==0])

        '''
        inds = np.arange(len(self))
        values = np.array(self.copy())
        invalid_bool = np.isnan(values)
        valid = np.ones((len(self)),'int')
        valid[ invalid_bool==True ] = 0
        invalid = 1 - valid
        if verbose:
            print " ... values = ", values
            print " ... values.dtype = ", values.dtype
            print " ... type(values) = ", type(values)
        firstIndex = valid.argmax()
        lastIndex = valid.shape[0]-valid[::-1].argmax()

        #valid = valid[firstIndex:lastIndex]
        #invalid = invalid[firstIndex:lastIndex]
        valid = valid[valid >= firstIndex]
        valid = valid[valid <= lastIndex]
        invalid = invalid[invalid >= firstIndex]
        invalid = invalid[invalid <= lastIndex]
        if verbose:
            print " ... inside interpolate .... firstIndex = ", firstIndex

        #inds = inds[firstIndex:]
        result = values.copy()
        #result[firstIndex:lastIndex][invalid[firstIndex:lastIndex]==1] = np.interp(inds[firstIndex:lastIndex][invalid[firstIndex:lastIndex]==1], inds[firstIndex:lastIndex][valid[firstIndex:lastIndex]==1],values[firstIndex:lastIndex][valid[firstIndex:lastIndex]==1])
        if len( invalid[invalid==1] ) > 0:
            result[invalid==1] = np.interp(inds[invalid==1], inds[valid==1],values[valid==1])
        '''

        if verbose:
            print(" ... interpolate (old) finished")

    else:

        if verbose:
            print(" ... inside interpolate (new) .... len(self) = ", len(self))
        inds = np.arange(len(self))
        values = np.array(self.copy())
        if verbose:
            print(" ... values = ", values)
            print(" ... values.dtype = ", values.dtype)
            print(" ... type(values) = ", type(values))

        invalid_bool = np.isnan(values)
        valid = np.ones((len(self)),'int')
        valid[ invalid_bool==True ] = 0
        invalid = 1 - valid
        firstIndex = valid.argmax()
        lastIndex = valid.shape[0]-valid[::-1].argmax()

        if verbose:
            print(" ... inside interpolate .... len(valid) = ", len(valid))
            print(" ... inside interpolate .... len(invalid) = ", len(invalid))
            print(" ... inside interpolate .... firstIndex,lastIndex = ", firstIndex,lastIndex)

        #valid = valid[firstIndex:lastIndex]
        #invalid = invalid[firstIndex:lastIndex]
        valid = valid[valid >= firstIndex]
        valid = valid[valid <= lastIndex]
        invalid = invalid[invalid >= firstIndex]
        invalid = invalid[invalid <= lastIndex]

        #inds = inds[firstIndex:]
        result = values.copy()
        #result[firstIndex:lastIndex][invalid[firstIndex:lastIndex]==1] = np.interp(inds[firstIndex:lastIndex][invalid[firstIndex:lastIndex]==1], inds[firstIndex:lastIndex][valid[firstIndex:lastIndex]==1],values[firstIndex:lastIndex][valid[firstIndex:lastIndex]==1])
        if len(invalid[invalid==1]) > 0:
            result[invalid==1] = np.interp(inds[invalid==1], inds[valid==1],values[valid==1])

    return result

#----------------------------------------------
'''
def interpolate(self, method='linear'):
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

    verbose = False

    if verbose:
        print " ... inside interpolate (old).... len(self) = ", len(self)

    inds = np.arange(len(self))
    values = np.array(self.copy())

    if verbose:
        print " ... values = ", values
        print " ... values.dtype = ", values.dtype
        print " ... type(values) = ", type(values)

    invalid = np.isnan(values)
    valid = -1 * invalid
    firstIndex = valid.argmax()

    if verbose:
        print " ... inside interpolate .... firstIndex = ", firstIndex

    valid = valid[firstIndex:]
    invalid = invalid[firstIndex:]

    if verbose:
        print " ... inside interpolate .... len(valid) = ", len(valid)
        print " ... inside interpolate .... len(invalid) = ", len(invalid)

    inds = inds[firstIndex:]
    result = values.copy()
    result[firstIndex:][invalid] = np.interp(inds[invalid], inds[valid],values[firstIndex:][valid])

    return result
'''


#----------------------------------------------
def cleantobeginning(self):
    """
    Copy missing values (to all dates prior the first valid value)

    Usage: infill NaN values at beginning with copy of first valid value
    """

    import sys

    verbose = False

    if sys.version_info < (2,7,12):

        if verbose:
            print(" ... inside cleantobeginning (old) .... len(self) = ", len(self))

        inds = np.arange(len(self))
        values = self.copy()
        if verbose:
            print(" ... type(values) = ", type(values))
        invalid = np.isnan(values)
        valid = -1*invalid
        firstIndex = valid.argmax()
        if verbose:
            print(" ... inside cleantobeginning (old) .... firstIndex = ", firstIndex)

        for i in range(firstIndex):
            values[i]=values[firstIndex]

        if verbose:
            print(" ... cleantobeginning (old) finished\n")

    else:

        if verbose:
            print(" ... inside cleantobeginning (new) .... len(self) = ", len(self))

        inds = np.arange(len(self))
        values = self.copy()
        invalid_bool = np.isnan(values)
        valid = np.ones((len(self)),'int')
        valid[ invalid_bool==True ] = 0
        invalid = 1 - valid
        firstIndex = valid.argmax()
        if verbose:
            print(" ... inside cleantobeginning (old) .... firstIndex = ", firstIndex)
        for i in range(firstIndex):
            values[i]=values[firstIndex]

    return values



#----------------------------------------------

def cleantoend(self):
    """
    Copy missing values (to all dates after the last valid value)

    Usage: infill NaN values at end with copy of last valid value
    """
    # reverse input 1D array and use cleantobeginning
    reverse = self[::-1]
    reverse = cleantobeginning(reverse)
    return reverse[::-1]

#----------------------------------------------

def clean_signal(array1D,symbol_name):
    ### clean input signals (again)
    quotes_before_cleaning = array1D.copy()
    adjClose = interpolate( array1D )
    adjClose = cleantobeginning( adjClose )
    adjClose = cleantoend( adjClose )
    adjClose_changed = False in (adjClose==quotes_before_cleaning)
    print("   ... inside clean_signal ... symbol, did cleaning change adjClose? ", symbol_name, adjClose_changed)
    return adjClose

#----------------------------------------------

def cleanspikes(x,periods=20,stddevThreshold=5.0):
    # remove outliers from gradient of x (in 2 directions)
    x_clean = np.array(x).copy()
    test = np.zeros(x.shape[0],'float')
    #gainloss_f = np.ones((x.shape[0]),dtype=float)
    #gainloss_r = np.ones((x.shape[0]),dtype=float)
    #print gainloss_f[1:],x[1:].shape,x[:-1].shape
    #print " ...inside cleanspikes... ", x[1:].shape, x[:-1].shape
    #gainloss_f[1:] = x[1:] / x[:-1]
    #gainloss_r[:-1] = x[:-1] / x[1:]
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

    """
    for i in range( 1,x.shape[0]-2 ):
         minx = max(0,i-periods/2)
         maxx = min(x.shape[0],i+periods/2)
         #Stddev_f = np.std(gainloss_f[minx:maxx]) + 1.e-5
         #Stddev_r = np.std(gainloss_r[minx:maxx]) + 1.e-5
         if gainloss_f[i-1]/Stddev_f > stddevThreshold and gainloss_r[i]/Stddev_r > stddevThreshold:
            x_clean[i] = np.nan
    """
    return x_clean

#----------------------------------------------

def percentileChannel(x,minperiod,maxperiod,incperiod,lowPct,hiPct):
    periods = np.arange(minperiod,maxperiod,incperiod)
    minchannel = np.zeros(len(x),dtype=float)
    maxchannel = np.zeros(len(x),dtype=float)
    for i in range(len(x)):
        divisor = 0
        for j in range(len(periods)):
            minx = max(1,i-periods[j])
            if len(x[minx:i]) < 1:
                minchannel[i] = minchannel[i] + x[i]
                maxchannel[i] = maxchannel[i] + x[i]
                divisor += 1
            else:
                minchannel[i] = minchannel[i] + np.percentile(x[minx:i+1],lowPct)
                maxchannel[i] = maxchannel[i] + np.percentile(x[minx:i+1],hiPct)
                divisor += 1
        minchannel[i] /= divisor
        maxchannel[i] /= divisor
    return minchannel,maxchannel
#----------------------------------------------
def percentileChannel_2D_optimized(x, minperiod, maxperiod, incperiod, lowPct, hiPct, verbose=True):
    """
    Optimized version of percentileChannel_2D using vectorized operations.
    Should be 10-100x faster than the original implementation.
    """
    if verbose:
        print(" ... inside percentileChannel_2D_optimized ...  x min,mean,max = ", x.min(), x.mean(), x.max())
        print(f" ... Input parameters: minperiod={minperiod}, maxperiod={maxperiod}, incperiod={incperiod}")
        print(f" ... Array shape: {x.shape[0]} stocks, {x.shape[1]} time periods")
    
    periods = np.arange(minperiod, maxperiod, incperiod)
    if verbose:
        print(f" ... Computing for {len(periods)} period lengths: {periods}")
    
    minchannel = np.zeros((x.shape[0], x.shape[1]), dtype=float)
    maxchannel = np.zeros((x.shape[0], x.shape[1]), dtype=float)
    
    # Use scipy for faster rolling percentile if available
    try:
        from scipy.ndimage import uniform_filter1d
        use_scipy = True
    except ImportError:
        use_scipy = False
        if verbose:
            print(" ... scipy not available, using numpy (slower)")
    
    if verbose:
        print(f" ... Starting optimized computation...")
    
    # Pre-allocate arrays for all periods
    all_mins = np.zeros((len(periods), x.shape[0], x.shape[1]), dtype=float)
    all_maxs = np.zeros((len(periods), x.shape[0], x.shape[1]), dtype=float)
    
    # Compute percentiles for all periods in vectorized manner
    for j, period in enumerate(periods):
        if verbose and j % 2 == 0:  # Print every other period
            print(f" ... Computing period {period} ({j+1}/{len(periods)})")
        
        # For each time point, we need percentiles over the lookback window
        for i in range(x.shape[1]):
            minx = max(1, i - period)
            if i < minx + 1:  # Not enough data
                all_mins[j, :, i] = x[:, i]
                all_maxs[j, :, i] = x[:, i]
            else:
                # Vectorized percentile computation across all stocks at once
                window_data = x[:, minx:i+1]  # Shape: (stocks, window_size)
                all_mins[j, :, i] = np.percentile(window_data, lowPct, axis=1)
                all_maxs[j, :, i] = np.percentile(window_data, hiPct, axis=1)
    
    # Average across all periods (vectorized)
    minchannel = np.mean(all_mins, axis=0)
    maxchannel = np.mean(all_maxs, axis=0)
    
    if verbose:
        print(" ... percentileChannel_2D_optimized computation complete!")
        print(" minperiod,maxperiod,incperiod = ", minperiod, maxperiod, incperiod)
        print(" lowPct,hiPct = ", lowPct, hiPct)
        print(" x min,mean,max = ", x.min(), x.mean(), x.max())
    
    return minchannel, maxchannel


def percentileChannel_2D_ultra_fast(x, minperiod, maxperiod, incperiod, lowPct, hiPct, verbose=True):
    """
    Ultra-fast version using pandas rolling windows for maximum performance.
    This should be the fastest possible implementation.
    """
    if verbose:
        print(" ... inside percentileChannel_2D_ultra_fast ...  x min,mean,max = ", x.min(), x.mean(), x.max())
        print(f" ... Input parameters: minperiod={minperiod}, maxperiod={maxperiod}, incperiod={incperiod}")
        print(f" ... Array shape: {x.shape[0]} stocks, {x.shape[1]} time periods")
    
    import pandas as pd
    
    periods = np.arange(minperiod, maxperiod, incperiod)
    if verbose:
        print(f" ... Computing for {len(periods)} period lengths: {periods}")
    
    # Convert to pandas DataFrame for ultra-fast rolling operations
    df = pd.DataFrame(x.T)  # Transpose so time is rows, stocks are columns
    
    minchannel_sum = np.zeros_like(x, dtype=float)
    maxchannel_sum = np.zeros_like(x, dtype=float)
    
    if verbose:
        print(f" ... Starting ultra-fast computation using pandas rolling...")
    
    # Use pandas rolling for each period (much faster)
    for j, period in enumerate(periods):
        if verbose and (j % 2 == 0 or j == len(periods) - 1):
            print(f" ... Computing period {period} ({j+1}/{len(periods)}) - {100*(j+1)/len(periods):.1f}% complete")
        
        # Compute rolling percentiles for all stocks at once
        rolling_min = df.rolling(window=int(period), min_periods=1).quantile(lowPct/100.0)
        rolling_max = df.rolling(window=int(period), min_periods=1).quantile(hiPct/100.0)
        
        # Add to sum (transpose back to original orientation)
        minchannel_sum += rolling_min.values.T
        maxchannel_sum += rolling_max.values.T
    
    # Average across periods
    divisor = len(periods)
    minchannel = minchannel_sum / divisor  
    maxchannel = maxchannel_sum / divisor
    
    if verbose:
        print(" ... percentileChannel_2D_ultra_fast computation complete!")
        print(" minperiod,maxperiod,incperiod = ", minperiod, maxperiod, incperiod)
        print(" lowPct,hiPct = ", lowPct, hiPct)
        print(" divisor = ", divisor)
    
    return minchannel, maxchannel


#----------------------------------------------
def percentileChannel_2D(x,minperiod,maxperiod,incperiod,lowPct,hiPct,verbose=True):
    """
    Original slow implementation - kept for backward compatibility.
    Use percentileChannel_2D_ultra_fast() for much better performance.
    """
    # Try the ultra-fast version first
    try:
        return percentileChannel_2D_ultra_fast(x, minperiod, maxperiod, incperiod, lowPct, hiPct, verbose)
    except ImportError:
        if verbose:
            print(" ... pandas not available, using optimized numpy version")
        return percentileChannel_2D_optimized(x, minperiod, maxperiod, incperiod, lowPct, hiPct, verbose)
    except Exception as e:
        if verbose:
            print(f" ... Error in fast version ({e}), falling back to optimized version")
        return percentileChannel_2D_optimized(x, minperiod, maxperiod, incperiod, lowPct, hiPct, verbose)


#----------------------------------------------
def dpgchannel(x,minperiod,maxperiod,incperiod):
    periods = np.arange(minperiod,maxperiod,incperiod)
    minchannel = np.zeros(len(x),dtype=float)
    maxchannel = np.zeros(len(x),dtype=float)
    for i in range(len(x)):
        divisor = 0
        for j in range(len(periods)):
            minx = max(1,i-periods[j])
            if len(x[minx:i]) < 1:
                minchannel[i] = minchannel[i] + x[i]
                maxchannel[i] = maxchannel[i] + x[i]
                divisor += 1
            else:
                minchannel[i] = minchannel[i] + min(x[minx:i+1])
                maxchannel[i] = maxchannel[i] + max(x[minx:i+1])
                divisor += 1
        minchannel[i] /= divisor
        maxchannel[i] /= divisor
    return minchannel,maxchannel
#----------------------------------------------
def dpgchannel_2D(x,minperiod,maxperiod,incperiod):
    periods = np.arange(minperiod,maxperiod,incperiod)
    minchannel = np.zeros( (x.shape[0],x.shape[1]), dtype=float)
    maxchannel = np.zeros( (x.shape[0],x.shape[1]), dtype=float)
    for i in range( x.shape[1] ):
        divisor = 0
        for j in range(len(periods)):
            minx = max(1,i-periods[j])
            if len(x[0,minx:i]) < 1:
                minchannel[:,i] = minchannel[:,i] + x[:,i]
                maxchannel[:,i] = maxchannel[:,i] + x[:,i]
                divisor += 1
            else:
                minchannel[:,i] = minchannel[:,i] + np.min(x[:,minx:i+1],axis=-1)
                maxchannel[:,i] = maxchannel[:,i] + np.max(x[:,minx:i+1],axis=-1)
                divisor += 1
        minchannel[:,i] /= divisor
        maxchannel[:,i] /= divisor
    return minchannel,maxchannel
#----------------------------------------------
def selfsimilarity(hi,lo):

    from scipy.stats import percentileofscore
    HminusL = hi-lo

    periods = 10
    SMS = np.zeros( (hi.shape[0]), dtype=float)
    for i in range( hi.shape[0] ):
        minx = max(0,i-periods)
        SMS[i] = np.sum(HminusL[minx:i+1],axis=-1)

    # find the 10-day range (incl highest high and lowest low)
    range10day = MoveMax(hi,10) - MoveMin(lo,10)

    # normalize
    SMS /= range10day

    # compute quarterly (60-day) SMA
    SMS = SMA(SMS,60)

    # find percentile rank
    movepctrank = np.zeros( (hi.shape[0]), dtype=float)
    for i in range( hi.shape[0] ):
        minx = max(0,i-periods)
        movepctrank[i] = percentileofscore(SMS[minx:i+1],SMS[i])

    return movepctrank

#----------------------------------------------
def jumpTheChannelTest(x,minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28, offset=3):
    ###
    ### compute linear trend in upper and lower channels and compare
    ### actual stock price to forecast range
    ### return pctChannel for each stock
    ### calling function will use pctChannel as signal.
    ### - e.g. negative pctChannel is signal that down-trend begins
    ### - e.g. more than 100% pctChanel is sgnal of new up-trend beginning

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    # import warnings
    # warnings.simplefilter('ignore', np.RankWarning)

    pctChannel = np.zeros( (x.shape[0]), 'float' )
    # calculate linear trend over 'numdaysinfit' with 'offset'
    minchannel,maxchannel = dpgchannel(x,minperiod,maxperiod,incperiod)
    minchannel_trenddata = minchannel[-(numdaysinfit+offset):-offset]
    regression = np.polyfit(list(range(-(numdaysinfit+offset),-offset)), minchannel_trenddata, 1)
    minchannel_trend = regression[-1]
    maxchannel_trenddata = maxchannel[-(numdaysinfit+offset):-offset]
    regression = np.polyfit(list(range(-(numdaysinfit+offset),-offset)), maxchannel_trenddata, 1)
    maxchannel_trend = regression[-1]
    pctChannel = (x[-1]-minchannel_trend) / (maxchannel_trend-minchannel_trend)

    # calculate the stdev over the period
    gainloss_period = x[-(numdaysinfit+offset)+1:-offset+1] / x[-(numdaysinfit+offset):-offset]
    gainloss_period[np.isnan(gainloss_period)] = 1.
    gainloss_cumu = np.cumprod( gainloss_period )[-1] -1.
    gainloss_std = np.std( gainloss_period )

    # calculate the current quote as number of stdevs above or below trend
    currentMidChannel = (maxchannel_trenddata+minchannel_trend)/2.
    numStdDevs = (x[-1]/currentMidChannel[-1]-1.) / gainloss_std

    '''
    print "pctChannel = ", pctChannel
    print "gainloss_period = ", gainloss_period
    print "gainloss_cumu = ", gainloss_cumu
    print "gainloss_std = ", gainloss_std
    print "currentMidChannel = ", currentMidChannel[-1]
    print "numStdDevs = ", numStdDevs
    '''

    return pctChannel, gainloss_cumu, gainloss_std, numStdDevs

#----------------------------------------------
def recentChannelFit(x,minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28, offset=3):
    ###
    ### compute cumulative gain over fitting period and number of
    ### ratio of current quote to fitted trend. Rescale based on std dev
    ### of residuals during fitting period.
    ### - e.g. negative pctChannel is signal that down-trend begins
    ### - e.g. more than 100% pctChanel is signal of new up-trend beginning

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    # import warnings
    # warnings.simplefilter('ignore', np.RankWarning)

    ##pctChannel = np.zeros( (x.shape[0]), 'float' )
    # calculate linear trend over 'numdaysinfit' with 'offset'
    minchannel,maxchannel = dpgchannel(x,minperiod,maxperiod,incperiod)
    if offset == 0:
        minchannel_trenddata = minchannel[-(numdaysinfit+offset):]
        '''
        print "numdaysinfit = ", numdaysinfit
        print "offset = ", offset
        print "len(x) = ", len(x)
        print "len(minchannel) = ", len(minchannel)
        print "most recent quote = ", x[-1]
        print "quote[-offset:] = ", x[-offset:]
        print "quote[-(numdaysinfit+offset)+1:-offset+1] = ", x[-(numdaysinfit+offset):]
        print "len(quote[-(numdaysinfit+offset)+1:-offset+1]) = ", len(x[-(numdaysinfit+offset):])
        print "length of days = ", len(range(-(numdaysinfit+offset)+1,-offset+1))
        print "relative days = ", range(-(numdaysinfit+offset)+1,-offset+1)
        print "length of quotes = ", len(minchannel_trenddata)
        print "quotes = ", minchannel_trenddata
        '''
        regression1 = np.poly1d(np.polyfit(list(range(-(numdaysinfit+offset)+1,-offset+1)), minchannel_trenddata, 1))
        minchannel_trend = regression1[-1]
        maxchannel_trenddata = maxchannel[-(numdaysinfit+offset):]
        regression2 = np.poly1d(np.polyfit(list(range(-(numdaysinfit+offset)+1,-offset+1)), maxchannel_trenddata, 1))
    else:
        minchannel_trenddata = minchannel[-(numdaysinfit+offset)+1:-offset+1]
        '''
        print "numdaysinfit = ", numdaysinfit
        print "offset = ", offset
        print "len(x) = ", len(x)
        print "len(minchannel) = ", len(minchannel)
        print "most recent quote = ", x[-1]
        print "quote[-offset:] = ", x[-offset:]
        print "quote[-(numdaysinfit+offset)+1:-offset+1] = ", x[-(numdaysinfit+offset)+1:-offset+1]
        print "len(quote[-(numdaysinfit+offset)+1:-offset+1]) = ", len(x[-(numdaysinfit+offset)+1:-offset+1])
        print "length of days = ", len(range(-(numdaysinfit+offset)+1,-offset+1))
        print "relative days = ", range(-(numdaysinfit+offset)+1,-offset+1)
        print "length of quotes = ", len(minchannel_trenddata)
        print "quotes = ", minchannel_trenddata
        '''
        regression1 = np.poly1d(np.polyfit(list(range(-(numdaysinfit+offset)+1,-offset+1)), minchannel_trenddata, 1))
        minchannel_trend = regression1[-1]
        maxchannel_trenddata = maxchannel[-(numdaysinfit+offset)+1:-offset+1]
        regression2 = np.poly1d(np.polyfit(list(range(-(numdaysinfit+offset)+1,-offset+1)), maxchannel_trenddata, 1))
    ##maxchannel_trend = regression2[-1]
    ##pctChannel = (x[-1]-minchannel_trend) / (maxchannel_trend-minchannel_trend)

    return regression1, regression2

#----------------------------------------------
def recentTrendAndStdDevs(x,datearray,minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28, offset=3):

    ###
    ### compute linear trend in upper and lower channels and compare
    ### actual stock price to forecast range
    ### return pctChannel for each stock
    ### calling function will use pctChannel as signal.
    ### - e.g. numStdDevs < -1. is signal that down-trend begins
    ### - e.g. whereas  > 1.0 is signal of new up-trend beginning

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    # fit short-term recent trend channel for plotting
    lowerFit, upperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit,
                                           offset=offset)
    recentFitDates = datearray[-numdaysinfit-offset:-offset+1]
    relativedates = list(range(-numdaysinfit-offset,-offset+1))
    p = np.poly1d(upperFit)
    upperTrend = p(relativedates)
    currentUpper = p(0) * 1.
    p = np.poly1d(lowerFit)
    lowerTrend = p(relativedates)
    currentLower = p(0) * 1.
    midTrend = (upperTrend+lowerTrend)/2.
    #residuals = x[-numdaysinfit-offset:-offset+1] - midTrend
    #fitStdDev = np.std(residuals)
    fitStdDev = np.mean( upperTrend - lowerTrend )/2.
    #print ".....lowerFit, upperFit = ", lowerFit, upperFit
    #print ".....fitStdDev,currentUpper,currentLower,x[-1] = ", fitStdDev, currentUpper,currentLower,x[-1]
    currentResidual = x[-1] - (currentUpper + currentLower)/2.
    if fitStdDev != 0.:
        numStdDevs = currentResidual / fitStdDev
    else:
        numStdDevs = 0.

    # calculate gain or loss over the period
    gainloss_period = x[-(numdaysinfit+offset)+1:-offset+1] / x[-(numdaysinfit+offset):-offset]
    gainloss_period[np.isnan(gainloss_period)] = 1.
    gainloss_cumu = np.cumprod( gainloss_period )[-1] -1.

    # different method for gainloss over period using slope
    gainloss_cumu = midTrend[-1] / midTrend[0] -1.

    if currentUpper != currentLower:
        pctChannel = (x[-1]-currentUpper) / (currentUpper-currentLower)
    else:
        pctChannel = 0.

    return gainloss_cumu, numStdDevs, pctChannel

#----------------------------------------------
'''
def recentSharpeWithAndWithoutGap(x,numdaysinfit=28,numdaysinfit2=20, offset=3):

    from math import sqrt
    from scipy.stats import gmean

    ###
    ### - Cmpute sharpe ratio for recent prices with gap of 'offset' recent days
    ### - Compute 2nd sharpe ratio for recent prices recent days

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate sharpe with a gap
    # - 'numdaysinfit2' describes number of days over which to calculate sharpe without a gap
    # - 'offset'  describes number recent days to skip (e.g. the gap)

    # calculate gain or loss over the gapped period
    gainloss_period = x[-(numdaysinfit+offset)+1:-offset+1] / x[-(numdaysinfit+offset):-offset]
    gainloss_period[np.isnan(gainloss_period)] = 1.

    # sharpe ratio in period with a gap
    sharpe_withGap = ( gmean(gainloss_period)**252 -1. ) / ( np.std(gainloss_period)*sqrt(252) )

    # calculate gain or loss over the period without a gap
    gainloss_period = x[-numdaysinfit2+1:] / x[-numdaysinfit2:-1]
    gainloss_period[np.isnan(gainloss_period)] = 1.

    # sharpe ratio in period wihtout a gap
    sharpe_withoutGap = ( gmean(gainloss_period)**252 -1. ) / ( np.std(gainloss_period)*sqrt(252) )

    # combine sharpe ratios compouted over 2 different periods
    # - use an angle of 33 degrees instead of 45 to give slightly more weight the the "no gap" sharpe
    crossplot_rotationAngle = 33. * np.pi/180.
    sharpe2periods = sharpe_withGap*np.sin(crossplot_rotationAngle) + sharpe_withoutGap*np.cos(crossplot_rotationAngle)

    print("sharpe with, without gap, combined = ", sharpe_withGap, sharpe_withoutGap, sharpe2periods)
    return sharpe2periods
'''
#----------------------------------------------

def recentSharpeWithAndWithoutGap(x,numdaysinfit=504,offset_factor=.4):

    from math import sqrt
    from scipy.stats import gmean

    ###
    ### - Cmpute sharpe ratio for recent prices with gap of 'offset' recent days
    ### - Compute 2nd sharpe ratio for recent prices recent days

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate sharpe with a gap
    # - 'numdaysinfit2' describes number of days over which to calculate sharpe without a gap
    # - 'offset'  describes number recent days to skip (e.g. the gap)

    # calculate number of loops
    sharpeList = []
    for i in range(1,25):
        if i == 1:
            numdaysStart = numdaysinfit
            numdaysEnd = int(numdaysStart * offset_factor + .5)
        else:
            numdaysStart /= 2
            if numdaysStart/2 > 20:
                numdaysEnd = int(numdaysStart * offset_factor + .5)
            else:
                numdaysEnd = 0

        # calculate gain or loss over the gapped period
        numdaysStart = int(numdaysStart)
        numdaysEnd = int(numdaysEnd)
        numdays = numdaysStart - numdaysEnd
        offset = numdaysEnd
        if offset > 0:
            print("i,start,end = ", i, -(numdays+offset)+1, -offset+1)
            gainloss_period = x[-(numdays+offset)+1:-offset+1] / x[-(numdays+offset):-offset]
            gainloss_period[np.isnan(gainloss_period)] = 1.

            # sharpe ratio in period with a gap
            sharpe = ( gmean(gainloss_period)**252 -1. ) / ( np.std(gainloss_period)*sqrt(252) )
        else:
            print("i,start,end = ", i, -numdays+1, 0)
            # calculate gain or loss over the period without a gap
            gainloss_period = x[-numdays+1:] / x[-numdays:-1]
            gainloss_period[np.isnan(gainloss_period)] = 1.

            # sharpe ratio in period wihtout a gap
            sharpe = ( gmean(gainloss_period)**252 -1. ) / ( np.std(gainloss_period)*sqrt(252) )
        sharpeList.append(sharpe)
        if numdaysStart/2 < 20:
            break

    print("sharpeList = ", sharpeList)
    sharpeList = np.array(sharpeList)
    for i,isharpe in enumerate(sharpeList):
        if i == len(sharpeList)-1:
            if np.isnan(isharpe):
                sharpeList[i] = -999.
        else:
            if isharpe==np.nan:
                sharpeList[i] = 0.
    print("sharpeList = ", sharpeList)

    crossplot_rotationAngle = 33. * np.pi/180.
    for i,isharpe in enumerate(sharpeList):
        # combine sharpe ratios compouted over 2 different periods
        # - use an angle of 33 degrees instead of 45 to give slightly more weight the the "no gap" sharpe
        if i==0:
            continue
        elif i==1:
            sharpe_pair = [sharpeList[i-1],sharpeList[i]]
        else:
            sharpe_pair = [sharpe2periods,sharpeList[i]]
        sharpe2periods = sharpe_pair[0]*np.sin(crossplot_rotationAngle) + sharpe_pair[1]*np.cos(crossplot_rotationAngle)
        print("i, sharpe_pair, combined = " + str((i, sharpe_pair, sharpe2periods)))

    return sharpe2periods

#----------------------------------------------

def recentTrendAndMidTrendChannelFitWithAndWithoutGap(x,minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28,numdaysinfit2=20, offset=3):
    ###
    ### - Cmpute linear trend in upper and lower channels and compare
    ###   actual stock price to forecast range
    ### - Compute 2nd linear trend in upper and lower channels only for
    ###   small number of recent prices without gap
    ### - return pctChannel for each stock
    ### - calling function will use pctChannel as signal.
    ###   * e.g. numStdDevs < -1. is signal that down-trend begins
    ###   * e.g. whereas  > 1.0 is signal of new up-trend beginning

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    # fit short-term recent trend channel with offset from current date for plotting
    gappedLowerFit, gappedUpperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit,
                                           offset=offset)
    #recentFitDates = datearray[-numdaysinfit-offset:-offset+1]
    relativedates = list(range(-numdaysinfit-offset,-offset+1))
    p = np.poly1d(gappedUpperFit)
    upperTrend = p(relativedates)
    currentUpper = p(0) * 1.
    p = np.poly1d(gappedLowerFit)
    lowerTrend = p(relativedates)
    currentLower = p(0) * 1.
    midTrend = (upperTrend+lowerTrend)/2.
    #residuals = x[-numdaysinfit-offset:-offset+1] - midTrend
    #fitStdDev = np.std(residuals)
    fitStdDev = np.mean( upperTrend - lowerTrend )/2.
    #print ".....gappedLowerFit, gappedUpperFit = ", gappedLowerFit, gappedUpperFit
    #print ".....fitStdDev,currentUpper,currentLower,x[-1] = ", fitStdDev, currentUpper,currentLower,x[-1]
    currentResidual = x[-1] - (currentUpper + currentLower)/2.
    if fitStdDev != 0.:
        numStdDevs = currentResidual / fitStdDev
    else:
        numStdDevs = 0.

    # calculate gain or loss over the period (with offset)
    gainloss_period = x[-(numdaysinfit+offset)+1:-offset+1] / x[-(numdaysinfit+offset):-offset]
    gainloss_period[np.isnan(gainloss_period)] = 1.
    gainloss_cumu = np.cumprod( gainloss_period )[-1] -1.

    if currentUpper!=currentLower:
        pctChannel = (x[-1]-currentUpper) / (currentUpper-currentLower)
    else:
        pctChannel = 0.

    # fit shorter trend without offset
    NoGapLowerFit, NoGapUpperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit2,
                                           offset=0)
    #recentFitDates = datearray[-numdaysinfit2:]
    relativedates = list(range(-numdaysinfit2+1,1))
    p = np.poly1d(NoGapUpperFit)
    NoGapUpperTrend = p(relativedates)
    NoGapCurrentUpper = p(0) * 1.
    p = np.poly1d(NoGapLowerFit)
    NoGapLowerTrend = p(relativedates)
    NoGapCurrentLower = p(0) * 1.
    NoGapMidTrend = (NoGapUpperTrend+NoGapLowerTrend)/2.

    return lowerTrend, upperTrend, NoGapLowerTrend, NoGapUpperTrend

#----------------------------------------------

def recentTrendAndMidTrendWithGap(x,datearray,minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28,numdaysinfit2=20, offset=3):
    ###
    ### - Cmpute linear trend in upper and lower channels and compare
    ###   actual stock price to forecast range
    ### - Compute 2nd linear trend in upper and lower channels only for
    ###   small number of recent prices without gap
    ### - return pctChannel for each stock
    ### - calling function will use pctChannel as signal.
    ###   * e.g. numStdDevs < -1. is signal that down-trend begins
    ###   * e.g. whereas  > 1.0 is signal of new up-trend beginning

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    # fit short-term recent trend channel with offset from current date for plotting
    gappedLowerFit, gappedUpperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit,
                                           offset=offset)
    recentFitDates = datearray[-numdaysinfit-offset:-offset+1]
    relativedates = list(range(-numdaysinfit-offset,-offset+1))
    p = np.poly1d(gappedUpperFit)
    upperTrend = p(relativedates)
    currentUpper = p(0) * 1.
    p = np.poly1d(gappedLowerFit)
    lowerTrend = p(relativedates)
    currentLower = p(0) * 1.
    midTrend = (upperTrend+lowerTrend)/2.
    #residuals = x[-numdaysinfit-offset:-offset+1] - midTrend
    #fitStdDev = np.std(residuals)
    fitStdDev = np.mean( upperTrend - lowerTrend )/2.
    #print ".....gappedLowerFit, gappedUpperFit = ", gappedLowerFit, gappedUpperFit
    #print ".....fitStdDev,currentUpper,currentLower,x[-1] = ", fitStdDev, currentUpper,currentLower,x[-1]
    currentResidual = x[-1] - (currentUpper + currentLower)/2.
    numStdDevs = currentResidual / fitStdDev

    # calculate gain or loss over the period (with offset)
    gainloss_period = x[-(numdaysinfit+offset)+1:-offset+1] / x[-(numdaysinfit+offset):-offset]
    gainloss_period[np.isnan(gainloss_period)] = 1.
    gainloss_cumu = np.cumprod( gainloss_period )[-1] -1.

    pctChannel = (x[-1]-currentUpper) / (currentUpper-currentLower)


    # fit shorter trend without offset
    NoGapLowerFit, NoGapUpperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit2,
                                           offset=0)
    recentFitDates = datearray[-numdaysinfit2:]
    relativedates = list(range(-numdaysinfit2,1))
    p = np.poly1d(NoGapUpperFit)
    NoGapUpperTrend = p(relativedates)
    NoGapCurrentUpper = p(0) * 1.
    p = np.poly1d(NoGapLowerFit)
    NoGapLowerTrend = p(relativedates)
    NoGapCurrentLower = p(0) * 1.
    NoGapMidTrend = (NoGapUpperTrend+NoGapLowerTrend)/2.

    # calculate relative gain or loss over entire period
    gainloss_cumu2 = NoGapMidTrend[-1]/midTrend[0] -1.
    relative_GainLossRatio = (NoGapCurrentUpper + NoGapCurrentLower)/(currentUpper + currentLower)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pylab as plt
    plt.figure(1)
    plt.clf()
    plt.grid(True)
    plt.plot(datearray[-(numdaysinfit+offset+20):],x[-(numdaysinfit+offset+20):],'k-')
    relativedates = list(range(-numdaysinfit-offset,-offset+1))
    plt.plot(datearray[np.array(relativedates)],upperTrend,'y-')
    plt.plot(datearray[np.array(relativedates)],lowerTrend,'y-')
    plt.plot([datearray[-1]],[(upperTrend[-1]+lowerTrend[-1])/2.],'y.',ms=30)
    relativedates = list(range(-numdaysinfit2,0))
    plt.plot(datearray[np.array(relativedates)],NoGapUpperTrend,'c-')
    plt.plot(datearray[np.array(relativedates)],NoGapLowerTrend,'c-')
    plt.plot([datearray[-1]],[(NoGapUpperTrend[-1]+NoGapLowerTrend[-1])/2.],'c.',ms=30)
    plt.show()

    return gainloss_cumu, gainloss_cumu2, numStdDevs, relative_GainLossRatio

#----------------------------------------------

def recentTrendComboGain(x,
                         datearray,
                         minperiod=4,
                         maxperiod=12,
                         incperiod=3,
                         numdaysinfit=28,
                         numdaysinfit2=20,
                         offset=3):
    ###
    ### - Cmpute linear trend in upper and lower channels and compare
    ###   actual stock price to forecast range
    ### - Compute 2nd linear trend in upper and lower channels only for
    ###   small number of recent prices without gap
    ### - return pctChannel for each stock
    ### - calling function will use pctChannel as signal.
    ###   * e.g. numStdDevs < -1. is signal that down-trend begins
    ###   * e.g. whereas  > 1.0 is signal of new up-trend beginning

    from scipy.stats import gmean

    # calculate dpgchannel for all stocks in x
    # - x[stock_number,date]
    # - 'numdaysinfit' describes number of days over which to calculate a linear trend
    # - 'offset'  describes number days to forecast channel trends forward

    # fit short-term recent trend channel with offset from current date for plotting
    gappedLowerFit, gappedUpperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit,
                                           offset=offset)
    recentFitDates = datearray[-numdaysinfit-offset:-offset+1]
    relativedates = list(range(-numdaysinfit-offset,-offset+1))
    p = np.poly1d(gappedUpperFit)
    upperTrend = p(relativedates)
    p = np.poly1d(gappedLowerFit)
    lowerTrend = p(relativedates)
    midTrend = (upperTrend+lowerTrend)/2.

    # calculate gain or loss over the period (no offset)
    gainloss_period = midTrend[1:] / midTrend[:-1]
    gainloss_period[np.isnan(gainloss_period)] = 1.
    gainloss_cumu = gmean( gainloss_period )**252 -1.

    # fit shorter trend without offset
    NoGapLowerFit, NoGapUpperFit = recentChannelFit( x,
                                           minperiod=minperiod,
                                           maxperiod=maxperiod,
                                           incperiod=incperiod,
                                           numdaysinfit=numdaysinfit2,
                                           offset=0)
    recentFitDates = datearray[-numdaysinfit2:]
    relativedates = list(range(-numdaysinfit2,1))
    p = np.poly1d(NoGapUpperFit)
    NoGapUpperTrend = p(relativedates)
    p = np.poly1d(NoGapLowerFit)
    NoGapLowerTrend = p(relativedates)
    NoGapMidTrend = (NoGapUpperTrend+NoGapLowerTrend)/2.

    # calculate gain or loss over the period (no offset)
    gainloss_period_nogap = NoGapMidTrend[1:] / NoGapMidTrend[:-1]
    gainloss_period_nogap[np.isnan(gainloss_period_nogap)] = 1.
    gainloss_cumu_nogap = gmean( gainloss_period_nogap )**252 -1.

    # calculate "combo gain" (defined as sum of gains rewarded for improvement, penalized for decline
    comboGain = (gainloss_cumu + gainloss_cumu_nogap)/2.
    comboGain *= (gainloss_cumu_nogap+1) / (gainloss_cumu+1)

    return comboGain

#----------------------------------------------

def textmessageOutsideTrendChannel(symbols, adjClose, json_fn):

    # temporarily skip this!!!!!!
    #return

    import datetime
    from functions.GetParams import get_json_params, get_holdings, GetEdition
    from functions.CheckMarketOpen import get_MarketOpenOrClosed
    #from functions.SendEmail import SendTextMessage
    from functions.SendEmail import SendEmail

    # send text message for held stocks if the lastest quote is outside
    # (to downside) the established channel

    # Get Credentials for sending email
    params = get_json_params(json_fn)
    print("")

    #print "params = ", params
    print("")
    username = str(params['fromaddr']).split("@")[0]
    emailpassword = str(params['PW'])

    subjecttext = "PyTAAA update - Pct Trend Channel"
    boldtext = "time is "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
    headlinetext = "market status: " + get_MarketOpenOrClosed()

    # Get Holdings from file
    # holdings = GetHoldings()
    holdings = get_holdings(json_fn)
    holdings_symbols = holdings['stocks']
    edition = GetEdition()

    # process symbols in current holdings
    downtrendSymbols = []
    channelPercent = []
    channelGainsLossesHoldings = []
    channelStdsHoldings = []
    channelGainsLosses = []
    channelStds = []
    currentNumStdDevs = []
    for i, symbol in enumerate(symbols):
        pctChannel,channelGainLoss,channelStd,numStdDevs = jumpTheChannelTest(adjClose[i,:],\
                                                                              #minperiod=4,\
                                                                              #maxperiod=12,\
                                                                              #incperiod=3,\
                                                                              #numdaysinfit=28,\
                                                                              #offset=3)
                                                   minperiod=params['minperiod'],
                                                   maxperiod=params['maxperiod'],
                                                   incperiod=params['incperiod'],
                                                   numdaysinfit=params['numdaysinfit'],
                                                   offset=params['offset'])
        channelGainsLosses.append(channelGainLoss)
        channelStds.append(channelStd)
        if symbol in holdings_symbols:
            #pctChannel = jumpTheChannelTest(adjClose[i,:],minperiod=4,maxperiod=12,incperiod=3,numdaysinfit=28, offset=3)
            print(" ... performing PctChannelTest: symbol = ",format(symbol,'5s'), "  pctChannel = ", format(pctChannel-1.,'6.1%'))
            '''
            if pctChannel < 1.:
                # send textmessage alert of possible new down-trend
                downtrendSymbols.append(symbol)
                channelPercent.append(format(pctChannel-1.,'6.1%'))
            '''
            # send textmessage alert of current trend
            downtrendSymbols.append(symbol)
            channelPercent.append(format(pctChannel-1.,'6.1%'))
            channelGainsLossesHoldings.append(format(channelGainLoss,'6.1%'))
            channelStdsHoldings.append(format(channelStd,'6.1%'))
            currentNumStdDevs.append(format(numStdDevs,'6.1f'))

    print("\n ... downtrending symbols are ", downtrendSymbols, "\n")

    if len(downtrendSymbols) > 0:
        #--------------------------------------------------
        # send text message
        #--------------------------------------------------
        #text_message = "PyTAAA/"+edition+" shows "+str(downtrendSymbols)+" in possible downtrend... \n"+str(channelPercent)+" % of trend channel."
        text_message = "PyTAAA/"+edition+" shows "+str(downtrendSymbols)+" current trend... "+\
                       "\nPct of trend channel  = "+str(channelPercent)+\
                       "\nperiod gainloss     = "+str(channelGainsLossesHoldings)+\
                       "\nperiod gainloss std = "+str(channelStdsHoldings)+\
                       "\ncurrent # std devs  = "+str(currentNumStdDevs)

        print(text_message +"\n\n")

        # send text message if market is open
        if 'close in' in get_MarketOpenOrClosed():
            #SendTextMessage( username,emailpassword,params['toSMS'],params['fromaddr'],text_message )
            SendEmail(username,emailpassword,params['toSMS'],params['fromaddr'],subjecttext,text_message,boldtext,headlinetext)

    return


#----------------------------------------------

def SMA(x,periods):
    SMA = np.zeros( (x.shape[0]), dtype=float)
    for i in range( x.shape[0] ):
        minx = np.max((0,i-periods))
        SMA[i] = np.mean(x[minx:i+1],axis=-1)
    return SMA


def hma(x, period):
    """Compute Hull moving average"""
    # convert ndarray to pandas dataframe
    # x should have shape x[num_companies, n_days]
    import pandas as pd
    col_labels = ['stock'+str(x) for x in np.arange(x.shape[0])]
    df = pd.DataFrame(
        data=x.T,    # values
        index=range(x.shape[1]),    # 1st column as index
        columns=col_labels
    )
    nday_half_range = np.arange(1, period//2+1)
    nday_range = np.arange(1, period + 1)
    #_x = df['stock']
    _x = df
    _func1 = lambda _x: np.sum(_x * nday_half_range) / np.sum(nday_half_range)
    _func2 = lambda _x: np.sum(_x * nday_range) / np.sum(nday_range)
    wma_1 = _x.rolling(period//2).apply(_func1, raw=True)
    wma_2 = _x.rolling(period).apply(_func2, raw=True)
    diff = 2 * wma_1 - wma_2
    hma = diff.rolling(int(np.sqrt(period))).mean()
    hma = hma.values.T
    return hma


def hma_pd(data, period):
    """
    Calculates the Hull Moving Average for a given dataset.

    Parameters:
    data: numpy array containing the price data
     - data should have shape data[num_companies, n_days]
    period: integer representing the HMA period

    Note: the elapsed time for this function is 3x that for function "hma"
    """
    import pandas as pd

    hma = np.zeros_like(data)
    for icompany in range(data.shape[0]):

        pd_data = pd.Series(list(data[icompany,:]))

        wma1 = pd_data.rolling(int(period/2)).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x)+1)),
            raw=True
        )
        wma2 = pd_data.rolling(period).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x)+1)),
            raw=True
        )

        hma_non_smooth = 2 * wma1 - wma2
        hma[icompany,:] = hma_non_smooth.rolling(int(np.sqrt(period))).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x)+1)),
            raw=True
        )

    return hma


# # create random dataset to test hma
# gain_loss = np.random.uniform(.95,1.05, (50,2500))
# x_val = gain_loss * 1.0
# x_val[:,0] = 1000.
# x_val = np.cumprod(x_val, axis=-1)



#----------------------------------------------

def SMS(x,periods):
    _SMS = np.zeros( (x.shape[0]), dtype=float)
    for i in range( x.shape[0] ):
        minx = np.max((0,i-periods))
        _SMS[i] = np.sum(x[minx:i+1],axis=-1)
    return _SMS


def SMA_2D(x,periods):
    SMA = np.zeros( (x.shape[0],x.shape[1]), dtype=float)
    for i in range( x.shape[1] ):
        minx = np.max((0,i-periods))
        SMA[:,i] = np.mean(x[:,minx:i+1],axis=-1)
    return SMA

# def hma_2D(x, periods):
#     HMA = np.zeros((x.shape[0],x.shape[1]), dtype=float)
#     for i in range(x.shape[0]):
#         HMA[i,:] = hma(x[i,:], periods)
#     return HMA

# def SMA_filtered_2D(x, periods, filt_min=-0.0125, filt_max=0.0125):
#     SMA = np.zeros((x.shape[0], x.shape[1]), dtype=float)
#     gainloss = np.zeros_like(x)
#     gainloss[:,1:] = x[:,1:] / x[:,:-1]
#     gainloss[np.isnan(x)] = 1.
#     gainloss -= 1.0
#     ii, jj =np.where((gainloss <= filt_min) | (gainloss >= filt_max))
#     x_count = np.zeros_like(x, dtype='int')
#     x_count[ii,jj] = 1
#     x_count = np.cumsum(x_count, axis=-1)
#     true_count = 0
#     for i in range(x.shape[0]):
#         indices = x_count[i,:]
#         for j in range(1, x.shape[1]):
#             end_indices = np.abs(indices - (indices[j]))
#             periods_end = np.max((0, np.argmin(end_indices)))
#             strt_indices = np.abs(indices - (indices[j] - periods))
#             periods_beg = np.max((0, int(np.argmin(strt_indices))))
#             if periods_end > periods_beg:
#                 SMA[i, j] = np.mean(x[i, periods_beg:periods_end], axis=-1)
#                 true_count += 1
#     return SMA


def SMA_filtered_2D(x, periods, filt_min=-0.0125, filt_max=0.0125):
    fsma = np.zeros((x.shape[0], x.shape[1]), dtype=float)
    gainloss = np.zeros_like(x)
    gainloss[:,1:] = x[:,1:] / x[:,:-1]
    gainloss[np.isnan(x)] = 1.
    gainloss -= 1.0
    ii, jj =np.where((gainloss <= filt_min) | (gainloss >= filt_max))
    x_count = np.zeros_like(x, dtype='int')
    x_count[ii,jj] = 1
    x_count = np.cumsum(x_count, axis=-1) - 1
    for i in range(x.shape[0]):
        indices = x_count[i,:]
        iii = np.where(indices[1:] != indices[:-1])[0] + 1
        SMA_sparse = SMA(x[i,iii], periods)
        if SMA_sparse.size > periods:
            fsma[i, iii[0]:iii[-1]+1] = np.interp(
                np.arange(iii[0], iii[-1]+1),
                iii,
                # x[i,iii]
                SMA_sparse
            )
            fsma[i,:iii[0]+1] = SMA_sparse[0]
            fsma[i,iii[-1]:] = SMA_sparse[-1]
    return fsma

'''
plt.clf();plt.grid()
plt.plot(x[0,:],'k-')
plt.plot(np.arange(x.shape[1])[iii], x[0,iii], 'ro')
plt.plot(fsma[0,:], 'b-')
plt.plot(sma[0,:], 'g-')

'''
#----------------------------------------------

def despike_2D(x,periods,stddevThreshold=5.0):
    # remove outliers from gradient of x (in 2nd dimension)
    gainloss = np.ones((x.shape[0],x.shape[1]),dtype=float)
    gainloss[:,1:] = x[:,1:] / x[:,:-1]
    for i in range( 1,x.shape[1] ):
        minx = max(0,i-periods)
        Stddev = np.std(gainloss[:,minx:i],axis=-1)
        Stddev *= stddevThreshold
        Stddev += 1.
        test = np.dstack( (Stddev, gainloss[:,i]) )
        gainloss[:,i] = np.min( test, axis=-1)
    gainloss[:,0] = x[:,0].copy()
    value = np.cumprod(gainloss,axis=1)
    return value


#----------------------------------------------

def MoveMax_2D(x,periods):
    MMax = np.zeros( (x.shape[0],x.shape[1]), dtype=float)
    for i in range( x.shape[1] ):
        minx = max(0,i-periods)
        MMax[:,i] = np.max(x[:,minx:i+1],axis=-1)
    return MMax


#----------------------------------------------

def MoveMax(x,periods):
    MMax = np.zeros( (x.shape[0]), dtype=float)
    for i in range( x.shape[0] ):
        minx = max(0,i-periods)
        MMax[i] = np.max(x[minx:i+1],axis=-1)
    return MMax

#----------------------------------------------

def MoveMin(x,periods):
    MMin = np.zeros( (x.shape[0]), dtype=float)
    for i in range( x.shape[0] ):
        minx = max(0,i-periods)
        MMin[i] = np.min(x[minx:i+1],axis=-1)
    return MMin


#----------------------------------------------

def move_sharpe_2D(adjClose,dailygainloss,period):
    """
    Compute the moving sharpe ratio
      sharpe_ratio = ( gmean(PortfolioDailyGains[-lag:])**252 -1. )
                   / ( np.std(PortfolioDailyGains[-lag:])*sqrt(252) )
      formula assume 252 trading days per year

    Geometric mean is simplified as follows:
    where the geometric mean is being used to determine the average
    growth rate of some quantity, and the initial and final values
    of that quantity are known, the product of the measured growth
    rate at every step need not be taken. Instead, the geometric mean
    is simply ( a(n)/a(0) )**(1/n), where n is the number of steps
    """
    from scipy.stats import gmean
    from math import sqrt
    from numpy import std
    #
    sharpe = np.zeros( (adjClose.shape[0],adjClose.shape[1]), dtype=float)
    for i in range( dailygainloss.shape[1] ):
        minindex = max( i-period, 0 )
        if i > minindex :
            sharpeValues = dailygainloss[:,minindex:i+1]
            sharpeValues[ np.isnan(sharpeValues) ] = 1.0
            numerator = gmean(sharpeValues,axis=-1)**252 -1.
            denominator = np.std(sharpeValues,axis=-1)*sqrt(252)
            denominator[ denominator == 0. ] = 1.e-5
            sharpe[:,i] = numerator / denominator
            '''
            sharpe[:,i] = ( gmean(dailygainloss[:,minindex:i+1],axis=-1)**252 -1. )     \
                   / ( np.std(dailygainloss[:,minindex:i+1],axis=-1)*sqrt(252) )
            '''
        else :
            sharpe[:,i] = 0.

    sharpe[sharpe==0]=.05
    sharpe[isnan(sharpe)] =.05

    return sharpe


#----------------------------------------------

def computeSignal2D( adjClose, gainloss, params ):

    print(" ... inside computeSignal2D ... ")
    print(" params = ",params)
    MA1 = int(params['MA1'])
    MA2 = int(params['MA2'])
    MA2offset = int(params['MA2offset'])

    narrowDays = params['narrowDays']
    mediumDays = params['mediumDays']
    wideDays = params['wideDays']

    lowPct = float(params['lowPct'])
    hiPct = float(params['hiPct'])
    sma2factor = float(params['MA2factor'])
    uptrendSignalMethod = params['uptrendSignalMethod']

    if uptrendSignalMethod == 'SMAs' :
        print("  ...using 3 SMA's for signal2D")
        print("\n\n ...calculating signal2D using '"+uptrendSignalMethod+"' method...")
        ########################################################################
        ## Calculate signal for all stocks based on 3 simple moving averages (SMA's)
        ########################################################################
        sma0 = SMA_2D( adjClose, MA2 )               # MA2 is shortest
        sma1 = SMA_2D( adjClose, MA2 + MA2offset )
        sma2 = sma2factor * SMA_2D( adjClose, MA1 )  # MA1 is longest

        signal2D = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
        for ii in range(adjClose.shape[0]):
            for jj in range(adjClose.shape[1]):
                if adjClose[ii,jj] > sma2[ii,jj] or ((adjClose[ii,jj] > min(sma0[ii,jj],sma1[ii,jj]) and sma0[ii,jj] > sma0[ii,jj-1])):
                    signal2D[ii,jj] = 1
                    if jj== adjClose.shape[1]-1 and isnan(adjClose[ii,-1]):
                        signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
            # take care of special case where constant share price is inserted at beginning of series
            index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1

            signal2D[ii,0:index] = 0

        dailyNumberUptrendingStocks = np.sum(signal2D,axis = 0)

        return signal2D

    if uptrendSignalMethod == 'HMAs' :
        print("  ...using 3 HMA's (hull moving average) for signal2D")
        print("\n\n ...calculating signal2D using '"+uptrendSignalMethod+"' method...")
        ########################################################################
        ## Calculate signal for all stocks based on 3 simple moving averages (HMA's)
        ########################################################################
        sma0 = hma(adjClose, MA2)               # MA2 is shortest
        sma1 = hma(adjClose, MA2 + MA2offset)
        sma2 = sma2factor * hma(adjClose, MA1)  # MA1 is longest

        signal2D = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
        for ii in range(adjClose.shape[0]):
            for jj in range(adjClose.shape[1]):
                if adjClose[ii,jj] > sma2[ii,jj] or ((adjClose[ii,jj] > min(sma0[ii,jj],sma1[ii,jj]) and sma0[ii,jj] > sma0[ii,jj-1])):
                    signal2D[ii,jj] = 1
                    if jj== adjClose.shape[1]-1 and isnan(adjClose[ii,-1]):
                        signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
            # take care of special case where constant share price is inserted at beginning of series
            index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1

            signal2D[ii,0:index] = 0

        dailyNumberUptrendingStocks = np.sum(signal2D,axis = 0)

        return signal2D

    elif uptrendSignalMethod == 'minmaxChannels' :
        print("  ...using 3 minmax channels for signal2D")
        print("\n\n ...calculating signal2D using '"+uptrendSignalMethod+"' method...")

        ########################################################################
        ## Calculate signal for all stocks based on 3 minmax channels (dpgchannels)
        ########################################################################

        # narrow channel is designed to remove day-to-day variability

        print("narrow days min,max,inc = ", narrowDays[0], narrowDays[-1], (narrowDays[-1]-narrowDays[0])/7.)
        narrow_minChannel, narrow_maxChannel = dpgchannel_2D( adjClose, narrowDays[0], narrowDays[-1], (narrowDays[-1]-narrowDays[0])/7. )
        narrow_midChannel = (narrow_minChannel+narrow_maxChannel)/2.

        medium_minChannel, medium_maxChannel = dpgchannel_2D( adjClose, mediumDays[0], mediumDays[-1], (mediumDays[-1]-mediumDays[0])/7. )
        medium_midChannel = (medium_minChannel+medium_maxChannel)/2.
        mediumSignal = ((narrow_midChannel-medium_minChannel)/(medium_maxChannel-medium_minChannel)-0.5)*2.0

        wide_minChannel, wide_maxChannel = dpgchannel_2D( adjClose, wideDays[0], wideDays[-1], (wideDays[-1]-wideDays[0])/7. )
        wide_midChannel = (wide_minChannel+wide_maxChannel)/2.
        wideSignal = ((narrow_midChannel-wide_minChannel)/(wide_maxChannel-wide_minChannel)-0.5)*2.0

        signal2D = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
        for ii in range(adjClose.shape[0]):
            for jj in range(adjClose.shape[1]):
                if mediumSignal[ii,jj] + wideSignal[ii,jj] > 0:
                    signal2D[ii,jj] = 1
                    if jj== adjClose.shape[1]-1 and isnan(adjClose[ii,-1]):
                        signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
            # take care of special case where constant share price is inserted at beginning of series
            index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1

            signal2D[ii,0:index] = 0

            '''
            # take care of special case where mp quote exists at end of series
            if firstTrailingEmptyPriceIndex[ii] != 0:
                signal2D[ii,firstTrailingEmptyPriceIndex[ii]:] = 0
            '''

        return signal2D

    elif uptrendSignalMethod == 'percentileChannels' :
        print("\n\n ...calculating signal2D using '"+uptrendSignalMethod+"' method...")
        signal2D = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
        lowChannel,hiChannel = percentileChannel_2D(adjClose,MA1,MA2+.01,MA2offset,lowPct,hiPct)
        for ii in range(adjClose.shape[0]):
            for jj in range(1,adjClose.shape[1]):
                if (adjClose[ii,jj] > lowChannel[ii,jj] and adjClose[ii,jj-1] <= lowChannel[ii,jj-1]) or adjClose[ii,jj] > hiChannel[ii,jj]:
                    signal2D[ii,jj] = 1
                elif (adjClose[ii,jj] < hiChannel[ii,jj] and adjClose[ii,jj-1] >= hiChannel[ii,jj-1]) or adjClose[ii,jj] < lowChannel[ii,jj]:
                    signal2D[ii,jj] = 0
                else:
                    signal2D[ii,jj] = signal2D[ii,jj-1]

                if jj== adjClose.shape[1]-1 and isnan(adjClose[ii,-1]):
                    signal2D[ii,jj] = 0                #### added to avoid choosing stocks no longer in index
            # take care of special case where constant share price is inserted at beginning of series
            index = np.argmax(np.clip(np.abs(gainloss[ii,:]-1),0,1e-8)) - 1
            signal2D[ii,0:index] = 0

        print(" finished calculating signal2D... mean signal2D = ", signal2D.mean())

        return signal2D, lowChannel, hiChannel


#----------------------------------------------

def nanrms(x, axis=None):
    from bottleneck import nanmean
    return np.sqrt(nanmean(x**2, axis=axis))

#----------------------------------------------


def move_informationRatio(dailygainloss_portfolio,dailygainloss_index,period):
    """
    Compute the moving information ratio

      returns for stock (annualized) = Rs
        -- assuming 252 days per year this is gmean(dailyGains)**252 -1
        -- Rs denotes stock's return

      excess return compared to bendmark = Expectation(Rp - Ri)
        -- assuming 252 days per year this is sum(Rp - Ri)/252, or just mean(Rp - Ri)
        -- Rp denotes active portfolio return
        -- Ri denotes index return

      tracking error compared to bendmark = sqrt(Expectation((Rp - Ri)**2))
        -- assuming 252 days per year this is sqrt(Sum((Rp - Ri)**2)/252), or just sqrt(mean(((Rp - Ri)**2)))
        -- Rp denotes active portfolio return
        -- Ri denotes index return

      information_ratio = ExcessReturn / TrackingError

      formula assume 252 trading days per year

    Geometric mean is simplified as follows:
    where the geometric mean is being used to determine the average
    growth rate of some quantity, and the initial and final values
    of that quantity are known, the product of the measured growth
    rate at every step need not be taken. Instead, the geometric mean
    is simply ( a(n)/a(0) )**(1/n), where n is the number of steps
    """
    from scipy.stats import gmean
    from math import sqrt
    from numpy import std
    from bottleneck import nanmean
    #
    infoRatio = np.zeros( (dailygainloss_portfolio.shape[0],dailygainloss_portfolio.shape[1]), dtype=float)

    for i in range( dailygainloss_portfolio.shape[1] ):

        minindex = max( i-period, 0 )

        if i > minindex :
            returns_portfolio = dailygainloss_portfolio[:,minindex:i+1] -1.
            returns_index =  dailygainloss_index[minindex:i+1] -1.
            excessReturn = nanmean( returns_portfolio - returns_index, axis = -1 )
            trackingError = nanrms( dailygainloss_portfolio[:,minindex:i+1] - dailygainloss_index[minindex:i+1], axis = -1 )

            infoRatio[:,i] = excessReturn / trackingError

            if i == dailygainloss_portfolio.shape[1]-1:
                print(" returns_portfolio = ", returns_portfolio)
                print(" returns_index = ", returns_index)
                print(" excessReturn = ", excessReturn)
                print(" infoRatio[:,i] = ", infoRatio[:,i])

        else :
            infoRatio[:,i] *= 0.

    infoRatio[infoRatio==0]=.0
    infoRatio[isnan(infoRatio)] =.0

    return infoRatio

#----------------------------------------------

def multiSharpe( datearray, adjClose, periods ):

    from allstats import allstats

    maxPeriod = np.max( periods )

    dates = datearray[maxPeriod:]
    sharpesPeriod = np.zeros( (len(periods),len(dates)), 'float' )
    #adjCloseSubset = adjClose[:,-len(dates):]

    for iperiod,period in enumerate(periods) :
        lenSharpe = period
        for idate in range( maxPeriod,adjClose.shape[1] ):
            sharpes = []
            for ii in range(adjClose.shape[0]):
                sharpes.append( allstats( adjClose[ii,idate-lenSharpe:idate] ).sharpe() )
            sharpes = np.array( sharpes )
            sharpes = sharpes[np.isfinite( sharpes )]
            if len(sharpes) > 0:
                sharpesAvg = np.mean(sharpes)
                if idate%1000 == 0:
                    print(period, datearray[idate],len(sharpes), sharpesAvg)
            else:
                sharpesAvg = 0.
            sharpesPeriod[iperiod,idate-maxPeriod] = sharpesAvg

    plotSharpe = sharpesPeriod[:,-len(dates):].copy()
    plotSharpe += .3
    plotSharpe /= 1.25
    signal = np.median(plotSharpe,axis=0)
    for i in range( plotSharpe.shape[0] ):
        signal += (np.clip( plotSharpe[i,:], -1., 2.) - signal)

    medianSharpe = np.median(plotSharpe,axis=0)
    signal = np.median(plotSharpe,axis=0) + 1.5 * (np.mean(plotSharpe,axis=0) - np.median(plotSharpe,axis=0))

    medianSharpe = np.clip( medianSharpe, -.1, 1.1 )
    signal = np.clip( signal, -.05, 1.05 )

    return dates, medianSharpe, signal


#----------------------------------------------

def move_martin_2D(adjClose,period):
    """
    Compute the moving martin ratio (ulcer performance index)

    martin ratio is based on ulcer index (rms drawdown over period)

    Reference: http://www.tangotools.com/ui/ui.htm


      martin_ratio = ( gmean(PortfolioDailyGains[-lag:])**252 -1. )
                   / ( np.std(PortfolioDailyGains[-lag:])*sqrt(252) )
      formula assume 252 trading days per year

    Geometric mean is simplified as follows:
    where the geometric mean is being used to determine the average
    growth rate of some quantity, and the initial and final values
    of that quantity are known, the product of the measured growth
    rate at every step need not be taken. Instead, the geometric mean
    is simply ( a(n)/a(0) )**(1/n), where n is the number of steps
    """
    from scipy.stats import gmean
    from math import sqrt
    from numpy import std
    #
    MoveMax = MoveMax_2D( adjClose, period )
    pctDrawDown = adjClose / MoveMax - 1.
    pctDrawDown = pctDrawDown ** 2

    martin = np.sqrt( SMA_2D( pctDrawDown, period )  )

    # reset NaN's to zero
    martin[ np.isnan(martin) ] = 0.

    return martin

#----------------------------------------------

def sharpeWeightedRank_2D(
        json_fn, datearray, symbols, adjClose, signal2D, signal2D_daily,
        LongPeriod, rankthreshold, riskDownside_min, riskDownside_max,
        rankThresholdPct, stddevThreshold=4.,
        is_backtest=True, makeQCPlots=False, verbose=False
):
    """
    Calculate Sharpe-weighted rankings for portfolio allocation.
    
    This function computes portfolio weights based on:
    1. Recent performance (momentum)
    2. Sharpe ratios over the LongPeriod lookback
    3. Risk-adjusted returns
    4. Signal strength from signal2D
    
    Returns:
        monthgainlossweight: Portfolio weights for each stock over time
    """
    import numpy as np
    try:
        import bottleneck as bn
        from scipy.stats import gmean
    except ImportError:
        print(" ... Warning: bottleneck not available, using numpy")
        import scipy.stats.mstats as bn
        from scipy.stats import gmean

    if verbose:
        print(" ... Inside sharpeWeightedRank_2D")
        print(f" ... adjClose shape: {adjClose.shape}")
        print(f" ... LongPeriod: {LongPeriod}, rankthreshold: {rankthreshold}")

    # Calculate gain/loss ratios
    gainloss = np.ones((adjClose.shape[0], adjClose.shape[1]), dtype=float)
    gainloss[:, 1:] = adjClose[:, 1:] / adjClose[:, :-1]
    gainloss[np.isnan(gainloss)] = 1.0

    # Initialize weights array
    monthgainlossweight = np.zeros((adjClose.shape[0], adjClose.shape[1]), dtype=float)
    
    # Calculate momentum-based gains over LongPeriod
    monthgainloss = np.ones((adjClose.shape[0], adjClose.shape[1]), dtype=float)
    monthgainloss[:, LongPeriod:] = adjClose[:, LongPeriod:] / adjClose[:, :-LongPeriod]
    monthgainloss[np.isnan(monthgainloss)] = 1.0

    # Calculate Sharpe ratios over rolling windows
    sharpe_scores = np.zeros_like(adjClose, dtype=float)
    
    for j in range(LongPeriod, adjClose.shape[1]):
        for i in range(adjClose.shape[0]):
            # Get recent returns for this stock
            recent_returns = gainloss[i, j-LongPeriod:j]
            
            # Only calculate if we have valid data and positive signal
            if signal2D[i, j] > 0 and len(recent_returns[recent_returns != 1.0]) > 10:
                try:
                    # Calculate annualized return and volatility
                    avg_return = gmean(recent_returns) ** 252 - 1.0
                    volatility = np.std(recent_returns) * np.sqrt(252)
                    
                    # Calculate Sharpe ratio (assuming 0% risk-free rate)
                    if volatility > 0:
                        sharpe_ratio = avg_return / volatility
                        # Apply momentum weighting
                        momentum = monthgainloss[i, j] - 1.0  # Convert to return
                        # Combine Sharpe ratio with momentum
                        sharpe_scores[i, j] = sharpe_ratio * (1.0 + momentum)
                    else:
                        sharpe_scores[i, j] = 0.0
                        
                except (ValueError, ZeroDivisionError, OverflowError):
                    sharpe_scores[i, j] = 0.0
            else:
                sharpe_scores[i, j] = 0.0

    # Apply risk constraints
    for j in range(LongPeriod, adjClose.shape[1]):
        for i in range(adjClose.shape[0]):
            # Apply downside risk constraint
            recent_returns = gainloss[i, j-LongPeriod:j] - 1.0  # Convert to actual returns
            downside_returns = recent_returns[recent_returns < 0]
            
            if len(downside_returns) > 0:
                downside_risk = np.std(downside_returns) * np.sqrt(252)
                # Penalize if downside risk is too high or too low
                if downside_risk < riskDownside_min or downside_risk > riskDownside_max:
                    sharpe_scores[i, j] *= 0.5  # Reduce weight for out-of-range risk

    # Rank stocks by Sharpe scores and assign weights
    for j in range(LongPeriod, adjClose.shape[1]):
        scores = sharpe_scores[:, j].copy()
        
        # Only consider stocks with positive signals
        valid_mask = (signal2D[:, j] > 0) & (scores > -np.inf) & np.isfinite(scores)
        
        if np.sum(valid_mask) > 0:
            # Get indices of valid stocks sorted by score (descending)
            valid_indices = np.where(valid_mask)[0]
            valid_scores = scores[valid_indices]
            
            # Sort by score (highest first)
            sort_idx = np.argsort(valid_scores)[::-1]
            sorted_indices = valid_indices[sort_idx]
            
            # Select top rankthreshold stocks
            n_selected = min(rankthreshold, len(sorted_indices))
            selected_indices = sorted_indices[:n_selected]
            
            if n_selected > 0:
                # Assign weights based on Sharpe scores (with floor)
                selected_scores = scores[selected_indices]
                selected_scores = np.maximum(selected_scores, 0.1)  # Floor to prevent negative weights
                
                # Normalize weights to sum to 1
                total_score = np.sum(selected_scores)
                if total_score > 0:
                    weights = selected_scores / total_score
                    monthgainlossweight[selected_indices, j] = weights
                else:
                    # Equal weights if all scores are zero
                    monthgainlossweight[selected_indices, j] = 1.0 / n_selected
            else:
                # No valid stocks - equal weight across all
                monthgainlossweight[:, j] = 1.0 / adjClose.shape[0]
        else:
            # No valid stocks - equal weight across all
            monthgainlossweight[:, j] = 1.0 / adjClose.shape[0]

    # Hold weights constant within calendar months
    for j in range(1, adjClose.shape[1]):
        if j < len(datearray) - 1:  # Ensure we don't go out of bounds
            current_month = datearray[j].month
            prev_month = datearray[j-1].month
            
            if current_month == prev_month:
                # Same month - maintain previous weights
                monthgainlossweight[:, j] = monthgainlossweight[:, j-1]

    # Ensure weights sum to 1 and handle NaNs
    for j in range(adjClose.shape[1]):
        weights_sum = np.sum(monthgainlossweight[:, j])
        if weights_sum > 0 and np.isfinite(weights_sum):
            monthgainlossweight[:, j] = monthgainlossweight[:, j] / weights_sum
        else:
            # Fallback to equal weights
            monthgainlossweight[:, j] = 1.0 / adjClose.shape[0]
    
    # Clean up any remaining NaNs or infinities
    monthgainlossweight[~np.isfinite(monthgainlossweight)] = 0.0
    
    # Final normalization check
    for j in range(adjClose.shape[1]):
        col_sum = np.sum(monthgainlossweight[:, j])
        if col_sum == 0:
            monthgainlossweight[:, j] = 1.0 / adjClose.shape[0]
        elif col_sum != 1.0:
            monthgainlossweight[:, j] = monthgainlossweight[:, j] / col_sum

    if verbose:
        print(f" ... Final weights shape: {monthgainlossweight.shape}")
        print(f" ... Weight range: [{monthgainlossweight.min():.6f}, {monthgainlossweight.max():.6f}]")
        print(f" ... Final column sums: min={np.sum(monthgainlossweight, axis=0).min():.6f}, max={np.sum(monthgainlossweight, axis=0).max():.6f}")

    return monthgainlossweight

#----------------------------------------------

def MAA_WeightedRank_2D(
        json_fn, datearray, symbols, adjClose ,signal2D ,signal2D_daily,
        LongPeriod,numberStocksTraded,
        wR, wC, wV, wS, stddevThreshold=4.
):

    # adjClose      --     # 2D array with adjusted closing prices (axes are stock number, date)
    # rankthreshold --     # select this many funds with best recent performance

    import numpy as np
    import nose
    import os
    import sys
    from matplotlib import pylab as plt
    import matplotlib.gridspec as gridspec
    try:
        import bottleneck as bn
        from bn import rankdata as rd
    except:
        import scipy.stats.mstats as bn

    from functions.GetParams import get_json_params
    params = get_json_params(json_fn)
    stockList = params['stockList']

    adjClose_despike = despike_2D( adjClose, LongPeriod, stddevThreshold=stddevThreshold )

    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    #gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
    gainloss[:,1:] = adjClose_despike[:,1:] / adjClose_despike[:,:-1]  ## experimental
    gainloss[isnan(gainloss)]=1.

    # convert signal2D to contain either 1 or 0 for weights
    signal2D -= signal2D.min()
    signal2D *= signal2D.max()

    ############################
    ###
    ### filter universe of stocks to exclude all that have return < 0
    ### - needed for correlation to "equal weight index" (EWI)
    ### - EWI is daily gain/loss percentage
    ###
    ############################

    EWI  = np.zeros( adjClose.shape[1], 'float' )
    EWI_count  = np.zeros( adjClose.shape[1], 'int' )
    for jj in np.arange(LongPeriod,adjClose.shape[1]) :
        for ii in range(adjClose.shape[0]):
            if signal2D_daily[ii,jj] == 1:
                EWI[jj] += gainloss[ii,jj]
                EWI_count[jj] += 1
    EWI = EWI/EWI_count
    EWI[np.isnan(EWI)] = 1.0

    ############################
    ###
    ### compute correlation to EWI
    ### - each day, for each stock
    ### - not needed for stocks on days with return < 0
    ###
    ############################

    corrEWI  = np.zeros( adjClose.shape, 'float' )
    for jj in np.arange(LongPeriod,adjClose.shape[1]) :
        for ii in range(adjClose.shape[0]):
            start_date = max( jj - LongPeriod, 0 )
            if adjClose_despike[ii,jj] > adjClose_despike[ii,start_date]:
                corrEWI[ii,jj] = normcorrcoef(gainloss[ii,start_date:jj]-1.,EWI[start_date:jj]-1.)
                if corrEWI[ii,jj] <0:
                    corrEWI[ii,jj] = 0.

    ############################
    ###
    ### compute weights
    ### - each day, for each stock
    ### - set to 0. for stocks on days with return < 0
    ###
    ############################

    weights  = np.zeros( adjClose.shape, 'float' )
    for jj in np.arange(LongPeriod,adjClose.shape[1]) :
        for ii in range(adjClose.shape[0]):
            start_date = max( jj - LongPeriod, 0 )
            returnForPeriod = (adjClose_despike[ii,jj]/adjClose_despike[ii,start_date])-1.
            if returnForPeriod  < 0.:
                returnForPeriod = 0.
            volatility = np.std(adjClose_despike[ii,start_date:jj])
            weights[ii,jj] = ( returnForPeriod**wR * (1.-corrEWI[ii,jj])**wC / volatility**wV ) **wS

    weights[np.isnan(weights)] = 0.0

    # make duplicate of weights for adjusting using crashProtection
    CPweights = weights.copy()
    CP_cashWeight = np.zeros(adjClose.shape[1], 'float' )
    for jj in np.arange(adjClose.shape[1]) :
        weightsToday = weights[:,jj]
        CP_cashWeight[jj] = float(len(weightsToday[weightsToday==0.])) / len(weightsToday)

    ############################
    ###
    ### compute weights ranking and keep best
    ### 'best' are numberStocksTraded*%risingStocks
    ### - weights need to sum to 100%
    ###
    ############################

    weightRank = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=int)

    weightRank = bn.rankdata(weights,axis=0)
    # reverse the ranks (low ranks are biggest gainers)
    maxrank = np.max(weightRank)
    weightRank -= maxrank-1
    weightRank *= -1
    weightRank += 2

    # set top 'numberStocksTraded' to have weights sum to 1.0
    for jj in np.arange(adjClose.shape[1]) :
        ranksToday = weightRank[:,jj].copy()
        weightsToday = weights[:,jj].copy()
        weightsToday[ranksToday > numberStocksTraded] = 0.
        if np.sum(weightsToday) > 0.:
            weights[:,jj] = weightsToday / np.sum(weightsToday)
        else:
            weights[:,jj] = 1./len(weightsToday)

    # set CASH to have weight based on CrashProtection
    cash_index = symbols.index("CASH")
    for jj in np.arange(adjClose.shape[1]) :
        CPweights[ii,jj] = CP_cashWeight[jj]
        weightRank[ii,jj] = 0
        ranksToday = weightRank[:,jj].copy()
        weightsToday = CPweights[:,jj].copy()
        weightsToday[ranksToday > numberStocksTraded] = 0.
        if np.sum(weightsToday) > 0.:
            CPweights[:,jj] = weightsToday / np.sum(weightsToday)
        else:
            CPweights[:,jj] = 1./len(weightsToday)

    # hold weights constant for month
    for jj in np.arange(LongPeriod,adjClose.shape[1]) :
        start_date = max( jj - LongPeriod, 0 )
        yesterdayMonth = datearray[jj-1].month
        todayMonth = datearray[jj].month
        if todayMonth == yesterdayMonth:
            weights[:,jj] = weights[:,jj-1]
            CPweights[:,jj] = CPweights[:,jj-1]

    # input symbols and company names from text file
    json_dir = os.path.split(json_fn)[0]
    if stockList == 'Naz100':
        companyName_file = os.path.join( json_dir, "symbols",  "companyNames.txt" )
    elif stockList == 'SP500':
        companyName_file = os.path.join( json_dir, "symbols",  "SP500_companyNames.txt" )
    with open( companyName_file, "r" ) as f:
        companyNames = f.read()

    print("\n\n\n")
    companyNames = companyNames.split("\n")
    ii = companyNames.index("")
    del companyNames[ii]
    companySymbolList  = []
    companyNameList = []
    for iname,name in enumerate(companyNames):
        name = name.replace("amp;", "")
        testsymbol, testcompanyName = name.split(";")
        companySymbolList.append(format(testsymbol,'5s'))
        companyNameList.append(testcompanyName)

    # print list showing current rankings and weights
    # - symbol
    # - rank (at begining of month)
    # - rank (most recent trading day)
    # - weight from sharpe ratio
    # - price
    import os
    rank_text = "<div id='rank_table_container'><h3>"+"<p>Current stocks, with ranks, weights, and prices are :</p></h3><font face='courier new' size=3><table border='1'> \
               </td><td>Rank (today) \
               </td><td>Symbol \
               </td><td>Company \
               </td><td>Weight \
               </td><td>CP Weight \
               </td><td>Price  \
               </td><td>Trend  \
               </td></tr>\n"
    for i, isymbol in enumerate(symbols):
        for j in range(len(symbols)):
            if int( weightRank[j,-1] ) == i :
                if signal2D_daily[j,-1] == 1.:
                    trend = 'up'
                else:
                    trend = 'down'

                # search for company name
                try:
                    symbolIndex = companySymbolList.index(format(symbols[j],'5s'))
                    companyName = companyNameList[symbolIndex]
                except:
                    companyName = ""

                rank_text = rank_text + \
                       "<tr><td>" + format(weightRank[j,-1],'6.0f')  + \
                       "<td>" + format(symbols[j],'5s')  + \
                       "<td>" + format(companyName,'15s')  + \
                       "<td>" + format(weights[j,-1],'5.03f') + \
                       "<td>" + format(CPweights[j,-1],'5.03f') + \
                       "<td>" + format(adjClose[j,-1],'6.2f')  + \
                       "<td>" + trend  + \
                       "</td></tr>  \n"
    rank_text = rank_text + "</table></div>\n"

    print("leaving function MAA_WeightedRank_2D...")

    """
    print " symbols = ", symbols
    print " weights = ", weights[:,-1]
    print " CPweights = ", CPweights[:,-1]

    print " number NaNs in weights = ", weights[np.isnan(weights)].shape
    print " number NaNs in CPweights = ", CPweights[np.isnan(CPweights)].shape

    print " NaNs in monthgainlossweight = ", weights[np.isnan(weights)].shape
    testsum = np.sum(weights,axis=0)
    print " testsum shape, min, and max = ", testsum.shape, testsum.min(), testsum.max()
    """

    return weights, CPweights


#----------------------------------------------
def UnWeightedRank_2D(datearray,adjClose,signal2D,LongPeriod,rankthreshold,riskDownside_min,riskDownside_max,rankThresholdPct):

    # adjClose      --     # 2D array with adjusted closing prices (axes are stock number, date)
    # rankthreshold --     # select this many funds with best recent performance

    import numpy as np
    import nose
    try:
        import bottleneck as bn
        from bn import rankdata as rd
    except:
        import scipy.stats.mstats as bn


    gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
    gainloss[isnan(gainloss)]=1.

    # convert signal2D to contain either 1 or 0 for weights
    signal2D -= signal2D.min()
    signal2D *= signal2D.max()

    # apply signal to daily gainloss
    gainloss = gainloss * signal2D
    gainloss[gainloss == 0] = 1.0

    value = 10000. * np.cumprod(gainloss,axis=1)

    # calculate gainloss over period of "LongPeriod" days
    monthgainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    monthgainloss[:,LongPeriod:] = adjClose[:,LongPeriod:] / adjClose[:,:-LongPeriod]
    monthgainloss[isnan(monthgainloss)]=1.

    monthgainlossweight = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)

    rankweight = 1./rankthreshold

    ########################################################################
    ## Calculate change in rank of active stocks each day (without duplicates as ties)
    ########################################################################
    monthgainlossRank = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=int)
    monthgainlossPrevious = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=float)
    monthgainlossPreviousRank = np.zeros((adjClose.shape[0],adjClose.shape[1]),dtype=int)

    monthgainlossRank = bn.rankdata(monthgainloss,axis=0)
    # reverse the ranks (low ranks are biggest gainers)
    maxrank = np.max(monthgainlossRank)
    monthgainlossRank -= maxrank-1
    monthgainlossRank *= -1
    monthgainlossRank += 2

    monthgainlossPrevious[:,LongPeriod:] = monthgainloss[:,:-LongPeriod]
    monthgainlossPreviousRank = bn.rankdata(monthgainlossPrevious,axis=0)
    # reverse the ranks (low ranks are biggest gainers)
    maxrank = np.max(monthgainlossPreviousRank)
    monthgainlossPreviousRank -= maxrank-1
    monthgainlossPreviousRank *= -1
    monthgainlossPreviousRank += 2

    # weight deltaRank for best and worst performers differently
    rankoffsetchoice = rankthreshold
    delta = -(monthgainlossRank - monthgainlossPreviousRank ) / (monthgainlossRank + rankoffsetchoice)

    # if rank is outside acceptable threshold, set deltarank to zero so stock will not be chosen
    #  - remember that low ranks are biggest gainers
    rankThreshold = (1. - rankThresholdPct) * ( monthgainlossRank.max() - monthgainlossRank.min() )
    for ii in range(monthgainloss.shape[0]):
        for jj in range(monthgainloss.shape[1]):
            if monthgainloss[ii,jj] > rankThreshold :
                delta[ii,jj] = -monthgainloss.shape[0]/2

    deltaRank = bn.rankdata(delta,axis=0)
    # reverse the ranks (low deltaRank have the fastest improving rank)
    maxrank = np.max(deltaRank)
    deltaRank -= maxrank-1
    deltaRank *= -1
    deltaRank += 2

    for ii in range(monthgainloss.shape[1]):
        if deltaRank[:,ii].min() == deltaRank[:,ii].max():
            deltaRank[:,ii] = 0.

    ########################################################################
    ## Hold values constant for calendar month (gains, ranks, deltaRanks)
    ########################################################################

    for ii in np.arange(1,monthgainloss.shape[1]):
        if datearray[ii].month == datearray[ii-1].month:
            monthgainloss[:,ii] = monthgainloss[:,ii-1]
            deltaRank[:,ii] = deltaRank[:,ii-1]

    ########################################################################
    ## Calculate number of active stocks each day
    ########################################################################

    # TODO: activeCount can be computed before loop to save CPU cycles
    # count number of unique values
    activeCount = np.zeros(adjClose.shape[1],dtype=float)
    for ii in np.arange(0,monthgainloss.shape[0]):
        firsttradedate = np.argmax( np.clip( np.abs( gainloss[ii,:]-1. ), 0., .00001 ) )
        activeCount[firsttradedate:] += 1

    minrank = np.min(deltaRank,axis=0)
    maxrank = np.max(deltaRank,axis=0)
    # convert rank threshold to equivalent percent of rank range

    rankthresholdpercentequiv = np.round(float(rankthreshold)*(activeCount-minrank+1)/adjClose.shape[0])
    ranktest = deltaRank <= rankthresholdpercentequiv

    ########################################################################
    ### calculate equal weights for ranks below threshold
    ########################################################################

    elsecount = 0
    elsedate  = 0
    for ii in np.arange(1,monthgainloss.shape[1]) :
        if activeCount[ii] > minrank[ii] and rankthresholdpercentequiv[ii] > 0:
            for jj in range(value.shape[0]):
                test = deltaRank[jj,ii] <= rankthresholdpercentequiv[ii]
                if test == True :
                    monthgainlossweight[jj,ii]  = 1./rankthresholdpercentequiv[ii]
                else:
                    monthgainlossweight[jj,ii]  = 0.
        elif activeCount[ii] == 0 :
            monthgainlossweight[:,ii]  *= 0.
            monthgainlossweight[:,ii]  += 1./adjClose.shape[0]
        else :
            elsedate = datearray[ii]
            elsecount += 1
            monthgainlossweight[:,ii]  = 1./activeCount[ii]

    aaa = np.sum(monthgainlossweight,axis=0)

    print("")
    print(" invoking correction to monthgainlossweight.....")
    print("")
    # find first date with number of stocks trading (rankthreshold) + 2
    activeCountAboveMinimum = activeCount
    activeCountAboveMinimum += -rankthreshold + 2
    firstTradeDate = np.argmax( np.clip( activeCountAboveMinimum, 0 , 1 ) )
    for ii in np.arange(firstTradeDate,monthgainloss.shape[1]) :
        if np.sum(monthgainlossweight[:,ii]) == 0:
            for kk in range(rankthreshold):
                indexHighDeltaRank = np.argmin(deltaRank[:,ii]) # remember that best performance is lowest deltaRank
                monthgainlossweight[indexHighDeltaRank,ii]  = 1./rankthreshold
                deltaRank[indexHighDeltaRank,ii] = 1000.


    print(" weights calculation else clause encountered :",elsecount," times. last date encountered is ",elsedate)
    rankweightsum = np.sum(monthgainlossweight,axis=0)

    monthgainlossweight[isnan(monthgainlossweight)] = 0.  # changed result from 1 to 0

    monthgainlossweight = monthgainlossweight / np.sum(monthgainlossweight,axis=0)
    monthgainlossweight[isnan(monthgainlossweight)] = 0.  # changed result from 1 to 0

    return monthgainlossweight








def hurst(X):
    """ Compute the Hurst exponent of X. If the output H=0.5,the behavior
    of the time-series is similar to random walk. If H<0.5, the time-series
    cover less "distance" than a random walk, vice verse.

    Parameters
    ----------
    X
        list
        a time series

    Returns
    -------
    H
        float
        Hurst exponent

    Examples
    --------
    >>> import pyeeg
    >>> from numpy.random import randn
    >>> a = randn(4096)
    >>> pyeeg.hurst(a)
    >>> 0.5057444

    ######################## Function contributed by Xin Liu #################
    https://code.google.com/p/pyeeg/source/browse/pyeeg.py
    Copyleft 2010 Forrest Sheng Bao http://fsbao.net
    PyEEG, a Python module to extract EEG features, v 0.02_r2
    Project homepage: http://pyeeg.org

    **Naming convention**

    Constants: UPPER_CASE_WITH_UNDERSCORES, e.g., SAMPLING_RATE, LENGTH_SIGNAL.
    Function names: lower_case_with_underscores, e.g., spectrum_entropy.
    Variables (global and local): CapitalizedWords or CapWords, e.g., Power.
    If a variable name consists of one letter, I may use lower case, e.g., x, y.

    """

    from numpy import zeros, log, array, cumsum, std
    from numpy.linalg import lstsq

    N = len(X)

    T = array([float(i) for i in range(1,N+1)])
    Y = cumsum(X)
    Ave_T = Y/T

    S_T = zeros((N))
    R_T = zeros((N))
    for i in range(N):
        S_T[i] = std(X[:i+1])
        X_T = Y - T * Ave_T[i]
        R_T[i] = max(X_T[:i + 1]) - min(X_T[:i + 1])

    R_S = R_T / S_T
    R_S = log(R_S)
    n = log(T).reshape(N, 1)
    H = lstsq(n[1:], R_S[1:])[0]
    return H[0]
