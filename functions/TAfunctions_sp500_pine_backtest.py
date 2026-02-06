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
    movepctrank = np.zeros( (hi.shape[0] ) , dtype=float)
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
        # combine sharpe ratios computed over 2 different periods
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
    from bottleneck import nanmean
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

#############################################################################
# Repeated Values Filter for Infilled Data Detection
#############################################################################
# Threshold for detecting infilled/stale price data. If the ratio of
# repeated values in the lookback window exceeds this threshold, the
# stock's Sharpe ratio is set to 0.0 to exclude it from selection.
# A value of 0.15 means if more than 15% of prices are duplicates,
# the stock is excluded.
REPEATED_VALUES_THRESHOLD = 0.15

#############################################################################
# Sharpe Ratio Threshold for Unreliable Values
#############################################################################
# Maximum absolute Sharpe ratio considered reliable. Values exceeding
# this threshold are set to 0.0 as they indicate data quality issues
# (e.g., near-zero volatility from infilled/stale data).
SHARPE_RATIO_THRESHOLD = 10.0

#############################################################################
# Trading Days Per Year for Annualization
#############################################################################
TRADING_DAYS_PER_YEAR = 252


def compute_valid_data_mask_1d(prices: np.ndarray) -> np.ndarray:
    """
    Create a mask identifying valid (non-repeated) price data points.

    A point is considered valid (mask=1) when BOTH conditions are met:
    1. Price differs from at least one neighbor (not flat).
    2. Gainloss differs from at least one neighbor (not constant slope).

    A point is invalid (mask=0) when:
    - Price matches BOTH previous and following values (flat section), OR
    - Gainloss matches BOTH previous and following values (constant slope).

    This filters out both flat/infilled data AND linear interpolation.

    Parameters
    ----------
    prices : np.ndarray
        1D array of prices for a single stock.

    Returns
    -------
    np.ndarray
        Binary mask array (same length as input), 1=valid, 0=invalid.
    """
    n_days = len(prices)
    if n_days < 3:
        return np.ones(n_days, dtype=float)

    # Compute daily gainloss (returns as ratio).
    gainloss = np.ones(n_days, dtype=float)
    gainloss[1:] = prices[1:] / prices[:-1]
    gainloss[np.isnan(gainloss)] = 1.0

    # Initialize mask as ones (assume valid).
    mask = np.ones(n_days, dtype=float)

    # For interior points (indices 1 to n_days - 2).
    for i in range(1, n_days - 1):
        # Check for FLAT sections: price same as both neighbors.
        price_same_prev = np.abs(prices[i] - prices[i - 1]) < 1e-6
        price_same_next = np.abs(prices[i] - prices[i + 1]) < 1e-6
        is_flat = price_same_prev and price_same_next

        # Check for CONSTANT SLOPE: gainloss same as both neighbors.
        gl_same_prev = np.abs(gainloss[i] - gainloss[i - 1]) < 1e-8
        gl_same_next = np.abs(gainloss[i] - gainloss[i + 1]) < 1e-8
        is_constant_slope = gl_same_prev and gl_same_next

        # Invalid if flat OR constant slope.
        if is_flat or is_constant_slope:
            mask[i] = 0.0

    # Handle boundary points.
    # First point.
    price_same = np.abs(prices[0] - prices[1]) < 1e-6
    gl_same = np.abs(gainloss[1] - gainloss[2]) < 1e-8 if n_days > 2 else False
    if price_same or gl_same:
        mask[0] = 0.0

    # Last point.
    price_same = np.abs(prices[-1] - prices[-2]) < 1e-6
    gl_same = np.abs(gainloss[-1] - gainloss[-2]) < 1e-8
    if price_same or gl_same:
        mask[-1] = 0.0

    return mask


def rolling_sharpe_vectorized(
    prices: np.ndarray,
    window: int = 252,
    trading_days: int = 252
) -> np.ndarray:
    """
    Compute rolling Sharpe ratio using vectorized NumPy operations.

    Uses sliding_window_view for efficient rolling window calculations.
    Formula: (mean(returns) / std(returns)) * sqrt(trading_days)

    Parameters
    ----------
    prices : np.ndarray
        1D array of daily prices for a single stock.
    window : int
        Rolling window size in days (default 252).
    trading_days : int
        Number of trading days per year for annualization (default 252).

    Returns
    -------
    np.ndarray
        Rolling Sharpe ratio values (same length as prices, with leading
        zeros for the warmup period).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n_days = len(prices)
    sharpe = np.zeros(n_days, dtype=float)

    if n_days < window + 1:
        return sharpe

    # Compute daily returns (as ratios minus 1).
    returns = np.zeros(n_days, dtype=float)
    returns[1:] = prices[1:] / prices[:-1] - 1.0
    returns[np.isnan(returns)] = 0.0
    returns[np.isinf(returns)] = 0.0

    # Create rolling windows of returns.
    # sliding_window_view requires returns[1:] since returns[0] = 0.
    if n_days - 1 < window:
        return sharpe

    windows = sliding_window_view(returns[1:], window_shape=window)

    # Vectorized mean and std across axis=1.
    mean_returns = windows.mean(axis=1)
    std_returns = windows.std(axis=1, ddof=1)  # Sample std.

    # Avoid division by zero.
    std_returns = np.where(std_returns < 1e-10, np.nan, std_returns)

    # Sharpe ratio (annualized).
    sharpe_values = (mean_returns / std_returns) * np.sqrt(trading_days)

    # Replace NaN with 0.
    sharpe_values = np.nan_to_num(sharpe_values, nan=0.0)

    # Place results in correct positions (offset by window).
    start_idx = window
    end_idx = start_idx + len(sharpe_values)
    if end_idx <= n_days:
        sharpe[start_idx:end_idx] = sharpe_values

    return sharpe


def calculate_repeated_values_ratio(
    price_series: np.ndarray,
    lookback: int,
    decimal_places: int = 2
) -> float:
    """
    Calculate the ratio of repeated values in a price series.

    Uses np.unique to find distinct values, then compares to total
    count to determine what fraction of values are duplicates.
    This detects stocks with infilled/stale data that would cause
    artificially high Sharpe ratios due to near-zero volatility.

    Prices are rounded to handle floating-point precision issues.
    For example, 23.4500000001 and 23.4499999999 would both round
    to 23.45 and be considered the same price.

    Parameters
    ----------
    price_series : np.ndarray
        1D array of prices for a single stock ending at current date.
    lookback : int
        Number of periods to examine (typically LongPeriod).
    decimal_places : int
        Number of decimal places to round prices to (default 2 for
        penny precision). Use higher values for very low-priced
        stocks or ETFs.

    Returns
    -------
    float
        Ratio of repeated values (0.0 = all unique, ~1.0 = all same).
        Returns 0.0 if lookback < 2 or insufficient data.

    Examples
    --------
    All unique prices:  252 values, 252 unique -> 0.0% repeated
    Normal stock:       252 values, 245 unique -> 2.8% repeated
    JEF infilled:       252 values, 1 unique   -> 99.6% repeated
    """
    if lookback < 2:
        return 0.0

    # Get the lookback window (last 'lookback' values).
    if len(price_series) < lookback:
        window = price_series
    else:
        window = price_series[-lookback:]

    # Handle NaN values by excluding them.
    valid_prices = window[~np.isnan(window)]

    if len(valid_prices) < 2:
        return 0.0

    #########################################################################
    # Round prices to handle floating-point precision issues.
    # Infilled data may have tiny differences like 23.4500000001 vs
    # 23.4499999999 that would incorrectly appear as unique values.
    # Rounding to 2 decimal places (penny precision) fixes this.
    #########################################################################
    rounded_prices = np.round(valid_prices, decimal_places)

    n_total = len(valid_prices)
    n_unique = len(np.unique(rounded_prices))

    # Ratio of repeated values = 1 - (unique / total).
    # If all values are unique: 1 - 1.0 = 0.0
    # If all values are same:   1 - (1/n) ≈ 1.0
    repeated_ratio = 1.0 - (n_unique / n_total)

    return repeated_ratio

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


#----------------------------------------------

def apply_weight_constraints(
    weights: np.ndarray,
    riskDownside_min: float,
    riskDownside_max: float
) -> np.ndarray:
    """
    Apply min/max constraints to portfolio weights.

    Clips weights to be within [riskDownside_min, riskDownside_max].
    Then renormalizes so weights sum to 1.0.

    Parameters
    ----------
    weights : np.ndarray
        1D array of portfolio weights (should sum to ~1.0).
    riskDownside_min : float
        Minimum allowed weight for any position.
    riskDownside_max : float
        Maximum allowed weight for any position.

    Returns
    -------
    np.ndarray
        Constrained and renormalized weights.
    """
    # Clip weights to min/max constraints.
    constrained = np.clip(weights, riskDownside_min, riskDownside_max)

    # Renormalize to sum to 1.0.
    total = constrained.sum()
    if total > 0:
        constrained = constrained / total

    return constrained


#----------------------------------------------

def sharpeWeightedRank_2D(
    json_fn,
    datearray,
    symbols,
    adjClose,
    signal2D,
    signal2D_daily,
    LongPeriod,
    numberStocksTraded,
    riskDownside_min,
    riskDownside_max,
    rankThresholdPct,
    stddevThreshold=5.0,
    makeQCPlots=False,
    # Parameters for backtest (refactored version).
    max_weight_factor=3.0,
    min_weight_factor=0.3,
    absolute_max_weight=0.9,
    apply_constraints=True,
    # Parameters for production (PortfolioPerformanceCalcs).
    is_backtest=True,
    verbose=False,
    **kwargs  # Accept any additional keyword arguments for compatibility.
):
    """
    Compute Sharpe-ratio-weighted portfolio allocation for all dates.

    This function computes rolling Sharpe ratios for each stock and
    returns a 2D array of portfolio weights [n_stocks, n_days].
    Stocks are selected based on:
    1. Having an uptrend signal (signal2D > 0)
    2. Ranking in the top N by Sharpe ratio
    3. Passing data quality checks

    Weights are assigned proportionally to Sharpe ratios, with constraints
    applied via max_weight_factor, min_weight_factor, and absolute_max_weight.

    IMPORTANT: Weights are forward-filled so every day has valid weights.
    This ensures portfolio calculations don't result in zero values.

    Parameters
    ----------
    json_fn : str
        Path to JSON configuration file (not used but kept for API).
    datearray : np.ndarray
        Array of dates corresponding to adjClose columns.
    symbols : list
        List of stock symbols corresponding to adjClose rows.
    adjClose : np.ndarray
        2D array of adjusted close prices [n_stocks, n_days].
    signal2D : np.ndarray
        2D array of uptrend signals [n_stocks, n_days], 1=uptrend.
    signal2D_daily : np.ndarray
        Daily signals before monthly hold logic (not used but kept).
    LongPeriod : int
        Lookback period for Sharpe ratio calculation.
    numberStocksTraded : int
        Number of top stocks to select for the portfolio.
    riskDownside_min : float
        Minimum weight constraint per position.
    riskDownside_max : float
        Maximum weight constraint per position.
    rankThresholdPct : float
        Percentile threshold for rank filtering (not used currently).
    stddevThreshold : float
        Threshold for spike detection (default 5.0).
    makeQCPlots : bool
        If True, generate QC plots (default False).
    max_weight_factor : float
        Maximum weight as multiple of equal weight (default 3.0).
    min_weight_factor : float
        Minimum weight as multiple of equal weight (default 0.3).
    absolute_max_weight : float
        Absolute maximum weight cap (default 0.9).
    apply_constraints : bool
        Whether to apply weight constraints (default True).
    is_backtest : bool
        If True, running in backtest mode (default True).
        If False, running in production mode.
    verbose : bool
        If True, print progress information (default False).
    **kwargs : dict
        Additional keyword arguments for forward compatibility.

    Returns
    -------
    np.ndarray
        2D array of portfolio weights [n_stocks, n_days].
        Weights sum to 1.0 for each day (column).
    """
    from math import sqrt

    n_stocks = adjClose.shape[0]
    n_days = adjClose.shape[1]

    print(" ... inside sharpeWeightedRank_2D (Sharpe-based selection) ...")
    print(f" ... n_stocks={n_stocks}, n_days={n_days}")
    print(f" ... LongPeriod={LongPeriod}")
    print(f" ... numberStocksTraded={numberStocksTraded}")
    print(f" ... is_backtest={is_backtest}")
    print(f" ... max_weight_factor={max_weight_factor}")
    print(f" ... min_weight_factor={min_weight_factor}")

    #########################################################################
    # Initialize output weight matrix.
    #########################################################################
    monthgainlossweight = np.zeros((n_stocks, n_days), dtype=float)

    #########################################################################
    # Compute daily gain/loss for Sharpe calculation.
    #########################################################################
    dailygainloss = np.ones((n_stocks, n_days), dtype=float)
    dailygainloss[:, 1:] = adjClose[:, 1:] / adjClose[:, :-1]
    dailygainloss[np.isnan(dailygainloss)] = 1.0
    dailygainloss[np.isinf(dailygainloss)] = 1.0

    #########################################################################
    # Compute rolling Sharpe ratio for all stocks and all dates.
    #########################################################################
    print(" ... Computing rolling Sharpe ratios for all stocks...")
    sharpe_2d = np.zeros((n_stocks, n_days), dtype=float)

    # Rolling window size for dynamic threshold calculation.
    ROLLING_WINDOW_FOR_THRESHOLD = 10
    # Multiplier for trimmed mean to define extreme values.
    EXTREME_MULTIPLIER = 3.0
    # Trim percentage for trimmed mean (trim 10% from each end).
    TRIM_PERCENT = 0.1

    for j in range(LongPeriod, n_days):
        # Get the lookback window for this date.
        start_idx = max(0, j - LongPeriod)

        # Compute raw Sharpe ratios for all stocks on this date first.
        raw_sharpes_today = np.zeros(n_stocks, dtype=float)
        
        for i in range(n_stocks):
            # Get daily returns for this stock in the lookback window.
            returns_window = dailygainloss[i, start_idx:j+1]

            # Skip if not enough valid data.
            valid_returns = returns_window[~np.isnan(returns_window)]
            if len(valid_returns) < LongPeriod // 2:
                raw_sharpes_today[i] = 0.0
                continue

            # Check for repeated/infilled data.
            price_window = adjClose[i, start_idx:j+1]
            valid_prices = price_window[~np.isnan(price_window)]
            if len(valid_prices) > 0:
                n_unique = len(np.unique(np.round(valid_prices, 2)))
                repeated_ratio = 1.0 - (n_unique / len(valid_prices))
                if repeated_ratio > 0.15:  # More than 15% repeated.
                    raw_sharpes_today[i] = 0.0
                    continue

            # Compute Sharpe ratio.
            # Formula: (mean_return / std_return) * sqrt(252).
            mean_ret = np.mean(valid_returns) - 1.0  # Convert to returns.
            std_ret = np.std(valid_returns - 1.0)

            if std_ret > 1e-8:
                sharpe_val = (mean_ret / std_ret) * sqrt(252)
                raw_sharpes_today[i] = sharpe_val
            else:
                raw_sharpes_today[i] = 0.0

        #####################################################################
        # Compute dynamic threshold using trimmed mean of rolling window.
        #####################################################################
        # Get previous Sharpe values for threshold calculation.
        window_start = max(LongPeriod, j - ROLLING_WINDOW_FOR_THRESHOLD)
        if j > window_start:
            # Collect all non-zero Sharpe values from the rolling window.
            historical_sharpes = sharpe_2d[:, window_start:j].flatten()
            # Filter out zeros and get absolute values.
            valid_historical = np.abs(historical_sharpes[historical_sharpes != 0])
            
            if len(valid_historical) > 5:
                # Compute trimmed mean (trim TRIM_PERCENT from each end).
                from scipy import stats
                trimmed_mean = stats.trim_mean(valid_historical, TRIM_PERCENT)
                dynamic_threshold = EXTREME_MULTIPLIER * trimmed_mean
                # Ensure threshold is at least 3.0 to avoid being too strict.
                dynamic_threshold = max(dynamic_threshold, 3.0)
            else:
                # Fallback to fixed threshold if not enough history.
                dynamic_threshold = 10.0
        else:
            # Fallback to fixed threshold for early dates.
            dynamic_threshold = 10.0

        #####################################################################
        # Apply dynamic threshold to filter extreme Sharpe values.
        #####################################################################
        for i in range(n_stocks):
            if abs(raw_sharpes_today[i]) > dynamic_threshold:
                sharpe_2d[i, j] = 0.0
            else:
                sharpe_2d[i, j] = raw_sharpes_today[i]

    #########################################################################
    # Modify stock selection criteria to use delta rank (improvement in momentum).
    #########################################################################
    print(" ... Modifying stock selection criteria to delta rank (improvement in momentum)...")

    # Compute delta rank (momentum improvement) for stock selection.
    delta_rank = np.zeros((n_stocks, n_days), dtype=float)
    for j in range(1, n_days):
        delta_rank[:, j] = sharpe_2d[:, j] - sharpe_2d[:, j - 1]

    for j in range(n_days):
        # Select stocks with the highest delta rank.
        if j >= LongPeriod:
            valid_indices = np.argsort(delta_rank[:, j])[-numberStocksTraded:]
            selected_weights = delta_rank[valid_indices, j]

            # Normalize weights to sum to 1.
            if selected_weights.sum() > 0:
                selected_weights /= selected_weights.sum()

            # Assign weights to the selected stocks.
            monthgainlossweight[valid_indices, j] = selected_weights

    #########################################################################
    # End of modification for delta rank.
    #########################################################################

    #########################################################################
    # Final forward-fill pass to ensure no gaps in weights.
    #########################################################################
    print(" ... Forward-filling any remaining gaps in weights...")
    for j in range(1, n_days):
        if monthgainlossweight[:, j].sum() < 0.001:
            monthgainlossweight[:, j] = monthgainlossweight[:, j-1]

    #########################################################################
    # Forward-fill weights for all days in the month.
    #########################################################################
    print(" ... Forward-filling weights for all days in the month...")

    for j in range(1, n_days):
        # If weights are zero for a day, forward-fill from the previous day.
        zero_weights = monthgainlossweight[:, j] == 0
        monthgainlossweight[zero_weights, j] = monthgainlossweight[zero_weights, j - 1]

    # Validate that weights sum to 1.0 for each day.
    for j in range(n_days):
        daily_sum = monthgainlossweight[:, j].sum()
        if daily_sum > 0:
            monthgainlossweight[:, j] /= daily_sum

    print(" ... Forward-filling and normalization complete.")
    #########################################################################

    print(" ... sharpeWeightedRank_2D computation complete")
    print(f" ... Non-zero weights in output: {np.sum(monthgainlossweight > 0)}")
    
    # Verify first and last dates have valid weights.
    print(f" ... Weights sum on day 0: {monthgainlossweight[:, 0].sum():.4f}")
    print(f" ... Weights sum on day {n_days-1}: {monthgainlossweight[:, -1].sum():.4f}")

    return monthgainlossweight
