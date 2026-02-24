"""Statistical analysis utilities for financial time series.

This module provides the ``allstats`` class, which wraps a NumPy array
and exposes a collection of descriptive and risk-adjusted statistics
commonly used in quantitative finance and portfolio analysis.

Example:
    from functions.allstats import allstats
    import numpy as np

    prices = np.cumprod(np.random.normal(1.001, 0.01, 252))
    stats = allstats(prices)
    print("Sharpe:", stats.sharpe())
    print("MAD:", stats.mad())
"""

import numpy as np
import math
from typing import List, Union


class allstats():
    """Container for statistical computations on a 1D numerical array.

    Wraps a NumPy array and provides methods for common descriptive
    statistics, robust dispersion measures, and risk-adjusted return
    metrics used in portfolio analysis.

    Attributes:
        data: The underlying 1D NumPy array supplied at construction.

    Example:
        >>> import numpy as np
        >>> from functions.allstats import allstats
        >>> prices = np.array([100., 102., 101., 105., 103.])
        >>> s = allstats(prices)
        >>> s.mean()
        102.2
    """

    def __init__(self, x: np.ndarray) -> None:
        """Initialise allstats with a 1D array.

        Args:
            x: 1D NumPy array of numerical values (e.g., prices or
                returns) on which statistics will be computed.
        """
        self.data = x

    def sharpe(self, periods_per_year: int = 252) -> float:
        """Calculate the annualised Sharpe ratio from a daily price series.

        Computes the geometric mean daily return minus 1, annualises it,
        and divides by the annualised daily return volatility.  Input is
        treated as a price series; daily returns are derived internally.

        Args:
            periods_per_year: Number of trading periods in a year used
                to annualise the ratio. Defaults to 252 (trading days).

        Returns:
            Annualised Sharpe ratio as a float, or an empty list if the
            data array is empty.

        Notes:
            - Near-zero prices are clamped to 1e-15 to avoid division
              by zero.
            - NaN and infinite return values are replaced with 1.0 (no
              gain/loss) before the ratio is computed.
        """
        x = self.data
        from scipy.stats import gmean
        # The mad of nothing is null
        if len(x) == 0:
            return []
        _x = x.astype('float')
        _x[_x<=0.] = 1.e-15
        dailygainloss = _x[1:] / _x[:-1]
        dailygainloss[ np.isnan(dailygainloss) ] = 1.
        dailygainloss[ np.isinf(dailygainloss) ] = 1.
        # Find the sharpe ratio of the list (assume x is daily prices)
        sharpe =  ( gmean(dailygainloss)**periods_per_year -1. ) / ( np.std(dailygainloss)*np.sqrt(periods_per_year) )
        return sharpe

    def monthly_sharpe(self) -> float:
        """Calculate the monthly Sharpe ratio from a monthly price series.

        Identical in structure to :meth:`sharpe` but uses 12 periods per
        year, making it suitable for monthly price or return data.

        Returns:
            Monthly-annualised Sharpe ratio as a float, or an empty list
            if the data array is empty.

        Notes:
            - Near-zero prices are clamped to 1e-15 to avoid division
              by zero.
            - NaN and infinite return values are replaced with 1.0 before
              the ratio is computed.
        """
        x = self.data
        from scipy.stats import gmean
        # The mad of nothing is null
        if len(x) == 0:
            return []
        _x = x.astype('float')
        _x[_x<=0.] = 1.e-15
        dailygainloss = _x[1:] / _x[:-1]
        dailygainloss[ np.isnan(dailygainloss) ] = 1.
        dailygainloss[ np.isinf(dailygainloss) ] = 1.
        # Find the sharpe ratio of the list (assume x is daily prices)
        sharpe =  ( gmean(dailygainloss)**12 -1. ) / ( np.std(dailygainloss)*np.sqrt(12) )
        return sharpe

    def sortino(self, risk_free_rate: float = 0., target_rate: float = 0.) -> float:
        """Calculate the Sortino ratio from a daily price series.

        The Sortino ratio measures risk-adjusted return using only
        downside volatility (lower partial moment of order 2) rather
        than total volatility, penalising losses more than gains.

        Adapted from turingfinance.com/computational-investing-with-python-week-one/

        Args:
            risk_free_rate: Annual risk-free rate used as the benchmark
                in the numerator. Defaults to 0.
            target_rate: Minimum acceptable return threshold used to
                compute the lower partial moment. Defaults to 0.

        Returns:
            Sortino ratio as a float, or an empty list if the data
            array is empty.

        Notes:
            - Input is treated as a price series; percent returns are
              derived internally.
            - If all returns are positive, the smallest gain is
              sign-reversed so that the lower partial moment is non-zero,
              preventing a division-by-zero.
            - NaN and infinite returns are zeroed before calculation.
        """
        # adapted from www.turingfinance.com/computational-investing-with-python-week-one/
        def lpm(returns, threshold, order):
            """Compute the lower partial moment of a return series.

            Args:
                returns: 1D array of return values.
                threshold: Minimum acceptable return threshold.
                order: Moment order (2 for semi-variance).

            Returns:
                Lower partial moment of the specified order.
            """
            # Create an array he same length as returns containing the minimum return threshold
            threshold_array = np.empty(len(returns))
            threshold_array.fill(threshold)
            # Calculate the difference between the threshold and the returns
            diff = threshold_array - returns
            # Set the minimum of each to 0
            diff = diff.clip(min=0)
            # Return the sum of the different to the power of order
            return np.sum(diff ** order) / len(returns)
        def sortino_ratio(er, returns, rf, target_rate=0):
            """Compute the Sortino ratio from expected return and LPM.

            Args:
                er: Expected (mean) return of the asset.
                returns: 1D array of return values used for LPM.
                rf: Risk-free rate.
                target_rate: Minimum acceptable return for LPM.
                    Defaults to 0.

            Returns:
                Sortino ratio as a float.
            """
            return (er - rf) / math.sqrt(lpm(returns, target_rate, 2))
        x = self.data
        # The mad of nothing is null
        if len(x) == 0:
            return []
        _x = x.astype('float')
        _x[_x<=0.] = 1.e-15
        dailypercentgainloss = _x[1:] / _x[:-1] - 1.
        if dailypercentgainloss.min() > 0.:
            # rescale so that smallest gain has sign reversed
            a_dailypercentgainloss = dailypercentgainloss.copy()
            a_dailypercentgainloss.sort()
            a_dailypercentgainloss[0] *= -1.
            dailypercentgainloss = a_dailypercentgainloss * 1.
            #print('    ... smallest values re-scaled')
        dailypercentgainloss[ np.isnan(dailypercentgainloss) ] = 0.
        dailypercentgainloss[ np.isinf(dailypercentgainloss) ] = 0.
        er = dailypercentgainloss.mean()
        # Find the sharpe ratio of the list (assume x is daily prices)
        sortino = sortino_ratio(er, dailypercentgainloss, risk_free_rate, target_rate=0.)
        return sortino

    def mad(self) -> float:
        """Calculate the Median Absolute Deviation (MAD) of the data.

        MAD is a robust measure of statistical dispersion that is more
        resilient to outliers than standard deviation.  It is defined as
        the median of the absolute deviations from the data's median.

        See Also:
            http://en.wikipedia.org/wiki/Median_absolute_deviation

        Returns:
            Median Absolute Deviation as a float, or an empty list if
            the data array is empty.
        """
        x = self.data
        if len(x) == 0:
            return []

        median_value = np.median(x)
        median_absolute_deviations = []

        # Make a list of absolute deviations from the median
        for i in range( len(x) ):
            median_absolute_deviations.append(np.abs(x[i] - median_value))

        #Find the median value of that list
        return np.median(median_absolute_deviations)

    def std(self) -> float:
        """Calculate the standard deviation of the data.

        Returns:
            Population standard deviation (ddof=0) as a float, or an
            empty list if the data array is empty.
        """
        x = self.data
        if len(x) == 0:
            return []
        return np.std(x)

    def z_score(self) -> np.ndarray:
        """Compute the Z-score (standard score) for each data point.

        The Z-score expresses how many population standard deviations
        each observation lies above or below the population mean.  A
        positive value indicates above-mean; a negative value indicates
        below-mean.

        See Also:
            http://en.wikipedia.org/wiki/Standard_score

        Returns:
            NumPy array of Z-scores with the same shape as the input
            data, or an empty list if the data array is empty.
        """
        x = self.data
        mean = np.mean(x)
        stddev = np.std(x)
        if len(x) == 0:
            return []
        #return np.hstack( ((0), ((x - mean)/stddev)) )
        return (x - mean)/stddev

    def med_score(self) -> np.ndarray:
        """Compute a robust median-based standardised score for each data point.

        Analogous to the Z-score but uses the median and Median Absolute
        Deviation (MAD) instead of mean and standard deviation, making
        it resistant to outliers.

        See Also:
            http://en.wikipedia.org/wiki/Standard_score

        Returns:
            NumPy array of median-based scores with the same shape as
            the input data, or an empty list if the data array is empty.
        """
        x = self.data
        if len(x) == 0:
            return []
        median_value = np.median(x)
        median_absolute_deviations = []
        # Make a list of absolute deviations from the median
        for i in range( len(x) ):
            median_absolute_deviations.append(np.abs(x[i] - median_value))
        mad = np.median(median_absolute_deviations)
        #return np.hstack( ((0), ((x - median_value)/mad)) )
        return (x - median_value)/mad

    def remove_medoutliers(self, num_stds: float = 1.) -> np.ndarray:
        """Return a copy of the data with median-based outliers removed.

        Uses :meth:`med_score` to identify outliers and filters them out.
        Points whose absolute median score exceeds ``num_stds`` times the
        standard deviation of those scores are considered outliers.

        Args:
            num_stds: Multiplier applied to the score's standard deviation
                to set the outlier threshold. Defaults to 1.0.

        Returns:
            1D NumPy array containing only the non-outlier elements, or
            an empty list if the input array is empty.
        """
        x = self.data
        if len(x) == 0:
            return []
        score = allstats(x).med_score()
        score = np.abs( score )
        x_no_outliers = x[score < num_stds*score.std()]
        return x_no_outliers

    def count_medoutliers(self, num_stds: float = 1.) -> int:
        """Count the number of median-based outliers in the data.

        Args:
            num_stds: Multiplier applied to the score's standard deviation
                to set the outlier threshold. Defaults to 1.0.

        Returns:
            Integer count of outlier elements, or an empty list if the
            input array is empty.
        """
        x = self.data
        if len(x) == 0:
            return []
        score = allstats(x).med_score()
        score = np.abs( score )
        outlier_count = x[score > num_stds*score.std()].shape[0]
        return outlier_count

    def return_medoutliers(self, num_stds: float = 1.) -> np.ndarray:
        """Return only the median-based outlier elements from the data.

        Args:
            num_stds: Multiplier applied to the score's standard deviation
                to set the outlier threshold. Defaults to 1.0.

        Returns:
            1D NumPy array of outlier elements, or an empty list if the
            input array is empty.
        """
        x = self.data
        if len(x) == 0:
            return []
        score = allstats(x).med_score()
        score = np.abs( score )
        x_outliers = x[score > num_stds*score.std()]
        return x_outliers

    def return_indices_medoutliers(self, num_stds: float = 1.) -> tuple:
        """Return the array indices of median-based outliers.

        Args:
            num_stds: Multiplier applied to the score's standard deviation
                to set the outlier threshold. Defaults to 1.0.

        Returns:
            Tuple of arrays (as returned by ``np.where``) containing the
            indices of outlier elements, or an empty list if the input
            array is empty.
        """
        x = self.data
        if len(x) == 0:
            return []
        score = allstats(x).med_score()
        score = np.abs( score )
        x_outliers_indices = np.where(score > num_stds*score.std())
        return x_outliers_indices

    def mean(self) -> float:
        """Calculate the arithmetic mean of the data.

        Returns:
            Mean value as a float, or an empty list if the data array
            is empty.
        """
        x = self.data
        if len(x) == 0:
            return []
        return np.mean(x)

    def median(self) -> float:
        """Calculate the median of the data.

        Returns:
            Median value as a float, or an empty list if the data array
            is empty.
        """
        x = self.data
        if len(x) == 0:
            return []
        return np.median(x)




if __name__ == "__main__":
    # do example
    x = np.array( ( 1,4,3,5,6,9,22,453,1,3,5,9,5,4,3,7,4,8,0,12,-12) )
    x = np.random.normal(loc=0.03, scale=1.0, size=1000)
    
    # print("mean = ", allstats(np.diff(x)).mean())
    # print("median = ", allstats(np.diff(x)).median())
    # print("MAD = ", allstats(x).mad())
    # print("sharpe = ", allstats(np.diff(x)).sharpe())
    # print("stddev = ", allstats(np.diff(x)).std())
    # print("Sortino Ratio =", allstats(x).sortino(risk_free_rate=0., target_rate=0.05))
    # print("z_score = ", allstats(np.diff(x)).z_score())
    # print("med_score = ", allstats(np.diff(x)).med_score())
    
    from matplotlib import pylab as plt
    plt.ion()
    
    plt.plot(x)
    plt.plot(allstats(np.diff(x)).z_score())
    plt.plot(allstats(np.diff(x)).med_score())
    plt.grid()
    
    
    # Returns from the portfolio (r) and market (m)
    r = np.random.uniform(0., 2., 5000)
    r = np.random.normal(1.036, .25, 5000)
    dr = np.gradient(r)
    r[dr>0.] = r[dr>0.0] ** 1.05

    # Risk-adjusted return based on Volatility
    print("mean = ", allstats(np.diff(r)).mean())
    print("median = ", allstats(np.diff(r)).median())
    print("MAD = ", allstats(r).mad())
    print("sharpe = ", allstats(np.diff(r)).sharpe())
    print("stddev = ", allstats(np.diff(r)).std())
    print("Sortino Ratio =", allstats(r).sortino(risk_free_rate=0.03, target_rate=0.))
    
    value = r.copy()
    value[0] = 10000.
    value = np.cumprod(value)
    print("final value = " + format(value[-1], '9,.0f'))

    plt.clf()
    plt.grid()
    plt.plot(value, 'k-')
    plt.yscale('log')
