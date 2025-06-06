import numpy as np
import math

class allstats():

    def __init__(self,x):
        self.data = x

    def sharpe(self, periods_per_year=252):
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

    def monthly_sharpe(self):
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

    def sortino(self, risk_free_rate = 0., target_rate=0.):
        # adapted from www.turingfinance.com/computational-investing-with-python-week-one/
        def lpm(returns, threshold, order):
            # This method returns a lower partial moment of the returns
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

    def mad(self):
        # [Median Absolute Deviation](http://en.wikipedia.org/wiki/Median_absolute_deviation)
        #
        # The Median Absolute Deviation (MAD) is a robust measure of statistical
        # dispersion. It is more resilient to outliers than the standard deviation.
        x = self.data
        # The mad of nothing is null
        if len(x) == 0:
            return []

        median_value = np.median(x)
        median_absolute_deviations = []

        # Make a list of absolute deviations from the median
        for i in range( len(x) ):
            median_absolute_deviations.append(np.abs(x[i] - median_value))

        #Find the median value of that list
        return np.median(median_absolute_deviations)

    def std(self):
        x = self.data
        # The mad of nothing is null
        if len(x) == 0:
            return []
        #Find the std dev value of that list
        return np.std(x)

    def z_score(self):
        # [Z-Score, or Standard Score](http://en.wikipedia.org/wiki/Standard_score)
        #
        # The z_score is the number of standard deviations an observation
        # or datum is above or below the mean. Thus, a positive standard score
        # represents a datum above the mean, while a negative standard score
        # represents a datum below the mean. It is a dimensionless quantity
        # obtained by subtracting the population mean from an individual raw
        # score and then dividing the difference by the population standard
        # deviation.
        #
        x = self.data
        mean = np.mean(x)
        stddev = np.std(x)
        # The z_score of nothing is null
        if len(x) == 0:
            return []
        #Find the z_score of that list
        #return np.hstack( ((0), ((x - mean)/stddev)) )
        return (x - mean)/stddev

    def med_score(self):
        # [Z-Score, or Standard Score](http://en.wikipedia.org/wiki/Standard_score)
        #
        # The z_score is the number of standard deviations an observation
        # or datum is above or below the mean. Thus, a positive standard score
        # represents a datum above the mean, while a negative standard score
        # represents a datum below the mean. It is a dimensionless quantity
        # obtained by subtracting the population mean from an individual raw
        # score and then dividing the difference by the population standard
        # deviation.
        #
        x = self.data
        # The z_score of nothing is null
        if len(x) == 0:
            return []
        median_value = np.median(x)
        median_absolute_deviations = []
        # Make a list of absolute deviations from the median
        for i in range( len(x) ):
            median_absolute_deviations.append(np.abs(x[i] - median_value))
        #Find the median value of that list
        mad = np.median(median_absolute_deviations)
        #Find the z_score of that list
        #return np.hstack( ((0), ((x - median_value)/mad)) )
        return (x - median_value)/mad

    def remove_medoutliers(self,num_stds=1.):
        x = self.data
        # The z_score of nothing is null
        if len(x) == 0:
            return []
        score = allstats(x).med_score()
        score = np.abs( score )
        x_no_outliers = x[score < num_stds*score.std()]
        return x_no_outliers

    def count_medoutliers(self,num_stds=1.):
        x = self.data
        # The z_score of nothing is null
        if len(x) == 0:
            return []
        score = allstats(x).med_score()
        score = np.abs( score )
        outlier_count = x[score > num_stds*score.std()].shape[0]
        return outlier_count

    def return_medoutliers(self,num_stds=1.):
        x = self.data
        # The z_score of nothing is null
        if len(x) == 0:
            return []
        score = allstats(x).med_score()
        score = np.abs( score )
        x_outliers = x[score > num_stds*score.std()]
        return x_outliers

    def return_indices_medoutliers(self,num_stds=1.):
        x = self.data
        # The z_score of nothing is null
        if len(x) == 0:
            return []
        score = allstats(x).med_score()
        score = np.abs( score )
        x_outliers_indices = np.where(score > num_stds*score.std())
        return x_outliers_indices

    def mean(self):
        x = self.data
        # The mean of nothing is null
        if len(x) == 0:
            return []
        #Find the mean value of that list
        return np.mean(x)

    def median(self):
        x = self.data
        # The median of nothing is null
        if len(x) == 0:
            return []
        #Find the median value of that list
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
