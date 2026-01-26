import os
import numpy as np
import datetime
from math import log10,sqrt
from scipy.stats import gmean

try:
    os.chdir(os.path.join(os.path.dirname(__file__),".."))
except:
    pass

from functions.TAfunctions import SMS
from functions.GetParams import get_json_params, get_symbols_file
from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
from functions.allstats import *

#----------------------------------------------

def newHighsAndLows(
        json_fn, num_days_highlow=252,
        num_days_cumu=21,
        HighLowRatio=2.,
        HighPctile=1.,
        HGamma=1.,
        LGamma=1.,
        makeQCPlots=True,
        outputStats=False
):

    ####################################################################
    ###
    ### count new highs and new lows for individual stocks over
    ### period of 'num_days_highlow'. Compute the sum over period of
    ### 'num_days_cumu' and over all stocks.
    ### - can use single values or tuple for num_days_highlow,
    ###   num_days_cumu, and HighLowRatio. Must be same length tuples.
    ### - using tuples invokes looping to calculate with each
    ###   set of parameters, then sum computed from all sets
    ###
    ####################################################################

    ###
    ### retrieve quotes with symbols and dates
    ###

    params = get_json_params(json_fn)
    stockList = params['stockList']
    #stockList = 'Naz100'

    # read list of symbols from disk.
    json_dir = os.path.split(json_fn)[0]
    symbol_directory = os.path.join(json_dir, "symbols")
    if stockList == 'Naz100':
        symbol_file = "Naz100_Symbols.txt"
    elif stockList == 'SP500':
        symbol_file = "SP500_Symbols.txt"
    symbols_file = os.path.join( symbol_directory, symbol_file )

    # get symbols file from json
    symbols_file = get_symbols_file(json_fn)

    print("\n\n")
    print(" ... inside functions/CountNewHighsAndLows/newHighsAndLows.py")
    print(".  . stockList = ", stockList)
    print("   . symbols_file = ", symbols_file)
    print("   . json_fn = ", json_fn)
    print("   . loading quotes from HDF5 file")
    
    if not os.path.isfile(symbols_file):
        raise ValueError("symbols file does not exist")

    adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF(symbols_file, json_fn)

    ###
    ### Count new highs and new lows over lookback period of number_days_highlow
    ### - indices in 2D input array: adjClose[number_symbols,number_days]
    ###

    num_indices_ignored = 500

    print("\n ... num_days_highlow = " + str(num_days_highlow))
    newLows_2D = np.zeros( (adjClose.shape[0],adjClose.shape[1],len(num_days_highlow)), 'float' )
    newHighs_2D = np.zeros_like( newLows_2D )
    for i in range( adjClose.shape[0] ):

        if type(num_days_highlow) == tuple and \
           type(num_days_cumu) == tuple and \
           len(num_days_highlow) == len(num_days_cumu):

            newLows = np.zeros( (adjClose.shape[1],len(num_days_highlow)), 'float' )
            newHighs = np.zeros( (adjClose.shape[1],len(num_days_highlow)), 'float' )

            for k in range(len(num_days_highlow)):
                for j in range( num_days_highlow[k], adjClose.shape[1] ):

                    index_lo = np.argmin( adjClose[i,j-num_days_highlow[k]:j] )
                    index_hi = np.argmax( adjClose[i,j-num_days_highlow[k]:j] )

                    if index_lo == num_days_highlow[k]-1:
                        newLows[j,k] += 1
                    if index_hi == num_days_highlow[k]-1:
                        newHighs[j,k] += HighLowRatio[k]

                # add to sum
                newLows_2D[i,:,k] += newLows[:,k]
                newHighs_2D[i,:,k] += newHighs[:,k]

        else:

            newLows = np.zeros_like( adjClose[i,:] )
            newHighs = np.zeros_like( adjClose[i,:] )

            for j in range( num_days_highlow, adjClose.shape[1] ):
                index_lo = np.argmin( adjClose[i,j-num_days_highlow:j] )
                index_hi = np.argmax( adjClose[i,j-num_days_highlow:j] )

                if j%500==0:
                    print(" ... i,j-num_days_highlow,j = ", i,j-num_days_highlow,j)

                if index_lo == num_days_highlow-1:
                    newLows[j] += 1
                if index_hi == num_days_highlow-1:
                    newHighs[j] += HighLowRatio

                # add to sum
                newLows_2D[i,:] += newLows
                newHighs_2D[i,:] += newHighs

    if type(num_days_highlow) == tuple and \
       type(num_days_cumu) == tuple and \
       len(num_days_highlow) == len(num_days_cumu):

        print("shape of newHighs_2D = ", newHighs_2D.shape)
        sumNewHighs =np.sum(newHighs_2D,axis=0)
        sumNewLows =np.sum(newLows_2D,axis=0)
        print("shape of sumNewHighs = ", sumNewHighs.shape)

        for k in range(len(num_days_cumu)):
            # compute simiple-moving-sum over num_days_cumu
            # - to make it easier to see clusters of new highs and lows
            sumNewHighs[:,k] = SMS(sumNewHighs[:,k],num_days_cumu[k])
            sumNewHighsMax = sumNewHighs[:,k].max()
            sumNewHighs[:,k] = ((sumNewHighs[:,k]/sumNewHighsMax)**(1./HGamma[k]))*sumNewHighsMax
            sumNewHighs[:,k] -= np.percentile(sumNewHighs[num_indices_ignored:,k],HighPctile[k])
            sumNewLows[:,k] = SMS(sumNewLows[:,k],num_days_cumu[k])
            sumNewLowsMax = sumNewLows[:,k].max()
            sumNewLows[:,k] = ((sumNewLows[:,k]/sumNewLowsMax)**(1./LGamma[k]))*sumNewLowsMax

        print("shape of sumNewHighs = ", sumNewHighs.shape)
        sumNewHighs =np.sum(sumNewHighs,axis=-1)
        print("shape of sumNewHighs = ", sumNewHighs.shape)
        sumNewLows = np.sum(sumNewLows,axis=-1)


    else:
        # compute simiple-moving-sum over num_days_cumu
        # - to make it easier to see clusters of new highs and lows
        if i%50==0:
            print("i=",i)
        sumNewHighs =np.sum(newHighs_2D,axis=0)
        sumNewLows =np.sum(newLows_2D,axis=0)

        sumNewHighs = SMS(sumNewHighs,num_days_cumu)
        sumNewHighs = sumNewHighs.max()
        sumNewHighs = ((sumNewHighs/sumNewHighsMax)**(1./HGamma))*sumNewHighsMax
        sumNewHighs -= np.percentile(sumNewHighs[num_indices_ignored:],HighPctile)
        sumNewLows = SMS(sumNewLows,num_days_cumu)
        sumNewLows = ((sumNewLows/sumNewLowsMax)**(1./LGamma))*sumNewLowsMax

    sumNewLows[:num_indices_ignored] = -1e5

    trade_signal = True
    if trade_signal:
        print('working on TradedValue...')
        gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)
        gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
        gainloss[np.isnan(gainloss)]=1.
        value = 10000. * np.cumprod(gainloss,axis=1)   ### used bigger constant for inverse of quotes
        BuyHoldValue = np.average(value,axis=0)

        TradedValue = np.zeros_like( adjClose )
        TradedValue[:,0] += 10000.
        for i in range(adjClose.shape[0]):
            for j in range( 1,gainloss.shape[1] ):
                if sumNewLows[j-1] > sumNewHighs[j-1]:
                    TradedValue[i,j] = TradedValue[i,j-1]
                else:
                    TradedValue[i,j] = TradedValue[i,j-1] * gainloss[i,j]

    sharpe_traded = allstats( np.mean(TradedValue,axis=0) ).sharpe()
    print(" sharpe_traded = ", format(sharpe_traded,'4.2f'))

    print("iterations done")
    if makeQCPlots:

        gainloss = np.ones((adjClose.shape[0],adjClose.shape[1]),dtype=float)

        gainloss[:,1:] = adjClose[:,1:] / adjClose[:,:-1]
        gainloss[np.isnan(gainloss)]=1.
        value = 10000. * np.cumprod(gainloss,axis=1)   ### used bigger constant for inverse of quotes
        BuyHoldValue = np.average(value,axis=0)

        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pylab as plt
        import matplotlib.gridspec as gridspec
        
        # json_dir = os.path.split(json_fn)[0]
        from functions.GetParams import get_webpage_store
        filepath = get_webpage_store(json_fn)
        # filepath = os.path.join(json_dir, "pyTAAA_web" )

        today = datetime.datetime.now()

        plt.clf()
        plt.grid(True)

        # set up to use dates for labels
        xlocs = []
        xlabels = []
        for i in range(1,len(datearray)):
            if datearray[i].year != datearray[i-1].year:
                xlocs.append(datearray[i])
                xlabels.append(str(datearray[i].year))
        #print "xlocs,xlabels = ", xlocs, xlabels
        if len(xlocs) < 12 :
            plt.xticks(xlocs, xlabels)
        else:
            plt.xticks(xlocs[::2], xlabels[::2])

        subplotsize = gridspec.GridSpec(2,1,height_ratios=[5,3])
        plt.subplot(subplotsize[0])
        plt.grid(True)
        plt.plot(datearray,BuyHoldValue,'k-')
        ymin = np.min(BuyHoldValue) * .75
        ymax = np.max(BuyHoldValue) * 1.5
        plt.ylim((ymin,ymax))
        plt.yscale('log')
        plt.title("counts of new highs and new lows"+"\n"+
                  "Uses "+str(num_days_highlow)+" days for window with "+
                  str(num_days_cumu)+" days sum for display", fontsize=11)

        # put text with stock list used at top left of plot
        # put text line with most recent date at bottom of plot
        # - get 7.5% of x-scale and y-scale for text location
        x_range = datearray[-1] - datearray[0]
        plotrange = log10(ymax / ymin)
        print("x,y for text = ", datearray[50],ymin*10**(.95*plotrange))
        plt.text(datearray[50],ymin*10**(.95*plotrange),stockList,fontsize=9)

        text_x = datearray[0] + datetime.timedelta( x_range.days / 20. )
        text_y = ymin*10**(.03*plotrange)
        plot_text = "most recent value from "+str(datearray[-1])+\
                    "\nplotted at "+today.strftime("%A, %d. %B %Y %I:%M%p")
        plt.text( text_x,text_y, plot_text, fontsize=8 )

        plt.plot(datearray,np.mean(TradedValue,axis=0),'b-')

        text_y1 = ymin*10**(.90*plotrange)
        plot_text1 = "num_days_highlow = ", num_days_highlow
        plt.text( text_x,text_y1, plot_text1, fontsize=8 )
        text_y2 = ymin*10**(.85*plotrange)
        plot_text2 = "num_days_cumu = ", num_days_cumu
        plt.text( text_x,text_y2, plot_text2, fontsize=8 )
        text_y3 = ymin*10**(.80*plotrange)
        plot_text3 = "HighLowRatio = ", HighLowRatio
        plt.text( text_x,text_y3, plot_text3, fontsize=8 )
        text_y4 = ymin*10**(.75*plotrange)
        plot_text4 = "HighPctile = ", HighPctile
        plt.text( text_x,text_y4, plot_text4, fontsize=8 )
        text_y5 = ymin*10**(.70*plotrange)
        plot_text5 = "HGamma = ", HGamma
        plt.text( text_x,text_y5, plot_text5, fontsize=8 )
        text_y5a = ymin*10**(.65*plotrange)
        plot_text5a = "LGamma = ", LGamma
        plt.text( text_x,text_y5a, plot_text5a, fontsize=8 )
        text_y6 = ymin*10**(.10*plotrange)
        plot_text6 = "traded sharpe = ", format(sharpe_traded,'4.2f')
        plt.text( datearray[-2500],text_y6, plot_text6, fontsize=10 )

        plt.subplot(subplotsize[1])
        plt.grid(True)

        plt.plot(datearray,sumNewHighs,'g-',label='new highs')
        plt.plot(datearray,sumNewLows,'r-',label='new lows')
        ymax = max(np.max(sumNewHighs),np.max(sumNewLows))
        #print "ymax=",ymax
        plt.ylim((0,ymax))
        plt.legend(loc=2,fontsize=9)

        plotfilepath = os.path.join( filepath, "PyTAAA_newHighs_newLows_count__"+today.strftime("%Y-%m-%d-%I.%M.%S%p" ) )
        #plotfilepath = plotfilepath.replace(":","-")
        #plotfilepath = plotfilepath.replace(".","-")
        #plotfilepath = plotfilepath.replace("C-","C:")
        plotfilepath = plotfilepath+"_"+stockList+".png"
        print("plotfilepath = ", plotfilepath)
        plt.savefig( plotfilepath, format='png' )
        plotfilepath = os.path.join( filepath, "PyTAAA_newHighs_newLows_count.png" )
        plt.savefig( plotfilepath, format='png' )

    if outputStats :

        outputfilepath = "CountNewHighsLows_stats.csv"
        from functions.GetParams import get_webpage_store
        webpage_dir = get_webpage_store(json_fn)
        output_file = os.path.join(webpage_dir, outputfilepath )
        PortfolioValue = np.mean(TradedValue,axis=0)

        PortfolioDailyGains = PortfolioValue[1:] / PortfolioValue[:-1]
        VarPctSharpe18Yr = ( gmean(PortfolioDailyGains[-18*252:])**252 -1. ) / ( np.std(PortfolioDailyGains[-18*252:])*sqrt(252) )
        VarPctSharpe10Yr = ( gmean(PortfolioDailyGains[-2520:])**252 -1. ) / ( np.std(PortfolioDailyGains[-2520:])*sqrt(252) )
        VarPctSharpe5Yr = ( gmean(PortfolioDailyGains[-1260:])**252 -1. ) / ( np.std(PortfolioDailyGains[-1260:])*sqrt(252) )
        VarPctSharpe3Yr = ( gmean(PortfolioDailyGains[-756:])**252 -1. ) / ( np.std(PortfolioDailyGains[-756:])*sqrt(252) )
        VarPctSharpe2Yr = ( gmean(PortfolioDailyGains[-504:])**252 -1. ) / ( np.std(PortfolioDailyGains[-504:])*sqrt(252) )
        VarPctSharpe1Yr = ( gmean(PortfolioDailyGains[-252:])**252 -1. ) / ( np.std(PortfolioDailyGains[-252:])*sqrt(252) )

        VarPctReturn18Yr = (PortfolioValue[-1] / PortfolioValue[-18*252])**(1/18.)
        VarPctReturn10Yr = (PortfolioValue[-1] / PortfolioValue[-2520])**(1/10.)
        VarPctReturn5Yr = (PortfolioValue[-1] / PortfolioValue[-1260])**(1/5.)
        VarPctReturn3Yr = (PortfolioValue[-1] / PortfolioValue[-756])**(1/3.)
        VarPctReturn2Yr = (PortfolioValue[-1] / PortfolioValue[-504])**(1/2.)
        VarPctReturn1Yr = (PortfolioValue[-1] / PortfolioValue[-252])

        MaxPortfolioValue = PortfolioValue * 0.
        for jj in range(PortfolioValue.shape[0]):
            MaxPortfolioValue[jj] = max(MaxPortfolioValue[jj-1],PortfolioValue[jj])
        PortfolioDrawdown = PortfolioValue / MaxPortfolioValue - 1.
        VarPctDrawdown18Yr = np.mean(PortfolioDrawdown[-18*252:])
        VarPctDrawdown10Yr = np.mean(PortfolioDrawdown[-2520:])
        VarPctDrawdown5Yr = np.mean(PortfolioDrawdown[-1260:])
        VarPctDrawdown3Yr = np.mean(PortfolioDrawdown[-756:])
        VarPctDrawdown2Yr = np.mean(PortfolioDrawdown[-504:])
        VarPctDrawdown1Yr = np.mean(PortfolioDrawdown[-252:])


        if os.path.isfile(output_file):
            textmessage = stockList + "," +\
                      today.strftime("%Y-%m-%d-%I.%M.%S%p") + "," +\
                      str(num_days_highlow) + "," +\
                      str(num_days_cumu) + "," +\
                      str(HighLowRatio) + "," +\
                      str(HighPctile) + "," +\
                      str(HGamma) + "," +\
                      str(LGamma) + "," +\
                      format(np.mean(TradedValue,axis=0)[-1],'9.0f') + "," +\
                      format(sharpe_traded,'4.2f') + "," +\
                      format(VarPctSharpe18Yr,'4.2f') + "," +\
                      format(VarPctSharpe10Yr,'4.2f') + "," +\
                      format(VarPctSharpe5Yr,'4.2f') + "," +\
                      format(VarPctSharpe3Yr,'4.2f') + "," +\
                      format(VarPctSharpe2Yr,'4.2f') + "," +\
                      format(VarPctSharpe1Yr,'4.2f') + "," +\
                      format(VarPctReturn18Yr,'4.2f') + "," +\
                      format(VarPctReturn10Yr,'4.2f') + "," +\
                      format(VarPctReturn5Yr,'4.2f') + "," +\
                      format(VarPctReturn3Yr,'4.2f') + "," +\
                      format(VarPctReturn2Yr,'4.2f') + "," +\
                      format(VarPctReturn1Yr,'4.2f') + "," +\
                      format(VarPctDrawdown18Yr,'4.2f') + "," +\
                      format(VarPctDrawdown10Yr,'4.2f') + "," +\
                      format(VarPctDrawdown5Yr,'4.2f') + "," +\
                      format(VarPctDrawdown3Yr,'4.2f') + "," +\
                      format(VarPctDrawdown2Yr,'4.2f') + "," +\
                      format(VarPctDrawdown1Yr,'4.2f') + "," +\
                      "\n"
        else:
            textmessage = 'stock list,timestamp,num day hi and low,num days for SMA,'+\
                          'high over low ratio,high percentile to subtract,'+\
                          'gamma for new lows,final traded value,'+\
                          'sharpe for traded values,'+\
                          'sharpe 18 yr,sharpe 10 yr,sharpe 5 yr,sharpe 3 yr,sharpe 2 yr,sharpe 1 yr,'+\
                          'return 18 yr,return 10 yr,return 5 yr,return 3 yr,return 2 yr,return 1 yr,'+\
                          'drawdown 18 yr,drawdown 10 yr,drawdown 5 yr,drawdown 3 yr,drawdown 2 yr,drawdown 1 yr,'+\
                          "\n"+\
                      stockList + "," +\
                      today.strftime("%Y-%m-%d-%I.%M.%S%p") + "," +\
                      str(num_days_highlow) + ","+\
                      str(num_days_cumu) + "," +\
                      str(HighLowRatio) + "," +\
                      str(HighPctile) + "," +\
                      str(HGamma) + "," +\
                      str(LGamma) + "," +\
                      format(np.mean(TradedValue,axis=0)[-1],'9.0f') + "," +\
                      format(sharpe_traded,'4.2f') + ","+\
                      format(VarPctSharpe18Yr,'4.2f') + ","+\
                      format(VarPctSharpe10Yr,'4.2f') + ","+\
                      format(VarPctSharpe5Yr,'4.2f') + ","+\
                      format(VarPctSharpe3Yr,'4.2f') + ","+\
                      format(VarPctSharpe2Yr,'4.2f') + ","+\
                      format(VarPctSharpe1Yr,'4.2f') + ","+\
                      format(VarPctReturn18Yr,'4.2f') + ","+\
                      format(VarPctReturn10Yr,'4.2f') + ","+\
                      format(VarPctReturn5Yr,'4.2f') + ","+\
                      format(VarPctReturn3Yr,'4.2f') + ","+\
                      format(VarPctReturn2Yr,'4.2f') + ","+\
                      format(VarPctReturn1Yr,'4.2f') + ","+\
                      format(VarPctDrawdown18Yr,'4.2f') + ","+\
                      format(VarPctDrawdown10Yr,'4.2f') + ","+\
                      format(VarPctDrawdown5Yr,'4.2f') + ","+\
                      format(VarPctDrawdown3Yr,'4.2f') + ","+\
                      format(VarPctDrawdown2Yr,'4.2f') + ","+\
                      format(VarPctDrawdown1Yr,'4.2f') + ","+\
                      "\n"
        with open( output_file, "a" ) as f:
            f.write(textmessage)

    return sumNewHighs, sumNewLows, np.mean(TradedValue,axis=0)[-1]

def HighLowIterate(iterations=100):
    import random
    finalValue = np.zeros((iterations),'float')
    times = []
    params = []
    for iter in range(iterations):
        print("\n*****************")
        print("* starting iteration ", iter, " of ", iterations)
        print("*****************")
        long_days1 = int(random.triangular(21,63,301))
        long_days2 = int(random.triangular(21,63,301))
        short_days1 = int(random.triangular(11,42,6*21))
        short_days2 = int(random.triangular(11,42,6*21))
        ratio1 = random.triangular(.2,.66,3.)
        ratio2 = random.triangular(.2,.66,3.)

        # params for Naz100
        long_days1 = int(random.triangular(45,86,125))
        long_days2 = int(random.triangular(135,275,375))
        short_days1 = int(random.triangular(27,54,80))
        short_days2 = int(random.triangular(75,150,180))
        ratio1 = random.triangular(.8,2.,2.65)
        ratio2 = random.triangular(.85,1.9,2.65)
        HighPctile1 = random.triangular(4.8,9.8,13)
        HighPctile2 = random.triangular(4.2,9.8,13)
        Gamma1 = random.triangular(.7,1.42, 1.8)
        Gamma2 = random.triangular(.95,1.46, 1.8)

        # params for SP500
        '''
        long_days1 = int(random.uniform(45,125) )
        long_days2 = int(random.uniform(140,375))
        short_days1 = int(random.uniform(25,85))
        short_days2 = int(random.uniform(70,170))
        ratio1 = random.uniform(.9,2.6)
        ratio2 = random.uniform(.9,2.4)
        HighPctile1 = random.uniform(4.5,13)
        HighPctile2 = random.uniform(4.5,13)
        HGamma1 = random.uniform(.4,1.9)
        HGamma2 = random.uniform(.4,1.9)
        LGamma1 = random.uniform(.4,1.9)
        LGamma2 = random.uniform(.6,1.9)
		'''
        '''
		elif params['stockList'] == 'SP500':
			_, _, _ = newHighsAndLows( num_days_highlow=(73,146),\
									num_days_cumu=(76,108),\
									HighLowRatio=(2.293,1.573),\
									HighPctile=(12.197,11.534),\
									HGamma=(1.157,.568),\
									LGamma=(.667,1.697),\
									makeQCPlots=True)
        '''
        '''
        long_days1 = int(random.uniform(45,125) )
        long_days2 = int(random.uniform(140,375))
        short_days1 = int(random.uniform(25,85))
        short_days2 = int(random.uniform(70,170))
        ratio1 = random.uniform(.9,2.6)
        ratio2 = random.uniform(.9,2.4)
        HighPctile1 = random.uniform(4.5,13)
        HighPctile2 = random.uniform(4.5,13)
        HGamma1 = random.uniform(.4,1.9)
        HGamma2 = random.uniform(.4,1.9)
        LGamma1 = random.uniform(.4,1.9)
        LGamma2 = random.uniform(.6,1.9)
		'''
        center_value = 73
        long_days1 = int(random.triangular(center_value-20,center_value,center_value+20))
        center_value = 146
        long_days2 = int(random.triangular(center_value-20,center_value,center_value+20))
        center_value = 76
        short_days1 = int(random.triangular(center_value-30,center_value,center_value+20))
        center_value = 108
        short_days2 = int(random.triangular(center_value-40,center_value,center_value+20))
        center_value = 2.293
        ratio1 = random.triangular(.85*center_value, center_value, 1.15*center_value)
        center_value = 1.573
        ratio2 = random.triangular(.85*center_value, center_value, 1.15*center_value)
        center_value = 12.197
        HighPctile1 = random.triangular(.85*center_value, center_value, 1.15*center_value)
        center_value = 11.534
        HighPctile2 = random.triangular(.85*center_value, center_value, 1.15*center_value)
        center_value = 1.157
        HGamma1 = random.triangular(.85*center_value, center_value, 1.15*center_value)
        center_value = 0.568
        HGamma2 = random.triangular(.85*center_value, center_value, 1.15*center_value)
        center_value = 0.667
        LGamma1 = random.triangular(.85*center_value, center_value, 1.15*center_value)
        center_value = 1.697
        LGamma2 = random.triangular(.85*center_value, center_value, 1.15*center_value)

        _,_, finalValue[iter] = newHighsAndLows(num_days_highlow=(long_days1,long_days2),\
                    num_days_cumu=(short_days1,short_days2),\
                    HighLowRatio=(ratio1,ratio2),\
                    HighPctile=(HighPctile1,HighPctile2),\
                    HGamma=(HGamma1,HGamma2),\
                    LGamma=(LGamma1,LGamma2),\
                    makeQCPlots=True,\
                    outputStats=True)
        params.append( [long_days1,long_days2,short_days1,short_days2,ratio1,ratio2,HighPctile1,HighPctile2,Gamma1,Gamma2] )
        today=datetime.datetime.now()
        times.append( today.strftime("%Y-%m-%d-%I.%M.%S%p" ) )
    return params, times, finalValue


if __name__ == "__main__":

    num_days_highlow = 73
    num_days_cumu = 76
    HighLowRatio = 2.293
    HighPctile = 12.197
    HGamma = 1.157
    LGamma = .667

    for i in range(250):
        num_days_highlow1 = int(np.random.triangular(20, 250, 300))
        num_days_highlow2 = int(np.random.triangular(20, 250, 300))
        if num_days_highlow1 > num_days_highlow2:
           num_days_highlow1, num_days_highlow2 = num_days_highlow2, num_days_highlow1
        num_days_cumu1 = int(np.random.triangular(20, 250, 300))
        num_days_cumu2 = int(np.random.triangular(20, 250, 300))
        if num_days_cumu1 > num_days_cumu2:
           num_days_cumu1, num_days_cumu2 = num_days_cumu2, num_days_cumu1
        HighLowRatio1 = np.random.triangular(1.5, 2.3, 2.8)
        HighLowRatio2 = np.random.triangular(1.5, 2.3, 2.8)
        HighPctile1 = np.random.triangular(6., 12., 25.)
        HighPctile2 = np.random.triangular(6., 12., 25.)
        HGamma1 = np.random.triangular(.7, 1.0, 1.4)
        HGamma2 = np.random.triangular(.7, 1.0, 1.4)
        LGamma1 = np.random.triangular(.4, .7, 1.1)
        LGamma2 = np.random.triangular(.4, .7, 1.1)
        num_days_highlow=(num_days_highlow1, num_days_highlow2)
        num_days_cumu=(num_days_cumu1, num_days_cumu2)
        HighLowRatio=(HighLowRatio1, HighLowRatio2)
        HighPctile=(HighPctile1, HighPctile2)
        HGamma=(HGamma1, HGamma2)
        LGamma=(LGamma1, LGamma2)
        params, _, finalValue = newHighsAndLows( num_days_highlow=num_days_highlow,\
                            num_days_cumu=num_days_cumu,\
                            HighLowRatio=HighLowRatio,\
                            HighPctile=HighPctile,\
                            HGamma=HGamma,\
                            LGamma=LGamma,\
                            makeQCPlots=True,\
                            outputStats=True)
        print(" i, finalValue = " + str((i, finalValue, (num_days_highlow,
                                                         num_days_cumu,
                                                         HighLowRatio,
                                                         HighPctile,
                                                         HGamma,
                                                         LGamma
                                                         ))))
