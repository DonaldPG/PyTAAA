import os
import numpy as np
import urllib
import datetime
from bs4 import BeautifulSoup
from matplotlib import pylab as plt

# local imports
from UpdateSymbols_inHDF5 import *
from TAfunctions import *

def getYieldCurve():
    ###
    ### get yield curve (interest rates) from US Treasury website
    ###
    print(" ... Making plot of US Treasury Rates ...")
    #try:
    base_url = 'http://data.treasury.gov/feed.svc/DailyTreasuryYieldCurveRateData'
    content = urllib.urlopen(base_url).read()
    soup = BeautifulSoup( content, 'xml' )
    interestRates = []
    # sample dates about once per month until last 2 years. sample every day of recent 2 years
    daterange = range(0, len(list(soup.find_all('content')))-500,int(365.25/12.))
    daterange = daterange + range( len(list(soup.find_all('content')))-499, len(list(soup.find_all('content'))) )
    for idate in daterange:
        for t in list(soup.find_all('content')[idate].descendants)[1].contents:
            #try:
            if t.name != None:
                #print " ... t.name = ", t.name
                if "NEW_DATE" in t.name:
                    if idate%250 == 0:
                        print(" i, date = ", idate, t.string.split('T')[0])
                    irate_date = datetime.datetime.strptime( t.string.split('T')[0], '%Y-%m-%d')
                    #irate_date = t.string.split('T')[0] )
                elif "1MONTH" in t.name:
                    if t.string == None:
                        intRate = '0'
                    else:
                        intRate = t.string
                    irate_1m = intRate
                elif "3MONTH" in t.name:
                    if t.string == None:
                        intRate = '0'
                    else:
                        intRate = t.string
                    irate_3m = intRate
                elif "6MONTH" in t.name:
                    if t.string == None:
                        intRate = '0'
                    else:
                        intRate = t.string
                    irate_6m = intRate
                elif "1YEAR" in t.name:
                    if t.string == None:
                        intRate = '0'
                    else:
                        intRate = t.string
                    irate_1y = intRate
                elif "2YEAR" in t.name:
                    if t.string == None:
                        intRate = '0'
                    else:
                        intRate = t.string
                    irate_2y = intRate
                elif "3YEAR" in t.name:
                    if t.string == None:
                        intRate = '0'
                    else:
                        intRate = t.string
                    irate_3y = intRate
                elif "5YEAR" in t.name:
                    if t.string == None:
                        intRate = '0'
                    else:
                        intRate = t.string
                    irate_5y = intRate
                elif "7YEAR" in t.name:
                    if t.string == None:
                        intRate = '0'
                    else:
                        intRate = t.string
                    irate_7y = intRate
                elif "10YEAR" in t.name:
                    if t.string == None:
                        intRate = '0'
                    else:
                        intRate = t.string
                    irate_10y = intRate
                elif "20YEAR" in t.name:
                    if t.string == None:
                        intRate = '0'
                    else:
                        intRate = t.string
                    irate_20y = intRate
                elif "30YEAR" in t.name and "BC_30YEARDISPLAY" not in t.name:
                    if t.string == None:
                        intRate = '0'
                    else:
                        intRate = t.string
                    irate_30y = intRate
        try:
            todayRates = np.array( ( irate_date, \
                                      float(irate_1m), \
                                      float(irate_3m), \
                                      float(irate_6m), \
                                      float(irate_1y), \
                                      float(irate_2y), \
                                      float(irate_3y), \
                                      float(irate_5y), \
                                      float(irate_7y), \
                                      float(irate_10y), \
                                      float(irate_20y), \
                                      float(irate_30y) ) )
            #print " idate = ", idate
            if interestRates == []:
                interestRates = todayRates
            else:
                interestRates = np.vstack( (interestRates, np.array(todayRates)) )
        except:
            pass

    # sort the interest rate information in-place using the date (in column 0)
    interestRates = interestRates[interestRates[:,0].argsort()]

    symbol_directory = os.path.join( os.getcwd(), "symbols" )
    symbol_file = "Naz100_Symbols.txt"
    adjClose, symbols, datearray, _, _ = loadQuotes_fromHDF( symbol_file )
    
    plt.figure(1,figsize=(11,9))
    plt.subplot(2,1,1)
    plt.title( 'US Treasury Rates compared to Nasdaq 100 performance' )
    plt.plot( datearray, np.mean(adjClose,axis=0) )
    plt.yscale('log')
    plt.ylim( (20,200) )
    plt.grid(True)
    plt.xlim((datetime.date(1990,1,1),datetime.datetime.now()))
    # put text line with most recent date at bottom of plot
    # - get 5% of x-scale and 70% of y-scale for text location
    x_range = datearray[-1] - datearray[0]
    text_x = datearray[0] + datetime.timedelta( x_range.days / 20. )
    text_y = ( 200 - 20 )* .70 + 20
    lastdate = str( datearray[-1] )
    plt.text( text_x,text_y, "most recent value from "+lastdate+"\nplotted "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"), fontsize=8 )
    plt.subplot(2,1,2)
    plt.grid(True)
    plt.plot( interestRates[:,0], interestRates[:,1],label='1m' )
    plt.plot( interestRates[:,0], interestRates[:,2],label='3m' )
    plt.plot( interestRates[:,0], interestRates[:,3],label='6m' )
    plt.plot( interestRates[:,0], interestRates[:,4],label='1y' )
    plt.plot( interestRates[:,0], interestRates[:,5],label='2y' )
    plt.plot( interestRates[:,0], interestRates[:,6],label='3y' )
    plt.plot( interestRates[:,0], interestRates[:,7],label='5y' )
    plt.plot( interestRates[:,0], interestRates[:,8],label='7y' )
    plt.plot( interestRates[:,0], interestRates[:,9],label='10y' )
    plt.plot( interestRates[:,0], interestRates[:,10],label='20y' )
    plt.plot( interestRates[:,0], interestRates[:,11],label='30y' )
    plt.ylabel('interest rate (%)')
    plt.legend(prop={'size':6})
    plt.grid(True)
    plt.xlim((datetime.date(1990,1,1),datetime.datetime.now()))
    # put text line with most recent date at bottom of plot
    # - get 5% of x-scale and 70% of y-scale for text location
    print("   interestRates[0,0] = ", interestRates[0,0])
    print("   interestRates[-1,0] = ", interestRates[-1,0])
    x_range = interestRates[-1,0] - interestRates[0,0]
    text_x = interestRates[0,0] + datetime.timedelta( x_range.days / 20. )
    text_y = ( interestRates[:,1:].max() - interestRates[:,1:].min() )* .05 + interestRates[:,1:].min()
    lastdate = str( interestRates[-1,0] )
    plt.text( text_x,text_y, "most recent value from "+lastdate+"\nplotted "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"), fontsize=8 )
    #plt.show()
    plt.savefig( os.path.join( os.getcwd(), 'pyTAAA_web', 'pyTAAA_NasdaqVsTreasuries.png' ), dpi=100, format='png' )

    '''
    except:
        pass
    '''
    return interestRates


interestRates = getYieldCurve()
