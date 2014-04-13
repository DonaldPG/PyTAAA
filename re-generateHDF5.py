
import os
import numpy as np

import datetime

import nose
#import bottleneck as bn
#import la
#import h5py

import pandas as pd
from pandas.io.data import get_data_yahoo

from scipy.stats import gmean
#from la.external.matplotlib import quotes_historical_yahoo

## local imports
#from functions.quotes_for_list_adjCloseVol import *
from functions.quotes_for_list_adjClose import *
from functions.TAfunctions import *

#---------------------------------------------
# Re-create the hdf5 file
#---------------------------------------------

##
##  Import list of symbols to process.
##

# read list of symbols from disk.
symbol_file = "Naz100_Symbols.txt"                     # plotmax = 1.e10, runnum = 902   Naz100_Symbols.txt
symbol_directory = os.path.join(os.getcwd(), 'symbols' )
filename = os.path.join(symbol_directory, symbol_file)

(shortname, extension) = os.path.splitext(symbol_file)

# set up to write quotes to disk.

listname = "Naz100_Symbols"

hdf5_directory = os.path.join(os.getcwd(), 'symbols' )
hdf5filename = os.path.join(hdf5_directory, listname + "_.hdf5")

print "symbol_directory = ",symbol_directory
print "symbol_file = ",symbol_file
print "shortname, extension = ",shortname, extension
print "hdf5filename = ",hdf5filename

##
## Get quotes for each symbol in list
## process dates.
## Clean up quotes.
## Make a plot showing all symbols in list
##
'''
firstdate=(1900,1,1)
import datetime
today = datetime.datetime.now()
lastdate = ( today.year, today.month, today.day )
'''
import datetime
firstdate=datetime.date(1900,1,1)
lastdate='2011-11-30'
lastdate='2013-6-1'
today = datetime.date.today()
lastdate = today

adjClose, symbols, datearray = arrayFromQuotesForList(filename, firstdate, lastdate)

print " security values check: ",adjClose[isnan(adjClose)].shape

dates = []
for i in range( len(datearray) ):
    dates.append(str(datearray[i]))
#quoteupdate = la.larry(adjClose, [symbols,dates], dtype=float)

# print first and last dates in dataframe
print "... first and last datearray are: ", datearray[0], datearray[-1]
print "... first and last dates are: ", dates[0], dates[-1]

# create pandas dataframe and write to hdf
quotes_df = pd.DataFrame( adjClose.swapaxes(0,1), index=dates, columns=symbols)
quotes_df.to_hdf( hdf5filename, listname, mode='a',format='table',append=False,complevel=5,complib='blosc')

'''io = la.IO(hdf5filename)
io[listname] = quoteupdate
'''
