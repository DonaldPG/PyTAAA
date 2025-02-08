
import os
import numpy as np

import datetime

import nose
import bottleneck as bn
import la
import h5py

from scipy.stats import gmean
from la.external.matplotlib import quotes_historical_yahoo

## local imports
from functions.quotes_for_list_adjCloseVol import *
from functions.TAfunctions import *

#---------------------------------------------
# Re-create the hdf5 file
#---------------------------------------------

##
##  Import list of symbols to process.
##

# read list of symbols from disk.
symbol_file = "Naz100_symbols.txt"                     # plotmax = 1.e10, runnum = 902   naz100_symbols.txt
symbol_directory = os.path.join(os.getcwd(), 'symbols' )
filename = os.path.join(symbol_directory, symbol_file)

(shortname, extension) = os.path.splitext(symbol_file)

# set up to write quotes to disk.

listname = "Naz100-Symbols"

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

firstdate=(2000,1,1)
import datetime
today = datetime.datetime.now()
lastdate = ( today.year, today.month, today.day )

adjClose, symbols, datearray = arrayFromQuotesForList(filename, firstdate, lastdate)

print " security values check: ",adjClose[isnan(adjClose)].shape

dates = []
for i in range(datearray.shape[0]):
    dates.append(str(datearray[i]))
quoteupdate = la.larry(adjClose, [symbols,dates], dtype=float)

io = la.IO(hdf5filename)
io[listname] = quoteupdate
