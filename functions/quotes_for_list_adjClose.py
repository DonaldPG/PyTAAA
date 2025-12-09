import numpy as np
import datetime

import datetime
from numpy import random
from scipy.stats import rankdata

from functions.quotes_adjClose import downloadQuotes
from functions.TAfunctions import (interpolate,
                        cleantobeginning,
                        cleantoend)
# from functions.readSymbols import readSymbolList
from functions.readSymbols import read_symbols_list_web
#from functions.quotes_adjClose_alphavantage import get_last_quote


# import pandas as pd
# class webpage_companies_extractor:
#     Url = None

#     def __init__(self, url):
#         self.__url = url

#     def get_companies_list(self, current_portfolio=None):
#         dfs = pd.read_html(self.__url, header=0)
#         first_table = dfs[2]
#         company_names = first_table
#         symbols = list(company_names['Ticker'].values)
#         companyNames = list(company_names['Company'].values)
#         return symbols, companyNames


# '''
# def get_SP500List(symbols_file, verbose=True):
#     ###
#     ### Query wikipedia.com for updated list of stocks in S&P 500 index.
#     ### Return list with stock tickers.
#     ###
#     #import urllib2
#     import urllib.request, urllib.error, urllib.parse
#     import requests
#     import re
#     from bs4 import BeautifulSoup
#     import os
#     import datetime
#     import unicodedata

#     ###
#     ### get symbol list from previous period
#     ###
#     # symbol_directory = os.path.join( os.getcwd(), "symbols" )
#     symbol_directory, symbol_file = os.path.split(symbols_file)
#     # symbol_file = "SP500_Symbols.txt"
#     # symbols_file = os.path.join( symbol_directory, symbol_file )

#     with open(symbols_file, "r+") as f:
#         old_symbolList = f.readlines()
#     for i in range( len(old_symbolList) ) :
#         old_symbolList[i] = old_symbolList[i].replace("\n","")

#     ###
#     ### get current symbol list from wikipedia website
#     ###

#     base_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
#     #content = urllib2.urlopen(base_url)
#     #c = content.read()
#     #print "\n\n\n content = ", content
#     #print "... got web content"

#     #soup = BeautifulSoup( urllib2.urlopen(base_url), "lxml" )
#     soup = BeautifulSoup( urllib.request.urlopen(base_url), "lxml" )
#     #soup = BeautifulSoup(content.read())
#     t = soup.find("table", {"class" : "wikitable sortable"})

#     print("... got web content")
#     print("... ran beautiful soup on web content")

#     """
#     symbolList = [] # store all of the records in this list
#     companyNamesList = []
#     industry = []
#     subIndustry = []
#     for row in t.findAll('tr'):
#         try:
#             col = row.findAll('td')
#             #print "\ncol = ", col
#             #
#             try:
#                 _ticker = col[0].string.strip()
#                 _ticker = _ticker.replace(".","-")
#             except:
#                 _ticker = str(col[0]).split(":")[-1].split('"')[0]
#                 _ticker = _ticker.split("/")[-1].upper()
#             try:
#                 _company = col[1].string.strip()
#             except:
#                 _company = str(col[1]).split('title="')[-1].split('">')[0]
#             #

#             # _company = col[0].string.strip()
#             # _company = _company.replace(".","-")
#             # try:
#             #     _ticker = col[1].string.strip()
#             # except:
#             #     _ticker = str(col[1]).split('title="')[-1].split('">')[0]

#             _sector = col[3].string.strip()
#             _subIndustry = col[4].string.strip()
#             symbolList.append(_ticker)
#             companyNamesList.append(_company)
#             industry.append(_sector)
#             subIndustry.append(_subIndustry)
#         except:
#             #print "/n could not add this row to SP500_Symbols.txt"
#             #print row
#             #print ""
#             #import pdb
#             #pdb.set_trace()
#             pass
#     """

#     symbolList = [] # store all of the records in this list
#     companyNamesList = []
#     data=[]
#     for row in t.findAll('tr'):
#         if str(row)==[]:
#             continue
#         col = row.findAll('td')
#         cols = row.find_all('td')
#         cols = [ele.text.strip() for ele in cols]
#         data.append([ele for ele in cols if ele])
#         if col==[]:
#             continue
#         company_name = data[-1][0].encode("utf8")
#         if company_name[:3]=='MON':
#             break
#         company_name = unicodedata.normalize('NFD',data[-1][1]).encode('ascii', 'ignore').decode('ascii')
#         symbol_name = data[-1][0].encode("utf8").decode('ascii')

#         #print(str(row)+"\n ...company_name = "+company_name+"\n ...symbol_name="+symbol_name+"\n")
#         print("  ...symbol_name="+symbol_name + " ...company_name = "+company_name)
#         companyNamesList.append(company_name)
#         symbolList.append(symbol_name)
#     print("... retrieved SP500 companies lists from internet")

#     companyName_file = os.path.join( symbol_directory, "SP500_companyNames.txt" )
#     with open( companyName_file, "w" ) as f:
#         for i in range( len(symbolList) ) :
#             f.write( symbolList[i] + ";" + companyNamesList[i] + "\n" )
#     print("... wrote SP500_companyNames.txt")

#     ###
#     ### compare old list with new list and print changes, if any
#     ###

#     # file for index changes history
#     symbol_change_file = "SP500_symbolsChanges.txt"
#     symbols_changes_file = os.path.join( symbol_directory, symbol_change_file )
#     if not os.path.isfile(symbols_changes_file):
#         open(symbols_changes_file, 'a').close()
#     with open(symbols_changes_file, "r+") as f:
#         old_symbol_changesList = f.readlines()
#     old_symbol_changesListText = ''
#     for i in range( len(old_symbol_changesList) ):
#         old_symbol_changesListText = old_symbol_changesListText + old_symbol_changesList[i]

#     # parse date
#     year = datetime.datetime.now().year
#     month = datetime.datetime.now().month
#     day = datetime.datetime.now().day
#     dateToday = str(year)+"-"+str(month)+"-"+str(day)

#     # compare lists to check for tickers removed from the index
#     # - printing will be suppressed if "verbose = False"
#     removedTickers = []
#     removedTickersText = ''
#     print("")
#     for i, ticker in enumerate( old_symbolList ):
#         if i == 0:
#             removedTickersText = ''
#         if ticker not in symbolList:
#             removedTickers.append( ticker )
#             if verbose:
#                 print(" Ticker ", ticker, " has been removed from the SP500 index")
#             removedTickersText = removedTickersText + "\n" + dateToday + " Remove " + ticker

#     # compare lists to check for tickers added to the index
#     # - printing will be suppressed if "verbose = False"
#     addedTickers = []
#     print("")
#     for i, ticker in enumerate( symbolList ):
#         if i == 0:
#             addedTickersText = ''
#         if ticker not in old_symbolList:
#             addedTickers.append( ticker )
#             if verbose:
#                 print(" Ticker ", ticker, " has been added to the SP500 index")
#             addedTickersText = addedTickersText + "\n" + dateToday + " Add    " + ticker

#     print("")
#     with open(symbols_changes_file, "w") as f:
#         f.write(addedTickersText)
#         f.write(removedTickersText)
#         f.write("\n")
#         f.write(old_symbol_changesListText)

#     print("****************")
#     print("addedTickers = ", addedTickers)
#     print("removedTickers = ", removedTickers)
#     print("****************")
#     ###
#     ### update symbols file with current list. Keep copy of of list.
#     ###

#     if removedTickers != [] or addedTickers != []:

#         # make copy of previous symbols list file
#         symbol_directory = os.path.join( os.getcwd(), "symbols" )
#         symbol_file = "SP500_Symbols.txt"
#         archive_symbol_file = "SP500_symbols__" + str(datetime.date.today()) + ".txt"
#         symbols_file = os.path.join( symbol_directory, symbol_file )
#         archive_symbols_file = os.path.join( symbol_directory, archive_symbol_file )

#         with open( archive_symbols_file, "w" ) as f:
#             for i in range( len(old_symbolList) ) :
#                 f.write( old_symbolList[i] + "\n" )

#         # make new symbols list file
#         with open( symbols_file, "w" ) as f:
#             for i in range( len(symbolList) ) :
#                 f.write( symbolList[i] + "\n" )

#     return symbolList, removedTickers, addedTickers
# '''

# def get_SP500List(symbols_file, verbose=True ):
#     ###
#     ### Query wikipedia.com for updated list of stocks in S&P 500 index.
#     ### Return list with stock tickers.
#     ###
#     #import urllib2
#     import urllib.request, urllib.error, urllib.parse
#     import requests
#     import re
#     from bs4 import BeautifulSoup
#     import os
#     import datetime
#     import unicodedata

#     ###
#     ### get symbol list from previous period
#     ###
#     # symbol_directory = os.path.join( os.getcwd(), "symbols" )

#     # symbol_file = "SP500_Symbols.txt"
#     # symbols_file = os.path.join( symbol_directory, symbol_file )
#     symbol_directory, symbol_file = os.path.split(symbols_file)

#     with open(symbols_file, "r+") as f:
#         old_symbolList = f.readlines()
#     for i in range( len(old_symbolList) ) :
#         old_symbolList[i] = old_symbolList[i].replace("\n","")

#     ###
#     ### get current symbol list from wikipedia website
#     ###

#     base_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
#     #content = urllib2.urlopen(base_url)
#     #c = content.read()
#     #print "\n\n\n content = ", content
#     #print "... got web content"

#     #soup = BeautifulSoup( urllib2.urlopen(base_url), "lxml" )
#     soup = BeautifulSoup( urllib.request.urlopen(base_url), "lxml" )
#     #soup = BeautifulSoup(content.read())
#     t = soup.find("table", {"class" : "wikitable sortable"})

#     print("... got web content")
#     print("... ran beautiful soup on web content")

#     symbolList = [] # store all of the records in this list
#     companyNamesList = []
#     data=[]
#     for row in t.findAll('tr'):
#         if str(row)==[]:
#             continue
#         col = row.findAll('td')
#         cols = row.find_all('td')
#         cols = [ele.text.strip() for ele in cols]
#         data.append([ele for ele in cols if ele])
#         if col==[]:
#             continue
#         company_name = data[-1][0].encode("utf8")
#         if company_name[:3]=='MON':
#             break
#         company_name = unicodedata.normalize('NFD',data[-1][1]).encode('ascii', 'ignore').decode('ascii')
#         symbol_name = data[-1][0].encode("utf8").decode('ascii')

#         #print(str(row)+"\n ...company_name = "+company_name+"\n ...symbol_name="+symbol_name+"\n")
#         print("  ...symbol_name="+symbol_name + " ...company_name = "+company_name)
#         companyNamesList.append(company_name)
#         symbolList.append(symbol_name)
#     print("... retrieved SP500 companies lists from internet")

#     companyName_file = os.path.join( symbol_directory, "SP500_companyNames.txt" )
#     with open( companyName_file, "w" ) as f:
#         for i in range( len(symbolList) ) :
#             f.write( symbolList[i] + ";" + companyNamesList[i] + "\n" )
#     print("... wrote SP500_companyNames.txt")

#     ###
#     ### compare old list with new list and print changes, if any
#     ###

#     # file for index changes history
#     symbol_change_file = "SP500_symbolsChanges.txt"
#     symbols_changes_file = os.path.join( symbol_directory, symbol_change_file )
#     if not os.path.isfile(symbols_changes_file):
#         open(symbols_changes_file, 'a').close()
#     with open(symbols_changes_file, "r+") as f:
#         old_symbol_changesList = f.readlines()
#     old_symbol_changesListText = ''
#     for i in range( len(old_symbol_changesList) ):
#         old_symbol_changesListText = old_symbol_changesListText + old_symbol_changesList[i]

#     # parse date
#     year = datetime.datetime.now().year
#     month = datetime.datetime.now().month
#     day = datetime.datetime.now().day
#     dateToday = str(year)+"-"+str(month)+"-"+str(day)

#     # compare lists to check for tickers removed from the index
#     # - printing will be suppressed if "verbose = False"
#     removedTickers = []
#     removedTickersText = ''
#     print("")
#     for i, ticker in enumerate( old_symbolList ):
#         if i == 0:
#             removedTickersText = ''
#         if ticker not in symbolList:
#             removedTickers.append( ticker )
#             if verbose:
#                 print(" Ticker ", ticker, " has been removed from the SP500 index")
#             removedTickersText = removedTickersText + "\n" + dateToday + " Remove " + ticker

#     # compare lists to check for tickers added to the index
#     # - printing will be suppressed if "verbose = False"
#     addedTickers = []
#     print("")
#     for i, ticker in enumerate( symbolList ):
#         if i == 0:
#             addedTickersText = ''
#         if ticker not in old_symbolList:
#             addedTickers.append( ticker )
#             if verbose:
#                 print(" Ticker ", ticker, " has been added to the SP500 index")
#             addedTickersText = addedTickersText + "\n" + dateToday + " Add    " + ticker

#     print("")
#     with open(symbols_changes_file, "w") as f:
#         f.write(addedTickersText)
#         f.write(removedTickersText)
#         f.write("\n")
#         f.write(old_symbol_changesListText)

#     print("****************")
#     print("addedTickers = ", addedTickers)
#     print("removedTickers = ", removedTickers)
#     print("****************")
#     ###
#     ### update symbols file with current list. Keep copy of of list.
#     ###

#     if removedTickers != [] or addedTickers != []:

#         # make copy of previous symbols list file
#         json_dir = os.path.split(json_fn)[0]
#         symbol_directory = os.path.join( os.getcwd(), "symbols" )
#         symbol_file = "SP500_Symbols.txt"
#         archive_symbol_file = "SP500_symbols__" + str(datetime.date.today()) + ".txt"
#         symbols_file = os.path.join( symbol_directory, symbol_file )
#         archive_symbols_file = os.path.join( symbol_directory, archive_symbol_file )

#         with open( archive_symbols_file, "w" ) as f:
#             for i in range( len(old_symbolList) ) :
#                 f.write( old_symbolList[i] + "\n" )

#         # make new symbols list file
#         with open( symbols_file, "w" ) as f:
#             for i in range( len(symbolList) ) :
#                 f.write( symbolList[i] + "\n" )

#     return symbolList, removedTickers, addedTickers



# def strip_accents(text):
#     import unicodedata
#     try:
#         text = unicode(text, 'utf-8')
#     except NameError: # unicode is a default on python 3
#         pass
#     try:
#         text = text.decode('ascii')
#     except:
#         pass
#     text = unicodedata.normalize('NFD', text)\
#            .encode('ascii', 'ignore')\
#            .decode("utf-8")
#     return str(text)


# '''
# def get_Naz100List(symbols_file, verbose=True ):
#     ###
#     ### Query nasdaq.com for updated list of stocks in Nasdaq 100 index.
#     ### Return list with stock tickers.
#     ###
#     #import urllib
#     import urllib.request, urllib.parse, urllib.error
#     from bs4 import BeautifulSoup
#     import os

#     ###
#     ### get symbol list from previous period
#     ###
#     # symbol_directory = os.path.join( os.getcwd(), "symbols" )

#     # symbol_file = "Naz100_Symbols.txt"
#     # symbols_file = os.path.join( symbol_directory, symbol_file )
#     # symbol_directory = os.path.join( os.getcwd(), "symbols" )
#     symbol_directory, symbol_file = os.path.split(symbols_file)

#     with open(symbols_file, "r+") as f:
#         old_symbolList = f.readlines()
#     for i in range( len(old_symbolList) ) :
#         old_symbolList[i] = old_symbolList[i].replace("\n","")

#     ###
#     ### get current symbol list from wikipedia website
#     ###
#     try:
#         base_url ='https://en.wikipedia.org/wiki/NASDAQ-100#Components'
#         soup = BeautifulSoup( urllib.request.urlopen(base_url).read(), "lxml" )
#         #t = soup.findAll("table", {"class" : "wikitable sortable"})[2] # 2020-09-04
#         #t = soup.findAll("table", {"id" : "constituents"}) # 2024-10-05
#         t = soup.find("table", {"id": "constituents"}) # 2024-10-05

#         print("... got web content")
#         print("... ran beautiful soup on web content")

#         symbolList = [] # store all of the records in this list
#         companyNamesList = []
#         data=[]
#         for row in t.find_all('tr'):
#             if str(row)==[]:
#                 continue
#             if str(row)=="\n":
#                 continue
#             col = row.findAll('td')
#             cols = row.find_all('td')
#             cols = [ele.text.strip() for ele in cols]
#             data.append([ele for ele in cols if ele])
#             if col==[]:
#                 continue
#             company_name = data[-1][0].encode("utf8")
#             if company_name[:3]=='MON':
#                 break

#             company_name = strip_accents(data[-1][0])
#             symbol_name = data[-1][1]

#             print(" ...symbol_name="+'{0: <5}'.format(symbol_name)+" ...company_name="+company_name)
#             companyNamesList.append(company_name)
#             symbolList.append(symbol_name)
#         print("... retrieved Naz100 companies lists from internet")
#         print("\nsymbolList = ", symbolList)


#         companyName_file = os.path.join( symbol_directory, "companyNames.txt" )
#         with open( companyName_file, "w" ) as f:
#             for i in range( len(symbolList) ) :
#                 f.write( symbolList[i] + ";" + companyNamesList[i] + "\n" )

#         ###
#         ### compare old list with new list and print changes, if any
#         ###

#         # file for index changes history
#         symbol_change_file = "Naz100_symbolsChanges.txt"
#         symbols_changes_file = os.path.join( symbol_directory, symbol_change_file )
#         with open(symbols_changes_file, "r+") as f:
#             old_symbol_changesList = f.readlines()
#         old_symbol_changesListText = ''
#         for i in range( len(old_symbol_changesList) ):
#             old_symbol_changesListText = old_symbol_changesListText + old_symbol_changesList[i]

#         # parse date
#         year = datetime.datetime.now().year
#         month = datetime.datetime.now().month
#         day = datetime.datetime.now().day
#         dateToday = str(year)+"-"+str(month)+"-"+str(day)

#         # compare lists to check for tickers removed from the index
#         # - printing will be suppressed if "verbose = False"
#         removedTickers = []
#         print("")
#         for i, ticker in enumerate( old_symbolList ):
#             if i == 0:
#                 removedTickersText = ''
#             if ticker not in symbolList:
#                 removedTickers.append( ticker )
#                 if verbose:
#                     print(" Ticker ", ticker, " has been removed from the Nasdaq100 index")
#                 removedTickersText = removedTickersText + "\n" + dateToday + " Remove " + ticker

#         # compare lists to check for tickers added to the index
#         # - printing will be suppressed if "verbose = False"
#         addedTickers = []
#         print("")
#         for i, ticker in enumerate( symbolList ):
#             if i == 0:
#                 addedTickersText = ''
#             if ticker not in old_symbolList:
#                 addedTickers.append( ticker )
#                 if verbose:
#                     print(" Ticker ", ticker, " has been added to the Nasdaq100 index")
#                 addedTickersText = addedTickersText + "\n" + dateToday + " Add    " + ticker

#         print("")
#         with open(symbols_changes_file, "w") as f:
#             f.write(addedTickersText)
#             f.write(removedTickersText)
#             f.write("\n")
#             f.write(old_symbol_changesListText)

#         print("****************")
#         print("addedTickers = ", addedTickers)
#         print("removedTickers = ", removedTickers)
#         print("****************")
#         ###
#         ### update symbols file with current list. Keep copy of of list.
#         ###

#         if removedTickers != [] or addedTickers != []:

#             # make copy of previous symbols list file
#             symbol_directory = os.path.join( os.getcwd(), "symbols" )
#             symbol_file = "Naz100_Symbols.txt"
#             archive_symbol_file = "Naz100_symbols__" + str(datetime.date.today()) + ".txt"
#             symbols_file = os.path.join( symbol_directory, symbol_file )
#             archive_symbols_file = os.path.join( symbol_directory, archive_symbol_file )

#             with open( archive_symbols_file, "w" ) as f:
#                 for i in range( len(old_symbolList) ) :
#                     f.write( old_symbolList[i] + "\n" )

#             # make new symbols list file
#             with open( symbols_file, "w" ) as f:
#                 for i in range( len(symbolList) ) :
#                     f.write( symbolList[i] + "\n" )
#     except:
#         ###
#         ### something didn't wor. print message and return old list.
#         ###
#         print("\n\n\n")
#         print("! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")
#         print(" Nasdaq sysmbols list did not get updated from web.")
#         print(" ... check quotes_for_list_adjCloseVol.py in function 'get_Naz100List' ")
#         print(" ... also check web at https://en.wikipedia.org/wiki/NASDAQ-100#Components")
#         print("! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")
#         print("\n\n\n")

#         symbolList = old_symbolList
#         removedTickers = []
#         addedTickers = []

#     return symbolList, removedTickers, addedTickers
# '''


# def get_Naz100List(symbols_file, verbose=True ):
#     ###
#     ### Query nasdaq.com for updated list of stocks in Nasdaq 100 index.
#     ### Return list with stock tickers.
#     ###
#     #import urllib
#     import urllib.request, urllib.parse, urllib.error
#     from bs4 import BeautifulSoup
#     import os

#     ###
#     ### get symbol list from previous period
#     ###
#     # symbol_directory = os.path.join( os.getcwd(), "symbols" )

#     # symbol_file = "Naz100_Symbols.txt"
#     # symbols_file = os.path.join( symbol_directory, symbol_file )
#     symbol_directory, symbol_file = os.path.split(symbols_file)

#     with open(symbols_file, "r+") as f:
#         old_symbolList = f.readlines()
#     for i in range( len(old_symbolList) ) :
#         old_symbolList[i] = old_symbolList[i].replace("\n","")

#     ###
#     ### get current symbol list from wikipedia website
#     ###
#     try:
#         base_url ='https://en.wikipedia.org/wiki/NASDAQ-100#Components'
#         soup = BeautifulSoup( urllib.request.urlopen(base_url).read(), "lxml" )
#         #t = soup.findAll("table", {"class" : "wikitable sortable"})[2] # 2020-09-04
#         #t = soup.findAll("table", {"id" : "constituents"}) # 2024-10-05
#         t = soup.find("table", {"id": "constituents"}) # 2024-10-05

#         print("... got web content")
#         print("... ran beautiful soup on web content")

#         symbolList = [] # store all of the records in this list
#         companyNamesList = []
#         data=[]
#         for row in t.find_all('tr'):
#             if str(row)==[]:
#                 continue
#             if str(row)=="\n":
#                 continue
#             col = row.findAll('td')
#             cols = row.find_all('td')
#             cols = [ele.text.strip() for ele in cols]
#             data.append([ele for ele in cols if ele])
#             if col==[]:
#                 continue
#             company_name = data[-1][0].encode("utf8")
#             if company_name[:3]=='MON':
#                 break

#             company_name = strip_accents(data[-1][0])
#             symbol_name = data[-1][1]

#             print(" ...symbol_name="+'{0: <5}'.format(symbol_name)+" ...company_name="+company_name)
#             companyNamesList.append(company_name)
#             symbolList.append(symbol_name)
#         print("... retrieved Naz100 companies lists from internet")
#         print("\nsymbolList = ", symbolList)


#         companyName_file = os.path.join( symbol_directory, "companyNames.txt" )
#         with open( companyName_file, "w" ) as f:
#             for i in range( len(symbolList) ) :
#                 f.write( symbolList[i] + ";" + companyNamesList[i] + "\n" )

#         ###
#         ### compare old list with new list and print changes, if any
#         ###

#         # file for index changes history
#         symbol_change_file = "Naz100_symbolsChanges.txt"
#         symbols_changes_file = os.path.join( symbol_directory, symbol_change_file )
#         with open(symbols_changes_file, "r+") as f:
#             old_symbol_changesList = f.readlines()
#         old_symbol_changesListText = ''
#         for i in range( len(old_symbol_changesList) ):
#             old_symbol_changesListText = old_symbol_changesListText + old_symbol_changesList[i]

#         # parse date
#         year = datetime.datetime.now().year
#         month = datetime.datetime.now().month
#         day = datetime.datetime.now().day
#         dateToday = str(year)+"-"+str(month)+"-"+str(day)

#         # compare lists to check for tickers removed from the index
#         # - printing will be suppressed if "verbose = False"
#         removedTickers = []
#         print("")
#         for i, ticker in enumerate( old_symbolList ):
#             if i == 0:
#                 removedTickersText = ''
#             if ticker not in symbolList:
#                 removedTickers.append( ticker )
#                 if verbose:
#                     print(" Ticker ", ticker, " has been removed from the Nasdaq100 index")
#                 removedTickersText = removedTickersText + "\n" + dateToday + " Remove " + ticker

#         # compare lists to check for tickers added to the index
#         # - printing will be suppressed if "verbose = False"
#         addedTickers = []
#         print("")
#         for i, ticker in enumerate( symbolList ):
#             if i == 0:
#                 addedTickersText = ''
#             if ticker not in old_symbolList:
#                 addedTickers.append( ticker )
#                 if verbose:
#                     print(" Ticker ", ticker, " has been added to the Nasdaq100 index")
#                 addedTickersText = addedTickersText + "\n" + dateToday + " Add    " + ticker

#         print("")
#         with open(symbols_changes_file, "w") as f:
#             f.write(addedTickersText)
#             f.write(removedTickersText)
#             f.write("\n")
#             f.write(old_symbol_changesListText)

#         print("****************")
#         print("addedTickers = ", addedTickers)
#         print("removedTickers = ", removedTickers)
#         print("****************")
#         ###
#         ### update symbols file with current list. Keep copy of of list.
#         ###

#         if removedTickers != [] or addedTickers != []:

#             # make copy of previous symbols list file
#             symbol_directory = os.path.join( os.getcwd(), "symbols" )
#             symbol_file = "Naz100_Symbols.txt"
#             archive_symbol_file = "Naz100_symbols__" + str(datetime.date.today()) + ".txt"
#             symbols_file = os.path.join( symbol_directory, symbol_file )
#             archive_symbols_file = os.path.join( symbol_directory, archive_symbol_file )

#             with open( archive_symbols_file, "w" ) as f:
#                 for i in range( len(old_symbolList) ) :
#                     f.write( old_symbolList[i] + "\n" )

#             # make new symbols list file
#             with open( symbols_file, "w" ) as f:
#                 for i in range( len(symbolList) ) :
#                     f.write( symbolList[i] + "\n" )
#     except:
#         ###
#         ### something didn't wor. print message and return old list.
#         ###
#         print("\n\n\n")
#         print("! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")
#         print(" Nasdaq sysmbols list did not get updated from web.")
#         print(" ... check quotes_for_list_adjCloseVol.py in function 'get_Naz100List' ")
#         print(" ... also check web at https://en.wikipedia.org/wiki/NASDAQ-100#Components")
#         print("! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")
#         print("\n\n\n")

#         symbolList = old_symbolList
#         removedTickers = []
#         addedTickers = []

#     return symbolList, removedTickers, addedTickers



def get_Naz100PlusETFsList( verbose=True ):
    ###
    ### Query nasdaq.com for updated list of stocks in Nasdaq 100 index.
    ### Return list with stock tickers.
    ###
    import urllib.request, urllib.parse, urllib.error
    import re
    import os
    import datetime

    ###
    ### get symbol list from previous period
    ###
    symbol_directory = os.path.join( os.getcwd(), "symbols" )
    symbol_file = "Naz100PlusETFs_symbols.txt"
    symbols_file = os.path.join( symbol_directory, symbol_file )
    with open(symbols_file, "r+") as f:
        old_symbolList = f.readlines()
    for i in range( len(old_symbolList) ) :
        old_symbolList[i] = old_symbolList[i].replace("\n","")

    ###
    ### get current symbol list from nasdaq website
    ###
    base_url = 'http://www.nasdaq.com/quotes/nasdaq-100-stocks.aspx'
    content = urllib.request.urlopen(base_url).read()
    m = re.search('var table_body.*?>*(?s)(.*?)<.*?>.*?<', content).group(0).split("],[")
    # handle exceptions in format for first and last entries in list
    m[0] = m[0].split("[")[2]
    m[-1] = m[-1].split("]")[0].split("[")[0]
    # parse list items for symbol name
    symbolList = []
    for i in range( len(m) ):
        symbolList.append( m[i].split(",")[0].split('"')[1] )

    ###
    ### compare old list with new list and print changes, if any
    ###

    # compare lists to check for tickers removed from the index
    # - printing will be suppressed if "verbose = False"
    removedTickers = []
    print("")
    for i, ticker in enumerate( symbolList ):
        if ticker not in old_symbolList:
            removedTickers.append( ticker )
            if verbose:
                print(" Ticker ", ticker, " has been removed from the Nasdaq100 index")

    # add GTAA asset classes to Naz100 tickers for extra diversity
    ETF_List = ['AGG', 'CEW', 'DBC', 'EEM', 'EMB', 'FXE', 'GLD', 'HYG', 'IVV', 'LQD', 'TIP', 'TLT', 'USO', 'VNQ', 'XLF', 'XWD.TO' ]
    for i in range( len(ETF_List) ) :
        symbolList.append( ETF_List[i] )

    # compare lists to check for tickers added to the index
    # - printing will be suppressed if "verbose = False"
    addedTickers = []
    print("")
    for i, ticker in enumerate( old_symbolList ):
        if ticker not in symbolList:
            addedTickers.append( ticker )
            if verbose:
                print(" Ticker ", ticker, " has been added to the Nasdaq100 index")

    print("")
    ###
    ### update symbols file with current list. Keep copy of of list.
    ###

    if removedTickers != [] or addedTickers != []:
        # make copy of previous symbols list file
        symbol_directory = os.path.join( os.getcwd(), "symbols" )
        symbol_file = "Naz100_Symbols.txt"
        archive_symbol_file = "Naz100_Symbols__" + str(datetime.date.today()) + ".txt"
        symbols_file = os.path.join( symbol_directory, symbol_file )
        archive_symbols_file = os.path.join( symbol_directory, archive_symbol_file )

        with open( archive_symbols_file, "w" ) as f:
            for i in range( len(old_symbolList) ) :
                f.write( old_symbolList[i] + "\n" )

        # make new symbols list file
        with open( symbols_file, "w" ) as f:
            for i in range( len(symbolList) ) :
                f.write( symbolList[i] + "\n" )

    return symbolList.sort(), removedTickers, addedTickers


def get_ETFList( verbose=True ):
    ###
    ### Return list with stock tickers.
    ###
    import urllib.request, urllib.parse, urllib.error
    import re
    import os
    import datetime

    ###
    ### get symbol list from previous period
    ###
    symbol_directory = os.path.join( os.getcwd(), "symbols" )
    symbol_file = "ETF_Symbols.txt"
    symbols_file = os.path.join( symbol_directory, symbol_file )
    with open(symbols_file, "r+") as f:
        old_symbolList = f.readlines()
    for i in range( len(old_symbolList) ) :
        old_symbolList[i] = old_symbolList[i].replace("\n","")

    return old_symbolList.sort(), [], []


def arrayFromQuotesForList(symbolsFile, json_fn, beginDate, endDate):
    '''
    read in quotes and process to 'clean' ndarray plus date array
    - prices in array with dimensions [num stocks : num days ]
    - process stock quotes to show closing prices adjusted for splits, dividends
    - single ndarray with dates common to all stocks [num days]
    - clean up stocks by:
       - infilling empty values with linear interpolated value
       - repeat first quote to beginning of series
    '''

    print(" ... inside quotes_for_list_adjClose/arrayFromQuotesForList ...")

    from functions.TAfunctions import interpolate
    from functions.TAfunctions import cleantobeginning

    # read symbols list
    # symbols = readSymbolList(symbolsFile, json_fn, verbose=True)
    _, symbols = read_symbols_list_web(json_fn, verbose=False)

    # get quotes for each symbol in list (adjusted close)
    quote = downloadQuotes(symbols,date1=beginDate,date2=endDate,adjust=True,Verbose=True)

    # clean up quotes for missing values and varying starting date
    #x = quote.as_matrix().swapaxes(0,1)
    quote = quote.convert_objects(convert_numeric=True)   ### test
    x = quote.values.T
    ###print "x = ", x
    date = quote.index
    date = [d.date().isoformat() for d in date]
    datearray = np.array(date)
    symbolList = list(quote.columns.values)

    # Clean up input quotes
    #  - infill interior NaN values using nearest good values to linearly interpolate
    #  - copy first valid quote to from valid date to all earlier positions
    for ii in range(x.shape[0]):
        x[ii,:] = np.array(x[ii,:]).astype('float')
        print(" progress-- ", ii, " of ", x.shape[0], " symbol = ", symbols[ii], x.shape, x[ii,:])
        #print " line 283........."
        x[ii,:] = interpolate(x[ii,:])
        x[ii,:] = cleantobeginning(x[ii,:])

    return x, symbolList, datearray

def arrayFromQuotesForListWithVol(symbolsFile, json_fn, beginDate, endDate):
    '''
    read in quotes and process to 'clean' ndarray plus date array
    - prices in array with dimensions [num stocks : num days ]
    - process stock quotes to show closing prices adjusted for splits, dividends
    - single ndarray with dates common to all stocks [num days]
    - clean up stocks by:
       - infilling empty values with linear interpolated value
       - repeat first quote to beginning of series
    '''

    # read symbols list
    # symbols = readSymbolList(symbolsFile, json_fn, verbose=True)
    _, symbols = read_symbols_list_web(json_fn, verbose=False)

    # get quotes for each symbol in list (adjusted close)
    quote = downloadQuotes(symbols,date1=beginDate,date2=endDate,adjust=True,Verbose=True)

    # clean up quotes for missing values and varying starting date
    x=quote.copyx()
    x=quote.as_matrix().swapaxes(0,1)
    date = quote.getlabel(2)
    datearray = np.array(date)


    # Clean up input quotes
    #  - infill interior NaN values using nearest good values to linearly interpolate
    #  - copy first valid quote to from valid date to all earlier positions
    for ii in range(x.shape[0]):
        print(" line 315.........")
        x[ii,0,:] = interpolate(x[ii,0,:].values)
        x[ii,0,:] = cleantobeginning(x[ii,0,:].values)

    return x, symbols, datearray

'''
def get_quote_google( symbol ):
    def scrape_quote(_symbol):
        import urllib
        import re
        base_url = 'http://finance.google.com/finance?q=NASDAQ%3A'
        content = urllib.urlopen(base_url + _symbol).read()
        m = re.search('class="pr".*?>*(?s)(.*?)<.*?>.*?<', content).group(0).split(">")[-1].split("<")[0]
        if m :
            quote = m
        else:
            quote = 'no quote available for: ' + symbol
        return quote
    for i in range(200):
        try:
            quote = scrape_quote(symbol)
            break
        except:
            print ".",
    return quote
'''


def get_quote_google( symbol ):
    " use alpha_vantage instead of google "
    from quotes_adjClose_alphavantage import get_last_quote
    if symbol == 'CASH':
        last_quote = 1.0
    else:
        last_quote = get_last_quote(symbol)
    return last_quote


def get_quote_alphavantage( symbol ):
    " use alpha_vantage instead of google "
    try:
        from functions.quotes_adjClose_alphavantage import get_last_quote
        from functions.quotes_adjClose_alphavantage import get_last_quote_daily
    except:
        from quotes_adjClose_alphavantage import get_last_quote
        from quotes_adjClose_alphavantage import get_last_quote_daily
    try:
        _quote = get_last_quote(symbol)
    except:
        _quote = get_last_quote_daily(symbol)
    return _quote


def get_pe_google( symbol ):
    import urllib.request, urllib.parse, urllib.error
    import re
    base_url = 'http://finance.google.com/finance?q=NASDAQ%3A'
    content = urllib.request.urlopen(base_url + symbol).read()
    try:
        m = float(content.split("pe_ratio")[1].split('\n')[2].split(">")[-1])
        quote = m
    except:
        quote = ""
    return quote


def get_pe_finviz(symbol: str, verbose: bool = False) -> float:
    """
    Use finviz to get calculated P/E ratios for a given symbol.
    Handles errors gracefully and returns np.nan for unavailable data.

    Args:
        symbol: The stock ticker symbol.
        verbose: If True, print detailed output.

    Returns:
        The P/E ratio as a float, or np.nan if unavailable.
    """
    import numpy as np
    import bs4 as bs
    import requests
    import os
    import csv
    import time
    import logging

    logger = logging.getLogger(__name__)

    try:
        url = "https://finviz.com/quote.ashx?t=" + symbol.upper()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            )
        }
        r = requests.get(url, headers=headers)
        
        # Handle rate limiting
        if r.status_code == 429:
            logger.warning(f"Rate limited by finviz for {symbol}. Sleeping and retrying...")
            time.sleep(5)
            r = requests.get(url, headers=headers)
        
        # Handle HTTP 404 and other errors gracefully
        if r.status_code == 404:
            logger.warning(f"HTTP 404 from finviz for {symbol} - symbol may be delisted")
            if verbose:
                print(f"HTTP 404 from finviz for {symbol}")
            return np.nan
        elif r.status_code != 200:
            logger.warning(f"HTTP {r.status_code} from finviz for {symbol}")
            if verbose:
                print(f"HTTP {r.status_code} from finviz for {symbol}")
            return np.nan
            
        html = r.text
        soup = bs.BeautifulSoup(html, "lxml")
        table = soup.find("table", class_="snapshot-table2")
        
        if table is None:
            logger.warning(f"Could not find snapshot-table2 in finviz HTML for {symbol}")
            if verbose:
                print(f"Could not find data table for {symbol}")
            return np.nan
            
        values = []
        for tr in table.find_all("tr")[1:3]:
            td = tr.find_all("td")[1]
            value = td.text
            if "B" in value:
                value = value.replace("B", "")
                value = float(value.strip())
                value = value * 1000000000
                values.append(value)
            elif "M" in value:
                value = value.replace("M", "")
                value = float(value.strip())
                value = value * 1000000
                values.append(value)
            else:
                values.append(0)
                
        market_cap = values[0]
        earnings = values[1]
        
        if float(earnings) != 0.0:
            pe = market_cap / earnings
        else:
            pe = np.nan
            
        if verbose:
            print(f"{symbol} market cap, earnings, P/E = {market_cap}, {earnings}, {pe}")
        return pe
        
    except ValueError as e:
        # Handle specific parsing errors
        if "could not convert string to float" in str(e):
            logger.warning(f"Failed to parse P/E data for {symbol}: {e}")
            if verbose:
                print(f"Failed to fetch P/E for {symbol}: {e}")
        return np.nan
    except Exception as exc:
        # Handle all other exceptions gracefully
        logger.warning(f"Failed to fetch P/E for {symbol}: {exc}")
        if verbose:
            print(f"Failed to fetch P/E for {symbol}: {exc}")
        return np.nan


def get_SectorAndIndustry_google( symbol ):
    import urllib.request, urllib.parse, urllib.error
    import re
    base_url = 'http://finance.google.com/finance?q=NASDAQ%3A'
    content = urllib.request.urlopen(base_url + symbol).read()
    try:
        m = content.split("Sector:")[1].split('>')[1].split("<")[0].replace("&amp;","and")
        sector = m
    except:
        sector = ""
    try:
        m = content.split("Industry:")[1].split('>')[1].split("<")[0].replace("&amp;","and").replace(" - NEC","")
        industry = m
    except:
        industry = ""
    return sector, industry


def get_SectorAndIndustry_google( symbol ):
    ' use finviz to get sector and industry '
    import requests
    import bs4 as bs
    import os
    import csv
    #Get source table
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'} # This is chrome, you can set whatever browser you like
    url = 'https://finviz.com/quote.ashx?t='+symbol.upper()
    r = requests.get(url, headers=headers)
    html = r.text
    soup = bs.BeautifulSoup(html, 'lxml')
    table = soup.find('table', class_= 'fullview-title')

    try:
        for tr in table.find_all('tr')[2:3]:
            td = tr.find_all('td')[0]
            value = td.text
            print(value)
        values = value.split(' | ')[:-1]
        #print values
        industry = values[0]
        sector = values[1]
    except:
        industry= "unknown"
        sector = "unknown"
    return sector, industry


def get_SectorAndIndustry_google( symbol ):
    ' use finviz from package finvizfinance to get sector and industry '
    import pandas as pd
    from finvizfinance.quote import finvizfinance

    try:
        stock = finvizfinance(symbol)
        stock_fundament = stock.ticker_fundament()
        industry = stock_fundament["Industry"]
        sector = stock_fundament["Sector"]
    except:
        industry = "unknown"
        sector = "unknown"
    return sector, industry


def LastQuotesForSymbolList( symbolList ):
    """
    read in latest (15-minute delayed) quote for each symbol in list.
    Use alpha_vantage for each symbol's quote.
    """
    from time import sleep
    #from functions.quote_adjClose_alphavantage import get_last_quote

    def scrape_quote(_symbol):
        quote = get_quote_alphavantage( _symbol )
        #print "ticker, quote = ", _symbol, quote
        # Remove comma from quote
        if type(quote) == 'str' and ',' in quote:
            quote = quote.replace(",", "")
        return quote

    quotelist = []
    for itick, ticker in enumerate( symbolList ):

        if ticker == 'CASH':
            print("ticker, quote = CASH 1.0")
            quotelist.append(1.0)
        else:
            for i in range(200):
                try:
                    quote = scrape_quote(ticker)
                    break
                except:
                    print(".", end=' ')
                    quote = scrape_quote(ticker)
            print("ticker, quote = ", ticker, quote)
            quotelist.append( quote )

    return quotelist


def LastQuotesForSymbolList_hdf(symbolList, symbols_file, json_fn):
    """
    read in latest (15-minute delayed) quote for each symbol in list.
    Use quotes on hdf for each symbol's quote.
    """
    import os
    from functions.UpdateSymbols_inHDF5 import loadQuotes_fromHDF
    from functions.GetParams import get_symbols_file
    # get last quote in pyTAAA hdf file
    print(" ... inside LastQuotesForSymbolList_hdf")
    print(" ... symbolList = " + str(symbolList))
    print(" ... symbols_file = " + str(symbols_file))
    filename = get_symbols_file(json_fn)
    
    _, _, _, quote, _ = loadQuotes_fromHDF(filename, json_fn)
    quotelist = []
    for itick, ticker in enumerate( symbolList ):
        quotelist.append(float(quote[ticker].values[-1]))
    print(" ... inside LastQuotesForSymbolList_hdf")
    # print("   . quote[-10:] = " +str(quote[-10:]))
    return quotelist


def LastQuotesForList( symbols_list ):

    from time import sleep
    #from functions.StockRetriever import *
    import StockRetriever

    stocks = StockRetriever()

    # remove 'CASH' from symbols_list, if present. Keep track of position in list to re-insert
    cash_index = None
    try:
        cash_index = symbols_list.index('CASH')
        if cash_index >= 0 and cash_index <= len(symbols_list)-1 :
            symbols_list.remove('CASH')
    except:
        pass

    attempt = 1
    NeedQuotes = True
    while NeedQuotes:
        try:
            a=stocks.get_current_info( symbols_list )
            print("inside LastQuotesForList 1, symbols_list = ", symbols_list)
            print("inside LastQuotesForList 1, attempt = ", attempt)
            # convert from strings to numbers and put in a list
            quotelist = []
            for i in range(len(a)):
                singlequote = np.float((a[i]['LastTradePriceOnly']).encode('ascii','ignore'))
                quotelist.append(singlequote)
            print(symbols_list, quotelist)
            NeedQuotes = False
        except:
            attempt += 1
            sleep(attempt)

    print("inside LastQuotesForList... location  2")

    # re-insert CASH in original position and also add curent price of 1.0 to quotelist
    if cash_index != None:
        if cash_index < len(symbols_list):
            symbols_list[cash_index:cash_index] = 'CASH'
            quotelist[cash_index:cash_index] = 1.0
        else:
            symbols_list.append('CASH')
            quotelist.append(1.0)

    print("attempts, sysmbols_list,quotelist =", attempt, symbols_list, quotelist)
    return quotelist


def diagnose_finviz_format(symbol: str = "AAPL", save_html: bool = True) -> dict:
    """
    Diagnostic function to check current Finviz webpage format.
    
    Args:
        symbol: Stock symbol to test (default: AAPL - a reliable stock)
        save_html: Whether to save the HTML for manual inspection
        
    Returns:
        Dictionary with diagnostic information about the webpage format
    """
    import numpy as np
    import bs4 as bs
    import requests
    import os
    import datetime
    import logging
    
    logger = logging.getLogger(__name__)
    
    print(f"############### Diagnosing Finviz format for {symbol} ###############")
    
    try:
        url = f"https://finviz.com/quote.ashx?t={symbol.upper()}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            )
        }
        
        print(f"Fetching URL: {url}")
        r = requests.get(url, headers=headers)
        print(f"HTTP Status Code: {r.status_code}")
        
        if r.status_code != 200:
            return {
                "status": "error",
                "http_code": r.status_code,
                "message": f"HTTP {r.status_code} error"
            }
        
        html = r.text
        print(f"HTML length: {len(html)} characters")
        
        # Save HTML for manual inspection if requested
        if save_html:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"finviz_{symbol}_{timestamp}.html"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"HTML saved to: {filename}")
        
        soup = bs.BeautifulSoup(html, "lxml")
        
        # Check for various table formats Finviz might use
        tables_found = {}
        
        # Original format we expect
        table_snapshot2 = soup.find("table", class_="snapshot-table2")
        tables_found["snapshot-table2"] = table_snapshot2 is not None
        
        # Alternative table formats to check
        table_formats = [
            ("table", {"class": "snapshot-table"}),
            ("table", {"class": "fullview-title"}),
            ("div", {"class": "snapshot-td2"}),
            ("div", {"class": "quote-price"}),
            ("table", {"id": "js-quote-summary"}),
            ("div", {"class": "quote-summary"}),
        ]
        
        for tag, attrs in table_formats:
            element = soup.find(tag, attrs)
            key = f"{tag}_{list(attrs.values())[0]}"
            tables_found[key] = element is not None
            if element:
                print(f"Found element: {tag} with {attrs}")
        
        # Look for P/E related text in the HTML
        pe_mentions = html.lower().count("p/e")
        ratio_mentions = html.lower().count("ratio")
        print(f"P/E mentions in HTML: {pe_mentions}")
        print(f"Ratio mentions in HTML: {ratio_mentions}")
        
        # Try to find all tables and their classes
        all_tables = soup.find_all("table")
        table_classes = []
        for table in all_tables:
            if table.get("class"):
                table_classes.extend(table.get("class"))
        
        print(f"All table classes found: {set(table_classes)}")
        
        # Check if we can find the expected data structure
        diagnostic_result = {
            "status": "success",
            "http_code": r.status_code,
            "html_length": len(html),
            "pe_mentions": pe_mentions,
            "ratio_mentions": ratio_mentions,
            "tables_found": tables_found,
            "table_classes": list(set(table_classes)),
            "snapshot_table2_exists": table_snapshot2 is not None
        }
        
        # Try to extract data using current method
        if table_snapshot2:
            print("snapshot-table2 found - testing data extraction...")
            try:
                values = []
                for tr in table_snapshot2.find_all("tr")[1:3]:
                    td = tr.find_all("td")[1]
                    value = td.text
                    values.append(value)
                    print(f"Extracted value: '{value}'")
                
                diagnostic_result["extracted_values"] = values
                diagnostic_result["extraction_success"] = True
                
            except Exception as e:
                print(f"Data extraction failed: {e}")
                diagnostic_result["extraction_success"] = False
                diagnostic_result["extraction_error"] = str(e)
        else:
            print("snapshot-table2 NOT FOUND - format may have changed")
            diagnostic_result["extraction_success"] = False
            diagnostic_result["extraction_error"] = "snapshot-table2 not found"
        
        return diagnostic_result
        
    except Exception as e:
        logger.error(f"Diagnostic failed for {symbol}: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


def get_pe_finviz_with_diagnostics(symbol: str, verbose: bool = False) -> float:
    """
    Enhanced version of get_pe_finviz with detailed format checking.
    
    Args:
        symbol: The stock ticker symbol
        verbose: If True, print detailed diagnostic output
        
    Returns:
        The P/E ratio as a float, or np.nan if unavailable
    """
    import numpy as np
    import bs4 as bs
    import requests
    import time
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        url = f"https://finviz.com/quote.ashx?t={symbol.upper()}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            )
        }
        
        if verbose:
            print(f"Fetching {symbol} from: {url}")
            
        r = requests.get(url, headers=headers)
        
        # Handle rate limiting
        if r.status_code == 429:
            logger.warning(f"Rate limited by finviz for {symbol}. Sleeping and retrying...")
            if verbose:
                print(f"Rate limited for {symbol}, waiting 5 seconds...")
            time.sleep(5)
            r = requests.get(url, headers=headers)
        
        # Handle HTTP errors gracefully
        if r.status_code == 404:
            logger.warning(f"HTTP 404 from finviz for {symbol} - symbol may be delisted")
            if verbose:
                print(f"HTTP 404 from finviz for {symbol}")
            return np.nan
        elif r.status_code != 200:
            logger.warning(f"HTTP {r.status_code} from finviz for {symbol}")
            if verbose:
                print(f"HTTP {r.status_code} from finviz for {symbol}")
            return np.nan
            
        html = r.text
        soup = bs.BeautifulSoup(html, "lxml")
        
        if verbose:
            print(f"HTML length: {len(html)} characters")
            
        # Try original format first
        table = soup.find("table", class_="snapshot-table2")
        
        if table is None:
            if verbose:
                print("snapshot-table2 not found, checking alternative formats...")
                
            # Try alternative table formats
            alternative_selectors = [
                ("table", {"class": "snapshot-table"}),
                ("div", {"class": "snapshot-td2"}),
                ("table", {"class": "fullview-title"}),
            ]
            
            for tag, attrs in alternative_selectors:
                table = soup.find(tag, attrs)
                if table:
                    if verbose:
                        print(f"Found alternative format: {tag} with {attrs}")
                    break
            
            if table is None:
                logger.warning(f"Could not find any data table for {symbol} - format may have changed")
                if verbose:
                    print(f"No recognized data table found for {symbol}")
                    # Show available table classes for debugging
                    all_tables = soup.find_all("table")
                    if all_tables:
                        print("Available table classes:")
                        for i, t in enumerate(all_tables[:5]):  # Show first 5 tables
                            print(f"  Table {i}: {t.get('class', 'no class')}")
                return np.nan
                
        # Try to extract P/E data
        try:
            values = []
            for tr in table.find_all("tr")[1:3]:
                td = tr.find_all("td")[1]
                value = td.text.strip()
                if verbose:
                    print(f"Raw value extracted: '{value}'")
                
                if "B" in value:
                    value = value.replace("B", "")
                    value = float(value.strip()) * 1000000000
                    values.append(value)
                elif "M" in value:
                    value = value.replace("M", "")
                    value = float(value.strip()) * 1000000
                    values.append(value)
                else:
                    try:
                        values.append(float(value))
                    except ValueError:
                        if verbose:
                            print(f"Could not parse value: '{value}'")
                        values.append(0)
                        
            if len(values) >= 2:
                market_cap = values[0]
                earnings = values[1]
                
                if float(earnings) != 0.0:
                    pe = market_cap / earnings
                else:
                    pe = np.nan
                    
                if verbose:
                    print(f"{symbol}: market_cap={market_cap}, earnings={earnings}, P/E={pe}")
                return pe
            else:
                if verbose:
                    print(f"Insufficient data extracted for {symbol}")
                return np.nan
                
        except Exception as e:
            logger.warning(f"Data extraction failed for {symbol}: {e}")
            if verbose:
                print(f"Data extraction failed for {symbol}: {e}")
            return np.nan
        
    except ValueError as e:
        # Handle specific parsing errors
        if "could not convert string to float" in str(e):
            logger.warning(f"Failed to parse P/E data for {symbol}: {e}")
            if verbose:
                print(f"Failed to parse P/E data for {symbol}: {e}")
        return np.nan
    except Exception as exc:
        # Handle all other exceptions gracefully
        logger.warning(f"Failed to fetch P/E for {symbol}: {exc}")
        if verbose:
            print(f"Failed to fetch P/E for {symbol}: {exc}")
        return np.nan


def get_pe_finviz_with_smart_limiting(symbol: str, verbose: bool = False, 
                                      global_state: dict = None) -> float:
    """
    Enhanced P/E fetching with smart error limiting and caching.
    
    Args:
        symbol: The stock ticker symbol
        verbose: If True, print detailed output
        global_state: Dictionary to track global state across calls
        
    Returns:
        The P/E ratio as a float, or np.nan if unavailable or limit reached
    """
    import numpy as np
    import bs4 as bs
    import requests
    import time
    import logging
    import os
    import json
    from datetime import datetime, timedelta
    
    logger = logging.getLogger(__name__)
    
    # Initialize global state if not provided
    if global_state is None:
        global_state = {
            'consecutive_errors': 0,
            'total_requests': 0,
            'rate_limit_count': 0,
            'delisted_cache_file': 'finviz_delisted_cache.json',
            'delisted_symbols': set(),
            'last_request_time': 0,
            'backoff_delay': 1
        }
    
    # Load delisted symbols cache
    if os.path.exists(global_state['delisted_cache_file']):
        try:
            with open(global_state['delisted_cache_file'], 'r') as f:
                cache_data = json.load(f)
                global_state['delisted_symbols'] = set(cache_data.get('symbols', []))
                if verbose:
                    print(f"Loaded {len(global_state['delisted_symbols'])} known delisted symbols from cache")
        except Exception as e:
            logger.warning(f"Could not load delisted cache: {e}")
    
    # Check if symbol is in delisted cache
    if symbol.upper() in global_state['delisted_symbols']:
        if verbose:
            print(f"{symbol} is in delisted cache - skipping")
        return np.nan
    
    # Check consecutive error limit
    if global_state['consecutive_errors'] >= 10:
        logger.warning("Stopping P/E fetching after 10 consecutive errors")
        if verbose:
            print("Stopping P/E fetching after 10 consecutive errors")
        return np.nan
    
    # Check total rate limit count
    if global_state['rate_limit_count'] >= 5:
        logger.warning("Stopping P/E fetching after 5 rate limits")
        if verbose:
            print("Stopping P/E fetching after 5 rate limits")
        return np.nan
    
    # Implement smart delays between requests
    current_time = time.time()
    time_since_last = current_time - global_state['last_request_time']
    
    # Minimum delay between requests (increases with rate limits)
    min_delay = global_state['backoff_delay']
    if time_since_last < min_delay:
        sleep_time = min_delay - time_since_last
        if verbose:
            print(f"Waiting {sleep_time:.1f}s before next request...")
        time.sleep(sleep_time)
    
    global_state['last_request_time'] = time.time()
    global_state['total_requests'] += 1
    
    try:
        url = f"https://finviz.com/quote.ashx?t={symbol.upper()}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            )
        }
        
        if verbose:
            print(f"Fetching {symbol} (#{global_state['total_requests']})...")
            
        r = requests.get(url, headers=headers, timeout=10)
        
        # Handle rate limiting with exponential backoff
        if r.status_code == 429:
            global_state['rate_limit_count'] += 1
            global_state['backoff_delay'] = min(global_state['backoff_delay'] * 2, 30)  # Cap at 30s
            
            logger.warning(f"Rate limited by finviz for {symbol} ({global_state['rate_limit_count']}/5). "
                          f"Increasing delay to {global_state['backoff_delay']}s")
            if verbose:
                print(f"Rate limited for {symbol}, waiting {global_state['backoff_delay']}s...")
            
            time.sleep(global_state['backoff_delay'])
            
            # Retry once
            r = requests.get(url, headers=headers, timeout=10)
            
            if r.status_code == 429:
                global_state['consecutive_errors'] += 1
                return np.nan
        
        # Handle HTTP 404 - symbol delisted
        if r.status_code == 404:
            logger.warning(f"HTTP 404 from finviz for {symbol} - symbol may be delisted")
            if verbose:
                print(f"HTTP 404 from finviz for {symbol}")
            
            # Add to delisted cache
            global_state['delisted_symbols'].add(symbol.upper())
            _save_delisted_cache(global_state)
            
            global_state['consecutive_errors'] += 1
            return np.nan
            
        elif r.status_code != 200:
            logger.warning(f"HTTP {r.status_code} from finviz for {symbol}")
            if verbose:
                print(f"HTTP {r.status_code} from finviz for {symbol}")
            global_state['consecutive_errors'] += 1
            return np.nan
            
        # Successfully got data - reset consecutive error count
        global_state['consecutive_errors'] = 0
        
        # Reduce backoff delay on success
        if global_state['backoff_delay'] > 1:
            global_state['backoff_delay'] = max(global_state['backoff_delay'] * 0.8, 1)
            
        html = r.text
        soup = bs.BeautifulSoup(html, "lxml")
        table = soup.find("table", class_="snapshot-table2")
        
        if table is None:
            logger.warning(f"Could not find snapshot-table2 in finviz HTML for {symbol}")
            if verbose:
                print(f"Could not find data table for {symbol}")
            return np.nan
            
        values = []
        for tr in table.find_all("tr")[1:3]:
            td = tr.find_all("td")[1]
            value = td.text.strip()
            
            if "B" in value:
                value = value.replace("B", "")
                value = float(value.strip()) * 1000000000
                values.append(value)
            elif "M" in value:
                value = value.replace("M", "")
                value = float(value.strip()) * 1000000
                values.append(value)
            else:
                try:
                    values.append(float(value))
                except ValueError:
                    values.append(0)
                    
        if len(values) >= 2:
            market_cap = values[0]
            earnings = values[1]
            
            if float(earnings) != 0.0:
                pe = market_cap / earnings
            else:
                pe = np.nan
                
            if verbose:
                print(f"{symbol}: market_cap={market_cap}, earnings={earnings}, P/E={pe}")
            return pe
        else:
            if verbose:
                print(f"Insufficient data extracted for {symbol}")
            return np.nan
        
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching {symbol}")
        if verbose:
            print(f"Timeout fetching {symbol}")
        global_state['consecutive_errors'] += 1
        return np.nan
        
    except ValueError as e:
        if "could not convert string to float" in str(e):
            logger.warning(f"Failed to parse P/E data for {symbol}: {e}")
            if verbose:
                print(f"Failed to parse P/E data for {symbol}: {e}")
        global_state['consecutive_errors'] += 1
        return np.nan
        
    except Exception as exc:
        logger.warning(f"Failed to fetch P/E for {symbol}: {exc}")
        if verbose:
            print(f"Failed to fetch P/E for {symbol}: {exc}")
        global_state['consecutive_errors'] += 1
        return np.nan


def _save_delisted_cache(global_state: dict) -> None:
    """Save delisted symbols cache to disk."""
    import logging
    
    try:
        from datetime import datetime
        import json
        
        cache_data = {
            'symbols': list(global_state['delisted_symbols']),
            'last_updated': datetime.now().isoformat()
        }
        with open(global_state['delisted_cache_file'], 'w') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not save delisted cache: {e}")


def get_pe_finviz_batch(symbols: list, verbose: bool = False, 
                       max_symbols: int = 50) -> dict:
    """
    Fetch P/E ratios for a batch of symbols with smart rate limiting.
    
    Args:
        symbols: List of stock symbols
        verbose: If True, print detailed output
        max_symbols: Maximum number of symbols to process before stopping
        
    Returns:
        Dictionary mapping symbols to P/E ratios
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Shared state across all requests
    global_state = {
        'consecutive_errors': 0,
        'total_requests': 0,
        'rate_limit_count': 0,
        'delisted_cache_file': 'finviz_delisted_cache.json',
        'delisted_symbols': set(),
        'last_request_time': 0,
        'backoff_delay': 1
    }
    
    results = {}
    processed_count = 0
    
    print(f"############### Starting P/E batch fetch for {len(symbols)} symbols ###############")
    
    for i, symbol in enumerate(symbols):
        if processed_count >= max_symbols:
            print(f"Reached maximum symbol limit ({max_symbols}), stopping...")
            break
            
        if global_state['consecutive_errors'] >= 10:
            print(f"Stopping after 10 consecutive errors")
            break
            
        if global_state['rate_limit_count'] >= 5:
            print(f"Stopping after 5 rate limits")
            break
        
        if verbose:
            print(f"Processing {symbol} ({i+1}/{len(symbols)})...")
        
        pe_ratio = get_pe_finviz_with_smart_limiting(symbol, verbose=False, global_state=global_state)
        results[symbol] = pe_ratio
        processed_count += 1
        
        # Progress update every 10 symbols
        if (i + 1) % 10 == 0:
            success_count = sum(1 for v in results.values() if not (v != v))  # Count non-NaN values
            print(f"Progress: {i+1}/{len(symbols)} symbols processed, "
                  f"{success_count} successful, "
                  f"{global_state['consecutive_errors']} consecutive errors, "
                  f"{global_state['rate_limit_count']} rate limits")
    
    # Final summary
    success_count = sum(1 for v in results.values() if not (v != v))  # Count non-NaN values
    print(f"############### Batch complete ###############")
    print(f"Total processed: {processed_count}")
    print(f"Successful P/E fetches: {success_count}")
    print(f"Failed/delisted: {processed_count - success_count}")
    print(f"Rate limits hit: {global_state['rate_limit_count']}")
    print(f"Cached delisted symbols: {len(global_state['delisted_symbols'])}")
    
    return results
