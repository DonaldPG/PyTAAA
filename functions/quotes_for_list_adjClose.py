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


def get_pe_finviz( symbol, verbose=False ):
    ' use finviz to get calculated P/E ratios '
    import numpy as np
    import bs4 as bs
    import requests
    import os
    import csv

    try:
        # Get source table
        url = 'https://finviz.com/quote.ashx?t='+symbol.upper()
        r = requests.get(url)
        html = r .text
        soup = bs.BeautifulSoup(html, 'lxml')
        table = soup.find('table', class_= 'snapshot-table2')

        # Split by row and extract values
        values = []
        for tr in table.find_all('tr')[1:3]:
            td = tr.find_all('td')[1]
            value = td.text

            #Convert to numeric
            if 'B' in value:
                value = value.replace('B',"")
                value = float(value.strip())
                value = value * 1000000000
                values.append(value)

            elif 'M' in value:
                value = value.replace('M',"")
                value = float(value.strip())
                value = value * 1000000
                values.append(value)

            #Account for blank values
            else:
                values.append(0)

        #Append to respective lists
        market_cap = values[0]
        earnings = values[1]
        if float(earnings) != 0.:
            pe = market_cap / earnings
        else:
            pe = np.nan
    except:
        market_cap = 0.
        earnings = 0.
        pe = np.nan
    if verbose:
        print(symbol+' market cap, earnings, P/E = '+str(market_cap)+', '+str(earnings)+', '+str(pe))
    return pe


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
