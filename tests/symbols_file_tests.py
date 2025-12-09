#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 19:43:59 2025

@author: donaldpg
"""

import os
import datetime
import pandas as pd
import numpy as np

from functions.UpdateSymbols_inHDF5 import (
    listvals_tostring,
    df_index_columns_tostring,
    loadQuotes_fromHDF,
    compareHDF_and_newquotes
)
from functions.quotes_adjClose import downloadQuotes
from functions.GetParams import get_symbols_file, get_json_params
import yfinance

def test_symbols_fn():
    _symbols_fn = "/Users/donaldpg/pyTAAA_data/Naz100/symbols/Naz100_symbols.txt"
    
    # get symbols list
    json_fn = "/Users/donaldpg/pyTAAA_data/naz100_pi/pytaaa_naz100_pi.json"
    symbols_fn = get_symbols_file(json_fn)
    
    assert symbols_fn == _symbols_fn, "test_symbols_fn failed"


def test_get_hdf_fn():
    
    # get hdf fn
    json_fn = "/Users/donaldpg/pyTAAA_data/naz100_pi/pytaaa_naz100_pi.json"
    symbols_fn = get_symbols_file(json_fn)
    symbols_dir = os.path.split(symbols_fn)[0]
    params = get_json_params(json_fn)
    hdf_fn = os.path.join(symbols_dir, params["stockList"] + "_Symbols_.hdf5")
    
    assert os.path.isfile(hdf_fn), "test_load_hdf failed"
    
    
def test_yfinance_quotes():
    
    # get quotes from hdf fn
    # json_fn = "/Users/donaldpg/pyTAAA_data/naz100_pi/pytaaa_naz100_pi.json"
    # symbols_fn = get_symbols_file(json_fn)
    # symbols_dir = os.path.split(symbols_fn)[0]
    # params = get_json_params(json_fn)
    # hdf_fn = os.path.join(symbols_dir, params["stockList"] + "_Symbols_.hdf5")
    
    symbols= ["AMZN", "NVDA", "F"]
    quote_df = yfinance.download(
        symbols,
        start='2022-12-01',
        end  ='2022-12-06',
    )
    
    
    import yfinance as yf

    # Adjust timeout to a longer period
    yf.shared._EXCHANGE_TIMEOUT = 10  # in seconds

    # read symbols list
    symbols= ["AMZN", "NVDA", "F"]
    start_date = str(
        datetime.datetime.now() - datetime.timedelta(days=5)
    ).split(' ')[0]
    end_date = str(datetime.datetime.now()).split(' ')[0]

    data = yf.download(
        symbols, start=start_date, end=end_date, auto_adjust=False,
        repair=True, timeout=15
    )
    try:
        # for multiple symbols
        symbolList = data['Adj Close'].columns
    except:
        # for single symbol
        symbolList = symbols
    datearray = data['Adj Close'].index
    x = data['Adj Close'].values
    newdates = []
    for i in range(datearray.shape[0]):
        newdates.append(str(datearray[i]).split(' ')[0])
    newdates = np.array(newdates)

    if x.ndim==1:
        x = x.reshape(x.size, 1)

    # ensure values are strings
    symbolList = [str(s) for s in symbolList]
    newdates = [str(s) for s in newdates]

    
    return x, symbolList, newdates
    

if __name__ == "__main__":
    
    test_symbols_fn()
    test_get_hdf_fn()
    
    
    