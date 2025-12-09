# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 17:43:03 2021

@author: Don
"""

import os
from time import sleep
import datetime

#%pylab
import matplotlib
from matplotlib import pylab as plt
plt.rcParams['figure.dpi'] = 150

import os
# os.chdir('C:\\Users\\Don\\raspberrypi\\Py3TAAADL_tracker')

import numpy as np
import networkx as nx
import sklearn


# from functions.info_theory_attributes import information_mutual_normalised
# from functions.mutual_info import mutual_information
from functions.GetParams import GetEdition, GetSymbolsFile, GetHoldings
from functions.GetParams import get_symbols_file
from functions.GetParams import get_holdings
from functions.GetParams import get_json_params
from functions.GetParams import get_performance_store

#from functions.quotes_for_list_adjClose import *
from functions.TAfunctions import (strip_accents,
                                   SMA_2D,
                                   SMA,
                                   dpgchannel_2D,
                                   computeSignal2D,
                                   percentileChannel_2D,
                                   sharpeWeightedRank_2D)

from functions.TAfunctions import (cleantobeginning,
                                   cleantoend,
                                   interpolate)
#from functions.UpdateSymbols_inHDF5 import UpdateHDF5, loadQuotes_fromHDF
from functions.UpdateSymbols_inHDF5 import UpdateHDF_yf, loadQuotes_fromHDF

from networkx.drawing.nx_agraph import to_agraph

#---------------------------------------------

def make_networkx_spanning_tree_plot(json_fn, plot_name):
    print("*************************************************************")
    print("*************************************************************")
    print("***                                                       ***")
    print("***                                                       ***")
    print("***  daily minimum-spanning tree analysis                 ***")
    print("***                                                       ***")
    print("***                                                       ***")
    print("*************************************************************")
    print("*************************************************************")

    sleep(5)

    # number of monte carlo scenarios
    randomtrials = 31
    randomtrials = 12
    randomtrials = 4
    edition = GetEdition()
    if edition == 'pi' or edition == 'pine64':
        randomtrials = 12
    elif edition == 'Windows32':
        randomtrials = 25
    elif edition == 'Windows64':
        randomtrials = 51
        randomtrials = 15

    # get symbol list of for current holdings
    holdings_dict = get_holdings(json_fn)
    holdings_symbols =  holdings_dict['stocks']
    holdings_symbols = list(set(holdings_symbols))
    holdings_symbols = [h for h in holdings_symbols if h != "CASH"]

    # get data_store fn
    data_store_fn = get_performance_store(json_fn)
    data_store_fn = get_symbols_file(json_fn)

    ##
    ##  Import list of symbols to process.
    ##
    # symbols_file = GetSymbolsFile()
    symbols_file = get_symbols_file(json_fn)
    current_stock_holdings = list(set(get_holdings(json_fn)['stocks']))
    current_stock_holdings = [x for x in current_stock_holdings if x != "CASH"]

    adjClose, adjSymbols, datearray, _, _ = loadQuotes_fromHDF(symbols_file, json_fn)
    firstdate = datearray[0]

    for iii in range(len(adjSymbols)):
        print(
            " i,symbols[i],datearray[-1],adjClose[i,-1] = ",
            iii,adjSymbols[iii],datearray[-1],adjClose[iii,-1]
        )

    # Clean up missing values in input quotes
    #  - infill interior NaN values using nearest good values to linearly interpolate
    #  - copy first valid quote from valid date to all earlier positions
    #  - copy last valid quote from valid date to all later positions
    for ii in range(adjClose.shape[0]):
        adjClose[ii,:] = interpolate(adjClose[ii,:])
        adjClose[ii,:] = cleantobeginning(adjClose[ii,:])
        adjClose[ii,:] = cleantoend(adjClose[ii,:])

    gainloss= adjClose[:,1:]/adjClose[:,:-1]
    # gainloss20 = gainloss[:,-20:]

    # # get company names
    # # input symbols and company names from text file
    # if 'Naz100' in symbols_file:
    #     companyName_file = os.path.join( os.getcwd(), "symbols",  "companyNames.txt" )
    # elif 'SP500' in symbols_file:
    #     companyName_file = os.path.join( os.getcwd(), "symbols",  "SP500_companyNames.txt" )
    # with open( companyName_file, "r" ) as f:
    #     companyNames = f.read()

    # print("\n\n\n")
    # companyNames = companyNames.split("\n")
    # ii = companyNames.index("")
    # del companyNames[ii]
    # companySymbolList  = []
    # companyNameList = []
    # for iname,name in enumerate(companyNames):
    #     name = name.replace("amp;", "")
    #     testsymbol, testcompanyName = name.split(";")
    #     testcompanyName = strip_accents(testcompanyName)
    #     companySymbolList.append(testsymbol)
    #     companyNameList.append(testcompanyName)

    # get data_store fn
    data_store_fn = get_performance_store(json_fn)
    data_store_fn = get_symbols_file(json_fn)

    params = get_json_params(json_fn)
    stockList = params['stockList']

    # input symbols and company names from text file
    if stockList == 'Naz100':
        companyName_file = os.path.join(
            os.path.split(data_store_fn)[0],  "companyNames.txt"
        )
    elif stockList == 'SP500':
        # companyName_file = os.path.join( os.getcwd(), "symbols",  "SP500_companyNames.txt" )
        companyName_file = os.path.join(
            os.path.split(data_store_fn)[0],  "SP500_companyNames.txt"
        )
    with open( companyName_file, "r" ) as f:
        companyNames = f.read()

    print("\n\n\n")
    companyNames = companyNames.split("\n")
    ii = companyNames.index("")
    del companyNames[ii]
    companySymbolList  = []
    companyNameList = []
    symbol_dict = {}
    for iname,name in enumerate(companyNames):
        name = name.replace("amp;", "")
        testsymbol, testcompanyName = name.split(";")
        companySymbolList.append(format(testsymbol,'s'))
        companyNameList.append(format(testcompanyName,'s'))
        if testsymbol != "CASH":
            symbol_dict[ testsymbol ] = format(testcompanyName,'s')
    print(" ... symbol_dict = ", symbol_dict)

    symbols = companySymbolList[:]
    names = companyNameList[:]

    for i,ticker in enumerate(symbols):
        symbols[i] = ticker.replace('.','-')

    print(" ... length of symbols = ", len(symbols))
    print(" ... length of names = ", len(names))


    # create nodes for a knowledge graph
    G = nx.Graph()

    # compute pair-wise correlation coefficients
    num_active = 1
    elist = []
    gains = {}
    xcorr_arr = np.zeros((len(companySymbolList), len(companySymbolList)), float)
    for ii, isymb in enumerate(companySymbolList):
        i = adjSymbols.index(isymb)
        if gainloss[i,-20:].mean() != 1.0:
            G.add_node(isymb)
            gains[isymb] = np.cumprod(gainloss[i,-20:])[-1]
            G.nodes[isymb]['gains'] = gains[isymb]
        for jj, jsymb in enumerate(companySymbolList):
            j = adjSymbols.index(jsymb)
            if gainloss[i,-20:].mean() != 1.0 and \
               gainloss[j,-20:].mean() != 1.0 and \
               j > i:
                xcorr = np.corrcoef(gainloss[i,-20:], gainloss[j,-20:])[0,1]
                if ~np.isnan(xcorr):
                    elist.append((isymb, jsymb, xcorr))
                    print(format(num_active, '4d') + "  "+\
                            format(isymb,'5s')+ "-- " + \
                            format(jsymb,'5s')+ \
                            format(xcorr,'7.3f'))
                    xcorr_arr[ii,jj] = xcorr
                    xcorr_arr[jj,ii] = xcorr
                    num_active += 1


    '''
    num_active = 1
    elist = []
    gains = {}
    xcorr_arr = np.zeros((len(companySymbolList), len(companySymbolList)), float)
    icount = 0
    for i,isymb in enumerate(symbols):
        if isymb not in companySymbolList:
            continue
        if gainloss[icount,-20:].mean() != 1.0:
            G.add_node(isymb)
            gains[isymb] = np.cumprod(gainloss[icount,-20:])[-1]
            G.nodes[isymb]['gains'] = gains[isymb]
        jcount = 0
        for j,jsymb in enumerate(symbols):
            if gainloss[icount,-20:].mean() != 1.0 and \
               gainloss[jcount,-20:].mean() != 1.0 and \
               jcount > icount:
                xcorr = np.corrcoef(gainloss[icount,-20:], gainloss[jcount,-20:])[0,1]
                if ~np.isnan(xcorr):
                    elist.append((isymb, jsymb, xcorr))
                    print(format(num_active, '4d') + "  "+\
                    format(isymb,'5s')+ "-- " + \
                    format(jsymb,'5s')+ \
                    format(xcorr,'7.3f'))
                    xcorr_arr[icount,jcount] = xcorr
                    xcorr_arr[jcount,icount] = xcorr
                    num_active += 1
                    icount += 1
                    jcount += 1
    '''

    # create a plot of the knowledge graph
    # add edges between nodes
    # plt.figure(1, figsize=(12,9))
    # plt.clf()
    # plt.imshow(xcorr_arr, cmap='jet', interpolation='nearest')
    # plt.colorbar()
    # plot_fn = "stock_correlations_table.png"
    # plt.savefig(plot_fn, dpi=150, format="png")

    # create a histogram of all stock correlations
    xcorr_vals = xcorr_arr[xcorr_arr!=0]
    plt.figure(2, figsize=(12,9))
    plt.clf()
    plt.hist(xcorr_vals, 101)
    plot_fn = "stock_correlations_histogram.png"
    plt.savefig(plot_fn, dpi=150, format="png")


    # # create a plot with tress of stock cluster
    # G.add_weighted_edges_from(elist)
    # edge_width = [xcorr_arr[xcorr_arr!=0].min() + 3. * G[u][v]['weight'] for u, v in G.edges()]

    # node_size = [25.8*2. +  100.* nx.get_node_attributes(G, 'gains')[v] for v in G]

    # plt.figure(3, figsize=(12,9))
    # plt.clf()
    # nx.draw_networkx(
    #     G, node_size = node_size,
    #     with_labels = True,
    #     edge_color ='.4', cmap = plt.cm.Blues
    # )
    # plot_fn = "stock_clusters_3.png"
    # plt.savefig(plot_fn, dpi=150, format="png")


    ### ------------------------------------------------------
    ###
    ### ------------------------------------------------------
    GG = nx.Graph()
    current_symbols = []
    num_symbols_i_graph = 0
    for i,isymb in enumerate(companySymbolList):
        ii = adjSymbols.index(isymb)
        correlations = xcorr_arr[i,:] * 1.
        indx_most_correlated = np.argmax(correlations)
        iindx_most_correlated = adjSymbols.index(companySymbolList[indx_most_correlated])
        if xcorr_arr[i,indx_most_correlated] != 1.:
            GG.add_node(isymb)
            current_symbols.append(isymb)
            #gains[isymb] = np.cumprod(gainloss[i,-20:])[-1]
            GG.nodes[isymb]['gains'] = np.cumprod(gainloss[ii,-20:])[-1]
            if companySymbolList[indx_most_correlated] not in current_symbols:
                GG.add_node(companySymbolList[indx_most_correlated])
                GG.nodes[isymb]['gains'] = np.cumprod(gainloss[ii,-20:])[-1]
            print(format(num_symbols_i_graph,' 5d')+ "  " + \
                  format(isymb,'5s')+ "--" + \
                  format(companySymbolList[indx_most_correlated],'5s')+ \
                  format(np.corrcoef(gainloss[ii,-20:],
                                     gainloss[iindx_most_correlated,-20:])[0,1],'7.3f'))
            GG.add_edge(isymb, companySymbolList[indx_most_correlated],
                        weight = xcorr_arr[i,indx_most_correlated])
            num_symbols_i_graph += 1

    '''
    GG = nx.Graph()
    current_symbols = []
    num_symbols_i_graph = 0
    for i,isymb in enumerate(symbols):
        correlations = xcorr_arr[i,:] * 1.
        if correlations.sum() == 0. and correlations.mean() == 0.:
            continue
        correlations[np.isnan(correlations)]=-100.
        indx_most_correlated = np.argmax(correlations)
        if xcorr_arr[i,indx_most_correlated] != 1.:
            GG.add_node(isymb)
            current_symbols.append(isymb)
            #gains[isymb] = np.cumprod(gainloss[i,-20:])[-1]
            GG.nodes[isymb]['gains'] = np.cumprod(gainloss[i,-20:])[-1]
            if symbols[indx_most_correlated] not in current_symbols:
                GG.add_node(symbols[indx_most_correlated])
                GG.nodes[isymb]['gains'] = np.cumprod(gainloss[i,-20:])[-1]
            print(format(num_symbols_i_graph,' 5d')+ "  " + \
            format(isymb,'5s')+ "--" + \
            format(symbols[indx_most_correlated],'5s')+ \
            format(np.corrcoef(gainloss[i,-20:], gainloss[indx_most_correlated,-20:])[0,1],'7.3f'))
            GG.add_edge(isymb, symbols[indx_most_correlated],
                        weight = xcorr_arr[i,indx_most_correlated])
            num_symbols_i_graph += 1
    '''

    # node_size = [25.8*2. +  100.* nx.get_node_attributes(GG, 'gains')[v] for v in GG]
    node_size = [.8*2. +  1.* nx.get_node_attributes(GG, 'gains')[v] for v in GG]

    edge_width = [.5 + 1* (GG[u][v]['weight']) for u, v in GG.edges()]

    plt.figure(4, figsize=(18,10))
    plt.clf()
    plt.title('stock clusters (forest of trees)')
    nx.draw_networkx(
        GG, node_size = node_size,
        with_labels = True, width = edge_width,
        edge_color ='.4', cmap = plt.cm.Blues
    )
    plot_fn = "stock_clusters.png"
    plt.savefig(plot_fn, dpi=150, format="png")


    ### ------------------------------------------------------
    ### minimum spanning tree
    ### ------------------------------------------------------
    G3 = nx.Graph()
    current_symbols = []
    current_pairs = []
    num_symbols_i_graph = 0
    gain_scalar = 1.
    for i,isymb in enumerate(companySymbolList):

        percent_progress = i / float(len((companySymbolList)))
        print("\r ... compute correlations. progress " + format(percent_progress, "5.1%"), end="")

        ii = adjSymbols.index(isymb)
        icorrelations = xcorr_arr[i,:] * 1.
        if icorrelations.sum() == 0. and icorrelations.mean() == 0.:
            continue
        icorrelations[np.isnan(icorrelations)]=-100.
        for j,jsymb in enumerate(companySymbolList):

            jj = adjSymbols.index(jsymb)
            jcorrelations = xcorr_arr[i,j] * 1.
            if jcorrelations.sum() == 0. and jcorrelations.mean() == 0.:
                continue
            indx_most_correlated = np.argmax(icorrelations)
            if xcorr_arr[i,indx_most_correlated] != 1.:
                if isymb not in current_symbols:
                    print("adding node "+isymb)
                    G3.add_node(isymb)
                    G3.nodes[isymb]['gains'] = np.cumprod(gainloss[ii,-20:])[-1] * gain_scalar
                    current_symbols.append(isymb)
                if jsymb not in current_symbols:
                    print("adding node "+jsymb)
                    G3.add_node(jsymb)
                    G3.nodes[jsymb]['gains'] = np.cumprod(gainloss[jj,-20:])[-1] * gain_scalar
                    current_symbols.append(jsymb)
                if (isymb,jsymb) in current_pairs or (jsymb,isymb) in current_pairs:
                    continue
                '''
                print(format(num_symbols_i_graph,' 5d')+ "  " + \
                format(isymb,'5s')+ "-- " + \
                format(jsymb,'5s')+ \
                format(np.corrcoef(gainloss[i,-20:], gainloss[j,-20:])[0,1],'7.3f'))
                '''


                G3.add_edge(isymb, jsymb, weight = np.sqrt(2 * (1. - xcorr_arr[i,j])))
                current_pairs.append((isymb, jsymb))
                num_symbols_i_graph += 1


                '''
                nmus = sklearn.metrics.normalized_mutual_info_score
                from sklearn.metrics import explained_variance_score
                nmus = sklearn.metrics.explained_variance_score
                nmus_val = (nmus(gainloss[i,-20:], gainloss[j,-20:]) + \
                            nmus(gainloss[j,-20:], gainloss[i,-20:])) /2.
                print(str((i,j))+" symbols "+symbols[i]+", "+symbols[j]+" = "+format(nmus_val,'5.2f'))
                G3.add_edge(isymb, jsymb, weight = np.sqrt(1.0 - nmus_val))
                current_pairs.append((isymb, jsymb))
                num_symbols_i_graph += 1
                '''


    T=nx.minimum_spanning_tree(G3)
    node_size = [25.8*2. +  100.* nx.get_node_attributes(G3, 'gains')[v] for v in G3]
    edge_width = [2. + 5* (G3[u][v]['weight']) for u, v in G3.edges()]

    """
    # minimum spanning tree
    G3 = nx.Graph()
    current_symbols = []
    current_pairs = []
    num_symbols_i_graph = 0
    gain_scalar = 1.
    for i,isymb in enumerate(symbols):
        icorrelations = xcorr_arr[i,:] * 1.
        if icorrelations.sum() == 0. and icorrelations.mean() == 0.:
            continue
        icorrelations[np.isnan(icorrelations)]=-100.
        for j,jsymb in enumerate(symbols):
            jcorrelations = xcorr_arr[i,j] * 1.
            if jcorrelations.sum() == 0. and jcorrelations.mean() == 0.:
                continue
            indx_most_correlated = np.argmax(icorrelations)
            if xcorr_arr[i,indx_most_correlated] != 1.:
                if isymb not in current_symbols:
                    print("adding node "+isymb)
                    G3.add_node(isymb)
                    G3.nodes[isymb]['gains'] = np.cumprod(gainloss[i,-20:])[-1] * gain_scalar
                    current_symbols.append(isymb)
                if jsymb not in current_symbols:
                    print("adding node "+jsymb)
                    G3.add_node(jsymb)
                    G3.nodes[jsymb]['gains'] = np.cumprod(gainloss[j,-20:])[-1] * gain_scalar
                    current_symbols.append(jsymb)
                if (isymb,jsymb) in current_pairs or (jsymb,isymb) in current_pairs:
                    continue
                '''
                print(format(num_symbols_i_graph,' 5d')+ "  " + \
                format(isymb,'5s')+ "-- " + \
                format(jsymb,'5s')+ \
                format(np.corrcoef(gainloss[i,-20:], gainloss[j,-20:])[0,1],'7.3f'))
                '''

                '''
                G3.add_edge(isymb, jsymb, weight = np.sqrt(2 * (1. - xcorr_arr[i,j])))
                current_pairs.append((isymb, jsymb))
                num_symbols_i_graph += 1
                '''

                nmus = sklearn.metrics.normalized_mutual_info_score
                from sklearn.metrics import explained_variance_score
                nmus = sklearn.metrics.explained_variance_score
                nmus_val = (nmus(gainloss[i,-20:], gainloss[j,-20:]) + \
                            nmus(gainloss[j,-20:], gainloss[i,-20:])) /2.
                print(str((i,j))+" symbols "+symbols[i]+", "+symbols[j]+" = "+format(nmus_val,'5.2f'))
                G3.add_edge(isymb, jsymb, weight = np.sqrt(1.0 - nmus_val))
                current_pairs.append((isymb, jsymb))
                num_symbols_i_graph += 1


    T=nx.minimum_spanning_tree(G3)
    node_size = [25.8*2. +  100.* nx.get_node_attributes(G3, 'gains')[v] for v in G3]
    edge_width = [2. + 5* (G3[u][v]['weight']) for u, v in G3.edges()]
    """
    '''
    plt.figure(4, figsize=(16,12))
    plt.clf()
    nx.draw_networkx(T, with_labels=True,
                     pos=nx.spring_layout(T),
                     #node_color=range(len(current_symbols)//2),
                     node_size=100,
                     #node_size=node_size,
                     #vmin=np.array(node_size).min(), vmax=np.array(node_size).max(),
                     #cmap=plt.cm.jet,
                     #node_size=range(len(current_symbols)//2)
                     #cmap=plt.cm.Blues
                     )
    plt.show()
    '''

    plt.figure(5, figsize=(16,12))
    plt.clf()
    #labels = {i : node_names[i][1] for i in T.nodes()}
    #colors = {i : node_attributes[labels[i]] for i in T.nodes()}
    # cmap = matplotlib.cm.get_cmap('gnuplot_r')
    cmap = matplotlib.colormaps.get_cmap('gnuplot_r')
    rgba = [cmap(i) for i in np.linspace(0.,1.3,len(list(T.nodes())))]
    node_color = [cmap(nx.get_node_attributes(G3, 'gains')[v]*gain_scalar) for v in T]

    node_color_list = []
    for i,inn in enumerate(T.nodes()):
        T.nodes[inn]["color"] = 'white'
        T.nodes[inn]["style"] = "filled"
        print("\n"+str((i,inn,inn in current_stock_holdings)))
        if inn in current_stock_holdings:
            node_color_list.append((1.,.7,.4))
        else:
            node_color_list.append((.5,.5,1.))
    color=nx.get_node_attributes(T,'color')
    fillcolor=nx.get_node_attributes(T,'fillcolor')
    #H=nx.relabel_nodes(T,labels)
    nx.draw(T, #scale=30,
            nodelist=T.nodes(), linewidths=0,
            #node_color=range(len(current_symbols)),
            node_color=node_color_list,
            cmap=plt.cm.gnuplot_r,
            with_labels = True, node_size=1200,font_size=12, font_color='white')

    _t = to_agraph(T)
    #_t.graph_attr.update(landscape='true', size="12,9", dpi='300', fontsize='18')
    _t.graph_attr.update(landscape='false', size="12,9", dpi='300', fontsize='18')
    #_t.write('_t.dot')
    #_t.node_attr['shape']='circle'
    #_t.node_attr['style']='filled'
    #_t.node_attr['fillcolor']='#%02x%02x%02x' % (180, 180, 255)
    _t.node_attr['fillcolor']='#%02x%02x%02x%02x' % (255, 255, 255, 100)
    _t.node_attr["fontsize"] = "18"
    _t.node_attr["fontname"] = "Helvetica"
    for inode in current_stock_holdings:
        if inode == 'CASH':
            continue
        n = _t.get_node(inode)
        #n.attr['shape']='circle'
        n.attr['style']='filled'
        n.attr['fillcolor']='#%02x%02x%02x%02x' % (255, 220, 150, 100)
    try:
        os.remove("minimum_spanning_tree.png")
    except:
        pass
    dt = datetime.datetime.now()
    _t.graph_attr["fontsize"] = "18."
    _t.graph_attr["fontname"] = "Helvetica"
    _t.graph_attr.update(label='minimum spanning tree for Nasdaq 100 stocks '+\
            '%s/%s/%s' % (dt.month, dt.day, dt.year), fontsize="36", fontcolor="red")
    #_t.graph_attr.update(fontsize=18)
    _t.draw(plot_name, prog="neato")



    ### -----------------------------------------------------
    ### print list of most-connected stocks
    ### -----------------------------------------------------

    mst = nx.minimum_spanning_edges(G3, algorithm="kruskal", data=False)
    edgelist = list(mst)
    sorted(sorted(e) for e in edgelist)

    print("\n\nMost connected companies in minimum spanning-tree graph")
    num_connections_list = []
    for i,isymb in enumerate(symbols):
        num_connections = len([e for e in edgelist if isymb in e])
        num_connections_list.append(num_connections)
    symbols_sorted_by_connections = [x for _,x in sorted(zip(num_connections_list,symbols))]
    num_connections_list.sort()
    # reverse both lists
    symbols_sorted_by_connections = symbols_sorted_by_connections[::-1]
    num_connections_list = num_connections_list[::-1]

    last_index = num_connections_list.index(3)
    for inum in range(100,0,-1):
        try:
            first_index = num_connections_list.index(inum)
            last_index = num_connections_list.index(inum-1)
            if last_index > first_index:
                print("\n"+str(inum)+" connections:")
                for i in range(first_index, last_index):
                    isymb = symbols_sorted_by_connections[i]
                    symbol_index = companySymbolList.index(isymb)
                    print(format(isymb,'5s'), companyNameList[symbol_index], str(num_connections_list[i]))
        except:
            pass

    ### -----------------------------------------------------
    ### print list of least-connected stocks (most anomalous?)
    ### -----------------------------------------------------

    def most_common(lst):
        return max(set(lst), key=lst.count)
    maxst = nx.maximum_spanning_edges(G3, algorithm="kruskal", data=False)
    edgelist = list(maxst)
    maxst_nodes = []
    for e in edgelist:
        maxst_nodes.append(e[0])
        maxst_nodes.append(e[1])

    print("\n\nleast connected companies from maximum spanning-tree graph")
    for i in range(10):
        highest_count_element = most_common(maxst_nodes)
        symbol_index = companySymbolList.index(highest_count_element)
        print(i, highest_count_element, companyNameList[symbol_index], maxst_nodes.count(highest_count_element))
        maxst_nodes = [e for e in maxst_nodes if e != highest_count_element]

    return



if __name__ == "__main__":

    json_fn = "/Users/donaldpg/pyTAAA_data/naz100_pi/pytaaa_naz100_pi.json"
    json_fn = "/Users/donaldpg/pyTAAA_data/sp500_hma/pytaaa_sp500_hma.json"

    plot_name = "networkx_spanning_tree.png"

    folder = os.path.join(os.path.dirname(__file__), "..")
    print(" ... folder = " + folder)

    os.chdir(folder)

    make_networkx_spanning_tree_plot(json_fn, plot_name)




