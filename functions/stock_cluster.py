
def dailyStockClusters():
    import datetime
    import os
    import numpy as np
    import pandas.io.data as web
    from pandas import DataFrame
    from matplotlib import pylab as pl
    from matplotlib import finance
    from matplotlib.collections import LineCollection
    
    from sklearn import cluster, covariance, manifold
    ########################################################################
    ###
    ### This example employs several unsupervised learning techniques to 
    ### extract the stock market structure from variations in historical quotes.
    ### The quantity that we use is the daily variation in quote price: 
    ### quotes that are linked tend to co-fluctuate during a day.
    ###
    ### stocks used are all Nasdaq 100 stocks that have one year of history
    ### from the current date.
    ###
    ### adopted from example at:
    ### http://scikit-learn.org/0.14/auto_examples/applications/plot_stock_market.html
    ###
    ########################################################################
    # Retrieve the data from Internet
    
    # Choose a time period reasonnably calm (not too long ago so that we get
    # high-tech firms, and before the 2008 crash)
    today = datetime.datetime.now()
    d1 = datetime.datetime(today.year-1, today.month, today.day)
    d2 = datetime.datetime(today.year, today.month, today.day)
    
    # input symbols and company names from text file
    companyName_file = os.path.join( os.getcwd(), "symbols",  "companyNames.txt" )
    with open( companyName_file, "r" ) as f:
        companyNames = f.read()
    
    print "\n\n\n"
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
    print " ... symbol_dict = ", symbol_dict
    
    
    symbols = companySymbolList[:]
    names = companyNameList[:]
    
                       
    all_data = {}
    for ticker in symbols:
        try:
            all_data[ticker] = web.get_data_yahoo(ticker, d1, d2)
            qclose = DataFrame({tic: data['Close']
                        for tic, data in all_data.iteritems()})
            qopen = DataFrame({tic: data['Open']
                        for tic, data in all_data.iteritems()})
        except:
            print "Cant find ", ticker
    
    symbols_edit = []
    names_edit = []
    for i, ticker in enumerate( symbols ):
        if True in np.isnan(np.array(qclose[ticker])).tolist():
            print ticker, " nans found, ticker removed"
            del qclose[ticker]
            del qopen[ticker]
        else:
            symbols_edit.append(ticker)
            names_edit.append( names[i] )
    
    # The daily variations of the quotes are what carry most information
    variation = qclose - qopen
    variation[ np.isnan(variation) ] = 0.
    
    
    ###############################################################################
    # Learn a graphical structure from the correlations
    edge_model = covariance.GraphLassoCV()
    
    # standardize the time series: using correlations rather than covariance
    # is more efficient for structure recovery
    X = variation.copy()
    #X = variation.copy().T
    X /= X.std(axis=0)
    edge_model.fit(X)
    
    ###############################################################################
    # Cluster using affinity propagation
    
    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    n_labels = labels.max()
    
    for i in range(n_labels + 1):
        print "Cluster "+str(i)+":"
        for j in range(len(labels)):
            if labels[j] == i:
                print " ... "+names_edit[j]
        #print('Cluster %i: %s' % ((i + 1), ', '.join(names_edit[labels == i])))

    for i in range(n_labels + 1):
        print "Cluster "+str(i)+":"
        for j in range(len(labels)):
            if labels[j] == i:
                print " ... "+names_edit[j]
                
    figure7path = 'Clustered_companyNames.png'  # re-set to name without full path
    figure7_htmlText = "\n<br><h3>Daily stock clustering analyis. Based on one year performance correlations.</h3>\n"
    figure7_htmlText = figure7_htmlText + "\nClustering based on daily variation in Nasdaq 100 quotes.\n"
    figure7_htmlText = figure7_htmlText + '''<br><img src="'''+figure7path+'''" alt="PyTAAA by DonaldPG" width="850" height="500"><br>\n'''

        
    ###############################################################################
    # Find a low-dimension embedding for visualization: find the best position of
    # the nodes (the stocks) on a 2D plane
    
    # We use a dense eigen_solver to achieve reproducibility (arpack is
    # initiated with random vectors that we don't control). In addition, we
    # use a large number of neighbors to capture the large-scale structure.
    node_position_model = manifold.LocallyLinearEmbedding(
        n_components=2, eigen_solver='dense', n_neighbors=6)
    
    embedding = node_position_model.fit_transform(X.T).T
    
    ###############################################################################
    # Visualization
    pl.figure(1, facecolor='w', figsize=(10, 8))
    pl.clf()
    ax = pl.axes([0., 0., 1., 1.])
    pl.axis('off')
    
    # Display a graph of the partial correlations
    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)
    
    # Plot the nodes using the coordinates of our embedding
    pl.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
               cmap=pl.cm.spectral)
    
    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    #a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments,
                        zorder=0, cmap=pl.cm.hot_r,
                        norm=pl.Normalize(0, .7 * values.max()))
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)
    
    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    for index, (name, label, (x, y)) in enumerate(
            zip(names, labels, embedding.T)):
    
        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002
        pl.text(x, y, name, size=10,
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment,
                bbox=dict(facecolor='w',
                          edgecolor=pl.cm.spectral(label / float(n_labels)),
                          alpha=.6))
    
    pl.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
            embedding[0].max() + .10 * embedding[0].ptp(),)
    pl.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
            embedding[1].max() + .03 * embedding[1].ptp())
    
    pl.savefig( os.path.join( os.getcwd(), "pyTAAA_web",  "Clustered_companyNames.png" ), format='png' )
    
    return figure7_htmlText
