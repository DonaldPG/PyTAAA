import numpy as np

#----------------------------------------------

def newHighsAndLows(datearray,\
                    adjClose,\
                    num_days_highlow=252,\
                    num_days_cumu=21,\
                    makeQCPlots=True):
    ###
    ### Count new highs and new lows over lookback period of number_days_highlow
    ### - indices in 2D input array: adjClose[number_symbols,number_days]

    newLows_2D = np.zeros_like( adjClose )
    newHighs_2D = np.zeros_like( adjClose )
    for i in range( adjClose.shape[0] ):

        newLows = np.zeros_like( adjClose[i,:] )
        newHighs = np.zeros_like( adjClose[i,:] )
        for j in range( num_days_highlow, adjClose.shape[1] ):

            index_lo = np.argmin( adjClose[i,j-num_days_highlow:j] )
            index_hi = np.argmax( adjClose[i,j-num_days_highlow:j] )

            if index_lo == j:
                newLows[index_lo] += 1
            if index_hi == j:
                newHighs[index_hi] += 1

        # compute simiple-moving-sum over num_days_cumu
        # - to make it easier to see clusters of new highs and lows
        newLows_2D[i,:] = SMS(newLows,periods)
        newHighs_2D[i,:] = SMS(newHighs,periods)

    if makeQCPlots:

        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pylab as plt
        filepath = os.path.join( os.getcwd(), "pyTAAA_web" )

        today = datetime.datetime.now()

        try:
            plt.clf()
            plt.grid(True)
            plt.plot(datearray,np.sum(newHighs_2D,asix=0),label='new highs')
            plt.plot(datearray,np.sum(newLows_2D,asix=0),label='new lows')
            plt.legend()
            # put text line with most recent date at bottom of plot
            # - get 7.5% of x-scale and y-scale for text location
            x_range = datearray[-1] - datearray[0]
            text_x = datearray[0] + datetime.timedelta( x_range.days / 20. )
            plt.ylim((-20,20))
            text_y = ( 40. * .085 -20.)
                    # set up to use dates for labels
            xlocs = []
            xlabels = []
            for i in xrange(1,len(datearray)):
                if datearray[i].year != datearray[i-1].year:
                    xlocs.append(i)
                    xlabels.append(str(datearray[i].year))
            print "xlocs,xlabels = ", xlocs, xlabels
            if len(xlocs) < 12 :
                xticks(xlocs, xlabels)
            else:
                xticks(xlocs[::2], xlabels[::2])
            plot_text = "most recent value from "+str(datearray[-1])+\
                        "\nplotted at "+today.strftime("%A, %d. %B %Y %I:%M%p")
            plt.text( text_x,text_y, plot_text, fontsize=8 )
            plt.title("newHighs_newLos_counts of new highs and new lows"+"\n"+
                      "Uses "+str(num_days_highlow)+" days for window with "+
                      str(num_days_cumu)+" days sum for display", fontsize=11)
            plotfilepath = os.path.join( filepath, "newHighs_newLos_count.png" )
            plt.savefig( plotfilepath, format='png' )
        except:
            pass

    return newHighs_2D, newLows_2D
