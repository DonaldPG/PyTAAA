

def ftpMoveDirectory(  ):

    # based on a demo in ptyhon package paramiko.
    #
    import base64
    import datetime
    import getpass
    import os
    import socket
    import sys
    import traceback

    # local imports
    from functions.GetParams import *

    import paramiko

    # get hostname and credentials
    ftpparams = GetFTPParams()
    hostname = ftpparams['ftpHostname']
    hostIP   = ftpparams['remoteIP']
    username = ftpparams['ftpUsername']
    password = ftpparams['ftpPassword']

    if hostname == '' :
        hostname = raw_input('Hostname: ')
    if len(hostname) == 0:
        print '*** Hostname required.'
        sys.exit(1)
    port = 22
    if hostname.find(':') >= 0:
        hostname, portstr = hostname.split(':')
        port = int(portstr)

    # get username
    if username == '':
        default_username = getpass.getuser()
        username = raw_input('Username [%s]: ' % default_username)
        if len(username) == 0:
            username = default_username

    # now, connect and use paramiko Transport to negotiate SSH2 across the connection
    try:
        print ' connecting to remote server'
        t = paramiko.Transport((hostIP, port))
        t.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(t)

        # dirlist on remote host
        dirlist = sftp.listdir('.')

        # copy this demo onto the server
        remote_path = "/var/www/mysite/pyTAAA_web/"
        try:
            sftp.mkdir( remote_path )
        except IOError:
            print '  ...'+remote_path+' already exists)'

        sftp.open( os.path.join(remote_path, 'README'), 'w').write('This was created by pyTAAA/WriteWebPage.py on DonXPS\n')

        transfer_list = os.listdir("./pyTAAA_web")

        # remove png files from transfer_list except after 10pm
        today = datetime.datetime.now()
        hourOfDay = today.hour
        if hourOfDay < 22 :
            for i in range( len(transfer_list)-1,-1,-1 ):
                name, extension = os.path.splitext( transfer_list[i] )
                if extension == ".png" and name != "PyTAAA_value" or extension == '.db':
                    transfer_list.pop(i)

        for i, local_file in enumerate(transfer_list):
            remote_file = os.path.join( remote_path, local_file )
            sftp.put( os.path.join("./pyTAAA_web/",local_file), remote_file )
            print '  ...created '+remote_file+' on piDonaldPG'

        t.close()

    except Exception, e:
        print '*** Caught exception: %s: %s' % (e.__class__, e)
        traceback.print_exc()
        try:
            t.close()
        except:
            pass
        sys.exit(1)


def piMoveDirectory(  ):

    import shutil
    import os
    import sys
    import datetime

    # local imports
    from functions.GetParams import *

    # create list of files to move and put them in web-accessible folder
    # - nothing here is 'mission critical'. fail without aborting.
    try:
        # get remote path location
        ftpparams = GetFTPParams()
        remote_path = ftpparams['remotepath']
        
        print "\n\n ... diagnostic:  ftpparams = ", ftpparams
              
        # create a target directory if it does not exist already
        try:
            os.mkdirs( remote_path )
        except:
            print '  ...'+remote_path+' already exists)'

        print "\n\n ... diagnostic:  remote_path = ", remote_path
        
        # create README in target directory
        try:
            with open( os.path.join(remote_path, 'README'), 'w') as f:
                f.write('This was created by pyTAAA/WriteWebPage_pi.py on piDonaldPG\n')
                print '  ...'+os.path.join(remote_path, 'README')+' created'
        except:
            #pass
            print '  ...'+os.path.join(remote_path, 'README')+' could not be created. Maybe already exists?'

        # create a list of files to be copied
        source_directory = "./pyTAAA_web"
        transfer_list = os.listdir( source_directory )
        
        print "\n\n ... diagnostic:  transfer_list = ", transfer_list

        # remove png files from transfer_list except after 10pm
        today = datetime.datetime.now()
        hourOfDay = today.hour
        if hourOfDay < 22 :
            for i in range( len(transfer_list)-1,-1,-1 ):
                name, extension = os.path.splitext( transfer_list[i] )
                if extension == ".png" and name != "PyTAAA_value" :
                    transfer_list.pop(i)
                    
        print "\n\n ... updated diagnostic:  transfer_list = ", transfer_list
        
        for f in transfer_list:
            local_file = os.path.join( source_directory, f )
            remote_file = os.path.join( remote_path, f )
            print "\n ... diagnostic:  local_file, remote_file = ", local_file, remote_file
            shutil.copyfile( local_file, remote_file )
            print '  ...created '+remote_file+' on piDonaldPG web server'

    except:
        print " Unable to create updated web page..."

    return


def writeWebPage( regulartext, boldtext, headlinetext, lastdate, last_symbols_text, last_symbols_weight, last_symbols_price ) :
    import smtplib
    import datetime
    import os
    import numpy as np
    # Local imports
    from functions.MakeValuePlot import *

    # message body preliminaries
    message = """<!DOCTYPE html>
<html>
<head>
<title>pyTAAA web</title>
</head>

<body id="w3s" bgcolor=#F2F2F2>

<style>
    body
    {
    background-image: -ms-linear-gradient(top left, #F2F2F2 0%, #94B0B3 80%);

    background-image: -moz-linear-gradient(top left, #F2F2F2 0%, #94B0B3 80%);

    background-image: -webkit-linear-gradient(top left, #F2F2F2 0%, #94B0B3 80%);

    background-image: linear-gradient(to bottom right, #F2F2F2 0%, #94B0B3 80%);
    }
    #rank_table_container
    {
      float:left;
      width:350px;
    }
    #indexchanges_table_container
    {
      position:absolute;
      right:450px;
      width:350px;
    }
</style>


<img src="PyTAAA_stock-chart-blue.png" alt="PyTAAA by DonaldPG" width="1000" height="350">

"""

    ##########################################
    # add current valuation table to message
    ##########################################
    message = message+"""
<h1>"""+headlinetext+"""</h1>
<b>"""+boldtext+"""</b>
<p> </p>
<p>"""+regulartext+"""</p>

"""

    """
    #print "last_symbols_price, type = ", last_symbols_price, type(last_symbols_price)

    # add current valuation table to message
    rankingsMessage = "<br><p>Symbol Rank Wght  Price"
    for i, symbol in enumerate( last_symbols_text ):
        #print "last_symbols_price, type = ", last_symbols_price[i], type(last_symbols_price[i]), format(symbol,'6s'),format(i,'5.0f'),format(last_symbols_weight[i],'5.3f'), format(last_symbols_price[i],'7.2f')
        rankingsMessage = rankingsMessage + "<br>"+format(symbol,'6s') \
                                          + format(i,'5.0f') \
                                          + format(last_symbols_weight[i],'5.3f') \
                                          + format(last_symbols_price[i],'7.2f')
    rankingsMessage = rankingsMessage + "<br>"
    """

    ##########################################
    # read valuations status file and make plot
    ##########################################

    figure_htmlText = makeValuePlot(  )


    ##########################################
    # write text for trading method back-test and insert plot
    ##########################################

    figure2path = "PyTAAA_backtest.png"
    figure_htmlText = figure_htmlText + "\n<br><h3>Original Monte-carlo Backtest plot</h3>\n"
    figure_htmlText = figure_htmlText + "\nHeavy black line is back-tested performance for model. Black shaded area shows performance with different modeling parameters.\n"
    figure_htmlText = figure_htmlText + "\n<br>Heavy red line is for equal-weighted basket of current Naz100 stocks. Red shaded area shows performance of individual stocks.\n"
    figure_htmlText = figure_htmlText + "\n<br>Lower graph shows number of up-trending stocks.\n"
    figure_htmlText = figure_htmlText + '''<br><img src="'''+figure2path+'''" alt="PyTAAA backtest by DonaldPG" width="850" height="500"><br><br>\n'''

    figure3path = "PyTAAA_backtest_updated.png"
    figure_htmlText = figure_htmlText + "\n<br><h3>Updated Monte-carlo Backtest plot</h3>\n"
    figure_htmlText = figure_htmlText + '''<br><img src="'''+figure3path+'''" alt="PyTAAA backtest by DonaldPG" width="850" height="500"><br><br>\n'''



    ##########################################
    # read uptrending stocks status file and make plot
    ##########################################

    figure4_htmlText = makeUptrendingPlot( )


    ##########################################
    # read performance dispersion status file and make plot
    ##########################################

    figure5_htmlText = makeTrendDispersionPlot( )


    ##########################################
    # make plot showing monte carlo backtest using variable percent Long trades
    ##########################################

    figure6_htmlText = makeDailyMonteCarloBacktest( )


    ##########################################
    # add current rankings table to message
    ##########################################

    filepath = os.path.join( os.getcwd(), "pyTAAA_web", "pyTAAAweb_RankList.txt" )
    try:
        with open( filepath, "r" ) as f:
            rankingMessage = f.read()
    except:
        print " Error: unable to read updates from pyTAAAweb_RankList.txt"
        print ""


    ##########################################
    # add table with Nasdaq100 index exchanges to message
    ##########################################

    filepath = os.path.join( os.getcwd(), "symbols", "Naz100_symbolsChanges.txt" )
    try:
        with open( filepath, "r" ) as f:
            input = f.read()
    except:
        print " Error: unable to read updates from pyTAAAweb_RankList.txt"
        print ""

    # separate and remove empty lines
    inputList0 = input.split("\n")

    # remove lines with just "\r"
    inputList = []
    for item in inputList0:
        if item != "\r":
            inputList.append(item)

    # remove "\r" at end of each item in list
    for i,item in enumerate(inputList):
        inputList[i] = item.split("\r")[0]

    # final filtering of each item in list
    inputListFiltered = []
    for item in inputList:
        if item:
            inputListFiltered.append(str(item))

    # print preliminary html tags
    indexExchangesMessage = '<div id="indexchanges_table_container">\n'
    indexExchangesMessage = indexExchangesMessage + "<h3><p>Recent Index Changes :  ..........</p></h3>\n"
    indexExchangesMessage = indexExchangesMessage + "<font face='courier new' size=3><table border='1'>\n"
    indexExchangesMessage = indexExchangesMessage + "<tr><td>Date</td><td>Change</td><td>Ticker</td></tr>\n"

    # print html table entries
    for istring in inputListFiltered :
        date, action, ticker = istring.split()
        indexExchangesMessage = indexExchangesMessage + "<tr><td>"+date \
                                          + "</td><td>"+action \
                                          + "</td><td>"+ticker \
                                          + "</td></tr>\n"
    indexExchangesMessage = indexExchangesMessage + "</table></div>"


    ##########################################
    # Create an updated html page
    ##########################################
    try:
        filepath = os.path.join( os.getcwd(), "pyTAAA_web", "pyTAAAweb.html" )
        with open( filepath, "w" ) as f:
            f.write(message)
            f.write(figure_htmlText)
            f.write(figure4_htmlText)
            f.write(figure5_htmlText)
            f.write(figure6_htmlText)
            f.write(rankingMessage)
            f.write(indexExchangesMessage)
        print " Successfully wrote updates to pyTAAAweb html ", datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
        print ""
    except :
        print " Error: unable to write updates to pyTAAAweb html"
        print ""


    ##########################################
    # Copy web pages to piDonaldPG using ftp (from pc) or using filecopy (from linux)
    ##########################################

    # check the operatiing system to determine whether to move files or use ftp
    import platform

    operatingSystem = platform.system()
    architecture = platform.uname()[4]
    computerName = platform.uname()[1]

    print "  ...platform: ", operatingSystem, architecture, computerName

    if operatingSystem == 'Linux' and architecture == 'armv6l' :
        print "  ...using piMoveDirectory"
        piMoveDirectory(  )
        try:
            piMoveDirectory(  )
        except:
            print "Could not ftp web files..."

    elif operatingSystem == 'Windows' and computerName == 'Don-XPS1530' :
        print "  ...using ftpMoveDirectory"
        try:
            ftpMoveDirectory(  )
        except:
            print "Could not ftp web files..."

    elif operatingSystem == 'Windows' and computerName == 'DonEnvy' :
        print "  ...using ftpMoveDirectory"
        try:
            ftpMoveDirectory(  )
        except:
            print "Could not ftp web files..."

    else:
        print "Could not place web files on server..."




