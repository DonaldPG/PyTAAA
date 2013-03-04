import scheduler
import datetime
import getpass
import os
import time
from functions.SendEmail import *
from functions.GetParams import *
from functions.UpdateSymbols_inHDF5 import *
from functions.CheckMarketOpen import *
from functions.PortfolioStatsOnDate import *

# Get Credentials for sending email
params = GetParams()
print ""
print "params = ", params
print ""
username = str(params['fromaddr']).split("@")[0]

#print "params['fromaddr'], params['toaddrs'], runtime, params['pausetime'], username = ", params['fromaddr'], params['toaddrs'], runtime, params['pausetime'], username
print ""
#pausetime = params['pausetime']
#toaddrs = params['toaddrs']
#fromaddr = params['fromaddr']
print "An email with updated analysis will be sent to ", params['toaddrs'], " every ", params['pausetime'], " seconds"
print params['pausetime'], " seconds is ", format(params['pausetime']/60/60.,'2.1f'), " hours, or ",  \
                                            format(params['pausetime']/60/60/24.,'3.1f'), " days."
print ""

emailpassword = getpass.getpass("Enter a password for : " + str(params['fromaddr']) + "\n")
print "you entered your email password"
print ""

def IntervalTask( ) :

    # Update symbols in HDF5 files based on symbols in list
    symbol_directory = os.getcwd() + "\\symbols"
    symbol_file = "Naz100_symbols.txt"
    symbols_file = os.path.join( symbol_directory, symbol_file )
    start_time = time.time()
    #UpdateHDF5( symbol_directory, symbols_file )
    marketOpen, lastDayOfMonth = CheckMarketOpen()
    elapsed_time = time.time() - start_time

    # TODO
    # Add computations to re-compute stock ranks and weightings
    StatDate = getLastDateFromHDF5( symbol_directory, symbol_file )
    print ""
    print "StatDate = ", StatDate
    print ""
    stockStats = PortfolioStatsOnDate( symbol_directory, symbol_file, params, StatDate )
    for i in range(len(stockStats)):
        print "i, stockStats[i] = ", i, stockStats[i]

    # send an email with status and updates (tries up to 10 times for each call).
    boldtext = "time is "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
    regulartext = "elapsed time was "+str(elapsed_time)

    # Customize and send email
    # - based on day of month and whether market is open or closed
    if marketOpen:
        headlinetext = "Regularly scheduled email (market is open)"
    else:
        headlinetext = "Regularly scheduled email (market is closed)"
    if lastDayOfMonth:
        subjecttext = "PyTAAA holdings update and trade suggestions"
    else:
        subjecttext = "PyTAAA status update"
    SendEmail(username,emailpassword,params['toaddrs'],params['fromaddr'],subjecttext,regulartext,boldtext,headlinetext)

'''
Main program
'''
if __name__ == '__main__':

    # Run scheduled tasks
    #runtime = 60 * 10
    #pausetime = 180

    # Create a scheduler
    my_scheduler = scheduler.Scheduler()

    # Add the mail task, a receipt is returned that can be used to drop the task from the scheduler
    mail_task = scheduler.Task("Interval_Task",
                          datetime.datetime.now(),
                          scheduler.every_x_secs(params['pausetime']),
                          scheduler.RunUntilSuccess( func=IntervalTask ) )

    mail_receipt = my_scheduler.schedule_task(mail_task)

    # Once started, the scheduler will identify the next task to run and execute it.
    my_scheduler.start()

    # Stop the scheduler after runtime
    from time import sleep
    #runtime = 60 * 10
    sleep(params['runtime'])
    my_scheduler.halt()

    # Give it a timeout to halt any running tasks and stop gracefully
    my_scheduler.join(100)




