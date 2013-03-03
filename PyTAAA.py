import scheduler
import datetime
import getpass
import os
import time
from functions.SendEmail import *
from functions.GetParams import *
from functions.UpdateSymbols_inHDF5 import *

# Get Credentials for sending email
fromaddr, toaddrs, max_uptime, seconds_between_runs = GetParams()
username = str(fromaddr).split("@")[0]

#print "fromaddr, toaddrs, max_uptime, seconds_between_runs, username = ", fromaddr, toaddrs, max_uptime, seconds_between_runs, username
print ""
print "An email with updated analysis will be sent to ", toaddrs, " every ", seconds_between_runs, " seconds"
print seconds_between_runs, " seconds is ", format(seconds_between_runs/60/60.,'2.1f'), " hours, or ",  \
                                            format(seconds_between_runs/60/60/24.,'3.1f'), " days."
print ""

emailpassword = getpass.getpass("Enter a password for : " + str(fromaddr) + "\n")
print "you entered your email password"
print ""

def IntervalTask( ) :

    # TODO
    # Add computations to re-compute stock ranks and weightings
    # Update symbols in HDF5 files based on symbols in list
    symbol_directory = os.getcwd() + "\\symbols"
    symbol_file = "Naz100_symbols.txt"
    symbols_file = os.path.join( symbol_directory, symbol_file )
    start_time = time.time()
    UpdateHDF5( symbol_directory, symbols_file )
    elapsed_time = time.time() - start_time

    # send an email with status and updates (tries up to 10 times for each call).
    subjecttext = "eScheduled SMTP HTML e-mail test"
    headlinetext = "Regularly scheduled email"
    boldtext = "time is "+datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
    regulartext = "elapsed time was "+str(elapsed_time)

    SendEmail(username,emailpassword,toaddrs,fromaddr,subjecttext,regulartext,boldtext,headlinetext)

'''
Main program
'''
if __name__ == '__main__':

    # Run scheduled tasks
    #max_uptime = 60 * 10
    #seconds_between_runs = 180

    # Create a scheduler
    my_scheduler = scheduler.Scheduler()

    # Add the mail task, a receipt is returned that can be used to drop the task from the scheduler
    mail_task = scheduler.Task("Interval_Task",
                          datetime.datetime.now(),
                          scheduler.every_x_secs(seconds_between_runs),
                          scheduler.RunUntilSuccess( func=IntervalTask ) )

    mail_receipt = my_scheduler.schedule_task(mail_task)

    # Once started, the scheduler will identify the next task to run and execute it.
    my_scheduler.start()

    # Stop the scheduler after max_uptime
    from time import sleep
    #max_uptime = 60 * 10
    sleep(max_uptime)
    my_scheduler.halt()

    # Give it a timeout to halt any running tasks and stop gracefully
    my_scheduler.join(100)




