##### Dependencies:
numpy
scipy
larry
nose
bottleneck
paramiko


##### Usage:

1. create an hdf file that holds stock quotes

   "python re-generateHDF5.py"
  
2. edit the file containing information for PyTAA to report results
   - options exist to send an email to your desired email from another email account (they don't have to be the same email account)
   - edit PyTAAA.params with a text editor and replace example values with your information

3. run PyTAAA with the command: "python PyTAAA.py"
   - the code updates quotes and re-runs every few hours. It runs un-interrupted for 2 weeks (duration can be changed in PyTAAA.params).
   - on the 2nd trading day of the month, PyTAAA recommends new stock holdings
   
4. It's up to the user to decide if they want to do anything with the recommendations. This is designed and provided for entertainment only. The author does not accept and responsibility for anything done by others with the recommendations.

5. To let the code know how to track a portfolio for you, manually update the stock holdings in "PyTAAA_holdings.params".

6. A web page is created in the 'pyTAAA_web' directory. In Windows, you can double-click pyTAAAweb.html to see the latest status and holdings, as recommended by PyTAAA.


##### Notes:

Backtest plots that start ca. 1991 contain different stocks for historical testing than those created by 're-generateHDF5.py'. Therefore backtest plots will not match those created by PyTAAA.py and shown on the created web page. This is due to changes in the Nasdaq 100 index over time.
