import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as sco
import math as m
import scipy as sp
import pandas_datareader as pd_data
import datetime

def mean_var_opt (weights, rtns, risk_aversion, covdata):
    #calculate expected returns = w.t + w0 rf
    #var(rp) = wt sum of weights we want to find
    rtnP = np.sum(weights*rtns.mean()*252)
    varP = np.dot(weights.T, np.dot(covdata.cov()*252,weights))
    return (rtnP - risk_aversion * (varP))

#For first period Jan 2011

start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2010, 12, 31)

symbols = ['AAPL','GS', 'GC=F', 'GE', 'BAC', 'PFE']
t1 = pd_data.DataReader(symbols,'yahoo',start_date,end_date)['Adj Close']

t1 = t1.dropna()
t1 = np.log(t1).diff().dropna()
t1['LIBOR'] = 0


start_date1 = '2011-01-01'
end_date1 = '2011-01-31'

symbols = ['AAPL','GS', 'GC=F', 'GE', 'BAC', 'PFE']
rtns1 = pd_data.DataReader(symbols,'yahoo',start_date1,end_date1)['Adj Close'] #importing data as dataframe
rtns1 = np.log(rtns1).diff().dropna()
rtns1['LIBOR'] = 0
nn = len(rtns1.columns)

cons = ({'type':'eq','fun': lambda x: np.sum(x)-1}) #add up to 1
bnds = tuple((0,1) for x in range(nn)) #only between 0 and 1
# Initialise weights
w0 =  [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]
w0 = np.array(w0)

def max_returnt1 (weights):
    return -(mean_var_opt(weights, rtns1, 2, t1)) 

wt1 = sco.minimize(max_returnt1, w0, method = 'SLSQP', bounds= bnds, constraints = cons)
display(wt1['x']) #for january 2011

#For second period Feb 2011
start_date = '2010-01-01'
end_date = '2011-01-31'

symbols = ['AAPL','GS', 'GC=F', 'GE', 'BAC', 'PFE']
t2 = pd_data.DataReader(symbols,'yahoo',start_date,end_date)['Adj Close']

t2 = t2.dropna()
t2 = np.log(t2).diff().dropna()
t2['LIBOR'] = 0


start_date2 = '2011-02-01'
end_date2 = '2011-02-31'

symbols = ['AAPL','GS', 'GC=F', 'GE', 'BAC', 'PFE']
rtns2 = pd_data.DataReader(symbols,'yahoo',start_date1,end_date1)['Adj Close'] #importing data as dataframe
rtns2 = np.log(rtns2).diff().dropna()
rtns2['LIBOR'] = 0


cons = ({'type':'eq','fun': lambda x: np.sum(x)-1}) #add up to 1
bnds = tuple((0,1) for x in range(nn)) #only between 0 and 1
# Initialise weights
w0 =  [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]
w0 = np.array(w0)

def max_returnt2 (weights):
    return -(mean_var_opt(weights, rtns2, 2, t2)) 

wt2 = sco.minimize(max_returnt2, w0, method = 'SLSQP', bounds= bnds, constraints = cons)
display(wt2['x']) #for february 2011


#through a loop
start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2010, 11, 30) # for the covariance matrix
a = 2 #number of cycles i.e. goes by months
weights_array = []

for i in range(a):
    
    end_date = end_date + datetime.timedelta(days=30) #not exactly a month
    #print(end_date)
    symbols = ['AAPL','GS', 'GC=F', 'GE', 'BAC', 'PFE']
    t1 = pd_data.DataReader(symbols,'yahoo',start_date,end_date)['Adj Close']

    t1 = t1.dropna()
    t1 = np.log(t1).diff().dropna()
    t1['LIBOR'] = 0
    #display(t1)


    start_date1 = datetime.datetime(2011, i+1, 1) #for the expected returns forecasts
    end_date1 = start_date1 + datetime.timedelta(days = 30) #average number of days in a month
    

    symbols = ['AAPL','GS', 'GC=F', 'GE', 'BAC', 'PFE']
    rtns1 = pd_data.DataReader(symbols,'yahoo',start_date1,end_date1)['Adj Close'] #importing data as dataframe
    rtns1 = np.log(rtns1).diff().dropna()
    rtns1['LIBOR'] = 0
    

    cons = ({'type':'eq','fun': lambda x: np.sum(x)-1}) #add up to 1
    bnds = tuple((0,1) for x in range(nn)) #only between 0 and 1
    # Initialise weights
    w0 =  [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]
    w0 = np.array(w0)

    def max_returnt1 (weights):
        return -(mean_var_opt(weights, rtns1, 2, t1)) 

    wt1 = sco.minimize(max_returnt1, w0, method = 'SLSQP', bounds= bnds, constraints = cons)
    display(wt1['x'].round(3))
    weights_array.append(wt1)



