import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as sco
import math as m
import scipy as sp
import pandas_datareader as pd_data


start_date = '2010-01-01'
end_date = '2013-01-04'

symbols = ['AAPL','GS', 'GC=F', 'GE', 'BAC', 'PFE']
df = pd_data.DataReader(symbols,'yahoo',start_date,end_date)['Adj Close']

df = df.dropna()
dfrtn = np.log(df).diff().dropna()

dfrtn['LIBOR'] = 0 

nn = len(dfrtn.columns) #saves number of columns of the log diff matrix in variable nn.

def mean_var_opt (weights, rtns, risk_aversion):
    #calculate expected returns = w.t + w0 rf
    #var(rp) = wt sum of weights we want to find
    rtnP = np.sum(weights*rtns.mean()*252)
    varP = np.dot(weights.T, np.dot(rtns.cov()*252,weights))
    return (rtnP - risk_aversion * (varP))

#scipy only does minimise so we return a negative value of the max return.
def max_return (weights):
    return -(mean_var_opt(weights, dfrtn, 2)) 

# Define constraints that all weights add up to 1, restricting their values between 0 and 1.

cons = ({'type':'eq','fun': lambda x: np.sum(x)-1}) #add up to 1
bnds = tuple((0,1) for x in range(nn)) #only between 0 and 1
# Initialise weights
w0 =  [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]
w0 = np.array(w0)


optv = sco.minimize(max_return, w0 , method = 'SLSQP', bounds= bnds, constraints = cons)
optv['x'].round(3)

