#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as sco
import math as m
import scipy as sp
import pandas_datareader as pd_data
import datetime


# In[2]:


# Will fix figure size for this notebook OK
plt.rcParams["figure.figsize"] = (8,6)

#Suppress warnings OK
import warnings
warnings.filterwarnings('ignore')


# # Imported functions for expected returns and covariance

# In[3]:


def returns_from_prices(prices, log_returns=False):
    """
    Calculate the returns given prices.

    :param prices: adjusted (daily) closing prices of the asset, each row is a
                   date and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param log_returns: whether to compute using log returns
    :type log_returns: bool, defaults to False
    :return: (daily) returns
    :rtype: pd.DataFrame
    """
    if log_returns:
        return np.log(prices).diff().dropna(how="all")
    else:
        return prices.pct_change().dropna(how="all")
        
def _pair_exp_cov(X, Y, span=180):
    """
    Calculate the exponential covariance between two timeseries of returns.

    :param X: first time series of returns
    :type X: pd.Series
    :param Y: second time series of returns
    :type Y: pd.Series
    :param span: the span of the exponential weighting function, defaults to 180
    :type span: int, optional
    :return: the exponential covariance between X and Y
    :rtype: float
    """
    covariation = (X - X.mean()) * (Y - Y.mean())
    # Exponentially weight the covariation and take the mean
    if span < 10:
        warnings.warn("it is recommended to use a higher span, e.g 30 days")
    return covariation.ewm(span=span).mean().iloc[-1]     

def exp_cov(prices, returns_data=False, span=180, frequency=252, **kwargs):
    """
    Estimate the exponentially-weighted covariance matrix, which gives
    greater weight to more recent data.

    :param prices: adjusted closing prices of the asset, each row is a date
                   and each column is a ticker/id.
    :type prices: pd.DataFrame
    :param returns_data: if true, the first argument is returns instead of prices.
    :type returns_data: bool, defaults to False.
    :param span: the span of the exponential weighting function, defaults to 180
    :type span: int, optional
    :param frequency: number of time periods in a year, defaults to 252 (the number
                      of trading days in a year)
    :type frequency: int, optional
    :return: annualised estimate of exponential covariance matrix
    :rtype: pd.DataFrame
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("data is not in a dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)
    assets = prices.columns
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices)
    N = len(assets)

    # Loop over matrix, filling entries with the pairwise exp cov
    S = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            S[i, j] = S[j, i] = _pair_exp_cov(
                returns.iloc[:, i], returns.iloc[:, j], span
            )
    cov = pd.DataFrame(S * frequency, columns=assets, index=assets)
    
    return cov


# # Mean Variance Optimiser

# In[4]:


def mean_var_opt (weights, rtns, risk_aversion, covdata):
    #calculate expected returns = w.t + w0 rf
    #var(rp) = wt sum of weights we want to find
    rtnP = np.sum(weights*rtns*252)
    varP = np.dot(weights.T, np.dot(covdata.cov()*252,weights))
    return (rtnP - risk_aversion * (varP))


# In[5]:


#For first period Jan 2011
start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2010, 12, 31)

#the historical period
symbols = ['AAPL','GS', 'GC=F', 'GE', 'BAC', 'PFE']
t1 = pd_data.DataReader(symbols,'yahoo',start_date,end_date)['Adj Close']

t1 = t1.dropna()

#look ahead period
start_date1 = datetime.datetime(2011, 1, 1)
end_date1 = datetime.datetime(2011, 1, 31)

symbols = ['AAPL','GS', 'GC=F', 'GE', 'BAC', 'PFE']
rtns1 = pd_data.DataReader(symbols,'yahoo',start_date1,end_date1)['Adj Close'] #importing data as dataframe
rtns1 = rtns1.dropna()
rtns1


# In[6]:


LIBOR_rates = pd.read_csv("LIBOR_USD.csv")
LIBOR_rates
LIBOR_rates['Date'] = pd.to_datetime(LIBOR_rates['Date'],format="%d.%m.%Y")

#for historical data
#converts rates to datetime
Rates = LIBOR_rates.loc[ (LIBOR_rates['Date'] >= start_date) & (LIBOR_rates['Date'] <= end_date) ]
Rates.sort_values(by='Date',ascending=True,inplace=True)
Rates.set_index('Date', inplace=True)


# In[7]:


#makes a dataframe including LIBOR rates for (future)
dft1 = t1.copy()
dft1['Cash'] = Rates['ON']/100/252
dft1 = dft1.dropna()
dft1 = dft1.pct_change().dropna()
dft1


# In[8]:


rtns1 = rtns1.pct_change().mean(axis = 0)
rtns1['Cash'] = Rates['ON'][end_date]/100/252
rtns1


# In[9]:


nn = len(rtns1)
print(nn)


# In[11]:



cons = ({'type':'eq','fun': lambda x: np.sum(x)-1}) #add up to 1
bnds = tuple((0,1) for x in range(nn)) #only between 0 and 1
# Initialise weights
w0 = [1.0/nn for i in range(nn)]
w0 = np.array(w0)

def max_returnt1 (weights):
    return -(mean_var_opt(weights, rtns1, 2, dft1)) 


wt1 = sco.minimize(max_returnt1, w0, method = 'SLSQP', bounds= bnds, constraints = cons)
wt1['x'].round(3)


# In[22]:


#have not put in exp_cov but other than that it seems to work.

start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2010, 11, 30) # for the covariance matrix
a = 12 #number of cycles i.e. goes by months
weights_array = []
dates_array = []
LIBOR_rates = pd.read_csv("LIBOR_USD.csv")
LIBOR_rates
LIBOR_rates['Date'] = pd.to_datetime(LIBOR_rates['Date'],format="%d.%m.%Y")


for i in range(a):
    
    end_date = end_date + datetime.timedelta(days=30) #not exactly a month
    symbols = ['AAPL','GS', 'GC=F', 'GE', 'BAC', 'PFE']
    t1 = pd_data.DataReader(symbols,'yahoo',start_date,end_date)['Adj Close']
    t1 = t1.dropna()
    
    #adding LIBOR rates into the historical data
    Rates = LIBOR_rates.loc[ (LIBOR_rates['Date'] >= start_date) & (LIBOR_rates['Date'] <= end_date) ]
    Rates.sort_values(by='Date',ascending=True,inplace=True)
    Rates.set_index('Date', inplace=True)
    
    #dft1 contains the historical data + Libor rates for that period
    dft1 = t1.copy()
    dft1['Cash'] = Rates['ON']/100/252
    dft1 = dft1.dropna()
    dft1 = dft1.pct_change().dropna()

    #look ahead period
    #will have to adjust depending on if want more than a year.
    start_date1 = datetime.datetime(2011, i+1, 1) #for the expected returns forecasts
    end_date1 = start_date1 + datetime.timedelta(days = 30) #average number of days in a month
    

    symbols = ['AAPL','GS', 'GC=F', 'GE', 'BAC', 'PFE']
    rtns1 = pd_data.DataReader(symbols,'yahoo',start_date1,end_date1)['Adj Close'] #importing data as dataframe
    rtns1 = rtns1.dropna()
    
    #rtns1 will store the mean value of the expected returns, with the libor rate being the date on the end date of the historical data
    rtns1 = rtns1.pct_change().mean(axis = 0)
    
    end_date_entry = end_date;
    rtns1['Cash'] = (Rates['ON'].iloc[-1])/100/252 
    
    nn = len(rtns1)
    
    cons = ({'type':'eq','fun': lambda x: np.sum(x)-1}) #add up to 1
    bnds = tuple((0,1) for x in range(nn)) #only between 0 and 1
    # Initialise weights
    w0 = [1.0/nn for i in range(nn)]
    w0 = np.array(w0)

    def max_returnt1 (weights):
        return -(mean_var_opt(weights, rtns1, 2, dft1)) 

    wt1 = sco.minimize(max_returnt1, w0, method = 'SLSQP', bounds= bnds, constraints = cons)
    
    #extrating the array data from the optimization function 
    weight = wt1['x'].round(3)
    weights_array.append(weight)
    dates_array.append(end_date1)
    print("The optimal weight on", end_date1," is ", weight)
    
 


# In[23]:


plt.rcParams["figure.figsize"] = (12,8)
weights_arr = np.stack(weights_array, axis=0)
dates_arr = np.stack(dates_array, axis = 0)
plt.plot(dates_arr, weights_arr)
plt.xticks(rotation=90)
plt.legend(symbols,bbox_to_anchor=(1.05, 1),loc='upper left')
plt.ylabel('Optimal weights',size=12)
plt.title('Robustness test for Mean Variance Optimization strategy',size=16)
plt.show()


# In[ ]:




