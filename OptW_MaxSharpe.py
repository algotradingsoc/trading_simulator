import scipy.optimize as sco
import numpy as np
import pandas as pd

# Need to define inputs (type: pd.Dataframe):
# 1. expRtns: "LIBOR rates & forecasted returns"
# 2. cov: "covariance based on historical data"
# 3. mu_f: "US treasury yield"

def Portfolio_stats(weights: "porportions of capital", 
                    expRtns: "LIBOR rates & forecasted returns", 
                    cov: "covariance based on historical data", 
                    mu_f: "US treasury yield") -> "portfolio returns,variance,volatility,sharpe_ratio":
    varP = np.dot(weights.T, np.dot(cov,weights))
    volP = np.sqrt(varP)
    rtnP = np.sum(weights*expRtns)
    sharpeP = (rtnP-mu_f)/volP
    return rtnP, varP, volP, sharpeP

def negative_sharpe(weights):
    return -Portfolio_stats(weights,expRtns,cov,mu_f)[3]

# total number of stocks (+cash)
nn = len(expRtns)

# Initialise weights
w0 = [1.0/nn for i in range(nn)]
w0 = np.array(w0)

# Constraints on weights
cons = ({'type':'eq','fun': lambda x: np.sum(x)-1}) #add up to 1
bnds = tuple((0,1) for x in range(nn)) #only between 0 and 1, i.e. no short-selling

# Maximise sharpes ratio
opts = sco.minimize(negative_sharpe, w0 , method = 'SLSQP', bounds= bnds, constraints = cons)

# Optimal weights
w_opt = opts['x'].round(3)
w_opt
