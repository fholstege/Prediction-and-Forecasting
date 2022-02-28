# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:01:34 2022

@author: flori
"""
import statsmodels.api as sm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def est_local_level_model(y, train_i, a_1=0, sigma2_eps=3.33, sigma2_eta=10, oos=True, h=1, forecast=True, z=1.96):
    
    
    # save here the parameters, set initial values
    a_est = [a_1]
    p_est = [sigma2_eps * 10**7]
    
    for i in range(train_i):
        print("step: {}".format(i))
        
        # use a_t, p_t, y_t for this iteration
        a_t = a_est[i]
        p_t = p_est[i]
        y_t = y.iloc[i]
        
        # calc v_t
        v_t = calc_v_t(y_t, a_t)
        
        # calc k_t
        k_t = calc_k_t(p_t, sigma2_eps)
                
        # update to a_t+1, p_t+1
        a_t_1 = calc_a_t_1(a_t, k_t, v_t)
        p_t_1 = calc_p_t_1(k_t,sigma2_eps, sigma2_eta)
        
        print('Param: a_t:{}, p_t:{}, v_t:{}, k_t:{}'.format(a_t,p_t, v_t, k_t))
        print('Next pred: {}'.format(a_t_1))

        # add to list
        a_est.append(a_t_1)
        p_est.append(p_t_1)
        
    
    if forecast:
    # move one forward to prevent foreknowledge
        a_est = [np.nan] + a_est[1:-1]
        p_est = [np.nan] + p_est[1:-1]
    else:
        a_est = a_est[1:]
        p_est = p_est[1:]
    
    # get oos prediction
    if oos:
        
        # define prediction
        pred = a_est[-1] 
        h_step_oos = [pred]*h
        
        # define P_{T|T} for confidence interval
        P_T=p_est[-1]-sigma2_eta
        oos_p = [P_T]*(h)
        
        print("P_T|T = {}".format({P_T}))
        
        # calculate  CI's
        CI_upper, CI_lower = calc_CI(pred, z, sigma2_eta, sigma2_eps, train_i, oos_p)
        
        return a_est, p_est, h_step_oos, CI_upper, CI_lower
    else:
        return a_est, p_est

def calc_CI(pred, z, sigma2_eta, sigma2_eps, n, oos_p):
    
    var_forecast = [(p + ((t+1) * sigma2_eta) + sigma2_eps) for t, p in enumerate(oos_p)]
    
    for i in range(len(var_forecast)):
        var = var_forecast[i]
        print('for h={}, sd = {}, var = {}'.format(i+1, np.sqrt(var), var))
    
    CI_upper =[ pred + z * (np.sqrt(var)/np.sqrt(n)) for t,var in enumerate(var_forecast)]
    CI_lower =[ pred - z * (np.sqrt(var)/np.sqrt(n)) for t,var in enumerate(var_forecast)]
    
    return CI_upper, CI_lower




def calc_f_t(p_t, sigma2_eps):
    return p_t + sigma2_eps

def calc_v_t(y_t, a_t):
    return y_t - a_t

def calc_k_t(p_t, sigma2_eps):
    
    return p_t/(p_t + sigma2_eps)

def calc_a_t_1(a_t,k_t, v_t):
    return a_t + k_t*v_t

def calc_p_t_1(k_t, sigma2_eps,sigma2_eta):
    
    return k_t*sigma2_eps + sigma2_eta


##### Load the data

#  gas sales
gas_data = pd.read_excel("Data/GasolineSales2.xlsx", header=None)
gas_data.columns = ['GasolineSales']

# non-responses
df_nonresponses = pd.read_excel("Data/NonResponse.xlsx")



##### replicate slide 19 + 22 for gas sales

# set the parameters for plt


# get our estimated a_{t} for gas sales
h=5
since_n = 0
last_n_obs= gas_data['GasolineSales'][since_n:]
total_n = last_n_obs.shape[0]
est_a, est_p, pred_a, CI_a_upper, CI_a_lower= est_local_level_model(last_n_obs, total_n, sigma2_eps=3.33, sigma2_eta = 10, h=h, forecast=True, z=2.65)

# slide 19
plt.figure(facecolor="white")
plt.scatter(range(1,23), gas_data['GasolineSales'], color='red', label='Gas sales')
plt.plot(range(since_n+1, 23), est_a, color='blue', label=r'$a_{t}$')
plt.ylim([15, 40])
plt.xlim(0, 24)   
plt.legend(loc='upper left')

# slide 22
plt.figure(facecolor="white")
plt.plot(range(1,23), gas_data['GasolineSales'], color='red', label='Gas sales')
plt.plot(range(23, 23+h), pred_a, color='blue', label=r'$y_{T+h}$ for $h=1,..,5$')
plt.plot(range(23, 23+h), CI_a_upper, color='black', linestyle='dashed', label = r'$95\%$ CI')
plt.plot(range(23, 23+h), CI_a_lower, color='black', linestyle='dashed')
plt.ylim([15, 40])
plt.xlim(0, 23+h+2)   
plt.legend(loc='upper left')


##### replicate slide 19 + 22 for non-responses

# do it for non responses
forecast_length = 24
total_length = df_nonresponses.shape[0]
train_i = total_length - forecast_length


# first, full sample (slide 19)
est_a_nr, est_p_nr, pred_a_nr, CI_a_upper_nr, CI_a_lower_nr= est_local_level_model(df_nonresponses['NonResponse'], total_length, sigma2_eps=30, sigma2_eta = 1.5, h=forecast_length, 
                                                                                   forecast=True)

# replicate slide 19 for non responses
plt.figure(facecolor="white")
plt.scatter(range(1, total_length+1),df_nonresponses['NonResponse'], color='red', label='Non-responses')
plt.plot(range(1, total_length+1), est_a_nr, color='blue', label=r'$a_{t}$')
plt.xlim(0, total_length)   
plt.legend(loc='upper left')

# get estimates for oos prediction
est_a_nr_split, est_p_nr_split, pred_a_nr_split, CI_a_upper_nr_split, CI_a_lower_nr_split= est_local_level_model(df_nonresponses['NonResponse'], train_i, sigma2_eps=30, sigma2_eta = 1.5, h=forecast_length, 
                                                                                   forecast=True)

# replication of slide 22 for non responses
plt.figure(facecolor="white")
plt.plot(range(1, train_i+1),df_nonresponses['NonResponse'][:train_i], color='red', label='Non-responses')
plt.plot(range(train_i, total_length), pred_a_nr_split, color='blue',  label=r'$y_{T+h}$ for $h=1,..,5$')
plt.plot(range(train_i, total_length), CI_a_upper_nr_split, color = 'black', linestyle='dashed', label = r'$95\%$ CI')
plt.plot(range(train_i, total_length), CI_a_lower_nr_split, color = 'black', linestyle='dashed')
plt.xlim(0, total_length)   
plt.legend(loc='upper right')
