# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:01:34 2022

@author: flori
"""
import statsmodels.api as sm
from matplotlib import pyplot as plt
import numpy as np

def est_local_level_model(y, train_i, a_1=0, sigma2_eps=3.33, sigma2_eta=10, oos=True, h=1, forecast=True):
    
    
    # save here the parameters, set initial values
    a_est = [a_1]
    p_est = [sigma2_eps * 10**7]
    
    for i in range(train_i):
        print("step: {}".format(i))
        
        # use a_t, p_t, y_t for this iteration
        a_t = a_est[i]
        p_t = p_est[i]
        f_t = calc_f_t(p_t,sigma2_eps)
        y_t = y[i]
        
        # calc v_t
        v_t = calc_v_t(y_t, a_t)
        
        # calc k_t
        k_t = calc_k_t(p_t, f_t)
                
        # update to a_t+1, p_t+1
        a_t_1 = calc_a_t_1(a_t, k_t, v_t)
        p_t_1 = calc_p_t_1(k_t,sigma2_eps, sigma2_eta)
        
        print('Param: a_t:{}, p_t:{}, f_t: {}, v_t:{}, k_t:{}'.format(a_t,p_t,f_t, v_t, k_t))
        print('Next pred: {}'.format(a_t_1))

        # add to list
        a_est.append(a_t_1)
        p_est.append(p_t_1)
        
    
    if oos:
        # starting value
        oos_p = [p_est[-1]]
        for j in range(h):
            
            # take k_t with time
            oos_p_j = oos_p[j]
            oos_f_j = calc_f_t(oos_p_j, sigma2_eps)
            k_t = calc_k_t(oos_p_j, oos_f_j)
            
            # add to list
            oos_p.append(k_t * sigma2_eps + sigma2_eta)
            
            CI_upper, CI_lower = calc_CI(a_est[-1], 1.96, sigma2_eta, sigma2_eps, train_i, oos_p)

    
    if forecast:
    # move one forward to prevent foreknowledge
        a_est = [np.nan] + a_est[1:-1]
        p_est = [np.nan] + p_est[1:-1]
    else:
        a_est = a_est[1:]
        p_est = p_est[1:]
    
    if oos:
        h_step_oos = [a_est[-1]]*h
        return a_est, p_est, h_step_oos, CI_upper, CI_lower
    else:
        return a_est, p_est

def calc_CI(pred, z, sigma2_eta, sigma2_eps, n, oos_p):
    
    
    CI_upper = [pred + z * (np.sqrt(p - sigma2_eta + t * sigma2_eta + sigma2_eps)/np.sqrt(t+ n)) for t, p in enumerate(oos_p[1:])]
    CI_lower = [pred - z * (np.sqrt(p - sigma2_eta + t * sigma2_eta + sigma2_eps)/np.sqrt(t+ n)) for t, p in enumerate(oos_p[1:])]
    
    return CI_upper, CI_lower

def calc_f_t(p_t, sigma2_eps):
    return p_t + sigma2_eps

def calc_v_t(y_t, a_t):
    return y_t - a_t

def calc_k_t(p_t, f_t):
    
    return p_t/f_t

def calc_a_t_1(a_t,k_t, v_t):
    return a_t + k_t*v_t

def calc_p_t_1(k_t, sigma2_eps,sigma2_eta):
    
    return k_t*sigma2_eps + sigma2_eta

class MLELocalLevel(sm.tsa.statespace.MLEModel):
    start_params = [1.0, 1.0]
    param_names = ['obs.var', 'level.var']

    def __init__(self, endog):
        super(MLELocalLevel, self).__init__(endog, k_states=1)

        self['design', 0, 0] = 1.0
        self['transition', 0, 0] = 1.0
        self['selection', 0, 0] = 1.0

        self.initialize_approximate_diffuse()
        self.loglikelihood_burn = 1

    def transform_params(self, params):
        return params**2

    def untransform_params(self, params):
        return params**0.5

    def update(self, params, **kwargs):
        # Transform the parameters if they are not yet transformed
        params = super(MLELocalLevel, self).update(params, **kwargs)

        self['obs_cov', 0, 0] = params[0]
        self['state_cov', 0, 0] = params[1]


# load data
gas_data = pd.read_excel("Data/GasolineSales2.xlsx", header=None)
gas_data.columns = ['GasolineSales']


# get our estimated a_{t+1}
h=5
est_a, est_p, pred_a, CI_a_upper, CI_a_lower= est_local_level_model(gas_data['GasolineSales'], 22, sigma2_eps=3.33, sigma2_eta = 10, h=h, forecast=False)


# slide 19
plt.scatter(range(1,23), gas_data['GasolineSales'], color='red', label='Gas sales')
plt.plot(range(1, 23), est_a, color='blue', label=r'$a_{t+1}$')
plt.ylim([15, 40])
plt.xlim(0, 24)   
plt.legend(loc='upper left')

# slide 22
plt.plot(range(1,23), gas_data['GasolineSales'], color='red', label='Gas sales')
plt.plot(range(23, 23+h), pred_a, color='blue', label='Prediction')
plt.plot(range(23, 23+h), CI_a_upper, color='black', linestyle='dashed')
plt.plot(range(23, 23+h), CI_a_lower, color='black', linestyle='dashed')
plt.ylim([15, 40])
plt.xlim(0, 23+h+2)   
plt.legend(loc='upper left')



# try with package
gas_model = MLELocalLevel(gas_data['GasolineSales'])
params={'obs.var':3.33,'level.var':10}
gas_model_results = gas_model.fit_constrained(params)
gas_forecast = gas_model_results.get_prediction(start=1, end =22)
a_package= gas_forecast.predicted_mean

# package
plt.plot(range(1, 23), a_package, color='blue', label=r'$a_{t+1}$ (package)')
plt.scatter(range(1,23), gas_data['GasolineSales'], color='red', label='Gas sales')
plt.ylim([15, 35])
plt.xlim(0, 24)  
plt.legend(loc='upper left')

