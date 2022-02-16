# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:13:34 2022

@author: flori
"""
from helpers_pf import * 
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from boxjenkins import *
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant





###### Regular gas sales

# read in gas sales data
gas_data = read_xlsx("Data/GasolineSales1.xlsx", header=None)
gas_data.columns = ['GasolineSales']

# create quick acf plot  - use train data
gas_data_train = gas_data['GasolineSales'][:7]
plot_acf(gas_data_train, lags=4, alpha=.05)

# check adf
augmented_dicky_fuller(gas_data_train, d=1)

    
# dataframe with gas sales  - h=1
df_compare_gas_h1 = evaluate_arma_forecasts_noValidation(gas_data['GasolineSales'], train_i=7, 
                                                      max_p=2, 
                                                      max_q=1,
                                                      max_d=1,
                                                      h=1, # one step ahead
                                                      d=1, # based on adf
                                                      criterion='aic', last_n=5, 
                                                      name_series= 'Gas sales' )
# dataframe with gas sales - h=2
df_compare_gas_h2 = evaluate_arma_forecasts_noValidation(gas_data['GasolineSales'], train_i=7,
                                                      max_p=1, 
                                                      max_q=1, 
                                                      h=2, # two-step ahead
                                                      d = 1, # based on adf
                                                      criterion='aic', last_n=5, 
                                                      name_series= 'Gas sales' ,
                                                      show_inSample=False,
                                                  name_series= 'Gas sales' )
# dataframe with gas sales - h=4
df_compare_gas_h4 = evaluate_arma_forecasts_noValidation(gas_data['GasolineSales'], train_i=7,
                                                      max_p=1, 
                                                      max_q=1, 
                                                      h=4, # four-step ahead
                                                      d = 1, # based on adf
                                                      criterion='aic', last_n=5, 
                                                      name_series= 'Gas sales' ,
                                                      show_inSample=False)


###### Extended gas sales

# Get the extended gas sales data
gas_data_ext = read_xlsx("Data/GasolineSales2.xlsx", header=None)
gas_data_ext.columns = ['GasolineSales']

# create quick acf plot  - use train data
gas_data_train_ext = gas_data_ext['GasolineSales'][:12]
plot_acf(gas_data_train_ext, lags=8, alpha=.05)


# check adf
augmented_dicky_fuller(gas_data_train_ext, d=0)


# dataframe with gas sales - extended
df_compare_gas_ext_h1 = evaluate_arma_forecasts_noValidation(gas_data_ext['GasolineSales'], train_i=12, 
                                                          max_p=4,
                                                          max_q=2, 
                                                          h=1, # one step ahead
                                                          d=0, # based on adf
                                                          criterion='aic', 
                                                          last_n=12,
                                                          name_series='Gas sales (ext.)',
                                                          ylim=[0,25])
# dataframe with gas sales - two-step ahead
df_compare_gas_ext_h2 = evaluate_arma_forecasts_noValidation(gas_data_ext['GasolineSales'], train_i=12, 
                                                          max_p=4,
                                                          max_q=2, 
                                                          h=2, # two-step ahead
                                                          d=0, # based on adf
                                                          criterion='aic', 
                                                          last_n=12,
                                                          name_series='Gas sales (ext.)',
                                                          ylim=[0,25],
                                                          show_inSample=False)
# dataframe with gas sales - four-step ahead
df_compare_gas_ext_h4 = evaluate_arma_forecasts_noValidation(gas_data_ext['GasolineSales'], train_i=12, 
                                                          max_p=4,
                                                          max_q=2, 
                                                          h=4, # four-step ahead,
                                                          d=0, # based on adf
                                                          criterion='aic', 
                                                          last_n=12,
                                                          name_series='Gas sales (ext.)',
                                                          ylim=[0,25],
                                                          show_inSample=False)





###### bike sales data

# load in bike sales data
bike = pd.read_excel('Data/BicycleSales.xlsx',header = None)
bike.columns = ['sales']

# create quick acf plot  - use train data
bike_sales_train = bike['sales'][:6]
plot_acf(bike_sales_train, lags=4, alpha=.05)


# adf test
time = list(bike_sales_train.index)
ct = add_constant(time)
constant_time_trend = OLS(endog=bike_sales_train, exog=ct).fit()
plt.scatter(time, constant_time_trend.resid)
augmented_dicky_fuller(constant_time_trend.resid, d=0)


# get the bike sales compared 
df_compare_bikes_h1 = evaluate_arma_forecasts_noValidation( bike['sales'], train_i=6, 
                                                        max_p=2, 
                                                        max_q=1, 
                                                        max_d=0,
                                                        h=1, # one step ahead
                                                        d=0,
                                                        trend='ct',
                                                        criterion='aic',
                                                        last_n=4,
                                                        name_series = 'Bike sales')

# two step forecast
df_compare_bikes_h2 = evaluate_arma_forecasts_noValidation( bike['sales'], train_i=6, 
                                                        max_p=2, 
                                                        max_q=1, 
                                                        max_d=0,
                                                        h=2, # two step ahead
                                                        d=0,
                                                        trend='ct',
                                                        criterion='aic',
                                                        last_n=4,
                                                        name_series = 'Bike sales')
# four step ahead forecast
df_compare_bikes_h4 = evaluate_arma_forecasts_noValidation( bike['sales'], train_i=6, 
                                                        max_p=2, 
                                                        max_q=1, 
                                                        max_d=0,
                                                        h=4, # four step ahead
                                                        d=0,
                                                        trend='ct', # based on adf
                                                        criterion='aic',
                                                        last_n=4,
                                                        name_series = 'Bike sales')



######  umbrella sales data

# load the umbrella sales data 
umbrella = read_xlsx("Data/Umbrella.xlsx")
umbrella.columns = ['year', 'season', 'sales']

# create seasonal dummies
seasonal_dummies = pd.get_dummies(umbrella['season'],prefix='season')

# create quick acf plot  - use train data
umbrella_sales_train = umbrella['sales'][:12]
plot_acf(umbrella_sales_train, lags=8, alpha=.05)

# check the umbrella sales
augmented_dicky_fuller(umbrella_sales_train, d=0)


# get results with seasonal dummies
df_compare_umbrella_h1 = evaluate_arma_forecasts_noValidation(umbrella['sales'], train_i=12,
                                                           max_p=3, 
                                                           max_q=1, 
                                                           max_d=0, # adf
                                                           d=0,
                                                           h=1, # one-step ahead
                                                           criterion='aic', last_n=7,
                                                           name_series = 'Umbrella sales',
                                                           ylim = [0,22],
                                                           exog=seasonal_dummies)

# two-step ahead
df_compare_umbrella_h2 = evaluate_arma_forecasts_noValidation(umbrella['sales'], train_i=12,
                                                           max_p=3, 
                                                           max_q=1, 
                                                           max_d=0, # adf
                                                           d=0,
                                                           h=2, # two-step ahead
                                                           criterion='aic', last_n=7,
                                                           name_series = 'Umbrella sales',
                                                           ylim = [0,22],
                                                           exog=seasonal_dummies)
# four-step ahead
df_compare_umbrella_h4 = evaluate_arma_forecasts_noValidation(umbrella['sales'], train_i=12,
                                                           max_p=3, 
                                                           max_q=1, 
                                                           max_d=0, # adf
                                                           d=0,
                                                           h=4, # four-step ahead
                                                           criterion='bic', last_n=7,
                                                           name_series = 'Umbrella sales',
                                                           ylim = [0,22],
                                                           exog=seasonal_dummies)
