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
import warnings





###### Regular gas sales

# read in gas sales data
gas_data = read_xlsx("Data/GasolineSales1.xlsx", header=None)
gas_data.columns = ['GasolineSales']

# create quick acf plot  - use train data
gas_data_train = gas_data['GasolineSales'][:7]
plot_acf(gas_data['GasolineSales'], lags=4, alpha=.05)

# check adf
augmented_dicky_fuller(gas_data_train, d=1)

    
# dataframe with gas sales  - h=1
df_compare_gas_h1 = evaluate_arma_forecasts_noValidation(gas_data['GasolineSales'], train_i=7, 
                                                      max_p=2, 
                                                      max_q=2,
                                                      max_d=1,
                                                      h=1, # one step ahead
                                                      d=1, # based on adf
                                                      criterion='aic', last_n=5, 
                                                      name_series= 'Gas sales' )

mean_prediction = np.mean(gas_data['GasolineSales'][:7])
oos_gas = gas_data['GasolineSales'][7:]
np.mean(get_squared_error(oos_gas, mean_prediction))



print(round(df_compare_gas_h1,2).to_latex())

# dataframe with gas sales - h=2
df_compare_gas_h2 = evaluate_arma_forecasts_noValidation(gas_data['GasolineSales'], train_i=7,
                                                      max_p=2, 
                                                      max_q=2, 
                                                      h=2, # two-step ahead
                                                      d = 1, # based on adf
                                                      criterion='aic', last_n=5, 
                                                      name_series= 'Gas sales' ,
                                                      show_inSample=False)

print(round(df_compare_gas_h2,2).to_latex())


# dataframe with gas sales - h=4
df_compare_gas_h4 = evaluate_arma_forecasts_noValidation(gas_data['GasolineSales'], train_i=7,
                                                      max_p=2, 
                                                      max_q=2, 
                                                      h=4, # four-step ahead
                                                      d = 1, # based on adf
                                                      criterion='aic', last_n=5, 
                                                      name_series= 'Gas sales' ,
                                                      show_inSample=False)

print(round(df_compare_gas_h4,2).to_latex())



###### Extended gas sales

# Get the extended gas sales data
gas_data_ext = read_xlsx("Data/GasolineSales2.xlsx", header=None)
gas_data_ext.columns = ['GasolineSales']

# create quick acf plot  - use train data
gas_data_train_ext = gas_data_ext['GasolineSales'][:12]
plot_acf(gas_data_ext['GasolineSales'], lags=8, alpha=.05)


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

print(round(df_compare_gas_ext_h1,2).to_latex())

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

print(round(df_compare_gas_ext_h2,2).to_latex())

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


print(round(df_compare_gas_ext_h4,2).to_latex())



###### bike sales data

# load in bike sales data
bike = pd.read_excel('Data/BicycleSales.xlsx',header = None)
bike.columns = ['sales']

# create quick acf plot  - use train data
bike_sales_train = bike['sales'][:6]
plot_acf(bike['sales'], lags=4, alpha=.05)


# adf test after ct
time = list(bike['sales'].index)
ct = add_constant(time)
constant_time_trend = OLS(endog=bike['sales'], exog=ct).fit()

# show residuals
plt.scatter(time, constant_time_trend.resid)
plt.xlim([0,9])
plt.hlines(0, xmin=-1, xmax=9, color='black')

# adf
augmented_dicky_fuller(bike['sales'], d=1)


# get the bike sales compared 
df_compare_bikes_h1 = evaluate_arma_forecasts_noValidation( bike['sales'], train_i=6, 
                                                        max_p=2, 
                                                        max_q=1, 
                                                        max_d=1,
                                                        h=1, # one step ahead
                                                        d=1,
                                                        criterion='aic',
                                                        last_n=4,
                                                        name_series = 'Bike sales')

print(round(df_compare_bikes_h1,2).to_latex())


# two step forecast
df_compare_bikes_h2 = evaluate_arma_forecasts_noValidation( bike['sales'], train_i=6, 
                                                        max_p=2, 
                                                        max_q=1, 
                                                        max_d=1,
                                                        h=2, # two step ahead
                                                        d=1,
                                                        criterion='aic',
                                                        last_n=4,
                                                        name_series = 'Bike sales',
                                                        show_inSample=False)

print(round(df_compare_bikes_h2,2).to_latex())

# four step ahead forecast
df_compare_bikes_h4 = evaluate_arma_forecasts_noValidation( bike['sales'], train_i=6, 
                                                        max_p=2, 
                                                        max_q=1, 
                                                        max_d=1,
                                                        h=4, # four step ahead
                                                        d=1,
                                                        criterion='aic',
                                                        last_n=4,
                                                        name_series = 'Bike sales',
                                                        show_inSample=False)
print(round(df_compare_bikes_h4,2).to_latex())



######  umbrella sales data

# load the umbrella sales data 
umbrella = read_xlsx("Data/Umbrella.xlsx")
umbrella.columns = ['year', 'season', 'sales']

# create seasonal dummies
seasonal_dummies = pd.get_dummies(umbrella['season'],prefix='season')

# create quick acf plot  - use train data
umbrella_sales_train = umbrella['sales'][:12]
plot_acf(umbrella['sales'], lags=8, alpha=.05)

# check the umbrella sales
augmented_dicky_fuller( umbrella['sales'], d=0)


# get results with seasonal dummies
df_compare_umbrella_h1 = evaluate_arma_forecasts_noValidation(umbrella['sales'], train_i=12,
                                                           max_p=4, 
                                                           max_q=2, 
                                                           max_d=0, # adf
                                                           d=0,
                                                           h=1, # one-step ahead
                                                           criterion='aic', last_n=7,
                                                           name_series = 'Umbrella sales',
                                                           ylim = [0,22],
                                                           exog=seasonal_dummies)

print(round(df_compare_umbrella_h1,2).to_latex())



# two-step ahead
df_compare_umbrella_h2 = evaluate_arma_forecasts_noValidation(umbrella['sales'], train_i=12,
                                                           max_p=4, 
                                                           max_q=2, 
                                                           max_d=0, # adf
                                                           d=0,
                                                           h=2, # two-step ahead
                                                           criterion='aic', last_n=7,
                                                           name_series = 'Umbrella sales',
                                                           ylim = [0,22],
                                                           exog=seasonal_dummies,
                                                           show_inSample=False)

print(round(df_compare_umbrella_h2,2).to_latex())


# four-step ahead
df_compare_umbrella_h4 = evaluate_arma_forecasts_noValidation(umbrella['sales'], train_i=12,
                                                           max_p=4, 
                                                           max_q=2, 
                                                           max_d=0, # adf
                                                           d=0,
                                                           h=4, # four-step ahead
                                                           criterion='aic', last_n=7,
                                                           name_series = 'Umbrella sales',
                                                           ylim = [0,22],
                                                           exog=seasonal_dummies,
                                                           show_inSample=False)

print(round(df_compare_umbrella_h4,2).to_latex())

