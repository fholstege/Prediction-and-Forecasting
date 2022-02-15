# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:13:34 2022

@author: flori
"""
from helpers_pf import * 
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from boxjenkins import *

###### Regular gas sales

# read in gas sales data
gas_data = read_xlsx("Data/GasolineSales1.xlsx", header=None)
gas_data.columns = ['GasolineSales']


# dataframe with gas sales 
df_compare_gas = evaluate_arma_forecasts_noValidation(gas_data['GasolineSales'], train_i=6, max_p=2, max_q=1, h=1,
                                                      criterion='hqic', last_n=6, 
                                                      name_series= 'Gas sales' )



###### Extended gas sales

# Get the extended gas sales data
gas_data_ext = read_xlsx("Data/GasolineSales2.xlsx", header=None)
gas_data_ext.columns = ['GasolineSales']

# dataframe with gas sales - extended
df_compare_gas_ext = evaluate_arma_forecasts_noValidation(gas_data_ext['GasolineSales'], train_i=12, max_p=4, max_q=1, h=1,
                                                          criterion='hqic', last_n=10,
                                                          name_series='Gas sales (ext.)',
                                                          ylim=[0,25])


###### bike sales data

# load in bike sales data
bike = pd.read_excel('Data/BicycleSales.xlsx',header = None)
bike.columns = ['sales']

# get the bike sales compared 
df_compare_bikes = evaluate_arma_forecasts_noValidation(bike['sales'], train_i=5, max_p=3, max_q=1, h=1, 
                                                        criterion='hqic', last_n=4,
                                                        name_series = 'Bike sales')

######  umbrella sales data

# load the umbrella sales data 
umbrella = read_xlsx("Data/Umbrella.xlsx")
umbrella.columns = ['year', 'season', 'sales']

df_compare_umbrella = evaluate_arma_forecasts_noValidation(umbrella['sales'], train_i=12, max_p=3, max_q=1, h=1, 
                                                           criterion='hqic', last_n=7,
                                                           name_series = 'Umbrella sales',
                                                           ylim = [0,22])
