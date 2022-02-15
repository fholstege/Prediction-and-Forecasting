# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 18:05:13 2022

@author: flori
"""
from helpers_pf import * 
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from boxjenkins import *


df_ts = read_xlsx("Data/DataAssignment1.xlsx")
ts_1 = df_ts['Var1']

table_inSample_series1, table_outSample_series1 =  evaluate_arma_comprehensive(ts_1, name_series = 'Series 1')

