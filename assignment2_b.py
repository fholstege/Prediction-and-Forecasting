# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 18:05:13 2022

@author: flori
"""
from helpers_pf import * 
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from boxjenkins import *
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings("ignore")


df_ts = read_xlsx("Data/DataAssignment1.xlsx")
ts_1 = df_ts['Var1']
ts_2 = df_ts['Var2']
ts_3 = df_ts['Var3']
ts_4 = df_ts['Var4']
ts_5 = df_ts['Var5']
ts_6 = df_ts['Var6']
ts_7 = df_ts['Var7']
ts_8 = df_ts['Var8']
ts_9 = df_ts['Var9']

# ADF results
augmented_dicky_fuller(ts_1, 0)
augmented_dicky_fuller(ts_2, 1)
augmented_dicky_fuller(ts_3, 1)
augmented_dicky_fuller(ts_4, 0)
augmented_dicky_fuller(ts_5, 1)
augmented_dicky_fuller(ts_6, 1)
augmented_dicky_fuller(ts_7, 1)
augmented_dicky_fuller(ts_8, 1)
augmented_dicky_fuller(ts_9, 1)



# results for h=1
table_inSample_series1_h1, table_outSample_series1_h1 =  evaluate_arma_comprehensive(ts_1, name_series = 'Series 1',criterion='aic', max_p = 4, max_q = 4, max_d = 0, d=0)
table_inSample_series2_h1, table_outSample_series2_h1 =  evaluate_arma_comprehensive(ts_2, name_series = 'Series 2', criterion='aic', max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series3_h1, table_outSample_series3_h1 =  evaluate_arma_comprehensive(ts_3, name_series = 'Series 3', criterion='aic', max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series4_h1, table_outSample_series4_h1 =  evaluate_arma_comprehensive(ts_4, name_series = 'Series 4', criterion='aic', max_p = 4, max_q = 4, max_d = 1, d=0)
table_inSample_series5_h1, table_outSample_series5_h1 =  evaluate_arma_comprehensive(ts_5, name_series = 'Series 5', criterion='aic', max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series5_h1, table_outSample_series6_h1 =  evaluate_arma_comprehensive(ts_6, name_series = 'Series 6', criterion='aic', max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series4_h1, table_outSample_series4_h1 =  evaluate_arma_comprehensive(ts_7, name_series = 'Series 7', criterion='aic', max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series5_h1, table_outSample_series5_h1 =  evaluate_arma_comprehensive(ts_8, name_series = 'Series 8', criterion='aic', max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series5_h1, table_outSample_series6_h1 =  evaluate_arma_comprehensive(ts_9, name_series = 'Series 9', criterion='aic', max_p = 4, max_q = 4, max_d = 1, d=1)


# results for h=2
table_inSample_series1_h2, table_outSample_series1_h2 =  evaluate_arma_comprehensive(ts_1, name_series = 'Series 1',criterion='aic',h=2, max_p = 4, max_q = 4, max_d = 0, d=0)
table_inSample_series2_h2, table_outSample_series2_h2 =  evaluate_arma_comprehensive(ts_2, name_series = 'Series 2',criterion='aic',h=2,max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series3_h2, table_outSample_series3_h2 =  evaluate_arma_comprehensive(ts_3, name_series = 'Series 3',criterion='aic',h=2, max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series4_h2, table_outSample_series4_h2 =  evaluate_arma_comprehensive(ts_4, name_series = 'Series 4',criterion='aic',h=2,max_p = 4, max_q = 4, max_d = 1, d=0)
table_inSample_series5_h2, table_outSample_series5_h2 =  evaluate_arma_comprehensive(ts_5, name_series = 'Series 5',criterion='aic',h=2,  max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series6_h2, table_outSample_series6_h2 =  evaluate_arma_comprehensive(ts_6, name_series = 'Series 6',criterion='aic',h=2, max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series4_h2, table_outSample_series7_h2 =  evaluate_arma_comprehensive(ts_7, name_series = 'Series 7',criterion='aic',h=2, max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series5_h2, table_outSample_series8_h2 =  evaluate_arma_comprehensive(ts_8, name_series = 'Series 8',criterion='aic',h=2, max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series6_h2, table_outSample_series9_h2 =  evaluate_arma_comprehensive(ts_9, name_series = 'Series 9',criterion='aic',h=2, max_p = 4, max_q = 4, max_d = 1, d=1)



# results for h=4
table_inSample_series1_h4, table_outSample_series1_h4 =  evaluate_arma_comprehensive(ts_1, name_series = 'Series 1',criterion='aic',h=4,max_p = 4, max_q = 4, max_d = 0, d=0)
table_inSample_series2_h4, table_outSample_series2_h4 =  evaluate_arma_comprehensive(ts_2, name_series = 'Series 2',criterion='aic',h=4, max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series3_h4, table_outSample_series3_h4 =  evaluate_arma_comprehensive(ts_3, name_series = 'Series 3',criterion='aic',h=4, max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series4_h4, table_outSample_series4_h4 =  evaluate_arma_comprehensive(ts_4, name_series = 'Series 4',criterion='aic',h=4,max_p = 4, max_q = 4, max_d = 0, d=0)
table_inSample_series5_h4, table_outSample_series5_h4 =  evaluate_arma_comprehensive(ts_5, name_series = 'Series 5',criterion='aic',h=4, max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series6_h4, table_outSample_series6_h4 =  evaluate_arma_comprehensive(ts_6, name_series = 'Series 6',criterion='aic',h=4, max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series4_h4, table_outSample_series7_h4 =  evaluate_arma_comprehensive(ts_7, name_series = 'Series 7',criterion='aic',h=4, max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series5_h4, table_outSample_series8_h4 =  evaluate_arma_comprehensive(ts_8, name_series = 'Series 8',criterion='aic',h=4, max_p = 4, max_q = 4, max_d = 1, d=1)
table_inSample_series6_h4, table_outSample_series9_h4 =  evaluate_arma_comprehensive(ts_9, name_series = 'Series 9',criterion='aic',h=4, max_p = 4, max_q = 4, max_d = 1, d=1)
