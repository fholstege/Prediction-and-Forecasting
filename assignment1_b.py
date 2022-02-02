# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:15:53 2022

@author: flori
"""

from helpers_pf import * 


df_ts = read_xlsx("Data/DataAssignment1.xlsx")
ts = df['Var1']
ts_validation = s[:40]


def produce_forecasts(df, train_i):
    
    # Running Average
    running_avg_pred = running_average(df, shift = 1)
    
    # Random Walk
    rw_pred = random_walk_forecast(df)
    
    # exponential smoothing - est.
    exp_pred_est = exponential_smoothing_est(df, train_i, n_values_gridsearch=10)
    
    # running trend
    run_trend_pred = running_trend(df, df.index + 1)
    
    # random walk with drift
    rw_drift_pred = random_walk_drift(df, df.index + 1)
    
    # holt winters
    hw_est = holt_winters_est(df, train_i, n_values_gridsearch=10)
    
    # dict of the predictions
    dict_predictions = {'running_avg': running_avg_pred,
                    'random_walk': rw_pred,
                    'exponential_smoothing': exp_pred_est,
                    'running_trend': run_trend_pred,
                    'random_walk_drift':rw_drift_pred, 
                    'holt_winters': hw_est}
    
    return dict_predictions


results_ts1_dict = produce_forecasts(ts_validation, train_i = 20)
t_ts1 = ts_validation.index + 1

list_of_results = [value for key, value in results_ts1_dict.items()]
list_evaluations =  [get_error, get_absolute_error, get_ape, get_squared_error]
list_names_method = ['Running Avg.', 'Random Walk','Exponential Smoothing (est.)', 'Running Trend', 'Random Walk + Drift', 'Holt Winters (est.)']
list_names_eval = ['Mean Residual', 'Mean Abs. Residual', 'Mean Abs. Perc. Residual', 'Mean Sq. Residual']

get_table_comparing_methods(ts_validation,
                            list_of_results,
                            list_names_method,
                            list_evaluations,
                            list_names_eval, 
                            last_n_forecasts=20)


standard_line_plot(t_ts1,[ts_validation, results_ts1_dict['exponential_smoothing'] ] ,['red', 'blue'], ['TS 1', 'RW'], [35,60],[0,40], 'Series 1')


prediction_exp_smoothing_ts1 = exponential_smoothing_est(ts, train_i = 40, n_values_gridsearch = 10)