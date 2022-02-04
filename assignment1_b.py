# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 10:15:53 2022

@author: flori
"""

from helpers_pf import * 


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



def create_prediction_plot(ts_index,ts, results_inSample, results_outSample, method_name,label_name, length_sample= 40, length_forecast = 10, length_validation= 20, series_name ='Series 1',  ylim = [0, 50]):
    
    
    xlim = [np.nanmin(ts) - 10, np.nanmax(ts) + 10]
    pred_outSample = ([np.nan] * length_sample) +list(results_outSample[method_name][length_sample:(length_sample + length_forecast)])
    pred_inSample = ([np.nan] * length_validation) +list(results_inSample[method_name][length_validation:length_sample]) + ([np.nan]*length_forecast)
    
    
    standard_line_plot(ts_index,[ts, pred_outSample,pred_inSample ] ,['red', 'blue', 'green'], [series_name, label_name + ' (in-sample)', label_name + ' (out-sample)'],['o', None, None], xlim,ylim, series_name, legend = True)
    

def evaluate_series_comprehensive(ts, train_i = 40, val_i = 20, name_series = 'Series 1'):
    
    ts_validation = ts[:train_i]

    
    results_ts1_dict_inSample = produce_forecasts(ts_validation, train_i = val_i)
    results_ts1_dict_outSample = produce_forecasts(ts, train_i = train_i)
    
    t_ts1_inSample = ts_validation.index + 1
    t_ts1_outSample = ts.index + 1
    
    list_of_results_inSample = [value for key, value in results_ts1_dict_inSample.items()]
    list_of_results_outSample = [value for key, value in results_ts1_dict_outSample.items()]
    
    list_evaluations =  [get_error, get_absolute_error, get_ape, get_squared_error]
    list_names_method = ['Running Avg.', 'Random Walk','Exponential Smoothing (est.)', 'Running Trend', 'Random Walk + Drift', 'Holt Winters (est.)']
    list_names_eval = ['Mean Residual', 'Mean Abs. Residual', 'Mean Abs. Perc. Residual', 'Mean Sq. Residual']
    
    # in sample
    table_inSample = get_table_comparing_methods(ts_validation,
                                list_of_results_inSample,
                                list_names_method,
                                list_evaluations,
                                list_names_eval, 
                                last_n_forecasts=20)
    #out of sample
    table_outSample = get_table_comparing_methods(ts,
                                list_of_results_outSample,
                                list_names_method,
                                list_evaluations,
                                list_names_eval, 
                                last_n_forecasts=10)
    
    list_of_methods = list(results_ts1_dict_inSample.keys())
    
    i = 0
    for name_method in list_of_methods:

        create_prediction_plot(t_ts1_outSample, ts,
                       results_ts1_dict_inSample,
                       results_ts1_dict_outSample,
                       name_method,
                       list_names_method[i],
                       series_name = name_series
                       )
        i = i + 1
    
    return round(table_inSample, 2), round(table_outSample,2)

table_inSample_series1, table_outSample_series1 =  evaluate_series_comprehensive(ts_1, name_series = 'Series 1')
table_inSample_series2, table_outSample_series2 =  evaluate_series_comprehensive(ts_2, name_series = 'Series 2')
table_inSample_series3, table_outSample_series3 =  evaluate_series_comprehensive(ts_3, name_series = 'Series 3')
table_inSample_series4, table_outSample_series4 =  evaluate_series_comprehensive(ts_4, name_series = 'Series 4')
table_inSample_series5, table_outSample_series5 =  evaluate_series_comprehensive(ts_5, name_series = 'Series 5')
table_inSample_series6, table_outSample_series6 =  evaluate_series_comprehensive(ts_6, name_series = 'Series 6')
table_inSample_series7, table_outSample_series7 =  evaluate_series_comprehensive(ts_7, name_series = 'Series 7')
table_inSample_series8, table_outSample_series8 =  evaluate_series_comprehensive(ts_8, name_series = 'Series 8')
table_inSample_series9, table_outSample_series9 =  evaluate_series_comprehensive(ts_9, name_series = 'Series 9')

