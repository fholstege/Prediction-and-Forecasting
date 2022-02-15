# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:15:12 2022

@author: flori
"""

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
from helpers_pf import *



def est_AR_model(train_y, test_y, param='est',h=1, p=1,return_inSample=False,**kwargs):
    
    # how many in train/test
    n_in_train = len(train_y)
    n_in_test = len(test_y)
    
    # save predictions here
    pred = [np.nan]*n_in_train
    
    # estimate parameters based on training set or pre-defined
    if param =='est':
        p = select_param_AR(train_y, **kwargs)
        trained_model = AutoReg(train_y, lags=p).fit()
    else:
        trained_model = AutoReg(train_y, lags=p).fit()
    
    # save in sample fit here
    inSample = list(trained_model.predict(0,n_in_train-1).values)
    inSample = inSample + [np.nan]*n_in_test
    
    # change train_y, test_y based on h-step forecast
    if h>1:
        train_y = train_y[:(n_in_train - h + 1)]
        test_y = pd.concat([ train_y[-(h-1):], test_y])
    
    # get the first prediction
    pred_first = trained_model.forecast(steps=h).values[-1]
    pred.append(pred_first)
        
    # get the other predictions
    for i in range(0, n_in_test):
        
         # select train data
         train_y = pd.Series(np.concatenate((train_y.values, [test_y.iloc[i]])))
        
         # fit the model again
         trained_model = AutoReg(train_y, lags=1).fit()
        
         # get predictions based on that alpha
         pred_at_i = trained_model.forecast(steps=h).values[-1]
        
         # add last prediction to list
         pred.append(pred_at_i)
    
    # shift back one
    pred = pred[:-1]
    
    if return_inSample:
        return pred, inSample
    else:
        return pred


def est_ARMA_model(train_y, test_y, param='est',h=1, p=1,q=1,return_inSample=False,**kwargs):
    
    # how many in train/test
    n_in_train = len(train_y)
    n_in_test = len(test_y)
    
    # save predictions here
    pred = [np.nan]*n_in_train
    d=0
    
    # estimate parameters based on training set or pre-defined
    if param =='est':
        p, q = select_param_ARMA(train_y, **kwargs)
        trained_model = ARIMA(train_y, order=(p,d,q)).fit(method='statespace')
    else:
        trained_model = ARIMA(train_y, order=(p,d,q)).fit(method='statespace')
    
    # save in sample fit here
    inSample = list(trained_model.predict(0,n_in_train-1).values)
    inSample = inSample + [np.nan]*n_in_test
    
    # change train_y, test_y based on h-step forecast
    if h>1:
        train_y = train_y[:(n_in_train - h + 1)]
        test_y = pd.concat([ train_y[-(h-1):], test_y])
    
    # get the first prediction
    pred_first = trained_model.forecast(steps=h).values[-1]
    pred.append(pred_first)
        
    # get the other predictions
    for i in range(0, n_in_test):
        
         # select train data
         train_y = pd.Series(np.concatenate((train_y.values, [test_y.iloc[i]])))
        
         # fit the model again
         trained_model = ARIMA(train_y, order=(p,d,q)).fit(method='statespace')
        
         # get predictions based on that p, q
         pred_at_i = trained_model.forecast(steps=h).values[-1]
        
         # add last prediction to list
         pred.append(pred_at_i)
    
    # shift back one
    pred = pred[:-1]
    
    if return_inSample:
        return pred, inSample
    else:
        return pred




def select_param_ARMA(train_y, criterion='aic',max_p=4, max_q=4):
    
    # starting parameters
    best_criterion_score = np.Inf
    best_p = 0
    best_q = 0
    d=0 # since no ARIMA
    
    # go over all parameter combinations
    for q in range(0, max_q+1):
        for p in range(0, max_p+1):
            
            # estimate the model at a lag
            arma_model_at_lag = ARIMA(train_y, order=(p,d,q)).fit(method='statespace')
            
            # get the score for a particular criterion
            if criterion=='aic':
                criterion_score_lag = arma_model_at_lag.aic
            elif criterion=='bic':
                criterion_score_lag = arma_model_at_lag.bic
            elif criterion =='hqic':
                criterion_score_lag = arma_model_at_lag.hqic
            else:
                print("Specify the criterion: one of aic, bic, hqic")
            
            print('p: {}, q:{} with a {} of {}'.format(p,q,criterion, criterion_score_lag))

            # if better score, save
            if criterion_score_lag < best_criterion_score:
                best_criterion_score = criterion_score_lag
                best_p = p
                best_q = q
                
    print('Best p/q: {}/{}, with a {} of {}'.format(best_p,best_q, criterion, best_criterion_score))

    return best_p, best_q



def select_param_AR(train_y, criterion='aic', max_p=4):
    
    # starting parameters
    best_criterion_score = np.Inf
    best_p = 0
    
    # go over all the lags
    for p in range(0, max_p+1):
        
        # estimate the model at a lag
        ar_model_at_lag = AutoReg(train_y,lags=p).fit()
        
        # get the score for a particular criterion
        if criterion=='aic':
            criterion_score_lag = ar_model_at_lag.aic
        elif criterion=='bic':
            criterion_score_lag = ar_model_at_lag.bic
        elif criterion =='hqic':
            criterion_score_lag = ar_model_at_lag.hqic
        else:
            print("Specify the criterion: one of aic, bic, hqic")
        
        print('p: {}, with a {} of {}'.format(p,criterion, criterion_score_lag))

        # if better score, save
        if criterion_score_lag < best_criterion_score:
            best_criterion_score = criterion_score_lag
            best_p = p
    
    print('Best p: {}, with a {} of {}'.format(best_p, criterion, best_criterion_score))

    return best_p
        
def produce_forecasts_arma(y, train_i, max_p=4, max_q=1,h=1, criterion='bic', return_inSample=False):
    
    # training for gas sales: first six observations
    train_y = y[:train_i]
    test_y = y[train_i:]
    
    # get predictions: ar(1), ma(1), arma(1,1)
    if return_inSample:
        ar1_prediction, ar1_inSample = est_AR_model(train_y, test_y, param='set',h=h, p=1, return_inSample=return_inSample)
        ma1_prediction, ma1_inSample = est_ARMA_model(train_y, test_y, param='set',h=h, p=0, q=1, return_inSample=return_inSample)
        arma1_prediction, arma1_inSample = est_ARMA_model(train_y, test_y, param='set',h=h, p=1, q=1,  return_inSample=return_inSample)
    else:
        
        ar1_prediction = est_AR_model(train_y, test_y, param='set',h=h, p=1, return_inSample=return_inSample)
        ma1_prediction = est_ARMA_model(train_y, test_y, param='set',h=h, p=0, q=1, return_inSample=return_inSample)
        arma1_prediction = est_ARMA_model(train_y, test_y, param='set',h=h, p=1, q=1,  return_inSample=return_inSample)
        

    # estimate the arma
    armaEst_prediction, armaEst_inSample = est_ARMA_model(train_y, test_y, param='est',h=h, max_p=max_p, max_q=max_q, criterion=criterion,  return_inSample=True)

    
    # dict of the predictions
    dict_predictions = {'ar_1': ar1_prediction,
                    'ma_1': ma1_prediction,
                    'arma_1_1': arma1_prediction,
                    'arma_est': armaEst_prediction}
    
    # if in-sample, get the in-sample predictions
    if return_inSample:
        dict_inSample = {'ar_1': ar1_inSample,
                     'ma_1': ma1_inSample,
                     'arma_1_1':arma1_inSample,
                     'arma_est':armaEst_inSample}
    
        return dict_predictions, dict_inSample
    else:
        return dict_predictions


def evaluate_arma_forecasts_noValidation(y, train_i, max_p=4, max_q=1,h=1, criterion='bic', last_n=10, ylim=[0,12],name_series='Series 1'):
    
    forecasts,inSample = produce_forecasts_arma(y, train_i = train_i,  max_p=max_p, max_q=max_q, h=h, criterion=criterion, return_inSample=True)
    list_forecasts = [value for key, value in forecasts.items()]
    list_inSample = [value for key, value in inSample.items()]
    
    # evaluate in different ways
    list_evaluations =  [get_error, get_absolute_error, get_ape, get_squared_error]
    list_names_method = ['AR(1)', 'MA(1)','ARMA(1,1)', 'ARMA(est.)']
    list_names_eval = ['Mean Residual', 'Mean Abs. Residual', 'Mean Abs. Perc. Residual', 'Mean Sq. Residual']
    
    # in sample
    table_results = get_table_comparing_methods(y,
                                list_forecasts,
                                list_names_method,
                                list_evaluations,
                                list_names_eval, 
                                last_n_forecasts=last_n)
    
    
    # get the name of methods
    list_of_methods = list(inSample.keys())
    t = list(range(1, len(y)+1))

    
    # go over each
    i = 0
    for name_method in list_of_methods:

        create_prediction_plot_arma(t, 
                                    y,
                       inSample,
                       forecasts,
                       name_method,
                       list_names_method[i],
                       series_name = name_series,
                       ylim=ylim
                       )
        i = i + 1
    
    return table_results


    


def create_prediction_plot_arma(ts_index,ts, results_inSample, results_outSample, method_name,label_name, length_sample= 40, length_forecast = 10, length_validation= 20, series_name ='Series 1',  ylim = [0, 50]):
    
    
    # get the xlim
    xlim = [np.nanmin(ts) - 10, np.nanmax(ts) + 10]
    
    # predictions out of sample
    #pred_outSample = ([np.nan] * length_sample) +list(results_outSample[method_name][length_sample:(length_sample + length_forecast)])
    pred_outSample = list(results_outSample[method_name])
    
    # predictions in sample
    #pred_inSample = ([np.nan] * length_validation) +list(results_inSample[method_name][length_validation:length_sample]) + ([np.nan]*length_forecast)
    pred_inSample = list(results_inSample[method_name])

    
    # create the lineplot
    standard_line_plot(ts_index,[ts, pred_outSample,pred_inSample ] ,['red', 'blue', 'green'], [series_name, label_name + ' (in-sample)', label_name + ' (out-sample)'],['o', None, None], xlim,ylim, series_name, legend = True)

def create_prediction_plot_arma_comprehensive(ts_index,ts, results_inSample, results_outSample, method_name,label_name, length_sample= 40, length_forecast = 10, length_validation= 20, series_name ='Series 1',  ylim = [0, 50]):
    
    # get the xlim
    xlim = [np.nanmin(ts) - 10, np.nanmax(ts) + 10]
    
    # predictions out of sample
    pred_outSample = ([np.nan] * length_sample) +list(results_outSample[method_name][length_sample:(length_sample + length_forecast)])
    
    # predictions in sample
    pred_inSample = ([np.nan] * length_validation) +list(results_inSample[method_name][length_validation:length_sample]) + ([np.nan]*length_forecast)

    
    # create the lineplot
    standard_line_plot(ts_index,[ts, pred_outSample,pred_inSample ] ,['red', 'blue', 'green'], [series_name, label_name + ' (in-sample)', label_name + ' (out-sample)'],['o', None, None], xlim,ylim, series_name, legend = True)


def evaluate_arma_comprehensive(ts, train_i = 40, val_i = 20, name_series = 'Series 1', max_p=4, max_q=4, h=1, criterion='bic'):
    
    # select validation set
    ts_validation = ts[:train_i]
    
    # build the insample and out of sample forecasts
    results_ts1_dict_inSample = produce_forecasts_arma(ts_validation, train_i = val_i, max_p=max_p, max_q=max_q, h=h,criterion=criterion, return_inSample=False)
    results_ts1_dict_outSample = produce_forecasts_arma(ts, train_i = train_i,  max_p=max_p, max_q=max_q, h=h, criterion=criterion, return_inSample=False)
    
    # set the indeces
    t_ts1_inSample = ts_validation.index + 1
    t_ts1_outSample = ts.index + 1
    
    # get the results in sample and out of sample in a list
    list_of_results_inSample = [value for key, value in results_ts1_dict_inSample.items()]
    list_of_results_outSample = [value for key, value in results_ts1_dict_outSample.items()]
    
    # evaluate in different ways
    list_evaluations =  [get_error, get_absolute_error, get_ape, get_squared_error]
    list_names_method = ['AR(1)', 'MA(1)','ARMA(1,1)', 'ARMA(est.)']
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

        create_prediction_plot_arma_comprehensive(t_ts1_outSample, ts,
                       results_ts1_dict_inSample,
                       results_ts1_dict_outSample,
                       name_method,
                       list_names_method[i],
                       series_name = name_series
                       )
        i = i + 1
    
    return round(table_inSample, 2), round(table_outSample,2)