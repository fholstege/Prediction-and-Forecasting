# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:19:15 2022

@author: flori
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm


def read_xlsx(filename, **kwargs):
    df = pd.read_excel(filename,**kwargs)
    print(df)
    return df

def read_csv(filename):
    df = np.genfromtxt(filename, delimiter=',')
    print(df)
    return df


def average_prediction(df):
    pred = np.mean(df)
    return pred


def running_average(df, shift=1):
    pred = np.cumsum(df) / range(1, df.shape[0] + 1)
    return pred.shift(shift)


def get_error(data, pred):
    return data - pred


def get_absolute_error(data, pred):
    return np.abs(data - pred)


def get_ape(data, pred):
    return (abs(data - pred) / abs(data)) * 100


def get_squared_error(data, pred):
    return (data - pred) ** 2


def random_walk_forecast(df):
    return df.shift(1)


def expert_forecast(df, value):
    return np.full(df.shape[0] - 1, value)


def exponential_smoothing(df, alpha):
    # Inefficient loop but hey
    predictions = [df[0]]
    for i in range(0, df.shape[0] - 1):
        predictions.append(alpha * df[i] + (1 - alpha) * predictions[-1])
        
    result = np.array(predictions)
    result[0] = np.nan
    return result

def exponential_smoothing_est(df, train_i, test_start_i=None, n_values_gridsearch = 100, return_param=False):
    
    if test_start_i is None:
        test_start_i=len(df)
    
    # save predictions
    pred = [np.nan]*train_i
    alpha = [np.nan]*train_i

    # get the predictions from train_i onwards
    for i in range(1, (test_start_i - train_i)+1):
        
        # select train data
        df_train = df[:(train_i+i)]
        
        # get alpha for train data 
        alpha_at_i = grid_search_alpha_exp_smoothing(df_train, n_values_gridsearch)
        alpha.append(alpha_at_i)
        
        # get predictions based on that alpha
        pred_at_i = exponential_smoothing(df_train, alpha_at_i)[-1]
        
        # add last prediction to list
        pred.append(pred_at_i)
    
    for j in range(1, len(df) - test_start_i+1):
        print('using best estimate of alpha: {}'.format(alpha_at_i))
        
        df_test = df[:(test_start_i+j)]
        

        pred_at_i = exponential_smoothing(df_test, alpha_at_i)[-1]
        alpha.append(alpha_at_i)

        pred.append(pred_at_i)
    
    if return_param:
        return pred, alpha
    else:
        return pred

def holt_winters_est(df, train_i,test_start_i=None, n_values_gridsearch = 10, return_param = False):
    
    if test_start_i is None:
        test_start_i=len(df)
    
    pred = [np.nan]*train_i
    alpha = [np.nan]*train_i
    beta = [np.nan]*train_i
    
    # get the predictions from train_i onwards
    for i in range(1, (test_start_i - train_i)+1):
        
        # select train data
        df_train = df[:(train_i+i)]
        
        # get alpha for train data 
        alpha_at_i, beta_at_i = grid_search_alpha_beta_hw(df_train, n_values_gridsearch)
        
        # get predictions based on that alpha
        pred_at_i = holt_winters_package(df_train, alpha_at_i, beta_at_i)[-1]
        alpha.append(alpha_at_i)
        beta.append(beta_at_i)
        
        # add last prediction to list
        pred.append(pred_at_i)
    
    for j in range(1, len(df) - test_start_i+1):
        print('using best estimate of alpha, beta: {}, {}'.format(alpha_at_i, beta_at_i))
        
        df_test = df[:(test_start_i+j)]
        
        # get predictions based on that alpha/beta combination
        pred_at_i = holt_winters_package(df_test, alpha_at_i, beta_at_i)[-1]
        alpha.append(alpha_at_i)
        beta.append(beta_at_i)
        
        pred.append(pred_at_i)
    
    if return_param:
        return pred, alpha, beta
    else:
        return pred
        
    

def grid_search_alpha_beta_hw(df, n_values = 10):
    
    possible_alpha = np.linspace(0.0, 1.0, n_values+1)
    possible_beta = np.linspace(0.0, 1.0, n_values+1)
    
    current_best = float('inf')
    best_alpha = 0.0
    best_beta = 0.0
    
    for alpha in possible_alpha:
        for beta in possible_beta:
            
            prediction_at_alpha_beta = holt_winters_package(df, alpha, beta)
            sq_error_at_alpha_beta = get_squared_error(df, prediction_at_alpha_beta)
            mse_at_alpha_beta = np.mean(sq_error_at_alpha_beta)
            
            if mse_at_alpha_beta < current_best:
                current_best = mse_at_alpha_beta
                best_alpha = alpha
                best_beta = beta
                
    print('Best alpha/beta combination: {}/{}, with an MSE of {}'.format(best_alpha,best_beta,current_best ))
    
    return best_alpha, best_beta

def grid_search_alpha_exp_smoothing(df, n_values = 100):


    possible_alpha =np.linspace(0.0,1.0,n_values + 1)
    current_best = float('inf') 
    best_alpha = 0.0

    for alpha in possible_alpha:
        
        prediction_at_alpha = exponential_smoothing(df, alpha)
        sq_error_at_alpha = get_squared_error(df,prediction_at_alpha)
        mse_at_alpha = np.mean(sq_error_at_alpha)
        
        if mse_at_alpha < current_best:
            current_best = mse_at_alpha
            best_alpha = alpha
    
    print('Best alpha: {}, with an MSE of {}'.format(best_alpha,current_best ))
    return best_alpha
    
def ar_1_model_pred(df, train_i):
    
    # save predictions
    pred = [np.nan]*train_i

    # get the predictions from train_i onwards
    df_train = df[:train_i]
    ar_model = AutoReg(df_train, lags=1).fit()
    
    for i in range(1, len(df) - train_i+1):
        
         # select train data
         df_train = df[:(train_i+i)]
        
         ar_model = AutoReg(df_train, lags=1).fit()
        
         # get predictions based on that alpha
         pred_at_i = ar_model.predict(start=len(df_train), end=len(df_train)).values[0]
        
         # add last prediction to list
         pred.append(pred_at_i)
    
    return pred[-1:]+pred[:-1]
    

def running_trend(df, t, return_param = False):
    
    
    predictions = [np.nan]*2
    a_values = [np.nan]*2
    b_values = [np.nan]*2
    
    
    for i in range(2, len(df)):
        
        Y = df[:i]
        t_i = t[:i]
        X = sm.add_constant(t_i)
        
        model = sm.OLS(Y,X)
        fitted_model = model.fit()
        params = fitted_model.params
        a = params[0]
        b = params[1]
        a_values.append(a)
        b_values.append(b)
        
        new_obs = [1., i+1]
        
        pred = fitted_model.predict(new_obs)
        
        predictions.append(pred[0])
    
    if return_param:
        return predictions, a_values, b_values
    else:
        return predictions


def random_walk_drift(df, t, return_param = False):
    
    predictions = [np.nan]*2
    diff = df - df.shift(1)
    c = [np.nan]*2
    
    for i in range(2, len(df)):
        
        diff_up_until = diff[:(i)]
                
        diff_up_until_avg = (1/ (i-1)) * np.sum(diff_up_until)
                
        pred = df[i-1] + diff_up_until_avg
        
        predictions.append(pred)
        c.append(diff_up_until_avg)

        
    if return_param:
        return predictions, c
    else:
        return predictions
    

def holt_winters_package(Y, alpha, beta):
    
  # try with package
  hw_fit = Holt(Y, initialization_method="known", initial_level=Y[0], initial_trend = Y[1]- Y[0]).fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)

  fcast = hw_fit.predict(start=2, end = len(Y)-1)
  start = [np.nan]*2
  fcast_package = start + list(fcast.values)
  
  return fcast_package






def holt_winters(Y, alpha, beta):
    
    # initialize values
    L_t_min1 = Y[0]
    G_t_min1 = Y[1] - Y[0]
    
    print('Initial: {}, {}'.format(L_t_min1,G_t_min1 ))
    
    # fill in predictions to this list
    predictions = [np.nan]*2
    level = [L_t_min1]
    trend = [G_t_min1]
    
    
    # first prediction
    Y_t_plus1 = L_t_min1 + G_t_min1
    
    # loop from 2 to T-1
    for i in range(1, len(Y)-1):
                
        print('t : {}'.format(i+1))
        
        # update L_t
        L_t = (alpha * Y[i]) + (1-alpha)*  ( L_t_min1 + G_t_min1)
        print('level: {}'.format(L_t))
        level.append(L_t)
        
        # update G_t
        G_t = beta *(L_t - L_t_min1) + (1-beta)*G_t_min1
        print('trend: {}'.format(G_t))
        trend.append(G_t)
        
        # update forecast
        Y_t_plus1 = L_t + G_t
        predictions.append(Y_t_plus1)
        
        # update L_t, G_t for next
        L_t_min1 = L_t
        G_t_min1 = G_t
    
    return predictions, level, trend
    


def standard_line_plot(x, y_series, color_lines,labels,markers,ylim,xlim,ylabel, legend = False, show = True, ticks=False, tick_labels = None, legend_pos = 'upper left'):
        
    plt.style.use('classic')
    plt.figure(facecolor="white")
    
    for i in range(len(y_series)):
        
        plt.plot(x, y_series[i], color = color_lines[i], label = labels[i], marker=markers[i])
        
    plt.xlabel('T')
    plt.ylabel(ylabel)
    plt.ylim(bottom=ylim[0], top=ylim[1])
    plt.xlim(xlim)
    
    if legend:
        plt.legend(loc=legend_pos)
        
    if ticks:
        number_years = np.ceil(len(y_series[0])/4)
        print(number_years)
        labels = [str(number) for number in range(1,int(number_years+1))]
        plt.xticks(np.arange(min(x), max(x)+1, 4.0), labels =labels )

    if show:
        plt.show()


def standard_residual_plot(x, residuals, color, ylim = None, ticks=False):
    
    plt.style.use('classic')
    plt.figure(facecolor="white")
    
    plt.scatter(x, residuals, color = 'black')
    
    for i in range(len(x)):
        plt.plot((x[i], x[i]), (0, residuals[i]), color)
    
    plt.axhline(0, color= 'black')
    
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
        
    if ticks:
        number_years = np.ceil(len(residuals)/4)
        labels = [str(number) for number in range(1,int(number_years+1))]
        plt.xticks(np.arange(min(x), max(x)+1, 4.0), labels =labels )
        
    plt.show()
    
    

def get_table_comparing_methods(y, predictions, pred_names, evaluation_methods, evaluation_names, last_n_forecasts = 'all'):
    
    
    n_predictions = len(predictions)
    n_evaluations = len(evaluation_methods)
    
    df_scores = pd.DataFrame(np.nan, index=pred_names, columns=evaluation_names)

    for i_pred in range(n_predictions):
        for j_eval in range(n_evaluations):
            
            if last_n_forecasts == 'all':
                prediction = predictions[i_pred]
            else:
                prediction = predictions[i_pred][-last_n_forecasts:]
                y = y[-last_n_forecasts:]
                
            evaluation = evaluation_methods[j_eval]
            
            score = evaluation(y, prediction)
            df_scores.iloc[i_pred, j_eval] = np.nanmean(score)
    
    return df_scores



def forecast_weights_exp(T, alpha, ylim = None):
    
    weights = []
    
    for j in range(0, T):
        
        weight = alpha * (1 - alpha)** j
        weights.append(weight)
    
    #last_weight = (1 - alpha) ** (T-1)
    #weights.append(last_weight)
    
    return weights

def calc_memory_index(alpha):
    return np.log(0.1)/np.log(1-alpha)

def weights_plot(T, weights, label, color, scale=False, width = 0.6, ylim =None, label2='0.1'):
    
    plt.style.use('classic')
    plt.figure(facecolor="white")
    if scale:
        plt.axhline(0.1, color = 'black', label = label2)
        plt.bar(range(1, T+1),  [x / weights[0] for x in weights], width=width, label = label, color = color, linewidth=0)
    else:
        plt.bar(range(1, T+1), weights, width=0.6, label = label, color = color)
    plt.xlim([0, T])
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend(loc="upper right")
    
def param_plot(t_index, params, colors, labels, width=0.3, xlim=[0,12], ylim = None, loc='lower right'):
    
    plt.style.use('classic')
    plt.figure(facecolor="white")
    
    for i in range(len(params)):
        param = params[i]
        label = labels[i]
        color = colors[i]
        
        plt.bar(t_index + (i * width), param , width, label=label, color = color)
    
    plt.xlim(xlim)
    
    if ylim is not None:
        plt.ylim(ylim)

   
    # Finding the best position for legends and putting it
    plt.legend(loc=loc)
    


    
    
    