# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:19:15 2022

@author: flori
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from matplotlib import pyplot as plt

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


def running_average(df):
    pred = np.cumsum(df) / range(1, df.shape[0] + 1)
    return pred


def get_error(data, pred):
    return data - pred


def get_absolute_error(data, pred):
    return np.abs(data - pred)


def get_ape(data, pred):
    return (abs(data - pred) / abs(data)) * 100


def get_squared_error(data, pred):
    return (data - pred) ** 2


def random_walk_forecast(df):
    return df[:-1]


def expert_forecast(df, value):
    return np.full(df.shape[0] - 1, value)


def exponential_smoothing(df, alpha):
    # Inefficient loop but hey
    predictions = [df[0]]
    for i in range(0, df.shape[0] - 1):
        predictions.append(alpha * df[i] + (1 - alpha) * predictions[-1])
    return np.array(predictions[1:])

def exponential_smoothing_est(df):
    fit = SimpleExpSmoothing(df, initialization_method="estimated").fit()
    pred = exponential_smoothing(df, fit.params['smoothing_level'])
    return pred


def standard_line_plot(x, y_series, color_lines,labels,ylim,xlim,ylabel, legend = False):
        
    plt.style.use('classic')
    plt.figure(facecolor="white")
    
    for i in range(len(y_series)):
        
        plt.plot(x, y_series[i], color = color_lines[i], label = labels[i])
        
    plt.xlabel('T')
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.show()
