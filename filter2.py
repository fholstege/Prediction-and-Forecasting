import warnings

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# -*- coding: utf-8 -*-
from ruben_file import get_error, get_absolute_error, get_ape, get_squared_error, random_walk_forecast, \
    exponential_smoothing_est

"""
Created on Tue Feb 15 12:15:12 2022

@author: flori
"""

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


# Code van jou Floris
def est_ARMA_model(train_y, test_y, param='est', h=1, p=1, q=1, d=0, return_inSample=False,
                   max_p=4, max_q=1, max_d=0, criterion='bic',
                   train_exog=None, test_exog=None,
                   **kwargs):
    # how many in train/test
    n_in_train = len(train_y)
    n_in_test = len(test_y)

    # save predictions here
    pred = [np.nan] * n_in_train

    # estimate parameters based on training set or pre-defined
    if param == 'est':
        p, q, d = select_param_ARMA(train_y, max_p=max_p, max_q=max_q, max_d=max_d, criterion=criterion,
                                    exog=train_exog)
        trained_model = ARIMA(train_y, order=(p, d, q), exog=train_exog, trend="c", **kwargs).fit(method='statespace')
    else:
        trained_model = ARIMA(train_y, order=(p, d, q), exog=train_exog, **kwargs).fit(method='statespace')

    # save in sample fit here

    inSample = list(trained_model.predict(0, n_in_train - 1).values)
    if d != 0:
        inSample[0] = np.nan
        inSample = inSample + [np.nan] * (n_in_test)
    else:
        inSample = [np.nan] + inSample[:-1] + [np.nan] * (n_in_test)

    # change train_y, test_y based on h-step forecast
    if h > 1:
        train_y = train_y[:(n_in_train - h + 1)]
        test_y = pd.concat([train_y[-(h - 1):], test_y])

        if train_exog is not None:
            train_exog = train_exog[:(n_in_train - h + 1)]
            test_exog = pd.concat([train_exog[-(h - 1):], test_exog])

    # get the first prediction
    if train_exog is not None:
        pred_first = trained_model.forecast(steps=h, exog=test_exog.iloc[0:h, :]).values[-1]
    else:
        pred_first = trained_model.forecast(steps=h).values[-1]

    # add to list
    pred.append(pred_first)

    # get the other predictions
    for i in range(0, n_in_test):

        # select train data
        train_y = pd.Series(np.concatenate((train_y.values, [test_y.iloc[i]])))

        # if exogenous include
        if train_exog is not None:

            # set training (exog)
            train_exog = pd.DataFrame(np.concatenate((train_exog.values, [test_exog.iloc[i]])))

            # fit the model again
            trained_model = ARIMA(train_y, order=(p, d, q), exog=train_exog, **kwargs).fit(method='statespace')

            # get prediction
            pred_at_i = trained_model.forecast(steps=h, exog=test_exog.iloc[i:(i + h), :]).values[-1]


        else:
            # train model
            trained_model = ARIMA(train_y, order=(p, d, q), **kwargs).fit(method='statespace')
            # get predictions based on that p, q
            pred_at_i = trained_model.forecast(steps=h).values[-1]

        # add last prediction to list
        pred.append(pred_at_i)

    # shift back one
    pred = pred[:-1]

    if return_inSample:
        return pred, inSample, trained_model
    else:
        return pred


def select_param_ARMA(train_y, criterion='aic', max_p=4, max_q=4, max_d=1, exog=None):
    # starting parameters
    best_criterion_score = np.Inf
    best_p = 0
    best_q = 0
    best_d = 0

    # go over all parameter combinations
    for d in range(0, max_d + 1):
        for q in range(0, max_q + 1):
            for p in range(0, max_p + 1):

                # estimate the model at a lag
                arma_model_at_lag = ARIMA(train_y, order=(p, d, q), exog=exog, trend='c').fit(method='statespace')

                # get the score for a particular criterion
                if criterion == 'aic':
                    criterion_score_lag = arma_model_at_lag.aic
                elif criterion == 'bic':
                    criterion_score_lag = arma_model_at_lag.bic
                elif criterion == 'hqic':
                    criterion_score_lag = arma_model_at_lag.hqic
                else:
                    print("Specify the criterion: one of aic, bic, hqic")

                print('p: {}, q:{}, d{}, with a {} of {}'.format(p, q, d, criterion, criterion_score_lag))

                # if better score, save
                if criterion_score_lag < best_criterion_score:
                    best_criterion_score = criterion_score_lag
                    best_p = p
                    best_q = q
                    best_d = d

    print('Best p/q/d: {}/{}/{}, with a {} of {}'.format(best_p, best_q, best_d, criterion, best_criterion_score))

    return best_p, best_q, best_d


# Hier begint mijn code, miss kleine veranderingen in je arma code gedaan maar in principe is t hetzelfde
def kalman_filter(y, training_observations, out_of_sample=True, sigma_epsilon=30.0, sigma_eta=1.5):
    """

    :param y: np.array with time Series we forecast
    :param training_observations: Number of observations for training / in-sample forecasting
    :param out_of_sample: Whether we do out of sample forecasts or not, requires y.shape[0] > training_observations
    :param sigma_epsilon: Given sigma of epsilon
    :param sigma_eta: Given sigma of eta
    :return: TODO
    """
    estimated_a = [0]
    estimated_p = [sigma_epsilon * 10 ** 7]
    out_sample_p = []
    for i in range(training_observations):
        vt = y[i] - estimated_a[i]
        kt = estimated_p[i] / (estimated_p[i] + sigma_epsilon)
        estimated_a.append(estimated_a[i] + kt * vt)
        estimated_p.append(kt * sigma_epsilon + sigma_eta)

    if out_of_sample:
        out_sample_p = [estimated_p[-1]]
        for j in range(y.shape[0] - training_observations):
            kt = out_sample_p[j] / (out_sample_p[j] + sigma_epsilon)
            out_sample_p.append(kt * sigma_epsilon + sigma_eta)

    # Confidence interval calculated using mu +/- z * sigma / sqrt(n)
    # We take sqrt(n) as total amount of observations included in both forecasting and training, however maybe better to
    # Only use training_obs
    # Sigma of forecast given by Kalman Filter as f(y_{T+h}|Y_{T}) = N(a_{T|T}, p_{T|T} + h * sigma_{eta} + sigma_{epsilon})
    # a_{T|T) is last estimated a, while p_{T|T} = p_{T+1} - \sigma_{eta} and estimated out of sample.
    confidence_interval_upper = [estimated_a[-1] + 1.96 * np.sqrt(p - sigma_eta + (t + 1) * sigma_eta + sigma_epsilon)
                                 / np.sqrt(training_observations) for t, p in enumerate(out_sample_p[1:])]
    confidence_interval_lower = [estimated_a[-1] - 1.96 * np.sqrt(p - sigma_eta + (t + 1) * sigma_eta + sigma_epsilon)
                                 / np.sqrt(training_observations) for t, p in enumerate(out_sample_p[1:])]

    return np.array(estimated_a), np.array(estimated_p), np.full((len(out_sample_p[1:])), estimated_a[-1]), \
           np.array(confidence_interval_upper), np.array(confidence_interval_lower)


def standard_line_plot(x, y_series, color_lines, labels, ylim, xlim, ylabel, legend=False, show=True,
                       ticks=False, tick_labels=None, legend_pos='upper left'):
    plt.style.use('classic')
    plt.figure(facecolor="white")

    for i in range(len(y_series)):
        plt.plot(x[i], y_series[i], color=color_lines[i], label=labels[i])

    plt.xlabel('T')
    plt.ylabel(ylabel)
    plt.ylim(bottom=ylim[0], top=ylim[1])
    plt.xlim(xlim)

    if legend:
        plt.legend(loc=legend_pos)

    if ticks:
        number_years = np.ceil(len(y_series[0]) / 4)
        labels = [str(number) for number in range(1, int(number_years + 1))]
        plt.xticks(np.arange(min(x), max(x) + 1, 4.0), labels=labels)

    if show:
        plt.show()


def add_row_to_df_non_season(actual, prediction, method, df):
    # Removes the first observation of the actual timeseries and the last of the prediction timeseries
    actual = np.array(actual)
    prediction = np.array(prediction)

    error = np.mean(get_error(actual[1:], prediction))
    ae = np.mean(get_absolute_error(actual[1:], prediction))
    ape = np.mean(get_ape(actual[1:], prediction))
    sq = np.mean(get_squared_error(actual[1:], prediction))
    df = df.append({'Method': method, 'ME': error, 'MAE': ae, 'MAPE': ape, 'MSE': sq}, ignore_index=True)
    return df


def create_error_table_non_season(data, train_obs, df_data):
    df = pd.DataFrame(columns=['Method', 'ME', 'MAE', 'MAPE', 'MSE'])
    # Here 20 is the num training obs
    rw_s_nd = random_walk_forecast(data[:train_obs])
    df = add_row_to_df_non_season(data[:train_obs], rw_s_nd, 'Random Walk', df)
    rw_s_d = exponential_smoothing_est(data[:train_obs])
    df = add_row_to_df_non_season(data[:train_obs], rw_s_d, 'Exponential Smoothing', df)
    out_sample_pred, in_sample_pred, model = est_ARMA_model(df_data['NonResponse'][:train_obs],
                                                            df_data['NonResponse'][train_obs:], return_inSample=True)
    out_sample_pred = model.forecast(steps=data.shape[0] - train_obs)
    print(model)
    print(model.summary())
    print(data[:train_obs][-1])
    in_sample_pred = np.array(in_sample_pred)
    in_sample_pred = in_sample_pred[~np.isnan(in_sample_pred)]
    print(in_sample_pred[-1])
    df = add_row_to_df_non_season(data[:train_obs], in_sample_pred, "ARMA Model", df)

    print(np.mean(data[:train_obs]))
    standard_line_plot([range(data.shape[0]), range(1, train_obs), range(train_obs, data.shape[0])],
                       [data, in_sample_pred, out_sample_pred], ['red', 'blue', 'green'],
                       ['Actual', "Estimate", "Out of Sample"], (-40, 80), (0, 130), "test")
    standard_line_plot([range(data.shape[0]), range(1, train_obs)], [data, rw_s_nd], ['red', 'blue'],
                       ['Actual', "Estimate"], (-40, 80), (0, 130), "test")
    standard_line_plot([range(data.shape[0]), range(1, train_obs)], [data, rw_s_d], ['red', 'blue'],
                       ['Actual', "Estimate"], (-40, 80), (0, 130), "test")


if __name__ == '__main__':
    ts = pd.read_excel("Data/NonResponse.xlsx")
    ts_gasoline = pd.read_excel("Data/GasolineSales2.xlsx", header=None)
    ts_gas = ts_gasoline[0]
    ts_non_response = np.array(ts['NonResponse'])
    num_train_obs_gas = 20
    num_train_obs_non = 100

    est_a, est_p, est_out_p, con_interval_up, con_interval_down = kalman_filter(ts_gas, num_train_obs_gas, sigma_eta=10,
                                                                                sigma_epsilon=3.33)
    est_a_non, est_p_non, est_out_p_non, con_interval_up_non, con_interval_down_non = \
        kalman_filter(ts_non_response, num_train_obs_non, sigma_eta=10, sigma_epsilon=3.33)
    # create_error_table_non_season(ts_gas, 12)
    df_results = create_error_table_non_season(ts_non_response, ts_non_response.shape[0] - 24, ts)
    # standard_line_plot(
    #     [range(0, num_train_obs_gas + 1), range(ts_gas.shape[0]), range(num_train_obs_gas + 1, ts_gas.shape[0] + 1),
    #      range(num_train_obs_gas + 1, ts_gas.shape[0] + 1), range(num_train_obs_gas + 1, ts_gas.shape[0] + 1)],
    #     [est_a, ts_gas, est_out_p, con_interval_up, con_interval_down],
    #     ['red', 'blue', 'black', 'purple', 'purple'],
    #     ['estimation', 'actual', 'out_sample', 'confidence_interval_up', 'confidence_interval_down'],
    #     (0, 50), (0, 30), "test")

    # standard_line_plot([range(1, num_train_obs_non+1), range(ts_non_response.shape[0]), range(num_train_obs_non+1, ts_non_response.shape[0]+1),
    #                     range(num_train_obs_non+1, ts_non_response.shape[0]+1), range(num_train_obs_non+1, ts_non_response.shape[0]+1)],
    #                    [est_a_non, ts_non_response, est_out_p_non, con_interval_up_non, con_interval_down_non], ['red', 'blue', 'black', 'purple', 'purple'],
    #                    ['estimation', 'actual', 'out_sample', 'confidence_interval', 'confidence_interval'],
    #                    (-40, 80), (0, 130), "test")
