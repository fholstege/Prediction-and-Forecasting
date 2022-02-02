from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


def read_csv(filename):
    df = np.genfromtxt(filename, delimiter=',')
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


def random_walk_seasonal(drift, df, season_size):
    season_lagged = df[season_size + 1:]
    last_season_removed = df[:-(season_size + 1)]
    if drift:
        seasonal_difference = []
        predictions = []
        for i in range(season_size, df.shape[0]):
            seasonal_difference.append(df[i] - df[i - season_size])
        drift_value = np.cumsum(seasonal_difference) / range(1, df.shape[0] - season_size + 1)
        for j in range(0, df.shape[0] - season_size):
            predictions.append(df[j] + drift_value[j])
        return predictions
    else:
        predictions = []
        for j in range(0, df.shape[0] - season_size):
            predictions.append(df[j])
        return predictions


def running_regression_seasonal(df):
    X = np.array(df)[:, 1:]
    y = np.array(df)[:, 0]
    beta_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.array(X).transpose(), np.array(X))), X.transpose()), y)
    return beta_hat


def make_forecast(df, season_size):
    predictions = []
    for i in range(season_size + 1, df.shape[0] - 1):
        params = running_regression_seasonal(df[:i])
        prediction = params @ df.iloc[i + 1][1:]
        predictions.append(prediction)
    return predictions


def seasonal_holt_winters(additive, df, s, alpha, beta, gamma):
    # df = df[s:]
    L = []
    G = []
    H = []
    predictions = []
    for j in range(s):
        L.append(np.sum(df[:s]) / s)
        G.append((np.sum(df[s:s + s]) / s - L[0]) / s)
        H.append(df[j] / L[j])
    if additive:
        for i in range(s, df.shape[0]):
            L.append(alpha * (df[i] - H[i - s]) + (1 - alpha) * (L[i - 1] + G[i - 1]))
            G.append(beta * (L[i] - L[i - 1]) + (1 - beta) * G[i - 1])
            H.append(gamma * (df[i] - L[i]) + (1 - gamma) * H[i - s])
            predictions.append(H[i - s] + L[i] + G[i])
    else:
        for i in range(s, df.shape[0]):
            L.append(alpha * (df[i] / H[i - s]) + (1 - alpha) * (L[i - 1] + G[i - 1]))
            G.append(beta * (L[i] - L[i - 1]) + (1 - beta) * G[i - 1])
            H.append(gamma * df[i] / L[i] + (1 - gamma) * H[i - s])
            predictions.append(H[i - s] * (L[i] + G[i]))
    return predictions


def plot_predictions(df, pred):
    plt.plot(range(1, df.shape[0]), pred)
    plt.plot(range(0, df.shape[0]), df)
    plt.show()


def plot_predictions_seasonal(df, pred, pred_us, season_size):
    plt.plot(range(season_size, df.shape[0]), pred)
    plt.plot(range(0, df.shape[0]), df)
    plt.plot(range(season_size, df.shape[0]), pred_us)
    plt.show()


def plot_errors(df, pred):
    errors = df - pred
    plt.bar(errors)
    plt.show()


def add_row_to_df(actual, prediction, method, df):
    # Removes the first observation of the actual timeseries and the last of the prediction timeseries
    error = np.mean(get_error(actual[4:], prediction))
    ae = np.mean(get_absolute_error(actual[4:], prediction))
    ape = np.mean(get_ape(actual[4:], prediction))
    sq = np.mean(get_squared_error(actual[4:], prediction))
    df = df.append({'Method': method, 'ME': error, 'MAE': ae, 'MAPE': ape, 'MSE': sq}, ignore_index=True)
    return df


def create_error_table(data, non_dummy):
    df = pd.DataFrame(columns=['Method', 'ME', 'MAE', 'MAPE', 'MSE'])
    rw_s_nd = random_walk_seasonal(False, non_dummy, 4)
    df = add_row_to_df(non_dummy, rw_s_nd, 'rw_without_drift', df)
    rw_s_d = random_walk_seasonal(True, non_dummy, 4)
    df = add_row_to_df(non_dummy, rw_s_d, 'rw_with_drift', df)
    rsg = make_forecast(data, 4)
    df = add_row_to_df(non_dummy[2:], rsg, 'seasonal_regression', df)
    print(df)


def holt_package(df):
    init_level = np.sum(df[:4]) / 4
    init_growth = (np.sum(df[4:4 + 4]) / 4 - (np.sum(df[:4]) / 4)) / 4
    init_season = []
    for j in range(4):
        init_season.append(df[j] / np.sum(df[:4]) / 4)
    fit = ExponentialSmoothing(df, trend='additive', seasonal='additive', seasonal_periods=4,
                               initialization_method='known', initial_level=init_level, initial_trend=init_growth,
                               initial_seasonal=init_season).fit(0.2, 0.2, 0.2, optimized=False)
    fcast = fit.predict(start=4, end=df.shape[0]-1)
    print(fcast)
    print(len(fcast))
    return fcast


if __name__ == '__main__':
    gas_data = read_csv("data/gasoline.csv")
    umbrella = np.array(pd.read_excel("data/Umbrella.xlsx")['Umbrella Sales'])
    umbrella_season = pd.read_excel("data/Umbrella.xlsx")[['Umbrella Sales', 'SeasIdx']]
    dummy_umbrella = pd.get_dummies(umbrella_season, columns=['SeasIdx'], drop_first=True)
    dummy_umbrella['t'] = range(1, dummy_umbrella.shape[0] + 1)
    dummy_umbrella['c'] = np.ones(dummy_umbrella.shape[0])
    create_error_table(dummy_umbrella, umbrella)
    test = seasonal_holt_winters(False, umbrella, 4, 0.2, 0.2, 0.2)
    #plot_predictions_seasonal(umbrella, test, 4)
    test1 = seasonal_holt_winters(True, umbrella, 4, 0.2, 0.2, 0.2)
    print(test1)
    print(len(test1))
    test2 = holt_package(umbrella)
    plot_predictions_seasonal(umbrella, test1, test2, 4)

