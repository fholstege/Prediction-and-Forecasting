from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


def read_csv(filename):
    df = np.genfromtxt(filename, delimiter=',')
    return df

def read_xlsx(filename):
    df = pd.read_excel(filename)
    df = df.dropna()
    return df

def average_prediction(df):
    pred = np.mean(df)
    return pred


def running_average(df):
    pred = np.cumsum(df) / range(1, df.shape[0] + 1)
    return pred


def get_error(data, pred):
    return pred - data


def get_absolute_error(data, pred):
    return np.abs(pred - data)


def get_ape(data, pred):
    return (abs(data - pred) / abs(data)) * 100


def get_squared_error(data, pred):
    return (pred - data) ** 2


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


def random_walk_seasonal(drift, df, season_size, start):
    season_lagged = df[season_size + 1:]
    last_season_removed = df[:-(season_size + 1)]
    if drift:
        seasonal_difference = []
        predictions = []
        for i in range(season_size, df.shape[0]):
            seasonal_difference.append(df[i] - df[i - season_size])
        drift_value = np.cumsum(seasonal_difference) / range(1, df.shape[0] - season_size + 1)

        for j in range(start, df.shape[0]):

            predictions.append(df[j-season_size] + drift_value[j-season_size])
        return predictions
    else:
        predictions = []
        for j in range(start, df.shape[0]):
            predictions.append(df[j-season_size])
        return predictions


def running_regression_seasonal(df):
    X = np.array(df)[:, 1:]
    y = np.array(df)[:, 0]
    beta_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.array(X).transpose(), np.array(X))), X.transpose()), y)
    return beta_hat


def make_forecast(df, season_size, start):
    predictions = []
    for i in range(start, df.shape[0]-1):
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
            H.append(gamma * (df[i] / L[i]) + (1 - gamma) * H[i - s])
            predictions.append(H[i - s] * (L[i] + G[i]))
    return predictions


def plot_predictions(df, pred):
    plt.plot(range(1, df.shape[0]), pred, label="Prediction")
    plt.plot(range(0, df.shape[0]), df, label="Actual")
    plt.show()


def plot_predictions_seasonal(df, pred, season_size, plot_name):
    plt.plot(range(season_size, df.shape[0]), pred, label="Prediction")
    plt.plot(range(0, df.shape[0]), df, label="Actual")
    plt.title(plot_name)
    plt.grid()
    plt.legend()
    #plt.plot(range(season_size, df.shape[0]), pred_us)
    plt.show()


def plot_errors(df, pred):
    errors = df - pred
    plt.bar(errors)
    plt.show()


def standard_line_plot(x, y_series, color_lines, labels, ylim, xlim, ylabel, legend=False, show=True, ticks=False,
                       tick_labels=None, name="temp"):
    plt.style.use('classic')
    plt.figure(facecolor="white")

    for i in range(len(y_series)):
        plt.plot(x[i], y_series[i], color=color_lines[i], label=labels[i])

    plt.xlabel('T')
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.xlim(xlim)

    if legend:
        plt.legend(loc="upper left")

    if ticks:
        number_years = np.ceil(len(y_series[0]) / 4)
        print(number_years)
        labels = [str(number) for number in range(1, int(number_years + 1))]
        plt.xticks(np.arange(min(x), max(x) + 1, 4.0), labels=labels)

    if show:
        plt.show()
    else:
        plt.savefig(name)


def add_row_to_df(actual, prediction, method, df, season_size, start):
    # Removes the first season_size observation of the actual timeseries
    print(len(prediction))
    error = np.mean(get_error(actual[start:], prediction))
    ae = np.mean(get_absolute_error(actual[start:], prediction))
    ape = np.mean(get_ape(actual[start:], prediction))
    sq = np.mean(get_squared_error(actual[start:], prediction))
    df = df.append({'Method': method, 'ME': error, 'MAE': ae, 'MAPE': ape, 'MSE': sq}, ignore_index=True)
    return df


def create_error_table(data, non_dummy, season_size, min_x, max_x, min_y, max_y, start=0):
    df = pd.DataFrame(columns=['Method', 'ME', 'MAE', 'MAPE', 'MSE'])

    rw_s_nd = random_walk_seasonal(False, non_dummy, season_size, start)
    df = add_row_to_df(non_dummy, rw_s_nd, 'rw_without_drift', df, season_size, start)

    rw_s_d = random_walk_seasonal(True, non_dummy, season_size, start)
    df = add_row_to_df(non_dummy, rw_s_d, 'rw_with_drift', df, season_size, start)

    rsg = make_forecast(data, season_size, start)
    df = add_row_to_df(non_dummy[1:], rsg, 'seasonal_regression', df, season_size, start)

    holt_fcast_add = holt_package(non_dummy, type_model='add', start=start, season_size=12)
    holt_fcast_mul = holt_package(non_dummy, type_model='mul', start=start, season_size=12)
    df = add_row_to_df(non_dummy, holt_fcast_add, 'holt_winters_add', df, season_size, start)
    df = add_row_to_df(non_dummy, holt_fcast_mul, 'holt_winters_mul', df, season_size, start)
    predictions = [rw_s_nd, rw_s_d, holt_fcast_add, holt_fcast_mul]
    plot_names = ["rw_s_nd", "rw_s_d", "holt_add", "holt_mul"]
    for i in range(4):
        standard_line_plot([range(0,non_dummy.shape[0]), range(start,non_dummy.shape[0])], [non_dummy, predictions[i]], ["Blue", "Red"], ["Actual", "Prediction"], [min_y, max_y], [min_x, max_x], "Umbrella Sales", legend=True, name=plot_names[i], show=False)
    standard_line_plot([range(0,non_dummy.shape[0]), range(start+1,non_dummy.shape[0])], [non_dummy, rsg], ["Blue", "Red"], ["Actual", "Prediction"], [min_y, max_y], [min_x, max_x], "Umbrella Sales", legend=True, name="rsg", show=False)


    # plot_predictions_seasonal(non_dummy, rsg, start+2, "Seasonal Regression Prediction vs Actual")
    # plot_predictions_seasonal(non_dummy, holt_fcast_add, start, "Additive Holt Winter Forecast vs Actual")
    # plot_predictions_seasonal(non_dummy, holt_fcast_mul, start,  "Multiplicative Holt Winter Forecast vs Actual")
    # plot_predictions_seasonal(non_dummy, rw_s_d, start, "Seasonal Random Walk With Drift vs Actual")
    # plot_predictions_seasonal(non_dummy, rw_s_nd, start, "Seasonal Random Walk Without Drift vs Actual")

    print(df.to_latex())

def create_error_table_os(data, non_dummy, season_size, min_x, max_x, min_y, max_y, start=0):
    pass


def holt_package(df, type_model,season_size, start):
    print(start)
    fit = ExponentialSmoothing(df, trend=type_model, seasonal=type_model, seasonal_periods=season_size,
                               initialization_method='heuristic').fit()
    fcast = fit.predict(start=start, end=df.shape[0]-1)
    return fcast

def random_walk_out_sample(drift, df, season_size, start):
    if drift:
        data_for_pred = []
        seasonal_difference = []
        predictions = []
        for i in range(season_size, df.shape[0]-start):
            seasonal_difference.append(df[i] - df[i - season_size])
        drift_value = np.cumsum(seasonal_difference) / range(1, df.shape[0] - season_size + 1-start)

        for z in range(start-12, start):
            data_for_pred.append(df[z])

        for year in range(int((df.shape[0]-start)/12)):
            for month in range(0,season_size):
                predictions.append(data_for_pred[month] + drift_value[-1])
        return predictions
    else:
        predictions = []
        data_for_pred = []
        for z in range(start-12, start):
            data_for_pred.append(df[z])

        for year in range(int((df.shape[0]-start)/12)):
            for month in range(0,season_size):
                predictions.append(data_for_pred[month])
        return predictions

def running_regression_out_sample(df, start, ):
    predictions = []
    params = running_regression_seasonal(df[:start])

    for i in range(start, df.shape[0] - 1):
        prediction = params @ df.iloc[i + 1][1:]
        predictions.append(prediction)
    return predictions

    pass

def holt_winters_out_sample(df, type_model, season_size, start):
    print(start)
    fit = ExponentialSmoothing(df[:start], trend=type_model, seasonal=type_model, seasonal_periods=season_size,
                               initialization_method='heuristic').fit()
    fcast = fit.forecast(steps=24)
    return fcast

if __name__ == '__main__':
    gas_data = read_csv("data/gasoline.csv")
    umbrella = np.array(pd.read_excel("data/Umbrella.xlsx")['Umbrella Sales'])
    umbrella_season = pd.read_excel("data/Umbrella.xlsx")[['Umbrella Sales', 'SeasIdx']]
    dummy_umbrella = pd.get_dummies(umbrella_season, columns=['SeasIdx'], drop_first=True)
    dummy_umbrella['t'] = range(1, dummy_umbrella.shape[0] + 1)
    dummy_umbrella['c'] = np.ones(dummy_umbrella.shape[0])

    df = pd.read_csv('data/time_series_champie.csv').dropna()
    seasons = np.array(df['Month'])
    seasons_monthly = []
    for i in range(0,seasons.shape[0]):
        seasons_monthly.append(str(seasons[i]).split('-')[1])
    seasonal_month = df
    seasonal_month['Month'] = seasons_monthly
    dummy_oil = pd.get_dummies(seasonal_month, columns=['Month'], drop_first=True)
    dummy_oil['t'] = range(1, dummy_oil.shape[0] + 1)
    dummy_oil['c'] = np.ones(dummy_oil.shape[0])
    data = np.array(df['Sales Millions'])

    #create_error_table(dummy_oil, data, 12, 0, 105, 1000, 15000, dummy_oil.shape[0]-24)
    create_error_table(dummy_umbrella, umbrella, 4, 0, 20, 50, 200, 4)

    #create_error_table_os(data, dummy_oil, 12, 0, 105, 1000, 15000, dummy_oil.shape[0]-24)
    error_df = pd.DataFrame(columns=['Method', 'ME', 'MAE', 'MAPE', 'MSE'])

    rw_nd = random_walk_out_sample(False, data, 12, dummy_oil.shape[0]-24)
    rw_d = random_walk_out_sample(True, data, 12, dummy_oil.shape[0]-24)
    reg_os = running_regression_out_sample(dummy_oil, dummy_oil.shape[0]-24)
    standard_line_plot([range(0, data.shape[0]), range(dummy_oil.shape[0]-24, data.shape[0])], [data, rw_nd],
                       ["Blue", "Red"], ["Actual", "Prediction"], [1000, 15000], [0, 105], "Champagne Sales",
                       legend=True, name="rw_nd_os", show=False)
    standard_line_plot([range(0, data.shape[0]), range(dummy_oil.shape[0] - 24, data.shape[0])], [data, rw_d],
                       ["Blue", "Red"], ["Actual", "Prediction"], [1000, 15000], [0, 105], "Champagne Sales",
                       legend=True, name="rw_d_os", show=False)
    standard_line_plot([range(0, data.shape[0]), range(dummy_oil.shape[0] - 24+1, data.shape[0])], [data, reg_os],
                       ["Blue", "Red"], ["Actual", "Prediction"], [1000, 15000], [0, 105], "Champagne Sales",
                       legend=True, name="reg_os", show=False)
    holt_fcast_add = holt_winters_out_sample(data, type_model='add', start=dummy_oil.shape[0] - 24, season_size=12)
    holt_fcast_mul = holt_winters_out_sample(data, type_model='mul', start=dummy_oil.shape[0] - 24, season_size=12)
    standard_line_plot([range(0, data.shape[0]), range(dummy_oil.shape[0] - 24, data.shape[0])], [data, holt_fcast_add],
                       ["Blue", "Red"], ["Actual", "Prediction"], [1000, 15000], [0, 105], "Champagne Sales",
                       legend=True, name="hwa_os", show=False)
    standard_line_plot([range(0, data.shape[0]), range(dummy_oil.shape[0] - 24, data.shape[0])], [data, holt_fcast_mul],
                       ["Blue", "Red"], ["Actual", "Prediction"], [1000, 15000], [0, 105], "Champagne Sales",
                       legend=True, name="hwm_os", show=False)
    error_df = add_row_to_df(data, rw_nd, 'rw_without_drift', error_df, 12, data.shape[0]-24)
    error_df = add_row_to_df(data, rw_d, 'rw_with_drift', error_df, 12, data.shape[0]-24)
    error_df = add_row_to_df(data[1:], reg_os, 'seasonal_regression', error_df, 12, data.shape[0]-24)
    error_df = add_row_to_df(data, holt_fcast_add, 'holt_winters_add', error_df, 12, data.shape[0]-24)
    error_df = add_row_to_df(data, holt_fcast_mul, 'holt_winters_mul', error_df, 12, data.shape[0]-24)
    print(error_df.to_latex())
