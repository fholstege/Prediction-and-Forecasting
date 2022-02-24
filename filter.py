import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def kalman_filter(y, training_observations, out_of_sample=True, sigma_epsilon=30, sigma_eta=1.5):
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
    confidence_interval_upper = [estimated_a[-1] + 1.96*(p -sigma_eta + t * sigma_eta + sigma_epsilon)
                                 /np.sqrt(t+training_observations) for t, p in enumerate(out_sample_p[1:])]
    confidence_interval_lower = [estimated_a[-1] - 1.96*(p -sigma_eta + t * sigma_eta + sigma_epsilon)
                                 /np.sqrt(t+training_observations) for t, p in enumerate(out_sample_p[1:])]

    return np.array(estimated_a[1:]), np.array(estimated_p[1:]), np.full((len(out_sample_p[1:])), estimated_a[-1]), \
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


if __name__ == '__main__':
    ts = pd.read_excel("Data/GasolineSales2.xlsx", header=None)
    ts_non_response = np.array(ts)

    est_a_ruben, est_p, est_out_p, con_interval_up, con_interval_down = kalman_filter(ts_non_response, 22, sigma_epsilon=3.33, sigma_eta = 10)
    
    h=3
    forecast = h*[est_a_ruben[-1][0]]
    
    plt.scatter(range(1, 23), ts_non_response, color='red')
    plt.plot(range(23,25+1), forecast, color='blue')
    
    
    # print(est_a)
    # print(est_p)
    # print(est_out_p)
    # print(confidence_interval)
    standard_line_plot([range(1, 23), range(1,23)],
                       [ts_non_response, est_a],
                       ['red', 'blue', 'black', 'purple', 'purple'],
                       ['estimation', 'actual', 'out_sample', 'confidence_interval_up', 'confidence_interval_down'],
                       (15, 40), (0, 22), "test")
