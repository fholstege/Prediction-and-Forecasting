from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from helpers_pf import * 



def gasoline_12(df):
    avg_pred = average_prediction(df)
    running_avg = running_average(df)
    error = get_error(df[1:], running_avg[:-1])
    ae = get_absolute_error(df[1:], running_avg[:-1])
    ape = get_ape(df[1:], running_avg[:-1])
    sq = get_squared_error(df[1:], running_avg[:-1])
    rw_forecast = random_walk_forecast(df)
    exp_forecast = expert_forecast(df, 19)
    exp_2 = exponential_smoothing(df, .2)
    mae_ex2 = np.mean(get_absolute_error(df[6:], exp_2[5:]))
    test = exponential_smoothing_est(df)
    

def plot_predictions(df, pred):
    plt.plot(range(1, df.shape[0]), pred)
    plt.plot(range(0, df.shape[0]), df)
    plt.show()



gas_data = read_xlsx("Data/GasolineSales1.xlsx", header=None)
gas_data.columns = ['GasolineSales']




gasoline_12(gas_data.GasolineSales)


t_gas = list(range(1, len(gas_data.GasolineSales)+1))


standard_line_plot(t_gas, [gas_data.GasolineSales], ['red'], ['Gasoline Sales'], [0,25], 'Gasoline Sales')
    
    
