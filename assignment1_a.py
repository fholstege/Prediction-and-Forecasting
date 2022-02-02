from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from helpers_pf import * 
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

# read in gas sales data
gas_data = read_xlsx("Data/GasolineSales1.xlsx", header=None)
gas_data.columns = ['GasolineSales']


# Time units for gas sales
t_gas = list(range(1, len(gas_data.GasolineSales)+1))

# Figure of Gas sales, slide 6/89
standard_line_plot(t_gas, [gas_data.GasolineSales], ['red'], ['Gasoline Sales'], [0,25],[0,14], 'Gasoline Sales')
    
# get the average prediction    
avg_pred_gas_sales = average_prediction(gas_data.GasolineSales)
avg_pred_gas_sales_series = [avg_pred_gas_sales]*len(gas_data.GasolineSales)

# Figure of Gas sales and average prediction, slide 8/89
standard_line_plot(t_gas, [gas_data.GasolineSales,avg_pred_gas_sales_series ], ['red', 'blue'], ['Gasoline Sales', 'Avg. Pred'], [0,25],[0,14], 'Gasoline Sales')

# Figure of Gas sales and average prediction, with extended prediction slide 9/89
standard_line_plot(t_gas, [gas_data.GasolineSales,avg_pred_gas_sales_series ], ['red', 'blue'], ['Gasoline Sales', 'Avg. Pred'], [0,25],[0,14], 'Gasoline Sales', show = False)
plt.plot([12, 13, 14],[avg_pred_gas_sales]*3, color = 'blue', linestyle = 'dashed' )
plt.show()

# running average prediction gas sales
running_avg_pred_gas_sales_present = running_average(gas_data.GasolineSales, shift=0)
running_avg_pred_gas_sales = running_average(gas_data.GasolineSales, shift=1)

# Figure of Gas sales and running average prediction, slide 11/89
standard_line_plot(t_gas, [gas_data.GasolineSales,running_avg_pred_gas_sales_present ], ['red', 'blue'], ['Gasoline Sales', 'Running Avg. Pred'], [0,25],[0,14], 'Gasoline Sales')


# Figure of Gas sales and running average prediction (lagged), slide 15/89
standard_line_plot(t_gas, [gas_data.GasolineSales,running_avg_pred_gas_sales ], ['red', 'blue'], ['Gasoline Sales', 'Running Avg. Pred'], [15,24],[0,14], 'Gasoline Sales', show = True)


# Calculate the errors of running average prediction and gasoline sales
running_avg_errors_gas_sales = gas_data.GasolineSales - running_avg_pred_gas_sales

# Figure of residuals for running average prediction, slide 15/89
standard_residual_plot(t_gas, running_avg_errors_gas_sales, 'black')

# Table of t, Y_t (gasoline sales), running average forecast, and residuals, slide 16/89
## Check if need to add details
df_slide_16 = pd.DataFrame({'t': list(range(1, len(t_gas)+1)), 'Y': gas_data.GasolineSales, 'Forecast': running_avg_pred_gas_sales, 'Residual': running_avg_errors_gas_sales })
df_slide_16_final = df_slide_16[1:]

# Table of t,  Y_t (gasoline sales), running average forecast, and several types of residuals, slide 19/89
df_slide_19 = df_slide_16
df_slide_19['abs_residual'] = get_absolute_error(gas_data.GasolineSales, running_avg_pred_gas_sales.shift(1) )
df_slide_19['perc_residual'] = get_ape(gas_data.GasolineSales, running_avg_pred_gas_sales.shift(1) )
df_slide_19['squ_residual'] = get_squared_error(gas_data.GasolineSales, running_avg_pred_gas_sales.shift(1) )


# Predictions for gas sales
random_walk_pred_gas_sales = random_walk_forecast(gas_data.GasolineSales)
expert_pred_gas_sales = list(expert_forecast(gas_data.GasolineSales, 19))
expert_pred_gas_sales[:0] = [np.nan]
 
# Table from slide 23
get_table_comparing_methods(gas_data.GasolineSales, [running_avg_pred_gas_sales,
                                                     random_walk_pred_gas_sales,
                                                     expert_pred_gas_sales],
                            ['Running Avg.', 'Random Walk', 'Expert'], [get_error, get_absolute_error, get_ape, get_squared_error],
                            ['Mean Residual', 'Mean Abs. Residual', 'Mean Abs. Perc. Residual', 'Mean Sq. Residual'])
 
# Table from slide 24   
get_table_comparing_methods(gas_data.GasolineSales, [running_avg_pred_gas_sales,
                                                     random_walk_pred_gas_sales,
                                                     expert_pred_gas_sales],
                            ['Running Avg.', 'Random Walk', 'Expert'], [get_error, get_absolute_error, get_ape, get_squared_error],
                            ['Mean Residual', 'Mean Abs. Residual', 'Mean Abs. Perc. Residual', 'Mean Sq. Residual'], last_n_forecasts=6)




## Forecast weight plots
T = 50
M = 10
running_avg_weights = [1/T]*T
moving_avg_weights = [1/M]*M +[0]*(T-M)
rw_weights = [1]*1 + [0]*(T-1)

alpha_weights_005 = forecast_weights_exp(T, 0.05)
alpha_weights_01 = forecast_weights_exp(T, 0.1)
alpha_weights_02 = forecast_weights_exp(T, 0.2)
alpha_weights_05 = forecast_weights_exp(T, 0.5)

# create plots slide 37/89
weights_plot(T,running_avg_weights, 'Running Avg.', color = 'blue' )
weights_plot(T,moving_avg_weights, 'Moving Avg.' , color = 'blue')
weights_plot(T,rw_weights, 'Random Walk.' , color = 'blue')
weights_plot(T,alpha_weights_02, 'Exponential Smoothing', color= 'blue' )

# create plots slide 38/89
weights_plot(T,alpha_weights_005, 'alpha=0.05', color = 'blue' )
weights_plot(T,alpha_weights_01, 'alpha=0.1' , color = 'green')
weights_plot(T,alpha_weights_02, 'alpha=0.2', color= 'pink' )
weights_plot(T,alpha_weights_05, 'alpha=0.5', color = 'purple')

# create plots slide 39/89
# TODO: change format
weights_plot(T,alpha_weights_005, 'alpha=0.05', color = 'blue', scale=True )
weights_plot(T,alpha_weights_01, 'alpha=0.1' , color = 'blue', scale=True )
weights_plot(T,alpha_weights_02, 'alpha=0.2', color= 'blue' , scale=True )
weights_plot(T,alpha_weights_05, 'alpha=0.5', color = 'blue', scale=True )

# change memory indeces
# TODO: add plots
memory_index_a_005 = calc_memory_index(0.05)
memory_index_a_01 = calc_memory_index(0.1)
memory_index_a_02 = calc_memory_index(0.2)
memory_index_a_05 = calc_memory_index(0.5)


# predictions with exponential smoothing: alpha = 0.2 and alpha = 0.8
exp_pred_gas_sales_02 = exponential_smoothing(gas_data.GasolineSales, .2)
exp_pred_gas_sales_08 = exponential_smoothing(gas_data.GasolineSales, .8)

# get residuals of exponential smoothing predictions
exp_pred_gas_sales_02_errors  = gas_data.GasolineSales - exp_pred_gas_sales_02
exp_pred_gas_sales_08_errors  = gas_data.GasolineSales - exp_pred_gas_sales_08

# figure from slide 42
standard_line_plot(t_gas, [gas_data.GasolineSales,exp_pred_gas_sales_02 ], ['red', 'blue'], ['Gasoline Sales', 'exp. smoothing, a=0.2'], [0,25],[0,14], 'Gasoline Sales')
standard_residual_plot(t_gas, exp_pred_gas_sales_02_errors, 'black')

# figure from slide 43
standard_line_plot(t_gas, [gas_data.GasolineSales,exp_pred_gas_sales_08 ], ['red', 'blue'], ['Gasoline Sales', 'exp. smoothing, a=0.2'], [0,25],[0,14], 'Gasoline Sales')
standard_residual_plot(t_gas, exp_pred_gas_sales_08_errors, 'black')


# Table from slide 44
get_table_comparing_methods(gas_data.GasolineSales, [running_avg_pred_gas_sales,
                                                     random_walk_pred_gas_sales,
                                                     expert_pred_gas_sales,  
                                                     exp_pred_gas_sales_02,
                                                     exp_pred_gas_sales_08],
                            ['Running Avg.', 'Random Walk', 'Expert', 'Exp. Smoothing (a=0.2)', 'Exp. Smoothing (a=0.8)'], [get_error, get_absolute_error, get_ape, get_squared_error],
                            ['Mean Residual', 'Mean Abs. Residual', 'Mean Abs. Perc. Residual', 'Mean Sq. Residual'], last_n_forecasts=6)


# estimate the optimal alpha (TODO: revise)
gas_train = gas_data.GasolineSales[:6]


# use exponential smoothing with estimated alpha
exp_pred_gas_sales_est = exponential_smoothing_est(gas_data.GasolineSales, 6, 100)

# Table from slide 44
get_table_comparing_methods(gas_data.GasolineSales, [running_avg_pred_gas_sales,
                                                     random_walk_pred_gas_sales,
                                                     expert_pred_gas_sales,  
                                                     exp_pred_gas_sales_02,
                                                     exp_pred_gas_sales_08,
                                                     exp_pred_gas_sales_est],
                            ['Running Avg.', 'Random Walk', 'Expert', 'Exp. Smoothing (a=0.2)', 'Exp. Smoothing (a=0.8)', 'Exp. Smoothing (est)'], [get_error, get_absolute_error, get_ape, get_squared_error],
                            ['Mean Residual', 'Mean Abs. Residual', 'Mean Abs. Perc. Residual', 'Mean Sq. Residual'], last_n_forecasts=6)


# Get the new gas sales data
gas_data_ext = read_xlsx("Data/GasolineSales2.xlsx", header=None)
gas_data_ext.columns = ['GasolineSales']
t_gas_ext = list(range(1, len(gas_data_ext.GasolineSales)+1))

# Predictions for gas sales  - extended
random_walk_pred_gas_sales_ext = random_walk_forecast(gas_data_ext.GasolineSales) # RW
expert_pred_gas_sales_ext = list(expert_forecast(gas_data_ext.GasolineSales, 19)) # Expert
expert_pred_gas_sales_ext[:0] = [np.nan]

running_running_avg_pred_gas_sales_ext = running_average(gas_data_ext.GasolineSales, shift=1) # Running avg.
exp_pred_gas_sales_02_ext = exponential_smoothing(gas_data_ext.GasolineSales, .2) # exp smooth with alpha = 0.2
exp_pred_gas_sales_08_ext = exponential_smoothing(gas_data_ext.GasolineSales, .8) # exp smooth with alpha = 0.8

exp_pred_gas_sales_est_ext = exponential_smoothing_est(gas_data_ext.GasolineSales, 6, 100) # est. alpha
ar_1_gas_sales_ext = ar_1_model_pred(gas_data_ext.GasolineSales, 6) # AR(1)


# all from slide 51/89
standard_line_plot(t_gas_ext, [gas_data_ext.GasolineSales,expert_pred_gas_sales_ext ], ['red', 'blue'], ['Gasoline Sales', 'Expert'], [10,40],[0,23], 'Gasoline Sales', legend=True)
standard_line_plot(t_gas_ext, [gas_data_ext.GasolineSales,running_running_avg_pred_gas_sales_ext ], ['red', 'blue'], ['Gasoline Sales', 'Running Avg.'], [10,40],[0,23], 'Gasoline Sales', legend=True)
standard_line_plot(t_gas_ext, [gas_data_ext.GasolineSales,random_walk_pred_gas_sales_ext ], ['red', 'blue'], ['Gasoline Sales', 'Random Walk'], [10,40],[0,23], 'Gasoline Sales', legend=True)
standard_line_plot(t_gas_ext, [gas_data_ext.GasolineSales,exp_pred_gas_sales_02_ext ], ['red', 'blue'], ['Gasoline Sales', 'Exp. Smoothing (a=0.2)'], [10,40],[0,23], 'Gasoline Sales', legend=True)
standard_line_plot(t_gas_ext, [gas_data_ext.GasolineSales,exp_pred_gas_sales_est_ext ], ['red', 'blue'], ['Gasoline Sales', 'Exp. Smoothing (a=est)'], [10,40],[0,23], 'Gasoline Sales', legend=True)
standard_line_plot(t_gas_ext, [gas_data_ext.GasolineSales,ar_1_gas_sales_ext ], ['red', 'blue'], ['Gasoline Sales', 'AR(1)'], [10,40],[0,23], 'Gasoline Sales', legend=True)

# Table from slide 52
get_table_comparing_methods(gas_data_ext.GasolineSales, [running_running_avg_pred_gas_sales_ext,
                                                     random_walk_pred_gas_sales_ext,
                                                     expert_pred_gas_sales_ext,  
                                                     exp_pred_gas_sales_02_ext,
                                                     exp_pred_gas_sales_08_ext,
                                                     exp_pred_gas_sales_est_ext],
                            ['Running Avg.', 'Random Walk', 'Expert', 'Exp. Smoothing (a=0.2)', 'Exp. Smoothing (a=0.8)', 'Exp. Smoothing (est)'], [get_error, get_absolute_error, get_ape, get_squared_error],
                            ['Mean Residual', 'Mean Abs. Residual', 'Mean Abs. Perc. Residual', 'Mean Sq. Residual'], last_n_forecasts=16)

# load in bike sales data
bike = pd.read_excel('Data/BicycleSales.xlsx',header = None)
bike.columns = ['sales']
t_bike = bike.index + 1

# running trend of bike sales
running_trend_bike_sales, a_bike, b_bike = running_trend(bike.sales, t_bike, return_param=True)

# Running average of bike sales
running_avg_bike_sales = running_average(bike.sales, shift=1)

# plot 54/89
standard_line_plot(t_bike,[bike.sales ] ,['red', 'blue'], ['Bike Sales', 'Running Avg.'], [15,35],[0,12], 'Bike Sales')

# plot 55/89
standard_line_plot(t_bike,[bike.sales,running_avg_bike_sales ] ,['red', 'blue'], ['Bike Sales', 'Running Avg.'], [15,35],[0,12], 'Bike Sales')

# plot 61/89
standard_line_plot(t_bike,[bike.sales,running_trend_bike_sales ] ,['red', 'blue'], ['Bike Sales', 'Running trend'], [15,35],[0,12], 'Bike Sales')

# plot 62/89
# TODO: color, latex
param_plot(t_bike, [a_bike, b_bike], ['darkgreen', 'purple'], ['a', 'b'], width = 0.3)

# random walk - bicycle sales
rw_bicycle = random_walk_forecast(bike.sales)

# random walk with drift- bicycle sales
rw_with_drift_bicycle, c = random_walk_drift(bike.sales, t_bike, return_param=True)

# Parameter plot: #TODO latex
param_plot(t_bike, [c], ['green'], 'c')


# slide 65/89
standard_line_plot(t_bike,[bike.sales,rw_bicycle ] ,['red', 'blue'], ['Bike Sales', 'RW'], [18,35],[0,12], 'Bike Sales')
standard_line_plot(t_bike,[bike.sales,rw_with_drift_bicycle ] ,['red', 'blue'], ['Bike Sales', 'RW with drift'], [18,35],[0,12], 'Bike Sales')


# Table from slide 66 TODO: check
get_table_comparing_methods(bike.sales, [running_avg_bike_sales,
                                         running_trend_bike_sales,
                                         rw_bicycle,
                                         rw_with_drift_bicycle],
                            ['Running Avg.', 'Running trend',  'Random Walk', 'Random Walk with Drift'], [get_error, get_absolute_error, get_ape, get_squared_error],
                            ['Mean Residual', 'Mean Abs. Residual', 'Mean Abs. Perc. Residual', 'Mean Sq. Residual'],last_n_forecasts=8)




# own
hw_bicycle_sales_a_02_b_01 = holt_winters_package(bike.sales, 0.2, 0.1)
hw_bicycle_sales_a_02_b_03 = holt_winters_package(bike.sales, 0.2, 0.3)
hw_bicycle_sales_est, alpha, beta = holt_winters_est(bike.sales, train_i = 2, n_values_gridsearch=100, return_param=True)

param_plot(t_bike, [alpha], ['green'],['HW_alpha'], ylim = [0,0.4])
param_plot(t_bike, [beta], ['purple'],['HW_beta'])


# plots from 71/89
standard_line_plot(t_bike,[bike.sales,hw_bicycle_sales_a_02_b_01 ] ,['red', 'blue'], ['Bike Sales', 'Holt winters'], [15,35],[0,12], 'Bike Sales')
standard_line_plot(t_bike,[bike.sales,hw_bicycle_sales_a_02_b_03 ] ,['red', 'blue'], ['Bike Sales', 'Holt winters'], [15,35],[0,12], 'Bike Sales')
standard_line_plot(t_bike,[bike.sales,hw_bicycle_sales_est ] ,['red', 'blue'], ['Bike Sales', 'Holt winters'], [15,35],[0,12], 'Bike Sales')



# exponential smoothing
exp_pred_bicycle_02 = exponential_smoothing(bike.sales, 0.2)
exp_pred_bicycle_08 = exponential_smoothing(bike.sales, 0.8)
exp_pred_bicycle_est = exponential_smoothing_est(bike.sales, 2)


# Table from slide 72/89
get_table_comparing_methods(bike.sales, [exp_pred_bicycle_02,
                                         exp_pred_bicycle_08,
                                         exp_pred_bicycle_est,
                                         hw_bicycle_sales_a_02_b_01,
                                         hw_bicycle_sales_a_02_b_03,
                                         hw_bicycle_sales_est
                                         ],
                            ['Exp. Smoothing (a=0.2)', 'Exp. Smoothing (a=0.8)', 'Exp. Smoothing (est)',  'HW (a =0.2, b=0.1)', 'HW (a =0.2, b=0.3)', 'HW (est.)'], 
                            [get_error, get_absolute_error, get_ape, get_squared_error],
                            ['Mean Residual', 'Mean Abs. Residual', 'Mean Abs. Perc. Residual', 'Mean Sq. Residual'], last_n_forecasts=8)


# Get the umbrella sales data 
umbrella = read_xlsx("Data/Umbrella.xlsx")
umbrella.columns = ['year', 'season', 'sales']
t_umbrella = list(range(1, len(umbrella.sales)+1))



Y = umbrella.sales
seasons = 4
init_level = np.sum(Y[:seasons]) / seasons
init_growth = (np.sum(Y[seasons:(seasons + seasons)]) / seasons - (np.sum(Y[:seasons]) / seasons)) / seasons
init_season = []
for j in range(seasons):
    init_season.append(Y[j] / np.sum(Y[:seasons]) / seasons)


fit = ExponentialSmoothing(Y, trend='additive', seasonal='additive', seasonal_periods=seasons,
                           initialization_method='estimated').fit()


from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(Y, model='additive', period = 4)


standard_line_plot(t_umbrella,[Y, fit.level ] ,['green','red'], ['sales','level'], [0,180],[0,20], 'Umbrella Sales', ticks = True)
standard_line_plot(t_umbrella,[ fit.seasonal ] ,['blue'], ['level'], [-30,30],[0,20], 'Umbrella Sales', ticks = True)
standard_residual_plot(t_umbrella, fit.resid, 'black')
                    


#, initial_level=init_level, initial_trend=init_growth,
                           #initial_seasonal=init_season).fit(0.2, 0.2, 0.2, optimized=False)
end_fcast = 28
start_fcast = 20

fcast = fit.predict(start=start_fcast, end=end_fcast).shift(1)
t_plot = list(range(1, end_fcast+2))
Y_plot = list(Y.values) + [np.nan]*(end_fcast - len(Y)+1)
fcast_plot = [np.nan]*(start_fcast) + list(fcast.values)

standard_line_plot(t_plot,[Y_plot,fcast_plot ] ,['green', 'blue'], ['Umbrella Sales', 'fcast'], [0,180],[0,30], 'Umbrella Sales', ticks=True)
