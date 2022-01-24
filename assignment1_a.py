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
                            ['Mean Residual', 'Mean Abs. Residual', 'Mean Abs. Perc. Residual', 'Mean Sq. Residual'], last_n_forecasts=6)
 
# Table from slide 24   
get_table_comparing_methods(gas_data.GasolineSales, [running_avg_pred_gas_sales,
                                                     random_walk_pred_gas_sales,
                                                     expert_pred_gas_sales],
                            ['Running Avg.', 'Random Walk', 'Expert'], [get_error, get_absolute_error, get_ape, get_squared_error],
                            ['Mean Residual', 'Mean Abs. Residual', 'Mean Abs. Perc. Residual', 'Mean Sq. Residual'], last_n_forecasts=6)



    
df_scores = pd.DataFrame(np.nan, index= ['Running Avg.', 'Random Walk'], columns=['Residual', 'Abs. Residual'])
    