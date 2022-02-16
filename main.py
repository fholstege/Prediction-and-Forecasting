import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings


def augmented_dicky_fuller(ts, d, regr_trend="c", criterion="aic"):
    while d > 0:
        ts = ts.diff().dropna()
        d = d - 1

    test_result = adfuller(ts, regression=regr_trend, autolag=criterion)
    if test_result[1] > .05:
        warnings.warn("The timeseries is not stationary, p-value: {}".format(test_result[1]))

    print(test_result[1])


df_ts = pd.read_excel("data/DataAssignment1.xlsx")
ts_1 = df_ts['Var1']
ts_2 = df_ts['Var2']
ts_3 = df_ts['Var3']
ts_4 = df_ts['Var4']
ts_5 = df_ts['Var5']
ts_6 = df_ts['Var6']
ts_7 = df_ts['Var7']
ts_8 = df_ts['Var8']
ts_9 = df_ts['Var9']

augmented_dicky_fuller(ts_1, 0)
augmented_dicky_fuller(ts_2, 1)
augmented_dicky_fuller(ts_3, 1)
augmented_dicky_fuller(ts_4, 0)
augmented_dicky_fuller(ts_5, 1)
augmented_dicky_fuller(ts_6, 1)
augmented_dicky_fuller(ts_7, 1)
augmented_dicky_fuller(ts_8, 1)
augmented_dicky_fuller(ts_9, 1)

