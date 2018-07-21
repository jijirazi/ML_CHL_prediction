# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from pandas import datetime
from pandas import datetime
from pandas import DataFrame
from pandas import read_csv
import statsmodels.api as sm
from statsmodels.tsa.stattools import  pacf
from statsmodels.tsa.stattools import acf
import statsmodels.tsa.stattools as st
from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_squared_error

def proper_model(data_ts, maxLag):
    init_bic = float("inf")
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARMA(data_ts, order=(p, q))
            try:
                result_ARMA = model.fit(disp=-1, method='css')
            except:
                continue
            bic = result_ARMA.bic
            if bic<init_bic:
                init_p = p
                init_q = q
                init_properModel = result_ARMA
                init_bic = bic
    return init_bic, init_p, init_q, init_properModel

#差分序列还原
def predict_recover(ts):
    ts = np.exp(ts)
    return ts

#数据读取
dateparse1 = lambda dates: datetime.strptime(dates, '%Y-%m-%d')
series_new = read_csv('ZS01-溶解氧.csv', parse_dates=[0], index_col=0, usecols=[0, 1], engine='python',
                  date_parser=dateparse1)

train_data = series_new.values[:750]
test_data = series_new.values[750:976]
# order = st.arma_order_select_ic(diff1, max_ar=50, max_ma=50, ic=['aic', 'bic', 'hqic'])
# model = ARMA(diff1, order.bic_min_order)
init_bic, init_p, init_q, init_properModel = proper_model(train_data, 5)
# result_arma = model.fit(disp=-1, method='css')

train_predict = init_properModel.predict()
length = len(train_predict)
length1 = len(train_data)
RMSE = np.sqrt(mean_squared_error(train_data[init_p:], train_predict))
print(RMSE)
plt.plot(train_predict, 'r', train_data[init_p:], 'b')
plt.show()

# 循环建模单步预测，效率太低
test_data = test_data.reshape(len(test_data))
train_data = train_data.reshape(len(train_data))
history = [x for x in train_data]
predictions = list()
# for t in range(len(test_data)):
for t in range(1):
    # output = init_properModel.forecast()
    init_bic, init_p, init_q, init_properModel = proper_model(history, 5)
    # model = ARMA(history, order=(init_p, init_q))

    # output = model_fit.forecast()
    output = init_properModel.forecast(steps=226)
    # yhat = output
    # predictions.append(yhat)
    # obs = np.array(test_data)[t]
    # history.append(obs)
    # print('predicted=%f, expected=%f' % (yhat, obs))
plt.plot(output[0], 'r', test_data, 'b')
plt.show()
# rmse = np.sqrt(mean_squared_error(predictions, train_data))
# print(rmse)
# # 查看时序是否平稳
# series_new.plot(figsize=(12, 8))

# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(111)
# diff1 = series_new.diff(1)
# diff1.plot(ax=ax1)
# plt.show()

# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(series_new, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(series_new, ax=ax2)
# plt.show()




# series_new = series_new[:2100]
# series_new = DataFrame(series_new)
#自相关图 偏自相关图
# fig = plt.figure(1, figsize=[12, 4])
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)
# data = series_new.values
# autocorr = acf(data)
# pac = pacf(data)
#
# x = [x for x in range(len(pac))]
# ax1.plot(x[1:], autocorr[1:])
# ax1.grid(True)
# ax1.set_xlabel('LAG')
# ax1.set_ylabel('Autocorrelation')
#
# ax2.plot(x[1:], pac[1:])
# ax2.grid(True)
# ax2.set_xlabel('LAG')
# ax2.set_ylabel('Partial Autocorrelation')
#
# plt.show()
# series_new_test = series_new[0:2100]
# #建模
# arima = ARIMA(series_new, order=[4, 1, 0])
# result = arima.fit(disp=None)
# pred = result.predict(start='2017-01-14 00:00:00', end='2017-12-29 12:00:00', typ='levels')
# # pred = result.predict(typ='levels')
# x = range(len(pred))
# # lens = len(x)
# # prs = len(pred)
# # df = len(series_new)
# plt.figure(figsize=(10, 5))
# plt.plot(x, series_new[2100:2800], label='data')
# # plt.plot(x, series_new[1:], label='data')
# plt.plot(x, pred, label='ARIMA')
# plt.xlabel('Times')
# plt.ylabel('views')
# plt.show()












