from pandas import DataFrame

import os
import re
import string
import requests
import numpy as np
import collections
import random
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas import concat
from pandas import DataFrame

import numpy
import time
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed,Bidirectional,GRU
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.layers.core import Dropout
from keras.layers import Dense, Activation,Convolution2D,MaxPooling2D,Flatten,SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error




#模型数据输入
# fubiao_num,dt_show,yls,zd
dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d %H:%M')
series = read_csv('NBshuizhi2013040120180316_正常_NULL.csv', parse_dates=[0], index_col=0, usecols=[0, 1, 2, 3, 4, 6, 7, 8, 9], engine='python',
                  date_parser=dateparse)
shiftDiff = 10
series.columns = ['fbnum', 'tmp', 'slt', 'o2', 'ph', 'chl', 'ntu', 'tbd']
series.replace(-9999, numpy.NaN, inplace=True)
series.dropna(inplace=True)
fbnums = series['fbnum'].unique()
fbnums = [value for value in fbnums if value is not numpy.NaN]
series_frame = DataFrame(series)
resample_res = '12H'

# 去重
series_drop = series_frame.groupby(series_frame.index).first()

resample1h = series_drop.resample('1H').mean()

# 利用插值方法补齐缺失值
series_time = resample1h.interpolate(method='time')
series_time.to_csv('logs/results/NBshuizhi2013040120180316_1H.csv')
# 绝对偏差
series_time['shift1'] = abs(series_time['chl'] - series_time['chl'].shift(1))
series_time['shift-1'] = abs(series_time['chl'] - series_time['chl'].shift(-1))
series_time['shift_diff'] = abs(
    series_time['chl'] - series_time['chl'].shift(1) + series_time['chl'] - series_time['chl'].shift(-1))

series_time['data_process'] = series_time['chl']
series_time['data_process'][
    (series_time['shift_diff'] >= shiftDiff) & (series_time['shift1'] >= shiftDiff / 2) & (
    series_time['shift-1'] >= shiftDiff / 2)] = numpy.NaN
series_time.dropna(inplace=True)
series_time1 = series_time.resample('1H').mean()
series_time2 = series_time1.interpolate(method='time')

#得到去重、补齐、重采样后的data
del series_time2['shift1']
del series_time2['shift-1']
del series_time2['shift_diff']
del series_time2['data_process']
series_time2.to_csv('logs/results/NBshuizhi2013040120180316_1H_7Feature.csv')
