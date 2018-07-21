# -*- coding: utf-8 -*-
#
# Implementing an LSTM RNN Model
# ------------------------------
#  Here we implement an LSTM model on all a data set of Shakespeare works.
#
#
#

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
from keras.layers import LSTM, TimeDistributed
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.layers.core import Dropout
from keras.layers import Dense, Activation,Convolution2D,MaxPooling2D,Flatten,SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error


ops.reset_default_graph()


def predict(coef, history):
    yhat = 0.0
    for i in range(1, len(coef) + 1):
        yhat += coef[i - 1] * history[-i]
    return yhat


def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return numpy.array(diff)


def interactive_legend(ax=None):
    if ax is None:
        ax = pyplot.gca()
    if ax.legend_ is None:
        ax.legend()

    return InteractiveLegend(ax.legend_)

def serials_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, 1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    agg = concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace=True)
    return agg





class InteractiveLegend(object):
    def __init__(self, legend):
        self.legend = legend
        self.fig = legend.axes.figure

        self.lookup_artist, self.lookup_handle = self._build_lookups(legend)
        self._setup_connections()

        self.update()

    def _setup_connections(self):
        for artist in self.legend.texts + self.legend.legendHandles:
            artist.set_picker(10)  # 10 points tolerance

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def _build_lookups(self, legend):
        labels = [t.get_text() for t in legend.texts]
        handles = legend.legendHandles
        label2handle = dict(zip(labels, handles))
        handle2text = dict(zip(handles, legend.texts))

        lookup_artist = {}
        lookup_handle = {}
        for artist in legend.axes.get_children():
            if artist.get_label() in labels:
                handle = label2handle[artist.get_label()]
                lookup_handle[artist] = handle
                lookup_artist[handle] = artist
                lookup_artist[handle2text[handle]] = artist

        lookup_handle.update(zip(handles, handles))
        lookup_handle.update(zip(legend.texts, handles))

        return lookup_artist, lookup_handle

    def on_pick(self, event):
        handle = event.artist
        if handle in self.lookup_artist:
            artist = self.lookup_artist[handle]
            artist.set_visible(not artist.get_visible())
            self.update()

    def on_click(self, event):
        if event.button == 3:
            return
            visible = False
        elif event.button == 2:
            visible = True
        else:
            return

        for artist in self.lookup_artist.values():
            artist.set_visible(visible)
        self.update()

    def update(self):
        for artist in self.lookup_artist.values():
            handle = self.lookup_handle[artist]
            if artist.get_visible():
                handle.set_visible(True)
            else:
                handle.set_visible(False)
        self.fig.canvas.draw()

    def show(self):
        pyplot.show()

# Start a session
sess = tf.Session()


# #模型数据输入
# # fubiao_num,dt_show,yls,zd
# dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d %H:%M')
# series = read_csv('NBshuizhi2013040120180316_正常_NULL.csv', parse_dates=[0], index_col=0, usecols=[0, 1, 2, 3, 4, 6, 7, 8, 9], engine='python',
#                   date_parser=dateparse)
# shiftDiff = 10
# series.columns = ['fbnum', 'tmp', 'slt', 'o2', 'ph', 'chl', 'ntu', 'tbd']
# series.replace(-9999, numpy.NaN, inplace=True)
# series.dropna(inplace=True)
# fbnums = series['fbnum'].unique()
# fbnums = [value for value in fbnums if value is not numpy.NaN]
# series_frame = DataFrame(series)

plot = False
error_plot = False
evaluation_plot = False
prediction_plot = False
b_diff = False

time_steps_lag = 60
time_steps_seq = 8
resample_res = '12H'
batch_size = 10

# # 去重
# series_drop = series_frame.groupby(series_frame.index).first()
#
# resample1h = series_drop.resample('1H').mean()
#
# # 利用插值方法补齐缺失值
# series_time = resample1h.interpolate(method='time')
#
# # 绝对偏差
# series_time['shift1'] = abs(series_time['chl'] - series_time['chl'].shift(1))
# series_time['shift-1'] = abs(series_time['chl'] - series_time['chl'].shift(-1))
# series_time['shift_diff'] = abs(
#     series_time['chl'] - series_time['chl'].shift(1) + series_time['chl'] - series_time['chl'].shift(-1))
#
# series_time['data_process'] = series_time['chl']
# series_time['data_process'][
#     (series_time['shift_diff'] >= shiftDiff) & (series_time['shift1'] >= shiftDiff / 2) & (
#     series_time['shift-1'] >= shiftDiff / 2)] = numpy.NaN
# series_time.dropna(inplace=True)
# series_time1 = series_time.resample('1H').mean()
# series_time2 = series_time1.interpolate(method='time')
#
# #得到去重、补齐、重采样后的data
# del series_time2['shift1']
# del series_time2['shift-1']
# del series_time2['shift_diff']
# del series_time2['data_process']
# series_time2.to_csv('logs/results/NBshuizhi2013040120180316_1H_7Feature.csv')
dateparse1 = lambda dates: datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
series_new = read_csv('logs/results/NBshuizhi2013040120180316_12H_7Feature_Edited1.csv', parse_dates=[0], index_col=0, usecols=[0, 1, 2, 3, 4, 5], engine='python',
                  date_parser=dateparse1)
series_time1 = series_new.resample('12H').mean()
series_time2 = series_time1.interpolate(method='time')
# series_new = series_new.resample(resample_res).mean()
# series_time2.to_csv('logs/results/NBshuizhi2013040120180316_12H_7Feature_Edited1.csv')
# datavalues = series_time2.values
# groups = [0, 1, 2, 3, 4]
# i = 1
# pyplot.figure()
# for group in groups:
#     pyplot.subplot(len(groups),1,i)
#     pyplot.plot(datavalues[:,group])
#     pyplot.title(series_time2.columns[group], y=0.5, loc='right')
#     i += 1
# pyplot.show()

# # 粗采样，默认采用mean值
# resampledata = series_time2.resample(resample_res).mean()
#
# dateparse1 = lambda dates: datetime.strptime(dates, '%Y/%m/%d %H:%M')
# series_new = read_csv('logs/results/resampledataStanderd1.csv', parse_dates=[0], index_col=0, usecols=[0, 1, 2, 3, 4, 5, 6], engine='python',
#                   date_parser=dateparse1)
# series_new = series_new.resample(resample_res).mean()
# print(series_new.head(10))
# #输出各要素的plot
# resampledatavalues = series_new.values
# groups = [0,1,2,3,4,5]
# i = 1
# pyplot.figure()
# for group in groups:
#     pyplot.subplot(len(groups),1,i)
#     pyplot.plot(resampledatavalues[:,group])
#     pyplot.title(series_new.columns[group],y=0.5,loc='right')
#     i += 1
# pyplot.show()

scaler1 = MinMaxScaler(feature_range=(0, 1))
series_chl = series_time2.values[:, 3].reshape(-1, 1)
resampledataScalerd1 = scaler1.fit_transform(series_chl)


scaler = MinMaxScaler(feature_range=(0,1))
resampledataScalerd = scaler.fit_transform(series_time2.values)
resampledataScalerd = DataFrame(resampledataScalerd)
# resampledataScalerd.to_csv('logs/results/resampledataScalerd.csv')
vocab_size = len(resampledataScalerd)
resample_aray = np.array(resampledataScalerd)[0:2810]

BATCH_START = 0
TIME_STEPS = 5
BATCH_SIZE = 32     #经测试 batch size对最后的结果影响不是特别大，但是训练轮数的影响较大
INPUT_SIZE = 5
OUTPUT_SIZE = 1
CELL_SIZE = 400    #隐藏单元个数对最后的结果影响很大
LR = 0.000001
TARGET_INDEX = 3

def get_batch():
    global BATCH_START, TIME_STEPS
    #arange函数用于创建等差数组
    # xs shape (50batch, 20steps)
    # xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    # resample_aray = np.reshape(resample_aray, [-1, 2, 11])
    batch_xy = resample_aray[BATCH_START:BATCH_START+BATCH_SIZE*(TIME_STEPS+1)]
    batch_xy = np.reshape(batch_xy, [-1, TIME_STEPS+1, INPUT_SIZE])
    seq = batch_xy[:, :TIME_STEPS, :]
    # seq = np.reshape(seq,[-1,9,1])
    res = batch_xy[:, TIME_STEPS:TIME_STEPS+1, TARGET_INDEX]
    # res = np.reshape(res,[-1,1])
    # seq = resample_aray[0,BATCH_START:BATCH_START+TIME_STEPS]
    # res = resample_aray[0,BATCH_START+1:BATCH_START+TIME_STEPS+2]
    BATCH_START += 1
    return [seq, res]
    # return [seq[:, :, np.newaxis], res[:, :, np.newaxis]]
    # return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

def generateTrain(resample_aray, batch_start, batch_size, time_steps, target_index, input_size):
    X_train=[]
    Y_train=[]
    X_test=[]
    Y_test=[]
    for i in range(2100):    #当全部数据用来做训练集（包括验证集）的时候，发现验证集的loss比训练集大很多  改成600之后 两者差不多
        batch_xy = resample_aray[batch_start:batch_start + time_steps + 1]
        seq = batch_xy[:time_steps, :]
        # seq = batch_xy[:time_steps, target_index]
        # seq = np.reshape(seq, [time_steps, INPUT_SIZE])
        res = batch_xy[time_steps, target_index]
        X_train.append(seq)
        Y_train.append(res)
        batch_start += 1

    for i in range(2100, 2800):
        batch_xy = resample_aray[batch_start:batch_start + time_steps + 1]
        seq2 = batch_xy[:time_steps, :]
        # seq2 = batch_xy[:time_steps, target_index]
        # seq2 = np.reshape(seq2, [time_steps, INPUT_SIZE])
        res2 = batch_xy[time_steps, target_index]
        X_test.append(seq2)
        Y_test.append(res2)
        batch_start += 1

    return np.array(X_train, dtype=np.float32), np.array(Y_train, dtype=np.float32), np.array(X_test, dtype=np.float32), np.array(Y_test, dtype=np.float32)

X_train, Y_train, X_test, Y_test = generateTrain(resample_aray, BATCH_START, BATCH_SIZE, TIME_STEPS, TARGET_INDEX, INPUT_SIZE)
model = Sequential()
# build a LSTM RNN
model.add(LSTM(
    # input_shape=(TIME_STEPS,INPUT_SIZE),
    CELL_SIZE,
    input_shape=( TIME_STEPS, INPUT_SIZE),
    # batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    # output_dim=CELL_SIZE,
    activation='relu',          #用relu会出现loss为NAN的情况
    # dropout=0.2,
    #return_sequences=True,      # True: output at all steps. False: output as last step.
    # stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
))


# model.add(Dropout(0.2))
# add output layer
# model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
# model.add(LSTM(
#     output_dim=CELL_SIZE,
#     activation='relu',          #用relu会出现loss为NAN的情况
#     return_sequences=False,      # True: output at all steps. False: output as last step.
#     # stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
# ))

# model.add(Dense(64))
# model.add(Activation('linear'))
# model.add(Dense(32))
# model.add(Activation('linear'))
model.add(Dense(OUTPUT_SIZE))
# model.add(Activation('linear'))
optim = Adam(LR)
#optim = RMSprop(lr=LR)
model.compile(optimizer=optim,
              loss='mse',)

print('====================Training ======================')
#history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=60, validation_split=0.5)
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=6000, validation_split=0.2)
score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
print(score)

pyplot.plot(history.history['loss'], label='Train_loss')
pyplot.plot(history.history['val_loss'], label='Validation_loss')
pyplot.legend()
pyplot.show()


#做预测与label的对比
yhat = model.predict(X_test)
Y_test = scaler1.inverse_transform(Y_test.reshape(-1, 1))
yhat = scaler1.inverse_transform(yhat)
rmse_scaler = numpy.sqrt(mean_squared_error(Y_test, yhat))
print(rmse_scaler)
plt.plot(yhat, 'r', Y_test, 'b')
plt.show()
# loss,accuracy = model.evaluate(X_test,Y_test)
# print('loss', loss)
# print('accuracy', accuracy)

# x_array = np.arange(0,10,1)
# x_array = np.reshape(x_array,[-1,1])
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler_y = scaler.fit_transform(x_array)
# print(scaler_y)
# result = scaler.inverse_transform(scaler_y)
# print(result)
#
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)