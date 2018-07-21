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

import pywt
import pywt.data

ecg = pywt.data.ecg()


