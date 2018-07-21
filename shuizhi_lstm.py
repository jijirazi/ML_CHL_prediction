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
import xlwt

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

# Set RNN Parameters
min_word_freq = 5  # Trim the less frequent words off
rnn_size = 12  # RNN Model size
embedding_size = 10  # Word embedding size
epochs = 10  # Number of epochs to cycle through data
batch_size = 10  # Train on this many examples at once
learning_rate = 0.001  # Learning rate
training_seq_len = 20  # how long of a word group to consider
embedding_size = rnn_size
save_every = 5  # How often to save model checkpoints
eval_every = 5  # How often to evaluate the test sentences
prime_texts = ['thou art more', 'to be or not to', 'wherefore art thou']

# # Download/store Shakespeare data
# data_dir = 'temp'
# data_file = 'shakespeare.txt'
# model_path = 'shakespeare_model'
# full_model_dir = os.path.join(data_dir, model_path)
#
# # Declare punctuation to remove, everything except hyphens and apostrophes
# punctuation = string.punctuation
# punctuation = ''.join([x for x in punctuation if x not in ['-', "'"]])
#
# # Make Model Directory
# if not os.path.exists(full_model_dir):
#     os.makedirs(full_model_dir)
#
# # Make data directory
# if not os.path.exists(data_dir):
#     os.makedirs(data_dir)
#
# print('Loading Shakespeare Data')
# # Check if file is downloaded.
# if not os.path.isfile(os.path.join(data_dir, data_file)):
#     print('Not found, downloading Shakespeare texts from www.gutenberg.org')
#     shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
#     # Get Shakespeare text
#     response = requests.get(shakespeare_url)
#     shakespeare_file = response.content
#     # Decode binary into string
#     s_text = shakespeare_file.decode('utf-8')
#     # Drop first few descriptive paragraphs.
#     s_text = s_text[7675:]
#     # Remove newlines
#     s_text = s_text.replace('\r\n', '')
#     s_text = s_text.replace('\n', '')
#
#     # Write to file
#     with open(os.path.join(data_dir, data_file), 'w') as out_conn:
#         out_conn.write(s_text)
# else:
#     # If file has been saved, load from that file
#     with open(os.path.join(data_dir, data_file), 'r') as file_conn:
#         s_text = file_conn.read().replace('\n', '')
#
# # Clean text
# print('Cleaning Text')
# s_text = re.sub(r'[{}]'.format(punctuation), ' ', s_text)
# s_text = re.sub('\s+', ' ', s_text).strip().lower()
#
#
# # Build word vocabulary function
# def build_vocab(text, min_word_freq):
#     word_counts = collections.Counter(text.split(' '))
#     # limit word counts to those more frequent than cutoff
#     word_counts = {key: val for key, val in word_counts.items() if val > min_word_freq}
#     # Create vocab --> index mapping
#     words = word_counts.keys()
#     vocab_to_ix_dict = {key: (ix + 1) for ix, key in enumerate(words)}
#     # Add unknown key --> 0 index
#     vocab_to_ix_dict['unknown'] = 0
#     # Create index --> vocab mapping
#     ix_to_vocab_dict = {val: key for key, val in vocab_to_ix_dict.items()}
#
#     return (ix_to_vocab_dict, vocab_to_ix_dict)
#
#
# # Build Shakespeare vocabulary
# print('Building Shakespeare Vocab')
# ix2vocab, vocab2ix = build_vocab(s_text, min_word_freq)
# vocab_size = len(ix2vocab) + 1
# print('Vocabulary Length = {}'.format(vocab_size))
# # Sanity Check
# assert (len(ix2vocab) == len(vocab2ix))
#
# # Convert text to word vectors
# s_text_words = s_text.split(' ')
# s_text_ix = []
# for ix, x in enumerate(s_text_words):
#     try:
#         s_text_ix.append(vocab2ix[x])
#     except:
#         s_text_ix.append(0)
# s_text_ix = np.array(s_text_ix)

#模型数据输入
# fubiao_num,dt_show,yls,zd
dateparse = lambda dates: datetime.strptime(dates, '%Y/%m/%d %H:%M')
series = read_csv('NBshuizhi2013040120180316top10000.csv', parse_dates=[0], index_col=0, usecols=[0, 1, 2], engine='python',
                  date_parser=dateparse)
shiftDiff = 10
series.columns = ['fbnum', 'data']
series.replace(-9999, numpy.NaN, inplace=True)

fbnums = series['fbnum'].unique()
fbnums = [value for value in fbnums if value is not numpy.NaN]

timeStr = time.strftime('%Y%m%d', time.localtime(time.time()))
f_log = open("logs/logARIMA" + timeStr + ".txt", "a+")

plot = False
error_plot = False
evaluation_plot = False
prediction_plot = False
b_diff = False

time_steps_lag = 60
time_steps_seq = 8
resample_res = '12H'
batch_size = 10
p_values = [15, 25, 20, 6, 15]

# fbnums = ['WZ01', 'ZS01', 'WZ02']

fbnums = ['NB01']

for fbindex in range(len(fbnums)):
    fbnum = fbnums[fbindex]
    pvalue = p_values[fbindex]
    print('fbnum:' + fbnum + 'pvalue:' + str(pvalue))
    f_log.write('fbnum:' + fbnum + 'pvalue:' + str(pvalue) + '\n')

    series_data = series['data'][series['fbnum'] == fbnum]
    series_data = DataFrame(series_data)

    book = xlwt.Workbook()
    sheet = book.add_sheet('result', cell_overwrite_ok=True)
    sheet_indicator = book.add_sheet('indicator', cell_overwrite_ok=True)
    sheet.write(0, 0, 'datetime')
    sheet.write(0, 1, 'data')
    sheet.write(0, 2, 'train')
    sheet_indicator.write(0, 0, 'train')

    for xindex in range(time_steps_seq):
        sheet.write(0, 3 + xindex, 'val_' + str(xindex + 1))
        sheet.write(0, 3 + time_steps_seq + xindex, 'test_' + str(xindex + 1))
        sheet_indicator.write(0, 1 + xindex, 'val_' + str(xindex + 1))
        sheet_indicator.write(0, 1 + time_steps_seq + xindex, 'test_' + str(xindex + 1))

    # 去重
    series_drop = series_data.groupby(series_data.index).first()

    # 重采样缺失值，默认用NaN填充
    resample1h = series_drop.resample('1H').mean()
    # 利用插值方法补齐缺失值
    # series_linear = resample1h.interpolate(method='linear')
    series_time = resample1h.interpolate(method='time')
    # series_nearest = resample1h.interpolate(method='nearest')
    # series_spline = resample1h.interpolate(method='spline', order=2)

    # series_slinear = resample1h.interpolate(method='slinear')
    # series_quadratic = resample1h.interpolate(method='quadratic')
    # series_barycentric = resample1h.interpolate(method='barycentric')
    # series_polynomial = resample1h.interpolate(method='polynomial', order=2)

    # 绝对偏差
    series_time['shift1'] = abs(series_time['data'] - series_time['data'].shift(1))
    series_time['shift-1'] = abs(series_time['data'] - series_time['data'].shift(-1))
    series_time['shift_diff'] = abs(
        series_time['data'] - series_time['data'].shift(1) + series_time['data'] - series_time['data'].shift(-1))

    # 相对偏差
    # series_time['shift1_percent'] = abs(series_time['shift1'] / series_time['data'] * 100)
    # series_time['shift-1_percent'] = abs(series_time['shift-1'] / series_time['data'] * 100)
    # series_time['shift_diff_percent'] = abs(series_time['shift_diff'] / series_time['data'] * 100)

    series_time['data_process'] = series_time['data']
    series_time['data_process'][
        (series_time['shift_diff'] >= shiftDiff) & (series_time['shift1'] >= shiftDiff / 2) & (
        series_time['shift-1'] >= shiftDiff / 2)] = numpy.NaN

    series_time['data_process_interp'] = series_time['data_process'].interpolate(method='time')

    series_time.to_csv('logs/results/series_time.csv')

    # fig, ax = pyplot.subplots()
    # ax.plot(series_time['data'], '-', label=fbnum+'_data', ms='3')
    # ax.plot(series_time['data_process'], '-o', label='data_process', ms='3')
    # ax.plot(series_time['data_process_interp'], '-o', label='data_process_interp', ms='3')
    # interactive_legend(ax).show()

    # series_time['x-Mean'] = abs(series_time['data'] - series_time['data'].mean())
    # series_time['1.96*std'] = 1.96 * series_time['data'].std()
    # series_time['Outlier2'] = abs(series_time['data'] - series_time['data'].mean()) > 1.96 * series_time['data'].std()
    # print(series_time.head(40))

    # fig, ax = pyplot.subplots()
    # ax.plot(series_drop, '-', label='Original')
    # ax.plot(series_spline, '-o', label='spline', ms='3')
    # ax.plot(series_time, '-o', label='time', ms='3')
    # ax.plot(series_nearest, '-o', label='nearest', ms='3')
    # ax.plot(series_linear, '-o', label='linear', ms='3')
    #
    # ax.plot(series_slinear, '-o', label='slinear', ms='3')
    # # ax.plot(series_quadratic, '-o', label='quadratic', ms='3')
    # ax.plot(series_barycentric, '-o', label='barycentric', ms='3')
    # # ax.plot(series_polynomial, '-o', label='polynomial', ms='3')
    # interactive_legend(ax).show()

    # 粗采样，默认采用mean值
    resampledata = series_time['data_process_interp'].resample(resample_res).mean()
    resampledata = DataFrame(resampledata)
    # resampledata.dropna(inplace=True)
    # # resample1d = series_time['data_process_interp'].resample('1D')
    #
    # for xindex in range(len(resampledata.values)):
    #     sheet.write(xindex + 1, 0, str(resampledata.index[xindex]))
    #     sheet.write(xindex + 1, 1, resampledata.values[xindex][0])
    #
    # pyplot.figure(figsize=(12, 6))
    # pyplot.plot(resampledata)
    # pyplot.savefig('logs/pic/' + str(fbindex + 1) + '_' + fbnum + '.png', dpi=196)
    # pyplot.close()
    #
    # if plot:
    #     fig, ax = pyplot.subplots()
    #     ax.plot(series_time['data'], label='Original')
    #     ax.plot(series_time['data_process_interp'], '-o', label='interp', ms='3')
    #     ax.plot(resampledata, '-o', label=resample_res, ms='3')
    #     ax.plot(resampledata.shift(1), label=resample_res + '-shift', ms='3')
    #     ax.plot(resampledata - resampledata.shift(1), label='shift-error', ms='3')
    #     # ax.plot(resample1d, '-o', label='1D', ms='3')
    #     interactive_legend(ax).show()

    # reframed = serials_to_supervised(resampledata, 1, 1)
    # print(reframed)
vocab_size = len(resampledata)
        # # Split train/test set
        # ix_cutoff = int(len(resampledata) * 0.80)
        # x_train, x_test = resampledata[:ix_cutoff], resampledata[ix_cutoff:]
        # y_train, y_test = resampledata[:ix_cutoff], resampledata[ix_cutoff:]
        # vocab_size = len(vocab_processor.vocabulary_)
        # print("Vocabulary Size: {:d}".format(vocab_size))
        # print("80-20 Train Test split: {:d} -- {:d}".format(len(y_train), len(y_test)))
# Define LSTM RNN Model
class LSTM_Model():
    def __init__(self, embedding_size, rnn_size, batch_size, learning_rate,
                 training_seq_len, vocab_size, infer_sample=False):
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.vocab_size = vocab_size
        self.infer_sample = infer_sample
        self.learning_rate = learning_rate

        if infer_sample:
            self.batch_size = 1
            self.training_seq_len = 1
        else:
            self.batch_size = batch_size
            self.training_seq_len = training_seq_len

        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
        self.initial_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)

        self.x_data = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        self.y_output = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])

        with tf.variable_scope('lstm_vars'):
            # Softmax Output Weights
            W = tf.get_variable('W', [self.rnn_size, self.vocab_size], tf.float32, tf.random_normal_initializer())
            b = tf.get_variable('b', [self.vocab_size], tf.float32, tf.constant_initializer(0.0))

            # Define Embedding
            embedding_mat = tf.get_variable('embedding_mat', [self.vocab_size, self.embedding_size],
                                            tf.float32, tf.random_normal_initializer())

            embedding_output = tf.nn.embedding_lookup(embedding_mat, self.x_data)
            rnn_inputs = tf.split(axis=1, num_or_size_splits=self.training_seq_len, value=embedding_output)
            rnn_inputs_trimmed = [tf.squeeze(x, [1]) for x in rnn_inputs]

        # If we are inferring (generating text), we add a 'loop' function
        # Define how to get the i+1 th input from the i th output
        def inferred_loop(prev, count):
            # Apply hidden layer
            prev_transformed = tf.matmul(prev, W) + b
            # Get the index of the output (also don't run the gradient)
            prev_symbol = tf.stop_gradient(tf.argmax(prev_transformed, 1))
            # Get embedded vector
            output = tf.nn.embedding_lookup(embedding_mat, prev_symbol)
            return (output)

        decoder = tf.contrib.legacy_seq2seq.rnn_decoder
        outputs, last_state = decoder(rnn_inputs_trimmed,
                                      self.initial_state,
                                      self.lstm_cell,
                                      loop_function=inferred_loop if infer_sample else None)
        # Non inferred outputs
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.rnn_size])
        # Logits and output
        self.logit_output = tf.matmul(output, W) + b
        self.model_output = tf.nn.softmax(self.logit_output)

        loss_fun = tf.contrib.legacy_seq2seq.sequence_loss_by_example
        loss = loss_fun([self.logit_output], [tf.reshape(self.y_output, [-1])],
                        [tf.ones([self.batch_size * self.training_seq_len])],
                        self.vocab_size)
        self.cost = tf.reduce_sum(loss) / (self.batch_size * self.training_seq_len)
        self.final_state = last_state
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), 4.5)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))

    # def sample(self, sess, words=ix2vocab, vocab=vocab2ix, num=10, prime_text='thou art'):
    #     state = sess.run(self.lstm_cell.zero_state(1, tf.float32))
    #     word_list = prime_text.split()
    #     for word in word_list[:-1]:
    #         x = np.zeros((1, 1))
    #         x[0, 0] = vocab[word]
    #         feed_dict = {self.x_data: x, self.initial_state: state}
    #         [state] = sess.run([self.final_state], feed_dict=feed_dict)
    #
    #     out_sentence = prime_text
    #     word = word_list[-1]
    #     for n in range(num):
    #         x = np.zeros((1, 1))
    #         x[0, 0] = vocab[word]
    #         feed_dict = {self.x_data: x, self.initial_state: state}
    #         [model_output, state] = sess.run([self.model_output, self.final_state], feed_dict=feed_dict)
    #         sample = np.argmax(model_output[0])
    #         if sample == 0:
    #             break
    #         word = words[sample]
    #         out_sentence = out_sentence + ' ' + word
    #     return (out_sentence)


# Define LSTM Model
lstm_model = LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate,
                        training_seq_len, vocab_size)

# Tell TensorFlow we are reusing the scope for the testing
with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    test_lstm_model = LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate,
                                 training_seq_len, vocab_size, infer_sample=True)

# Create model saver
saver = tf.train.Saver(tf.global_variables())

# Create batches for each epoch
num_batches = int(len(resampledata) / (batch_size * training_seq_len)) + 1
# Split up text indices into subarrays, of equal size
batches = np.array_split(resampledata, num_batches)
# Reshape each split into [batch_size, training_seq_len]
batches = [np.resize(x, [batch_size, training_seq_len]) for x in batches]

# Initialize all variables
init = tf.global_variables_initializer()
sess.run(init)

# Train model
train_loss = []
iteration_count = 1
for epoch in range(epochs):
    # Shuffle word indices
    random.shuffle(batches)
    # Create targets from shuffled batches
    targets = [np.roll(x, -1, axis=1) for x in batches]
    # Run a through one epoch
    print('Starting Epoch #{} of {}.'.format(epoch + 1, epochs))
    # Reset initial LSTM state every epoch
    state = sess.run(lstm_model.initial_state)
    for ix, batch in enumerate(batches):
        training_dict = {lstm_model.x_data: batch, lstm_model.y_output: targets[ix]}
        c, h = lstm_model.initial_state
        training_dict[c] = state.c
        training_dict[h] = state.h

        temp_loss, state, _ = sess.run([lstm_model.cost, lstm_model.final_state, lstm_model.train_op],
                                       feed_dict=training_dict)
        train_loss.append(temp_loss)

        # Print status every 10 gens
        if iteration_count % 10 == 0:
            summary_nums = (iteration_count, epoch + 1, ix + 1, num_batches + 1, temp_loss)
            print('Iteration: {}, Epoch: {}, Batch: {} out of {}, Loss: {:.2f}'.format(*summary_nums))

        # Save the model and the vocab
        # if iteration_count % save_every == 0:
        #     # Save model
        #     model_file_name = os.path.join(full_model_dir, 'model')
        #     saver.save(sess, model_file_name, global_step=iteration_count)
        #     print('Model Saved To: {}'.format(model_file_name))
        #     # Save vocabulary
        #     dictionary_file = os.path.join(full_model_dir, 'vocab.pkl')
        #     with open(dictionary_file, 'wb') as dict_file_conn:
        #         pickle.dump([vocab2ix, ix2vocab], dict_file_conn)
        #
        # if iteration_count % eval_every == 0:
        #     for sample in prime_texts:
        #         print(test_lstm_model.sample(sess, ix2vocab, vocab2ix, num=10, prime_text=sample))

        iteration_count += 1

# Plot loss over time
plt.plot(train_loss, 'k-')
plt.title('Sequence to Sequence Loss')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()