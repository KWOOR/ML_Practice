# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 18:37:48 2018

@author: 우람
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.set_random_seed(777)  # reproducibility

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
#한번에 다 하기엔 어려우니.. 짤라서 읽도록 한다.

char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10  # Any arbitrary number 내가 한 번에 얼마만큼의 글자를 읽을건지 정해라! 
#dynamic RNN을 하려면 이 sequence_length를 다르게 주면 된다... 예를 들어 'hello' 'hi' 'why'인 경우..
#sequence_length=[5,2,3] 이렇게 주면 된다!! 
learning_rate = 0.1

dataX = []
dataY = []
for i in range(0, len(sentence) - sequence_length): #문장이 너무 기니까 잘라서 읽어낸다 
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]  # x str to index
    y = [char_dic[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)

X = tf.placeholder(tf.int32, [None, sequence_length])  #None은 이제 배치 사이즈가 되겠지!
Y = tf.placeholder(tf.int32, [None, sequence_length])

# One-hot encoding
X_one_hot = tf.one_hot(X, num_classes)
print(X_one_hot)  # check out the shape


# Make a lstm cell with hidden_size (each unit output vector size)
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell
#cell= rnn.Basic:STMCell(hidden_size, state_is_tuple=True)
multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)
#cell= rnn.MultiRNNCell ([cell]*2, state_is_tuple=True) #... [cell]*2를 했으니까 RNN을 2번 돌리겠다고!

# outputs: unfolding size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size]) #ouput을 softmax 돌리기 위한 형태로 바꿔줌!! hidden_size앞의 -1은 알아서 쌓으라고 나머지는..
#softmax_w=tf.get_variable("softmax_w", [hidden_size, num_classes]) #hidden size는 input size, num_classes= ouput size
#softmax_b=tf.get_variable("softmax_b", [num_classes])
#ouputs=tf.matmul(X_for_fc, softmax_w)+softmax_b
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)
#여기서의 outputs은 softmax돌려서 나온 output, activation function이 없기에 sequence_loss에서 액티베이션 해준다!

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
#softmax돌려서 나온 output을 다시 재배열!! RNN에서 나온 ouput과 같은 형태겠지!! 

# All weights are 1 (equal weights)
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    _, l, results = sess.run(
        [train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print(i, j, ''.join([char_set[t] for t in index]), l)

# Let's print the last char of each result to check it works.. 배치에 있는거 다 모아서 출력하기
results = sess.run(outputs, feed_dict={X: dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:  # print all for the first result to make a sentence
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')

#%%
        

import tensorflow as tf
import numpy as np
import matplotlib
import os

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

'''주식 데이터 가지고 훈련시키는 것..! 
Open, High, Low, Volume, Close 이렇게 5개의 데이터가 총 7일 동안 있음 
이것들로 다음날의 close를 예측!!'''


# train Parameters
timestpes=7 #시퀀스길이랑 같음 
seq_length = 7 #7일 동안 있으니까 sequence length도 7개
data_dim = 5 #input size, open, high, low, volume, close 이렇게 5개
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 500
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\3차학기\\청강) 인공지능')

# Open, High, Low, Volume, Close
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)
#xy=MinMaxScaler(xy)
#x=xy
#y=xy[:,[-1]]  #종가를 label로 쓴다
#dataX=[]
#dataY=[]
'''for i in range(0, len(y)-seq_length):
    _x=x[i:i+seq_length]
    _y=y[i+seq_length] #다음날 종가
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)'''

# train/test split
train_size = int(len(xy) * 0.7)
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length:]  # Index from [train_size - seq_length] to utilize past sequence

# Scale each
train_set = MinMaxScaler(train_set)
test_set = MinMaxScaler(test_set)

# build datasets
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, [-1]]  # Next close price
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
#여기서 hidden_dim은 내 맘대로 설정해도 된다.. 출력 사이즈

outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output
#ouput중에 마지막 하나껏만 쓸거라서.. outputs[:,-1] 이렇게 썼다

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
#sequence loss가 아니라 값이 하나기 때문에 평균으로 하지 않았음

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    # Plot predictions
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()


