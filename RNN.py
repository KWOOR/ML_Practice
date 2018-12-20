# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:35:29 2018

@author: 우람
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
pp = pprint.PrettyPrinter(indent=4)
from __future__ import print_function
import os
from tensorflow.examples.tutorials.mnist import input_data
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\3차학기\\청강) 인공지능')

'''cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)''' #num_units= 출력의 사이즈
#Basic RNN 말고도 많이 있음.. GRU나 LSTM 등등
#cell=tf.contrib.rnn.BasicLSTM(num_units=hidden_size)
'''outputs,_states= tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)'''
#보통 dynamic_rnn 많이 씀. 위에서 만든 cell과 내가 갖고 있는 데이터를 넣음
#_states값을 웬만하면 직접 사용하지는 않는다

#%%
#input은 4, 출력은 2

sess=tf.InteractiveSession()
h=[1,0,0,0]
e=[0,1,0,0]
l=[0,0,1,0]
o=[0,0,0,1] #이 아이들이 인풋dimension을 결정한다!!
#인풋데이터가 [[[1,0,0,0]]] 이라면 shape=(1,1,4)
#output의 크기는 내가 맘대로 정해라.. 히든 사이즈를 무엇으로 정했는지에 따라 다르다
hidden_size=2
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size) #입력 사이즈가 무엇이든 출력 사이즈는 2다

x_data=np.array([[[1,0,0,0]]], dtype=np.float32) #shape는 (1,1,4)니까.. 셀을 4개 펼치겠지!! 한 셀에 1주고, 다른 셀에 0 주고, 0주고, 0주고...
outputs,_states= tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

sess.run(tf.global_variables_initializer())
pp.pprint(outputs.eval())
#print(outputs.eval())
#tf.reset_default_graph()
#input shape = (1,5,4) 라면.. 4는 input dimension. 5는 한번에 셀을 5개 펼치겠다는 뜻, 입력데이터의 모양에 따라 결정된다.
#output shape = (1,5,2) 라면.. 2는 히든레이어.. 출력 수, 5는 위에서 받은 값이 그대로 나옴. 

with tf.variable_scope('two_sequances') as scope:
    # One cell RNN input_dim (4) -> output_dim (2). sequence: 5
    hidden_size = 2
    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size) #hidden size가 2가 되는 셀을 만든다
    x_data = np.array([[h, e, l, l, o]], dtype=np.float32) #sequence 만들기! 지금 5개니까 size=5가 된다.
    print(x_data.shape)
    pp.pprint(x_data)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())  #입력했을 때 출력이 어떻게 나오는지 살펴보는 과정 


with tf.variable_scope('3_batches') as scope:
    # One cell RNN input_dim (4) -> output_dim (2). sequence: 5, batch 3
    # 3 batches 'hello', 'eolll', 'lleel'
    x_data = np.array([[h, e, l, l, o],  #한 줄씩 학습 시키면 시간이 오래걸리니까 한 번에 여러개를 학습!!
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=np.float32)
    pp.pprint(x_data)
    
    hidden_size = 2 #출력사이즈가 2
    cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(
        cell, x_data, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval()) #배치 사이즈도 똑같은 사이즈로 가져온다.
    
    
#%% 문자열을 입력했을 때, 다음 문자열이 뭐가 될지 예측하는 것!!
# hihello
    
tf.set_random_seed(777)  # reproducibility

idx2char = ['h', 'i', 'e', 'l', 'o']
# Teach hello: hihell -> ihello
x_data = [[0, 1, 0, 2, 3, 3]]   # hihell  (1,6)이니까 셀을 6개 펼치겠지! 그럼 시퀀스는 6
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0  인덱스를 붙이는 것.. 0=h, 1=i 등등..
              [0, 1, 0, 0, 0],   # i 1  원핫 인코딩으로 바꾼다!
              [1, 0, 0, 0, 0],   # h 0  인풋디멘션은 5.. (유니크한 문자열의 갯수?!)
              [0, 0, 1, 0, 0],   # e 2  시퀀스 사이즈는 6이 되겠지!!
              [0, 0, 0, 1, 0],   # l 3  히든 사이즈는 5개가 될 거고!
              [0, 0, 0, 1, 0]]]  # l 3  배치 사이즈는 문자가 1개밖에 없으니까 1..

y_data = [[1, 0, 2, 3, 3, 4]]    # ihello

num_classes = 5
input_dim = 5  # one-hot size
hidden_size = 5  # output from the LSTM. 5 to directly predict one-hot. 인풋디멘션과 맞춰줘야 바로 예측할 수 있겠지!?
batch_size = 1   # one sentence
sequence_length = 6  # |ihello| == 6
learning_rate = 0.1

X = tf.placeholder(
    tf.float32, [None, sequence_length, input_dim])  # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
#rnn_cell=rnn_cell.BasicRNNCell(rnn_size)  여기서 rnn_size=5가 되겠지 
initial_state = cell.zero_state(batch_size, tf.float32) #다 0으로 둔다 초기값을
outputs, _states = tf.nn.dynamic_rnn(
    cell, X, initial_state=initial_state, dtype=tf.float32)  #아웃풋을 코스트 펑션에서 로짓으로 사용한다!!

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
# fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
# fc_b = tf.get_variable("fc_b", [num_classes])
# outputs = tf.matmul(X_for_fc, fc_w) + fc_b
outputs = tf.contrib.layers.fully_connected(
    inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes]) 

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights) #logit은 우리의 예측이 된다. target은 정답인 데이터. weight은 1로 생각하고..
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2) #결과가 숫자로 나오겠지 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(500):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)] #숫자를 다시 문자로 바꾼다 
        print("\tPrediction str: ", ''.join(result_str))
    
    
#%%
#문자열을 일일이 다 적어서 내가 데이터를 가공할 수는 없으니까.. 편하게 하는 방법
        
sample = " if you want you"
idx2char = list(set(sample))  # index -> char  유니크한 문자열을 골라서 인덱스로 만드는 과정 
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex

# hyper parameters
dic_size = len(char2idx)  # RNN input size (one hot size)
hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]  # char to index  문자를 숫자로 바꾸는 과정 
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n) hello: ello

X = tf.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

x_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0  몇개의 one hot으로 만들지. num_class는 유일한 문자열의 갯수와 같다
# x_one_hot이 어떻게 생겼는지 보고싶을 땐, sess.run(x_one_hot, feed_dict={X: x_data})
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_size, state_is_tuple=True) #hidden size는 dictionary size랑 같지! 
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]

        print(i, "loss:", l, "Prediction:", ''.join(result_str))


tf.reset_default_graph()
















