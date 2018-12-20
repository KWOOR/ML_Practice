# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 15:32:34 2018

@author: 우람
"""

import numpy as np
import tensorflow as tf
import os
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\3차학기\\청강) 인공지능')



x_data=[[73., 89., 9.,],[13., 35., 78.],[ 100., 56., 55.],[35., 27., 21.], [96.,88., 65.]]
y_data=[[152.], [126.],[130.], [98.], [120.]]

X=tf.placeholder(tf.float32, shape=[None, 3])
Y=tf.placeholder(tf.float32, shape= [None, 1])
W=tf.Variable(tf.random_normal([3,1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name= 'bias')


hypothesis= X@W+b  #X@W 대신에 tf.matmul(X,W)

cost=tf.reduce_mean((hypothesis -Y)**2)  #  tf.square(hypothesis-Y) 가능

optimizer= tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)
#위에랑 아래랑 같은 함수임.. 그냥 한 번에 쓴거
train= tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(201):
    cost_val, hy_vol,_= sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})
    if step%10==0:
        print(step, "Cost:", cost_val, "Prediction", hy_vol,_)
#%%

xy=np.loadtxt('test.csv', delimiter=',', dtype=np.float32)   #xlsx 파일은 로딩 안 됨

x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]],batch_size=10)

sess=tf.Session()

coord=tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
coord.request_stop()
coord.join(threads)
#%%
        
filename_queue=tf.train.string_input_producer(['test.csv'], shuffle=False, name='filename_queue')
reader=tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults= [[0.], [0.], [0.], [0.] ]
xy=tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]],batch_size=10)

X=tf.placeholder(tf.float32, shape=[None, 3])
Y=tf.placeholder(tf.float32, shape= [None, 1])
W=tf.Variable(tf.random_normal([3,1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name= 'bias')

hypothesis= tf.matmul(X,W)+b

cost=tf.reduce_mean(tf.square(hypothesis-Y))

optimizer= tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

sess=tf.Session()
sess.run(key)

coord= tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess, coord=coord)
for step in range(2001):
 #   x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val ,_ = sess.run([cost, hypothesis, train], feed_dict={X:train_x_batch, Y:train_y_batch})
    if step%10==0:
        print(step, cost_val, hy_val)
coord.request_stop()
coord.join(threads)








