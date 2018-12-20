# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 23:04:44 2018

@author: 우람
"""

import numpy as np
import tensorflow as tf
import os
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\3차학기\\청강) 인공지능')

x_data=[[1,2],[2,3], [3,1], [4,3], [5,3], [6,2]]
y_data=[[0],[0],[0],[1],[1],[1]]

X=tf.placeholder(tf.float32, shape=[None, 2])
Y=tf.placeholder(tf.float32, shape=[None,1])


W=tf.Variable(tf.random_normal([2,1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')

hypothesis=tf.sigmoid(tf.matmul(X,W)+b) #둘 중 하나 구하기.. 0인가 1인

cost= -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))

train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted=tf.cast(hypothesis>0.5, dtype=tf.float32) #tf.float32로 두면 트루는 1, false는 0으로 나온다
accuracy= tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype= tf.float64))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
        if step%200 ==0:
            print(step, cost_val)
            
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\n Hypothesis:", h, c,a)
    
    
#%%
x_data=[[1,2,1,1],[2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]]

y_data=[[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

nb_classes=3

X=tf.placeholder(tf.float32, shape=[None, 4])
Y=tf.placeholder(tf.float32, shape=[None,3])
#Y=tf.placeholder(tf.int32, [None,1])  원핫이 아닌 수를 원핫으로 바꾸는 법!!
#Y_one_hot= tf.one_hot(Y, nb_classes) #1차원 더 커진다!! 주의!!
#Y_one_hot=tf.reshape(Y_one_hot, [-1, nb_classes])


W=tf.Variable(tf.random_normal([4,nb_classes]), name='weight')
b=tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis=tf.nn.softmax(tf.matmul(X,W)+b)  #Probabilities가 된다.

#cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))  아래와 같은 함수
logits=tf.matmul(X,W)+b
cost_i=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
cost=tf.reduce_mean(cost_i)

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        a=sess.run(hypothesis, feed_dict={X:[[1,11,7,9],[1,3,4,3]]}) #잘 맞추는지 보기 위해 넣는것..
        if step%200 ==0:
            print(step, sess.run(cost, feed_dict={X:x_data,Y:y_data}))
            print(a, sess.run(tf.arg_max(a,1)))
            # y_data.flatten()  [[1],[0]] ==> [1, 0] 이런식이 된다.
    
    
    
    
    
    
    
    
    
    
    
    
    