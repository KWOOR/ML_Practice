# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 18:30:22 2018

@author: 우람
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\3차학기\\청강) 인공지능')
from tensorflow.examples.tutorials.mnist import input_data
mnist= input_data.read_data_sets("MNIST_data/", one_hot=True)  
X=tf.placeholder(tf.float32, [None, 784])
nb_classes=10
Y=tf.placeholder(tf.float32, [None, nb_classes]) #0~9까지 총 10개잖아
W=tf.Variable(tf.random_normal([784, nb_classes]))
b= tf.Variable(tf.random_normal([nb_classes]))

hypothesis= tf.nn.softmax(tf.matmul(X,W)+b)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer=tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
  
predicton = tf.arg_max(hypothesis, 1) 
is_correct= tf.equal(predicton, tf.arg_max(Y,1))
accuracy= tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs=15  

batch_size=100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost=0
        total_batch= int(mnist.train.num_examples/batch_size) #전체 데이터숫자 나누기 배치사이
        for i in range(total_batch):
            batch_xs, batch_ys= mnist.train.next_batch(batch_size) #리턴값이 두개
            c , _=sess.run([cost, optimizer], feed_dict={X:batch_xs, Y:batch_ys})
            avg_cost += c/total_batch
        print("epoch:", '%04d' %(epoch +1), "cost=", "{:.9f}".format(avg_cost))
        
    print("Accuracy", accuracy.eval(session=sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

    r=random.randint(0,mnist.test.num_examples -1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
    print("Prediction:", sess.run(tf.arg_max(hypothesis,1), feed_dict={X:mnist.test.images[r:r+1]}))
    
    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    plt.show()
    
#%%  위의 것은 원래 있던 거고... 이걸 뉴럴 네트워크로 바꿔보자!!!
    ''' relu 사용'''
    ''' xavier 사용'''
    '''drop out 사용
        keep_prob은.. train 할 때는 보통 0.5~0.7로 실전에서는 1로!!'''
    ''' AdamOptimizer 사용'''
        
tf.reset_default_graph() #변수 초기화!! 여러번 돌릴 수 있음

keep_prob=tf.placeholder(tf.float32)  #학습할 때와 실전할 때 값을 다르게 주기 위해 place holder로 한다.

X=tf.placeholder(tf.float32, [None, 784])
Y=tf.placeholder(tf.float32, [None, 10]) #0~9까지 총 10개잖아
W1=tf.get_variable("W1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
#W1=tf.get_variable("W1", shape=[784, 256])
b1= tf.Variable(tf.random_normal([256]))
L1= tf.nn.relu(tf.matmul(X,W1)+b1)
L1= tf.nn.dropout(L1, keep_prob=keep_prob)

W2=tf.get_variable("W2", shape=[256, 256],  initializer=tf.contrib.layers.xavier_initializer())
#W2=tf.get_variable("W2", shape=[256, 256])
b2= tf.Variable(tf.random_normal([256]))
L2= tf.nn.relu(tf.matmul(L1,W2)+b2)
L2= tf.nn.dropout(L1, keep_prob=keep_prob)

W3=tf.get_variable("W3", shape=[256, 10], initializer=tf.contrib.layers.xavier_initializer())
#W3=tf.get_variable("W3", shape=[256, 10])
b3= tf.Variable(tf.random_normal([10]))
hypothesis= tf.matmul(L2,W3)+b3  #여기선 relu 벗겨줌!! 왜냐? 밑에 cost함수에서 써주니까..!! (relu는 아니지만..)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost) 
'''러닝레이트 중요함!! '''

predicton = tf.arg_max(hypothesis, 1) 
is_correct= tf.equal(predicton, tf.arg_max(Y,1))
accuracy= tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs=15  

batch_size=100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost=0
        total_batch= int(mnist.train.num_examples/batch_size) #전체 데이터숫자 나누기 배치사이
        for i in range(total_batch):
            batch_xs, batch_ys= mnist.train.next_batch(batch_size) #리턴값이 두개
            c , _=sess.run([cost, optimizer], feed_dict={X:batch_xs, Y:batch_ys, keep_prob:0.7}) #여긴 0.7
            avg_cost += c/total_batch
        print("epoch:", '%04d' %(epoch +1), "cost=", "{:.9f}".format(avg_cost))
    
    #이 밑으론 실전이니까 keep_prob:1 이 되어야함!!    
    print("Accuracy", accuracy.eval(session=sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob:1})) #여긴 1

    r=random.randint(0,mnist.test.num_examples -1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
    print("Prediction:", sess.run(tf.arg_max(hypothesis,1), feed_dict={X:mnist.test.images[r:r+1], keep_prob:1})) #여기도 1!!
    
    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    plt.show()



    
    
    
    