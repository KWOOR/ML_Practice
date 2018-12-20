# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:40:45 2018

@author: 우람
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\3차학기\\청강) 인공지능')

#%%  test 데이터와 training 데이터를 나눠서 돌리면서 검증한다!!

x_data=[[1,2,1],[1,3,2],[1,3,4],[1,5,5],[1,7,5],[1,2,5],[1,6,6],[1,7,7]]
y_data=[[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]
'''여기 있는 인풋데이터에 너무 큰 아웃라이어들이 있다면.. 최적화가 안 된다. 그럴 경우에
정규화를 써줘야한다. 혹은, 데이터가 너무 들쭉날쭉 할 때 사용..
eg.)  x=MinMaxScaler(x) '''
# 텐서플로우에서 float와 int 구분할것!! 웬만하면 float로 해야겠지!?

x_test=[[2,1,1],[3,1,2],[3,3,4]]
y_test=[[0,0,1],[0,0,1],[0,0,1]]

X=tf.placeholder("float", [None,3])
Y=tf.placeholder("float", [None,3])
W=tf.Variable(tf.random_normal([3,3]))
b=tf.Variable(tf.random_normal([3]))  #'''칼럼의 갯수에 맞춘다!'''

hypothesis= tf.nn.softmax(tf.matmul(X,W)+b)
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
'''러닝레이트는 너무 크지도 않고, 너무 작지도 않게 설정해야한다!!'''

predicton = tf.arg_max(hypothesis, 1)
is_correct= tf.equal(predicton, tf.arg_max(Y,1))
accuracy= tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        cost_val, W_val, _=sess.run([cost, W, optimizer], feed_dict={X:x_data, Y:y_data})
        print(step, cost_val, W_val)
    print("Prediction:", sess.run(predicton, feed_dict={X:x_test}))
    print("Accuracy:", sess.run(accuracy, feed_dict={X:x_test, Y:y_test}))
            
#%%
from tensorflow.examples.tutorials.mnist import input_data #나같은 놈을 위해 만들어줌.. MNIST데이터가 이 안에 있다.
mnist= input_data.read_data_sets("MNIST_data/", one_hot=True)  #MNIST 데이터라는 폴더를 만들고 그 안에서 다운로드함

X=tf.placeholder(tf.float32, [None, 784])
nb_classes=10
Y=tf.placeholder(tf.float32, [None, nb_classes]) #0~9까지 총 10개잖아
W=tf.Variable(tf.random_normal([784, nb_classes]))
b= tf.Variable(tf.random_normal([nb_classes]))

hypothesis= tf.nn.softmax(tf.matmul(X,W)+b)
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
  
predicton = tf.arg_max(hypothesis, 1) 
is_correct= tf.equal(predicton, tf.arg_max(Y,1))
accuracy= tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs=15  
'''전체 데이터 셋을 한 번 학습시키는걸 1epoch라고 부른다... 메모리가 달리니까 한 번에 너무
많은걸 넣기 힘들어서 그런다. 학습을 15번 한다! 전체 데이터셋이 3000개면 4만5천번 돌아가겠'''
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
    #.eval 많이 쓰임!! sess.run과 비슷한것!  쓸 때, sess=tf.InteractiveSession() 꼭 선언하고 쓰
    #label은 정답임.. 그냥 숫자데이터! 글로 쓴거 말고!
    r=random.randint(0,mnist.test.num_examples -1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
    print("Prediction:", sess.run(tf.arg_max(hypothesis,1), feed_dict={X:mnist.test.images[r:r+1]}))
    
    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    plt.show()
    
#%%
    
    # 행렬을 다시 만들어주는 것.. x의 싸이즈로 1이 가득한 행렬을 만들어줌 
x=np.array([[0,1,2], [2,1,0]])
sess=tf.InteractiveSession()
tf.ones_like(x).eval()
