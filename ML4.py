# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 21:11:30 2018

@author: 우람
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\3차학기\\청강) 인공지능')

#%% Neural Net...  레이어를 여러개 만들어서 합성하는 것.
x_data=[[0,0],[0,1], [1,0], [1,1]]
y_data=[[0],[1],[1],[0]]

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)


W1=tf.Variable(tf.random_normal([2,2]), name='weight1')
b1=tf.Variable(tf.random_normal([2]), name='bias1')
layer1= tf.sigmoid(tf.matmul(X,W1)+b1)

''' layer를 늘려서 많이 쌓고.. W의 갯수도 [2,2]가 아닌 [2,10], [10,10] 이런식으로 늘리면 더 정확해진다.'''


W2=tf.Variable(tf.random_normal([2,1]), name='weight2')
b2=tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis=tf.sigmoid(tf.matmul(layer1,W2)+b2) #둘 중 하나 구하기.. 0인가 1인



cost= -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted=tf.cast(hypothesis>0.5, dtype=tf.float32) #tf.float32로 두면 트루는 1, false는 0으로 나온다
accuracy= tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype= tf.float64))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step%100 ==0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
            
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\n Hypothesis:", h, c,a)
    
#%%  Tensorboard!!
    
# 그래프 이쁘게 보기
    
w2_hist = tf.summary.histogram("weights2", W2) #값이 여러개인 경우엔 histogram으로 불러오기 
cost_sum = tf.summary.scalar("cost", cost) #값일 경우엔 scalar
# 내가 기록하고 싶은 텐서들 고르기

summary1= tf.summary.merge_all()
# 요약한 것들 한 눈에 보도록 합치기

writer= tf.summary.FileWriter('C:\FEP/logs') # logs라는 이름을 가진 파일 생성!!
writer.add_graph(sess.graph) #그래프 추가

sess=tf.Session()
sess.run(tf.global_variables_initializer())

s, _ = sess.run([summary1, train], feed_dict= {X:x_data, Y:y_data}) #summary1은 텐서니까 작동시키려면 sess.run으로 해야지!
#writer.add_summary(s, global_step=global_step)  #요약한것 추가하기.. 스텝이 있으면 추가하
writer.add_summary(s)  #없으면 이거 쓰고...

tensorboard --logdir=./logs  #위에서 쓴 디렉토리 적어주기.. 프롬프트에서 입력!!!


#name_scope를 사용해서, 레이어별로 따로 보여준다!! 정리해서! 
with tf.name_scope("layer1") as scope: 
    W1=tf.Variable(tf.random_normal([2,2]), name='weight1')
    b1=tf.Variable(tf.random_normal([2]), name='bias1')
    layer1= tf.sigmoid(tf.matmul(X,W1)+b1)
    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist=tf.summary.histogram("biases1", b1)
    layer1_hist = tf.summary.histogram("layer1",layer1)

with tf.name_scope("layer2") as scope: 
    W2=tf.Variable(tf.random_normal([2,1]), name='weight2')
    b2=tf.Variable(tf.random_normal([1]), name='bias2')
    hypothesis= tf.sigmoid(tf.matmul(layer1,W2)+b2)
    w2_hist = tf.summary.histogram("weights2", W2)
    b2_hist=tf.summary.histogram("biases2", b2)
    hypothesis_hist = tf.summary.histogram("hypothesis",hypothesis)

summary2= tf.summary.merge_all()

sess=tf.Session()
sess.run(tf.global_variables_initializer())

writer= tf.summary.FileWriter('TB_SUMMARY_DIR')
writer.add_graph(sess.graph)

s,_=sess.run([summary2, ], feed_dict = feed_dict)  #실행시키고 나온 값 s를..
writer.add_summary(s, global_step= global_step)  #더해준다. 
global_step +=1

writer = tf.summary.FileWriter("./logs/xor_logs")
$ tensorboard --logdir = ./logs/xor_logs   #동시에 여러개를 돌리고 싶으면.. 얘처럼 logs 밑에 하위 폴더들을 만든다
# tensorboard --logdir= ./logs/xor_logs_r0_01 이렇게..
#그러고 나서 
#tensorboard --logdir=.logs 이렇게 상위폴더만 돌리면 하위폴더 다 돌아간다

# 여기 밑에 있는건 외워라 그냥... 이렇게 쓴다....
ssh -L local_port:127.0.0.1:remote_port username@server.com #로컬이름 적으면 됨
locar> $ ssh -L 7007:121.0.0.0:6006 kur7@server.com
server> $ tensorboard --logdir=.logs/xor_logs
#I can navigate to http://127.0.0.1:7007
    
    
    
    
    