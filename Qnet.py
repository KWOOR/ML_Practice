# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:01:34 2018

@author: 우람
"""


import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
import pprint
pp = pprint.PrettyPrinter(indent=4)
#from __future__ import print_function
import random as pr
from gym.envs.registration import register
import os
from tensorflow.examples.tutorials.mnist import input_data
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\3차학기\\청강) 인공지능')

register(id='FrozenLake-v3', entry_point='gym.envs.toy_text:FrozenLakeEnv',kwargs={'map_name':'4x4', 'is_slippery':False})
env=gym.make('FrozenLake-v0')

#%%

'''State가 총 16개니까... 첫 번째 state는 1,0,0...0, 두 번째 state는 0,1,0,....0이 되는 원핫 인코딩 만들자!'''

np.identity(16) #One-hot 인코딩 만들기
# state=np.identity(16)[s1:s1+1]

def one_hot(x):
    return np.identity(16)[x:x+1]

'''출력의 개수는 내가 선택할 수 있는 길이 위, 아래, 오른쪽, 왼쪽 4개니까 4개'''

input_size=env.observation_space.n  #16
output_size=env.action_space.n  #4
learning_rate=0.1

X=tf.placeholder(shape=[1,input_size], dtype=tf.float32)
W=tf.Variable(tf.random_uniform([input_size, output_size], 0,0.01)) #0, 0.01은 초기값

Qpred=tf.matmul(X,W)
Y=tf.placeholder(shape=[1,output_size], dtype=tf.float32)

loss=tf.reduce_sum(tf.square(Y-Qpred))
train=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

dis=0.99
num_episodes=2000

rList=[]

with tf.Session() as sess:   #E-greedy 사용!
    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        s=env.reset()
        e=1.0/((i/50)+10)
        rAll=0
        done=False
        local_loss=[]
        
        while not done:
            Qs=sess.run(Qpred, feed_dict={X:one_hot(s)})  #Qtable이 neuron으로 바뀌었다..
            if np.random.rand(1)<e:
                a=env.action_space.sample()
            else:
                a=np.argmax(Qs)
            s1,reward,done,_=env.step(a)
            if done:
                Qs[0,a]=reward
            else:
                Qs1=sess.run(Qpred, feed_dict={X:one_hot(s1)})
                Qs[0,a]=reward +dis*np.max(Qs1)
            sess.run(train, feed_dict={X:one_hot(s), Y:Qs})
            rAll+=reward
            s=s1
        rList.append(rAll)
        
print("Percent of successful episodes:" +str(sum(rList)/num_episodes)+"%")
plt.bar(range(len(rList)), rList, color='blue')
plt.show()
    
#%%    
''' 이건 안 돌리는 것.. 그냥 설명 써 놓은것임'''

Qs=sess.run(Qpred, feed_dict={X:one_hot(s)})  #Qtable이 neuron으로 바뀌었다..

if np.random.rand(1)<e:
    a=env.action_space.sample()
else:
    a=np.argmax(Qs)
    
if done:
    Qs[0,a]=reward  #Terminal state인 경우
else:
    Qs1=sess.run(Qpred, feed_dict={X:one_hot(s1)})
    Qs[0,a]=reward +dis*np.max(Qs1)
    

Qs[0,a]=reward+dis*np.max(Qs1)  #Qs1은 다음 상태의 값
sess.run(train, feed_dict={X:one_hot(s), Y:Qs})


#%%
'''CartPole Game'''

env=gym.make('CartPole-v0')
env.reset()
random_episodes=0
reward_sum=0
while random_episodes<10:
#    env.render()
    action=env.action_space.sample()
    observation, reward, done, _=env.step(action)
    print(observation, reward, done)
    reward_sum+=reward
    if done:
        random_episodes +=1
        print("Reward for this episode was:", reward_sum)
        reward_sum=0
        env.reset()
 
#%%       
learning_rate=1e-1
input_size=env.observation_space.shape[0]   #4
output_size=env.action_space.n  #2.. 오른쪽 왼쪽

X=tf.placeholder(tf.float32, [None, input_size], name="input_x") #1xinput size로 보자

W1=tf.get_variable("W1", shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
Qpred=tf.matmul(X,W1)

Y=tf.placeholder(shape=[None, output_size], dtype=tf.float32)

loss=tf.reduce_sum(tf.square(Y-Qpred))

train=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

num_episodes=2000
dis=0.9
rList=[]

with tf.Session() as sess:   #E-greedy 사용!
    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        s=env.reset()
        e=1.0/((i/10)+1)
        rAll=0
        step_count=0
        done=False
        
        while not done:
            step_count +=1
            x=np.reshape(s, [1,input_size])
            Qs=sess.run(Qpred, feed_dict={X:x}) 
            if np.random.rand(1)<e:
                a=env.action_space.sample()
            else:
                a=np.argmax(Qs)
            s1,reward,done,_=env.step(a)
            if done:
                Qs[0,a]= -100  #실패했다는 뜻임
            else:
                x1=np.reshape(s1, [1,input_size])
                Qs1=sess.run(Qpred, feed_dict={X:x1})
                Qs[0,a]=reward +dis*np.max(Qs1)
            sess.run(train, feed_dict={X:x, Y:Qs})  #학습 시키는 구간
            s=s1
        rList.append(step_count)
        
        print("Episode: {} steps: {}".format(i,step_count))
        if len(rList)>10 and np.mean(rList[-10:])>500:  #보통 200만 넘어도 됨. 10번연속이라는 의미
            break
        
    #%%
    
'''우리가 훈련시킨거 잘 되는지 보기'''    

sess=tf.Session()
sess.run(tf.global_variables_initializer())

observation=env.reset()
reward_sum=0
while True:
#    env.render()
    x=np.reshape(observation, [1,input_size])
    Qs=sess.run(Qpred, feed_dict={X:x})
    a=np.argmax(Qs)
    
    observation, reward, done, _=env.step(a)
    reward_sum+=reward
    if done:
        print("Total score: {}".format(reward_sum))
        break
        


tf.reset_default_graph()

''' 잘 안 맞는다... Weihgt layer가 1개고..  변수도 4개밖에 없어서 학습이 잘 안 됨..
딥하게 간다고 해도 sample간에 correlation이 있고, 안정적이지가 않음'''


#%%
s1, reward, done, _ = env.step(a)
if done:
    Qs[0,a]=-100  #넘어지면 게임이 끝난다. 이 때 잘못한 벌로 100을 깎자!
else:
    x1=np.reshape(s1, [1,input_size])  #원핫 인코딩 말고 그대로 쓴다.. 1x4 사이즈
    Qs1=sess.run(Qpred, feed_dict={X:x1})
    Qs[0,a]=reward + dis*np.max(Qs1)
sess.run(train, feed_dict={X:x, Y:Qs})















