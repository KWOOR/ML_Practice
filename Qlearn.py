# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 13:45:09 2018

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

def rargmax(vector):
    m=np.amax(vector)
    indices=np.nonzero(vector==m)[0]
    return pr.choice(indices)  #random 하게 간다

register(id='FrozenLake-v3', entry_point='gym.envs.toy_text:FrozenLakeEnv',kwargs={'map_name':'4x4', 'is_slippery':False})
env=gym.make('FrozenLake-v3')
#배경을 만들어주는 것들이다.



#%%

# Decaying E-greedy


Q=np.zeros([env.observation_space.n, env.action_space.n])
num_episodes=2000
learning_rate=0.85  #크면 빨리 학습, 작으면 느리게 학습 

dis=0.9 #discounted reward


rList=[]

for i in range(num_episodes):
    state=env.reset()
    rAll=0
    done=False
    e=1./((i//100)+1)   
    while not done:
        if np.random.rand(1)<e:
            action=env.action_space.sample()
        else:
            action=rargmax(Q[state,:])  #Random argmax 
        new_state, reward, done, _ =env.step(action)
        #Q[state,action]=reward+ dis * np.max(Q[new_state,:])  #Deterministic World 방법
        Q[state,action]=(1-learning_rate)*Q[state,action]+ learning_rate*(reward + dis*np.max(Q[new_state,:]))
        rAll+=reward
        state=new_state
    rList.append(rAll)
    
print("Success rate:" +str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("Left Down Right Up")
print(Q)
plt.bar(range(len(rList)), rList, color='blue')
plt.show()


#%%

# Add random noise  &&  Decay  ( (i+1)로 나눠주는 곳이 Decay하는 곳 ) && Discounted reward  

Q=np.zeros([env.observation_space.n, env.action_space.n])
num_episodes=2000
learning_rate=0.85  #크면 빨리 학습, 작으면 느리게 학습 

dis=0.99 #discounted reward

rList=[]

for i in range(num_episodes):
    state=env.reset()
    rAll=0
    done=False
    while not done:
        action=np.argmax(Q[state,:] + np.random.randn(1, env.action_space.n)/(i+1))  #Random argmax 
        new_state, reward, done, _ =env.step(action)
        #Q[state,action]=reward + dis*np.max(Q[new_state,:])   #Deterministic World 방법
        ''' Non-Stochastic World인데도, Stochastic World에서 쓰던 방법을 사용해도 잘 작동한다!!! '''
        Q[state,action]=(1-learning_rate)*Q[state,action]+ learning_rate*(reward + dis*np.max(Q[new_state,:]))
        rAll+=reward
        state=new_state
    rList.append(rAll)
print("Success rate:" +str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("Left Down Right Up")
print(Q)
plt.bar(range(len(rList)), rList, color='blue')
plt.show()


#%%

'''Stocahstic World'''

env=gym.make('FrozenLake-v0')


Q=np.zeros([env.observation_space.n, env.action_space.n])
num_episodes=2000
learning_rate=0.9  #크면 빨리 학습, 작으면 느리게 학습 
dis=0.99 #discounted reward

rList=[]

for i in range(num_episodes):
    state=env.reset()
    rAll=0
    done=False
    while not done:
        action=np.argmax(Q[state,:] + np.random.randn(1, env.action_space.n)/(i+1))  #Random argmax 
        new_state, reward, done, _ =env.step(action)
        Q[state,action]=(1-learning_rate)*Q[state,action]+ learning_rate*(reward + dis*np.max(Q[new_state,:]))
        rAll+=reward
        state=new_state
    rList.append(rAll)
print("Success rate:" +str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("Left Down Right Up")
print(Q)
plt.bar(range(len(rList)), rList, color='blue')
plt.show()












