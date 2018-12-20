# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:34:41 2018

@author: 우람
"""

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
import pprint
from collections import deque
pp = pprint.PrettyPrinter(indent=4)
#from __future__ import print_function
import random
from gym.envs.registration import register
import os
from tensorflow.examples.tutorials.mnist import input_data
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\3차학기\\청강) 인공지능')

class DQN:
    def __init__(self, session, input_size, output_size, name="main"): #session 받아와야함
        self.session=session
        self.input_size=input_size
        self.output_size=output_size
        self.net_name =name
        self._build_network()
    
    def _build_network(self, h_size=10, l_rate=1e-1):  #h_size: hidden layer size
        with tf.variable_scope(self.net_name):
            self._X=tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            
            W1=tf.get_variable("W1", shape=[self.input_size, h_size],
                               initializer=tf.contrib.layers.xavier_initializer())
            layer1= tf.nn.tanh(tf.matmul(self._X, W1))
            
            W2=tf.get_variable("W2", shape=[h_size, self.output_size],
                               initializer=tf.contrib.layers.xavier_initializer())
            self._Qpred=tf.matmul(layer1, W2)
         
            #레이어를 더 쌓아도 좋음
        
        self._Y=tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)
        
        self._loss= tf.reduce_mean(tf.square(self._Y - self._Qpred))
        
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)
 
    def predict(self, state):
        x=np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X:x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={
                self._X:x_stack, self._Y: y_stack}) 

    
    
    
env= gym.make('CartPole-v0')
env._max_episode_steps = 10001

input_size=env.observation_space.shape[0]  #4개
output_size=env.action_space.n  #2개의 액션 오른쪽? 왼쪽? 이렇게
dis=0.9
REPLAY_MEMORY=50000



def simple_replay_train(DQN, train_batch):
    x_stack=np.empty(0).reshape(0, DQN.input_size)
    y_stack=np.empty(0).reshape(0, DQN.output_size)  #한개씩이 아니라 모아서 학습시키기 위해 스택을 사용
    
    for state, action, reward, next_state, done in train_batch:
        Q=DQN.predict(state)
        
        if done:
            Q[0,action]=reward
        else:
            Q[0, action]=reward+dis*np.max(DQN.predict(next_state))
        y_stack=np.vstack([y_stack, Q])
        x_stack=np.vstack([x_stack, state])

    return DQN.update(x_stack, y_stack)
        

def bot_play(mainDQN):   #학습이 잘 된 네트웤을 받아서 스코어를 출력하는 함수
    s=env.reset()
    reward_sum=0
    while True:
        a=np.argmax(mainDQN.predict(s))
        s,reward, done, _ = env.step(a)
        reward_sum +=reward
        if done:
            print("Total score:{}".format(reward_sum))
            break
        
      

def main():
    max_episodes=5000
    
    replay_buffer=deque()
    
    with tf.Session() as sess:
        mainDQN= DQN(sess, input_size, output_size)
        tf.global_variables_initializer().run()
        for episode in range(max_episodes):
            e=1.0/((episode/10)+1)
            done=False
            step_count=0
            state=env.reset()  #루프 시작하기 전에 리셋!
            
            while not done:
                if np.random.rand(1)<e:
                    action= env.action_space.sample()
                else:
                    action= np.argmax(mainDQN.predict(state))  #액션을 먼저 정하고..
                    
                next_state, reward, done, _ =env.step(action)
                
                if done:
                    reward=-100
                replay_buffer.append((state, action, reward, next_state, done)) #결과물 저장
                if len(replay_buffer) >REPLAY_MEMORY:
                    replay_buffer.popleft()
                
                state=next_state
                step_count+=1
                if step_count >10000:
                    break
            
            print("Episode: {}  steps:{}".format(episode, step_count))
            if step_count>10000:
                pass
                break  #step이 무한대로 가면 끝나질 않으니까...
                
            if episode%10 ==1:  #에피소드 10번에 1번씩 우리가 저장해놨던 곳에서 랜덤하게 10개씩 꺼내서 학습시킨다
                for _ in range(50):
                    minibatch = random.sample(replay_buffer,10)
                    loss,_=simple_replay_train(mainDQN, minibatch)
                print("Loss:", loss)
        bot_play(mainDQN)
        
if __name__ == "__main__":
    main()
    
#main()    
        

tf.reset_default_graph()

#%%
'''실행 안 함'''

'''
replay_buffer=deque()   #여기다가 정보들 쌓어서 저장한

replay_buffer.append((state, action, reward, next_state, done))
if len(replay_buffer) > REPLAY_MEMORY:
    replay_buffer.popleft()  #replay buffer에 일정 수준 이상의 데이터가 쌓이면 오래된 것부터 지워버려라

if episode %10 ==1:  #episode 10번에 1번씩 작동함...
    for _ in range(50):
        minibatch = random.sample(replay_buffer, 10)  #미니배치에서 10개씩 데이터를 가져옴
        loss, _ =simple_replay_train(mainDQN, minibatch)
            
'''
#%%        
''' y를 만들때, target을 만들어두자
학습할 때, Y(=label, 정답)이 필요할텐데 target를 Y로 둔다.'''

def get_copy_var_ops(*, dest_scope_name='target', src_scope_name='main'):  #복사하는 함수
    op_holder=[]
    src_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    
    for src_var, dest_var in zip(src_vars, dest_vars): #src_var, dest_var는 tensor들임
        op_holder.append(dest_var.assign(src_var.value())) #dest_var = src_var로 복사가 된다.. 같아진다
    
    return op_holder



def replay_train(mainDQN, targetDQN, train_batch):
    x_stack=np.empty(0).reshape(0, input_size)
    y_stack=np.empty(0).reshape(0, output_size) 
    
    for state, action, reward, next_state, done in train_batch:
        Q=mainDQN.predict(state)
        
        if done:
            Q[0,action]=reward
        else:
            Q[0, action]=reward+dis*np.max(targetDQN.predict(next_state))
        y_stack=np.vstack([y_stack, Q])
        x_stack=np.vstack([x_stack, state])

    return mainDQN.update(x_stack, y_stack)

  
def main():
    max_episodes=5000
    
    replay_buffer=deque()
    
    with tf.Session() as sess:
        mainDQN= DQN(sess, input_size, output_size, name='main')
        targetDQN=DQN(sess, input_size, output_size, name='target')
        tf.global_variables_initializer().run()
        copy_ops=get_copy_var_ops(dest_scope_name='target', src_scope_name='main')
        
        sess.run(copy_ops)
        
        for episode in range(max_episodes):
            e=1.0/((episode/10)+1)
            done=False
            step_count=0
            state=env.reset()  #루프 시작하기 전에 리셋!
            
            while not done:
                if np.random.rand(1)<e:
                    action= env.action_space.sample()
                else:
                    action= np.argmax(mainDQN.predict(state))  #액션을 먼저 정하고..
                    
                next_state, reward, done, _ =env.step(action)
                
                if done:
                    reward=-100
                replay_buffer.append((state, action, reward, next_state, done)) #결과물 저장
                if len(replay_buffer) >REPLAY_MEMORY:
                    replay_buffer.popleft()
                
                state=next_state
                step_count+=1
                if step_count >10000:
                    break
                
            #여기까진 일단 학습이 일어나진 않는다
            
            print("Episode: {}  steps:{}".format(episode, step_count))
            if step_count>10000:
                pass
                break  #step이 무한대로 가면 끝나질 않으니까...
                
            if episode%10 ==1:  #에피소드 10번에 1번씩 우리가 저장해놨던 곳에서 랜덤하게 10개씩 꺼내서 학습시킨다
                for _ in range(50):
                    minibatch = random.sample(replay_buffer,10)
                    loss,_=replay_train(mainDQN,targetDQN, minibatch)
                print("Loss:", loss)
                sess.run(copy_ops) #메인과 타겟이 같아지도록 복사 
        bot_play(mainDQN)
        
if __name__ == "__main__":
    main()
          
        
tf.reset_default_graph()

           
    #%%
''' 실행 안 함'''

'''
       
with tf.Session() as sess:
    mainDQN=DQN(sess, input_size, output_size, name="main")
    targetDQN=DQN(sess, input_size, output_size, neme='target') 
    tf.global_variables_initializer().run()
    
    copy_ops =get_copy_var_ops(dest_scope_name='target', src_scope_name='main')
    sess.run(copy_ops)
    # mainDQN만 변하게 학습을 시키고, 일정 step마다 복사를 다시 해서 같게만든다
    
   '''     
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            