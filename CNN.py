# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 23:58:28 2018

@author: 우람
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
from tensorflow.examples.tutorials.mnist import input_data
os.chdir('C:\\Users\\우람\\Desktop\\kaist\\3차학기\\청강) 인공지능')

#%% 이미지 만들기
sess=tf.InteractiveSession()
image= np.array([[[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]]], dtype=np.float32)  
print(image.shape) #출력하면 (1,3,3,1)이 나오는데 맨 앞의 1은.. n개의 데이터 즉 1개의 데이터를 사용한다는 뜻
plt.imshow(image.reshape(3,3), cmap='Greys')
#image의 크기는 (1,3,3,1) 즉, 3X3X1인거지
#Filter 는 2,2,1,1 로 해보자... 3번째 값은 데이터의 맨 끝번째 값과 같아야지!!!! 맨 마지막값은 필터의 갯수!!!
#Filter를 거치고 나온 값이 나온다..!!

print("image.shape", image.shape)
weight=tf.constant([[[[1.]],[[1.]]],[[[1.]],[[1.]]]]) #Filter임!! (2x2 사이즈!!)
print("weight.shape", weight.shape) #(2,2,1,1) 나온다!
conv2d= tf.nn.conv2d(image, weight, strides=[1,1,1,1], padding='VALID')
conv2d_img=conv2d.eval()
print("conv2d_img.shape", conv2d_img.shape) #(1,2,2,1)이 나오지!! (3X3X1)에 Filter를 거치는데 Stride가 1씩이니까!
#밑에는 그냥.. 이렇게 쓴다고 보고 넘어가면 
conv2d_img=np.swapaxes(conv2d_img, 0,3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(2,2))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(2,2), cmap='gray')
''' 만약 위에서 padding를 "SAME"으로 주면.. stride가 (1x1)일 때 원래 데이터 값하고 같은 사이즈가 나옴!!'''
''' 입력 사이즈와 출력 사이즈가 같아진다!'''

weight=tf.constant([[[[1.,10.,-1.]],[[1.,10.,-1.]]],[[[1.,10.,-1.]],[[1.,10.,-1.]]]]) 
# 필터가 3개 일 때의 예시임!!!
print("weight.shape", weight.shape) #(2,2,1,3)이 나오는데.. 필터가 3장이니까!!

image= np.array([[[[4],[3]],[[2],[1]]]], dtype=np.float32)
pool=tf.nn.max_pool(image, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME') #ksize는 필터의 사이즈! 풀링할때는 커널이라고 부른다!
#ksize는 커널 사이즈를 정해주는 것임!!
# maxpooling이 cnn이랑 잘 맞는다
print(image.shape)
print(pool.shape)
print(pool.eval())

mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)
img=mnist.train.images[0].reshape(28,28)
img=img.reshape(-1,28,28,1) #-1은 너가 알아서 계산하라는 뜻임..데이터가 여러개 있으니까!! 맨 끝의 1은 한 색깔이라고
W1=tf.Variable(tf.random_normal([3,3,1,5], stddev=0.01)) #1은 색깔의 갯수(img의 끝값과 맞춰줘야지), 5개는 필터의 갯수!
conv2d=tf.nn.conv2d(img, W1, strides=[1,2,2,1], padding='SAME') #stride가 2x2라는 뜻임!
'''padding이 same인데 stride가 2x2니까 원래 데이터 크기가 반으로 줄어든다!!!'''
print(conv2d)
sess.run(tf.global_variables_initializer())
#역시나 밑에껀 알 필요 없음...
conv2d_img=conv2d.eval()
conv2d_img=np.swapaxes(conv2d_img,0,3)
for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14), cmap='gray')
    
pool=tf.nn.max_pool(conv2d, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# 입력값이 14x14인데 stride가 2,2고, 패딩이 SAME니까 7x7이 나온다!!
print(pool)
sess.run(tf.global_variables_initializer())
pool_img=pool.eval()
pool_img=np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(7,7), cmap='gray')



#%%
'''Conv 하고, pool하고, Conv하고, pool하고...    '''
tf.reset_default_graph()

X=tf.placeholder(tf.float32, [None, 784])
X_img=tf.reshape(X,[-1,28,28,1]) #이미지로 바꿔주는것! 28x28사이즈
Y=tf.placeholder(tf.float32, [None, 10])

W1=tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01)) #3x3의 필터, 색깔은 1개, 필터는 32개!!
L1=tf.nn.conv2d(X_img,W1, strides=[1,1,1,1], padding='SAME') # padding='SAME'고, stride가 1일 땐, 필터의 사이즈에 상관없이 입력의 사이즈로 출력한다!
L1=tf.nn.relu(L1)
L1=tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# 풀링에서 stride가 2x2이기에 풀링을 거치면 14x14로 나온다!!
# Conv = (?, 28,28,32)
# Relu = (?,28,28,32)
#Pool = (?,14,14,32)

W2=tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01)) #넘어오는 데이터의 끝 값이 32니까, 3번째 값도 32. 64는 필터의 갯수!
L2=tf.nn.conv2d(L1,W2, strides=[1,1,1,1], padding='SAME')
L2=tf.nn.relu(L2)
L2=tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2=tf.reshape(L2,[-1,7*7*64])  #Fully Connected 하기 위해서 플랫트닝 하는 것임!!!
# Conv= (?,14,14,64)
# Relu= (?, 14, 14, 64)
# Pool= (?, 7,7,64)
''' 보면 알겠지만.. 데이터의 맨 끝 값은 필터의 갯수를 따라가네..? (내 생각)'''
''' 사이즈가 얼마인지 잘 모를 때는 그냥 프린트 해봐라...'''
dense4=tf.layers.dense(inputs=flat, units=625, activiation=tf.nn.relu) #W3~hypothesis까지 대체해줌!!
W3= tf.get_variable("W2", shape=[7*7*64, 10], initializer= tf.contrib.layers.xavier_initializer()) 
# 7*7*64는 입력의 값!, 10은 출력의 값!! 0~9까지 찍어낼거니까 10개지!!
b=tf.Variable(tf.random_normal([10]))
hypothesis=tf.matmul(L2,W3)+b
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,labels=Y))
optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


sess=tf.Session()
sess.run(tf.global_variables_initializer())


is_correct= tf.equal(tf.arg_max(hypothesis,1), tf.arg_max(Y,1))
accuracy= tf.reduce_mean(tf.cast(is_correct, tf.float32))
training_epochs=15  
batch_size=50  #배치사이즈가 너무 커도 안 되고.. 그렇다고 너무 작아서 토탈배치가 커져도 안 된다!!!
total_batch=10  #배치사이즈 100개, 토탈배치 10개가 적당함... 정확도 96%는 나
for epoch in range(training_epochs):
    avg_cost=0
#    total_batch= int(mnist.train.num_examples/batch_size) #전체 데이터숫자 나누기 배치사이즈. 렉걸림...
    for i in range(total_batch):
        batch_xs, batch_ys= mnist.train.next_batch(batch_size) #리턴값이 두개
        c , _=sess.run([cost, optimizer], feed_dict={X:batch_xs, Y:batch_ys})
        avg_cost += c/total_batch
    print("epoch:", '%04d' %(epoch +1), "cost=", "{:.9f}".format(avg_cost))
    
print("Accuracy", accuracy.eval(session=sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

r=random.randint(0,mnist.test.num_examples -1)
print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
print("Prediction:", sess.run(tf.arg_max(hypothesis,1), feed_dict={X:mnist.test.images[r:r+1]}))
#NN이랑 똑같이.. dropout을 쓸 수도 있고, Fully Connected 층을 늘릴 수도 있다!!


plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
plt.show()




#%% Class, Layers, Ensemble로 CNN하기

''' Class '''


import tensorflow as tf
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100


class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.keep_prob = tf.placeholder(tf.float32)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])
            # img 28x28x1 (black/white)
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # L1 ImgIn shape=(?, 28, 28, 1)
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            #    Conv     -> (?, 28, 28, 32)
            #    Pool     -> (?, 14, 14, 32)
            L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
            '''
            Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
            Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
            Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
            Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
            '''

            # L2 ImgIn shape=(?, 14, 14, 32)
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            #    Conv      ->(?, 14, 14, 64)
            #    Pool      ->(?, 7, 7, 64)
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
            '''
            Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
            Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
            Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
            Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
            '''

            # L3 ImgIn shape=(?, 7, 7, 64)
            W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            #    Conv      ->(?, 7, 7, 128)
            #    Pool      ->(?, 4, 4, 128)
            #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                                1, 2, 2, 1], padding='SAME')
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)

            L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])
            '''
            Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
            Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
            Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
            Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
            Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
            '''

            # L4 FC 4x4x128 inputs -> 625 outputs
            W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([625]))
            L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
            L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)
            '''
            Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
            Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
            '''

            # L5 Final FC 625 inputs -> 10 outputs
            W5 = tf.get_variable("W5", shape=[625, 10],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([10]))
            self.logits = tf.matmul(L4, W5) + b5
            '''
            Tensor("add_1:0", shape=(?, 10), dtype=float32)
            '''

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})

    #%%
    
    ''' Class로 만든 모델 시험해보기!! 렉걸려...'''
    
# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))


#%%

'''Layers'''


L1= tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
# 위와 아래는 똑같은 뜻이다!! tf.layer를 사용해서 더 쉽게 사용함!!
conv1= tf.layers.conv2d(inputs=X_img, filter=32, kerner_size=[3,3], padding='SAME', activation=tf.nn.relu)

L1=tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
pool1= tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], padding='SAME', strides=2)

L1=tf.nn.dropout(L1, keep_prob=self.keep_prob)
dropout1= tf.layers.dropout(inputs=pool1, rate=0.7, training=self.training)
#training 대신에 testing을 쓰면 자동으로 레이트가 1로 바뀐다.

conv2=tf.layers.conv2d(inputs=dropout1,............)#위랑 똑같은 방식으로..

#fully connected 레이어도... 사이즈가 얼마인가 계산할 필요없이
dense4=tf.layers.dense(inputs=flat, units=625, activiation=tf.nn.relu) #unit=> 몇개를 출력할 것인지
#이 함수를 쓰면 알아서 다 해준다!! 웨이트의 계산이랑 다 할 필요 없음!!!


#%%

''' Ensemble'''

#모델을 하나만 학습시키는 것이 아니라.. 여러개의 모델을 학습시키는 것이다!!
# 각각의 모델들이 예측한 값을 합해서.. 가장 높게 나온 애로 예측한다(출력)

# For example....

tf.reset_default_graph()
sess=tf.Session()
# 모델 만들어놓고
models = []
num_models = 2
for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))
    
sess.run(tf.global_variables_initializer())

#모델한테 학습을 시킴!!!
batch_size= 50  #렉걸리니까... 각각 50, 10으로 둠...ㅜ
total_batch=10
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
#    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # train each model
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)



#이제 각각의 모델로 앙상블 ㄱㄱ!!
test_size=len(mnist.test.labels)
predictions= np.zeros(test_size*10).reshape(test_size,10)  #0으로 된 벡터 만들고..

for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy:', m.get_accuracy(mnist.test.images, mnist.test.labels))
    p=m.predict(mnist.test.images)
    predictions+=p #예측값을 계속 더해주고.. 밑에서 제일 큰 값을 골라낸다   
ensemble_correct_prediction= tf.equal(tf.argmax(predictions,1), tf.argmax(mnist.test.labels,1))
ensemble_accuracy=tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))













