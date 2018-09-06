from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import pickle as pc
import os

path=os.getcwd()+'/testing_TF/pickle_data.txt'

with open(path, 'r') as f:
	r_data=pc.load(f)


trX=r_data
#trX=np.matrix(r_data, dtype=float)
trY=np.zeros((9,9))

#trX = np.matrix(([3,5], [5,1],[10,2]), dtype=float)
#trY = np.matrix(([85], [82], [93]), dtype=float) # 3X1 matrix

# koi faida nahi
learning_rate = 0.01
training_epochs = 1000
display_step = 50


#n_samples = trX.shape[0]


# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
Wh = tf.Variable(tf.random_normal([2,20]))
Wo = tf.Variable(tf.random_normal([20,1]))


def model(X, w_h, w_o):
    z2 = tf.matmul(X, w_h)
    a2 = tf.nn.sigmoid(z2) 
    z3 = tf.matmul(a2, w_o)
    yHat = tf.nn.sigmoid(z3)
    return yHat 
'''
def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us
'''


py_x = model(X, Wh, Wo)

cost = tf.reduce_mean(tf.square(py_x - Y))
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost) # construct an optimizer
predict_op = py_x

init = tf.initialize_all_variables()

'''
sess.run(tf.initialize_all_variables())


sess.run(train_op, feed_dict={X: trX, Y: trY})

print sess.run(predict_op, feed_dict={X: trX})

sess.close()


pred = tf.add(tf.mul(X, W), b)


# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
'''


with tf.Session() as sess:
	sess.run(init)
    	#for i in range(n_samples):
	for (x, y) in zip(trX, trY):
		print(x,y)
            	sess.run(train_op, feed_dict={X: x, Y: y})



'''
for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
         print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX, Y: teY})))

'''
