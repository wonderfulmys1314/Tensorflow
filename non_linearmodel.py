# -*- coding:UTF-8 -*-

"""
    多层神经网络实现非线性问题求解
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


# generate data for train
def generate(num_classes, sample_size, mean, cov, diff):
    """
    :param sample_size: the number of node to be generated
    :param mean: the mean of all node
    :param cov: the covariance of the data
    :param diff: change the mean according to the class
    :param regression:
    :return:
    """
    samples_per_class = int(sample_size / num_classes)
    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)

    for c, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean+d, cov, samples_per_class)
        Y1 = (c+1) * np.ones(samples_per_class)

        X0 = np.concatenate((X1, X0))
        Y0 = np.concatenate((Y1, Y0))

    index = list(range(samples_per_class * num_classes))
    np.random.shuffle(index)
    X0 = X0[index]
    Y0 = Y0[index]

    return X0, Y0


# generate data
np.random.seed(142857)
num_classes = 2
mean = np.random.randn(num_classes)
cov = np.eye(num_classes)
sample_size = 200
diff = [3.0]
X, Y = generate(num_classes, sample_size, mean, cov, diff)
# have a outline about sample
colors = []
for i in Y:
    if i == 0:
        colors.append('b')
    else:
        colors.append('r')
# # # plt.scatter(X[:, 0], X[:, 1], c=colors)
# # # plt.xlabel("X")
# # # plt.ylabel("Y")
# # # plt.show()
# #
# # create network
input_dim = 2
label_dim = 1
input_feature = tf.placeholder(tf.float32, [None, input_dim])
input_labels = tf.placeholder(tf.float32, [None, label_dim])
#
w = tf.Variable(tf.random_normal([input_dim, label_dim]), name="weight")
b = tf.Variable(tf.zeros(label_dim), name="bias")
#
output = tf.nn.sigmoid(tf.add(tf.matmul(input_feature, w), b))
cross_entropy = -(input_labels * tf.log(output) + (1 - input_labels) * tf.log(1 - output))
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.04)
train = optimizer.minimize(loss)
#
# # # train
maxEpoches = 50
minBatchSize = 25
Y = np.reshape(Y, [sample_size, 1])
# #
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
#
    for epoch in range(maxEpoches):
        for i in range(np.int32(len(Y)/minBatchSize)):
            x1 = X[i*minBatchSize: (i+1)*minBatchSize, :]
            y1 = np.reshape(Y[i*minBatchSize: (i+1)*minBatchSize], [-1, 1])
            _, lossval, outputval = sess.run([train, loss, output],
                                             feed_dict={input_feature: x1, input_labels: y1})
        #
        print("Epoch:", '%d'%(epoch+1), "cost=", "{0}".format(sess.run(loss, feed_dict={input_feature: X, input_labels: Y})))
