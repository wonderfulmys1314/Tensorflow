# -*-coding:utf-8 -*-

import tensorflow as tf
import numpy as np

# create data for simulation
img = tf.Variable(tf.constant(1.0, shape=[1, 4, 4, 1]))
# create filter
filter = tf.Variable(tf.constant([1.0, 0, -1, -2], shape=[2, 2, 1, 1]))
# two operations to deal with img.One is VALID while another is SAME
# achieve convolution
conv = tf.nn.conv2d(img, filter, strides=[1, 2, 2, 1], padding="VALID")
cons = tf.nn.conv2d(img, filter, strides=[1, 2, 2, 1], padding="SAME")
print(conv.shape)
print(cons.shape)
# create the transpose of convolution
contv = tf.nn.conv2d_transpose(conv, filter, [1, 4, 4, 1], strides=[1, 2, 2, 1], padding="VALID")
conts = tf.nn.conv2d_transpose(cons, filter, [1, 4, 4, 1], strides=[1, 2, 2, 1], padding="SAME")

# start session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("conv:\n", sess.run([conv, filter]))
    print("cons:\n", sess.run([cons]))
    print("contv:\n", sess.run([contv]))
    print("conts:\n", sess.run([conts]))