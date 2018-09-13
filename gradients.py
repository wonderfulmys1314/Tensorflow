# -*- coding:utf-8 -*-

import tensorflow as tf

# set default graph
tf.reset_default_graph()

# initialize weights and bias
w1 = tf.get_variable("w1", shape=[2])
w2 = tf.get_variable("w2", shape=[2])
w3 = tf.get_variable("w3", shape=[2])
w4 = tf.get_variable("w4", shape=[2])

# calculate the result
y1 = w1 + w2 + w3
y2 = w3 + w4
# stop the gradients by w2 and w4
y3 = tf.stop_gradient(w2 + w4) + w3


# define the gradients
gradients = tf.gradients([y1, y2, y3], [w1, w2, w3, w4],
                         grad_ys=[tf.convert_to_tensor([1, 2]),
                         tf.convert_to_tensor([3, 4]), tf.convert_to_tensor([3, 4])])

# start session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(gradients))
