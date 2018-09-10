# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

# create queue
queue = tf.FIFOQueue(100, "float")

# initialize a variable
c = tf.Variable(0.0)
# operation to do with c
op = tf.assign_add(c, tf.constant(1.0))
# put the result into queue
enqueue_op = queue.enqueue(c)
# create a manager for queue
qr = tf.train.QueueRunner(queue, enqueue_ops=[op, enqueue_op])
# start session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    # create threads
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    # main thread
    for i in range(0, 10):
        print("----------------------")
        print(sess.run(queue.dequeue()))

    coord.request_stop()

# achieve one hot code
labels = list(np.arange(0, 10))
one_hot_label = np.eye(10, dtype=float)[labels]
print(one_hot_label)