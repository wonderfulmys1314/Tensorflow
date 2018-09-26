# -*- coding:utf-8 -*-

"""
    构建双向RNN模型
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

# 读取手写体数据
mnist = input_data.read_data_sets("/data/", one_hot=True)

# 定义参数
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# 网络参数模型
n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

# 重置网络
tf.reset_default_graph()

# 定义占位符
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# 拆分成列表形式
x1 = tf.unstack(x, n_steps, 1)

# 正向LSTM
lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
# 反向LSTM
lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
print(len(outputs), outputs[0].shape, outputs[1].shape)

# 输出处理
outputs = tf.concat(outputs, 2)
outputs = tf.transpose(outputs, [1, 0, 2])

# 预测结果
pred = tf.contrib.layers.fully_connected(outputs[-1], n_classes, activation_fn=None)
# 损失
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# 优化
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# 准确率
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # 计算批次数据的准确率
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print(loss)
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print(" Finished!")


