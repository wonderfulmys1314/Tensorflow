# -*- coding:utf-8 -*-
"""
    通过tensorflow完成rnn
"""

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

tf.set_random_seed(2020)
np.random.seed(2020)

# 设置参数
num_epochs = 5
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length // batch_size // truncated_backprop_length


# 生成数据
def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0: echo_step] = 0

    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))

    return x, y


# 输入数据设置
# shape设置为（5，15）
batchX_placeholder = tf.placeholder(tf.float32, shape=[batch_size, truncated_backprop_length])
bacthY_placeholder = tf.placeholder(tf.int32, shape=[batch_size, truncated_backprop_length])
# shape设置为（5,4）
# 初始记忆
init_state = tf.placeholder(tf.float32, [batch_size, state_size])

# 对输入数据进行切分，用于序列输入
input_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(bacthY_placeholder, axis=1)


current_state = init_state
predictions_series = []
losses = []

# 读取数据
for current_input, labels in zip(input_series, labels_series):
    # 输入，以前为单个，现在批处理
    current_input = tf.reshape(current_input, [batch_size, 1])
    # 添加记忆单元
    input_and_state_concatenated = tf.concat([current_input, current_state], 1)
    # 全连接
    next_state = tf.contrib.layers.fully_connected(input_and_state_concatenated, state_size,
                                                   activation_fn=tf.tanh)
    # 更新记忆状态
    current_state = next_state

    # 计算输出
    logits = tf.contrib.layers.fully_connected(next_state, num_classes, activation_fn=None)

    # 计算损失
    # 稀疏化求交叉熵
    # 标签为单值, 输入为全连接层的输出
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    # 添加损失
    losses.append(loss)

    # 预测概率
    predictions = tf.nn.softmax(logits)
    predictions_series.append(predictions)


# 绘图
def plot(_predictions_series, batchX, batchY):
    # 设置第一张子图
    plt.subplot(2, 3, 1)
    # clear current axe
    plt.cla()
    # plt.plot()

    for batch_series_idx in range(batch_size):
        # 取出每一行
        one_hot_out_series = np.array(_predictions_series)[:, batch_series_idx, :]
        # 判断每一行是0还是1
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_out_series])
        # 设置第二张子图
        plt.subplot(2, 3, batch_series_idx + 2)
        # 清除坐标信息
        plt.cla()
        # 横纵坐标
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        left_offset2 = range(echo_step, truncated_backprop_length + echo_step)

        label1 = "past values"
        label2 = "True echo value"
        label3 = "Predictions"

        plt.plot(left_offset2, batchX[batch_series_idx, :]*0.2+1.5, "o--b", label=label1)
        plt.plot(left_offset, batchY[batch_series_idx, :]*0.2+0.8, "x--b", label=label2)
        plt.plot(left_offset, single_output_series*0.2+0.1, "o--y", label=label3)

    plt.legend(loc="best")
    plt.draw()
    plt.pause(0.0001)


# 总损失
total_loss = tf.reduce_mean(losses)
# 不同的优选器也会产生差异
# 不妨试试Adam，图像没有比较过
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


# 启动会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 打开交互功能
    plt.ion()
    # 参数设置
    plt.figure()
    loss_list = []

    for epoch_idx in range(num_epochs):
        # 生成数据 shape=(5, 12)
        x, y = generateData()
        # 初始化记忆
        _current_state = np.zeros((batch_size, state_size))
        # 打印信息
        print("New data, epoch", epoch_idx)

        # 每次训练5个，每个长度为15，一共分为10000/15批
        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:, start_idx: end_idx]
            batchY = y[:, start_idx: end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder: batchX,
                    bacthY_placeholder: batchY,
                    init_state: _current_state
                }
            )

            # 增加总损失
            loss_list.append(_total_loss)

            # 打印损失
            if batch_idx%100 == 0:
                print("Step", batch_idx, "Loss", _total_loss)
                plot(_predictions_series, batchX, batchY)
    plt.ioff()
    plt.show()

