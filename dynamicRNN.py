# -*- coding:utf-8 -*-

"""
    动态RNN处理边长
"""
import tensorflow as tf
import numpy as np
np.random.seed(1)

# 重置图
tf.reset_default_graph()
# 创建输入数据

X = np.random.randn(2, 4, 5)
# 第二个样本长度为1
X[1, 1:] = 0
# 输入序列长度
seq_lengths = [4, 1]

# 创建LSTM和GRU的cell
cell = tf.contrib.rnn.BasicLSTMCell(num_units=3, state_is_tuple=True)
gru = tf.contrib.rnn.GRUCell(1)

# 指定输入类型
outputs, last_states = tf.nn.dynamic_rnn(cell, X, seq_lengths, dtype=tf.float64)
gruoutputs, grulast_states = tf.nn.dynamic_rnn(gru, X, seq_lengths, dtype=tf.float64)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
result, sta, gruout, grusta = sess.run([outputs, last_states, gruoutputs, grulast_states])

# 打印信息
print("LSTM全序列:", result[0])
print("LSTM短序列:", result[1])
print("LSTM全状态:", sta[0])
print("LSTM短状态:", sta[1])
print("GRU全序列:", gruout[0])
print("GRU短序列:", gruout[1])
print("GRU全序列状态:", grusta)
print("GRU短序列状态", grusta[1])