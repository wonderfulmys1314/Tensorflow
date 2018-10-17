# -*- coding:utf-8 -*-

import random
import math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 依据批次村叠加余弦与正弦
def do_generate_x_y(isTrain, batch_size, seqlen):
    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() + 0.1

        # 起始点、终止点、连接点数
        sin_data = amp_rand * np.sin(np.linspace(
            offset_rand, seqlen / 15.0 * freq_rand * 3.0 * math.pi + offset_rand, seqlen * 2))

        offset_rand = random.random() * 2 * math.pi
        freq_rand = (random.random() - 0.5) / 1.5 * 15 + 0.5
        amp_rand = random.random() * 1.2

        sin_data = amp_rand * np.cos(np.linspace(
            offset_rand, seqlen / 15.0 * freq_rand * 3.0 * math.pi + offset_rand, seqlen * 2)) + sin_data

        # [15, 1]
        batch_x.append(np.array([sin_data[: seqlen]]).T)
        batch_y.append(np.array([sin_data[seqlen:]]).T)

    # [15, 3, 1]
    batch_x = np.array(batch_x).transpose((1, 0, 2))
    batch_y = np.array(batch_y).transpose((1, 0, 2))

    return batch_x, batch_y


# 生成叠加曲线
def generate_data(isTrain, batch_size):
    seq_length = 15
    if isTrain:
        return do_generate_x_y(isTrain, batch_size, seqlen=seq_length)
    else:
        return do_generate_x_y(isTrain, batch_size, seqlen=seq_length*2)


# 生成叠加曲线
sample_now, sample_f = generate_data(isTrain=True, batch_size=3)
print("training examples:")
print(sample_now.shape)
print("(seq_length, batch_size, output_dim)")
seq_length = sample_now.shape[0]
batch_size = 10
output_dim = input_dim = sample_now.shape[-1]
hidden_dim = 12
layers_stacked_count = 2

# 学习率
learning_rate = 0.04
nb_iters = 100
lambda_l2_reg = 0.003
tf.reset_default_graph()
encoder_input = []
expected_output = []
decode_input = []

for i in range(seq_length):
    # 编码器输入
    # [seq_length, batch_size, input_dim]
    encoder_input.append(tf.placeholder(tf.float32, shape=[None, input_dim]))
    # 期待输出
    # [seq_length, batch_size, output_dim]
    expected_output.append(tf.placeholder(tf.float32, shape=[None, output_dim]))
    # 解码器输入
    # [seq_length, batch_size, input_dim]
    decode_input.append(tf.placeholder(tf.float32, shape=[None, input_dim]))

tcells = []
# 增加循环结构
for i in range(layers_stacked_count):
    tcells.append(tf.contrib.rnn.GRUCell(hidden_dim))

# 连接多种rnn细胞
Mcell = tf.contrib.rnn.MultiRNNCell(tcells)
# 序列模型
dec_outputs, dec_memeory = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(
    encoder_input, decode_input, Mcell)

# 外接全连接模型
reshape_outputs = []
for ii in dec_outputs:
    reshape_outputs.append(tf.contrib.layers.fully_connected(ii, output_dim, activation_fn=None))

# 计算损失值
output_loss = 0
for _y, _Y in zip(reshape_outputs, expected_output):
    output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

# 正则化loss值
reg_loss = 0
for tf_val in tf.trainable_variables():
    if not ("fully_connected" in tf_val.name):
        reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_val))

loss = output_loss + lambda_l2_reg * reg_loss
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 建立会话
sess = tf.InteractiveSession()


# 批训练
def train_batch(batch_size):
    # [15, 3, 1]
    X, Y = generate_data(isTrain=True, batch_size=batch_size)
    # 字典输入，一一匹配
    feed_dict = {encoder_input[t]: X[t] for t in range(len(encoder_input))}
    feed_dict.update({expected_output[t]: Y[t] for t in range(len(expected_output))})
    # 初始状态，除了第一个状态为0，其他基本后移
    c = np.concatenate(([np.zeros_like(Y[0])], Y[:-1]), axis=0)
    feed_dict.update({decode_input[t]: c[t] for t in range(len(c))})
    # 计算损失
    _, loss_t = sess.run([train_op, loss], feed_dict)
    return loss_t


# 批测试
def test_batch(batch_size):
    X, Y = generate_data(isTrain=True, batch_size=batch_size)
    feed_dict = {encoder_input[t]: X[t] for t in range(len(encoder_input))}
    feed_dict.update({expected_output[t]: Y[t] for t in range(len(expected_output))})
    c = np.concatenate(([np.zeros_like(Y[0])], Y[:-1]), axis=0)

    feed_dict.update({decode_input[t]: c[t] for t in range(len(c))})
    output_lossv, reg_lossv, loss_t = sess.run([output_loss, reg_loss, loss], feed_dict)
    print("----------")
    print(output_lossv, reg_lossv)
    return loss_t


# 训练
train_losses = []
test_losses = []

# 启动会话
sess.run(tf.global_variables_initializer())
for t in range(nb_iters+1):
    train_loss = train_batch(batch_size)
    train_losses.append(train_loss)
    if t%50 == 0:
        test_loss = test_batch(batch_size)
        test_losses.append(test_loss)
        print("Step {}/{}, train loss: {}, \tTest loss: {}".format(t, nb_iters, train_loss, test_loss))
    # print("Fin. train loss: {}, \tTest loss: {}".format(train_loss, test_loss))

plt.figure(figsize=(12, 6))
plt.plot(np.array(range(0, len(test_losses))) /
         float(len(test_losses) - 1) * (len(train_losses) - 1),
         np.log(test_losses), label="Test loss")

plt.plot(np.log(train_losses), label="Train loss")
plt.title("Training errors over time (on a logarithmic scale)")
plt.xlabel('Iteration')
plt.ylabel('log(Loss)')
plt.legend(loc="best")
plt.show()






