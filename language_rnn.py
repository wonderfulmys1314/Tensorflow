# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import random
import time
from collections import Counter

# 起始时间
start_time = time.time()


# 时间显示
def elapsed(sec):
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60 * 60)) + " hour"


# 重置计算图
tf.reset_default_graph()
# 文本地址
training_file = 'data/words'


# 处理多个中文文件
# 生成标签
def readalltxt(txt_files):
    labels = []
    for txt_file in txt_files:
        target = get_ch_label(txt_file)
        labels.append(target)
    return labels


# 处理汉字
# 好像就是编码处理
def get_ch_label(txt_file):
    labels = ""
    with open(txt_file, 'rb') as f:
        for label in f:
            labels = labels + label.decode('utf-8')

    return labels


# 转换文字为向量
def get_ch_label_v(txt_file, word_num_map, txt_label=None):
    words_size = len(word_num_map)
    # 函数
    to_num = lambda word: word_num_map.get(word, words_size)
    # 转码
    if txt_file is not None:
        txt_label = get_ch_label(txt_file)
    # map函数
    labels_vector = list(map(to_num, txt_label))
    return labels_vector


# 样本预处理
training_data = get_ch_label(training_file)
print(training_data)
print("Loaded training data...")
counter = Counter(training_data)
# print(counter)
words = sorted(counter)
print(words)
words_size = len(words)
# 创建字典
word_num_map = dict(zip(words, range(words_size)))
print("字体大小：", words_size)
# 转换为向量(序数)
wordlabel = get_ch_label_v(training_file, word_num_map)
print(wordlabel)

# 搭建模型
learning_rate = 0.001
training_iters = 10000
display_step = 1000
n_input = 4

# 隐藏变量
n_hidden1 = 256
n_hidden2 = 512
n_hidden3 = 512

# 占位符
# (batch, max_time, num)
x = tf.placeholder("float", [None, n_input, 1])
wordy = tf.placeholder("float", [None, words_size])

# 定义网络结构
x1 = tf.reshape(x, [-1, n_input])
# 按时序切割
x2 = tf.split(x1, n_input, 1)
# 网络结构
rnn_cell = rnn.MultiRNNCell([rnn.LSTMCell(n_hidden1), rnn.LSTMCell(n_hidden2), rnn.LSTMCell(n_hidden3)])
# 得到输出
outputs, states = rnn.static_rnn(rnn_cell, x2, dtype=tf.float32)

# 全连接
pred = tf.contrib.layers.fully_connected(outputs[-1], words_size, activation_fn=None)

# 定义loss和优选器
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=wordy))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 模型评估
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(wordy, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 训练
save_dir = "log/rnnword/"
saver = tf.train.Saver(max_to_keep=1)

# 启动Session
with tf.Session() as sess:
    # 初始化
    sess.run(tf.global_variables_initializer())
    # 步骤
    step = 0
    # 偏置
    offset = random.randint(0, n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    # 最后一次训练节点
    kpt = tf.train.latest_checkpoint(save_dir)
    print("kpt:", kpt)
    # 轮次
    startepoch = 0
    # 是否加载训练节点
    if kpt is not None:
        saver.restore(sess, kpt)
        ind = kpt.find("-")
        startepoch = int(kpt[ind+1:])
        step = startepoch

    while step < training_iters:
        # 如果取到末尾则回到最初的原点
        if offset > (len(training_data) - end_offset):
            offset = random.randint(0, n_input+1)
        # 输入汉字
        inwords = [[wordlabel[i] for i in range(offset, offset+n_input)]]
        inwords = np.reshape(np.array(inwords), [-1, n_input, 1])

        # one hot处理，制作标签
        out_onehot = np.zeros([words_size], dtype=float)
        out_onehot[wordlabel[offset+n_input]] = 1.0
        out_onehot = np.reshape(out_onehot, [1, -1])

        _, acc, lossval, onehot_pred = sess.run([optimizer, accuracy, loss, pred], feed_dict={x: inwords, wordy: out_onehot})
        loss_total += lossval
        acc_total += acc

        if (step+1) % display_step == 0:
            print("Iter=" + str(step+1) + ", Average Loss=" +
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " +
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            in2 = [words[wordlabel[i]] for i in range(offset, offset+n_input)]
            out2 = words[wordlabel[offset + n_input]]
            out_pred = words[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (in2, out2, out_pred))
            saver.save(sess, save_dir+"rnnwordtest.ckpt", global_step=step)
        step += 1
        offset += (n_input + 1)

    print("Finished!")
    saver.save(sess, save_dir + "rnnwordtest.ckpt", global_step=step)
    print("Elapsed time:", elapsed(time.time() - start_time))

    while True:
        prompt = "请输入%s个字: " % n_input
        sentence = input(prompt)
        inputword = sentence.strip()

        if len(inputword) != n_input:
            print("您输入的字符长度为：", len(inputword), "请输入4个字")
            continue
        try:
            inputword = get_ch_label_v(None, word_num_map, inputword)

            for i in range(32):
                keys = np.reshape(np.array(inputword), [-1, n_input, 1])
                onehot_pred = sess.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s%s" % (sentence, words[onehot_pred_index])
                inputword = inputword[1:]
                inputword.append(onehot_pred_index)
            print(sentence)
        except:
            print("该字我还没学会")


