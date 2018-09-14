# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
tf.reset_default_graph()

#创建输入数据
X = np.random(2,4,5)

#第二个样本长度为3
X[1,1:] = 0
seq_lengths = [4,1]

#分别建立一个LStm与GRU的cell,比较输入状态
