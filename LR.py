# -*- coding:utf-8 -*-

"""
    通过TensorFlow实现线性回归
    模型重载、查看模型文件内容
    TensorBoard
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


# create data
def create(value, num):
    """
    value: define the scope of x
    return:
        create some data with noise
    """
    np.random.seed(142857)
    train_x = np.linspace(-value, value, num)
    train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.3
    return train_x, train_y
    #
    # plt.plot(train_x, train_y, 'ro', label='Original data')
    # plt.legend()
    # plt.show()


# test the function of create
# create(1, 100)


# create model and train
def model(train_x, train_y):
    # define input and output
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    # the params of the weights and bias
    w = tf.Variable(tf.random_normal([1]), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")
    Z = tf.multiply(X, w) + b
    tf.summary.histogram('Z', Z)
    cost = tf.reduce_mean(tf.square(Y-Z))
    tf.summary.scalar('cost', cost)
    learning_rate = 0.01
    # choose the way to update the weights and bias
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # start session
    init_op = tf.global_variables_initializer()
    training_epochs = 20
    display_step = 2
    saver = tf.train.Saver(max_to_keep=5)
    save_path = "log/"
    load_epoch = 18

    # begin
    with tf.Session() as sess:
        sess.run(init_op)
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("summary", sess.graph)
        plotdata = {"batchsize": [], "loss": []}

        for epoch in range(training_epochs):
            for (x, y) in zip(train_x, train_y):
                sess.run(optimizer, feed_dict={X: x, Y: y})

            if epoch % display_step == 0:
                loss = sess.run(cost, feed_dict={X: train_x, Y: train_y})
                print("Epoch:", epoch+1, "cost=", loss, "W=", sess.run(w), "b=", sess.run(b))
                if loss is not "NA":
                    plotdata['loss'].append(loss)
                    plotdata['batchsize'].append(epoch)
            # use python library called matplotlib to make a draw about the change of cost
            saver.save(sess, save_path + "linearmodel.cpkt", global_step=epoch)
            summary_all = sess.run(merged, feed_dict={X: train_x, Y: train_y})
            summary_writer.add_summary(summary_all, epoch)
        plt.plot(plotdata['batchsize'], plotdata['loss'])
        plt.show()

    print("Finished project 1")

    # load model from store file
    with tf.Session() as sess2:
        sess2.run(tf.global_variables_initializer())
        saver.restore(sess2, save_path+"linearmodel.cpkt-" + str(load_epoch))
        print("x=0.2 z=", sess2.run(Z, feed_dict={X: 0.2}))


# view detail about params stored in .cpkt file
def view_cpkt(load_epoch):
    save_dir = "log/"
    print_tensors_in_checkpoint_file(save_dir+"linearmodel.cpkt-" + str(load_epoch), None, True)


# achieve
train_x, train_y = create(1, 100)
model(train_x, train_y)
view_cpkt(18)
