import tensorflow as tf


a = tf.constant(4.0)
g = tf.Graph()

with g.as_default():
    m = tf.constant(3.0)
    print(a.graph)
    print(g)
    print(m.graph)

    d = tf.get_default_graph()
    print(d)




