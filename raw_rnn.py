# -*-coding:utf-8 -*-

"""
    理解RNN过程
    真是值得反复品味
"""

import numpy as np

# set random seed
np.random.seed(142857)


# define sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))


# define derivative of sigmoid
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


# define the map between int and binary
# Only to calculate the subtraction in 256
int2binary = {}
binary_dim = 8
largest_num = pow(2, binary_dim)
binary = np.unpackbits(np.array([range(largest_num)], dtype=np.uint8).T, axis=1)
for i in range(largest_num):
    int2binary[i] = binary[i]


# define params
alpha = 0.9
input_dim = 2
hidden_dim = 16
output_dim = 1

# initialize network
synapse_0 = (2*np.random.random((input_dim, hidden_dim)) - 1)*0.05
synapse_1 = (2*np.random.random((hidden_dim, output_dim)) - 1)*0.05
synapse_h = (2*np.random.random((hidden_dim, hidden_dim)) - 1)*0.05

# store the updated weights and bias
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# train
for j in range(10000):
    # create number to subtract
    a_int = np.random.randint(largest_num)
    b_int = np.random.randint(largest_num)

    if a_int < b_int:
        t_int = b_int
        b_int = a_int
        a_int = t_int

    # get binary expression
    a = int2binary[a_int]
    b = int2binary[b_int]

    # right answer
    c_int = a_int - b_int
    c = int2binary[c_int]

    # store prediction
    d = np.zeros_like(c)
    overall = 0

    # store the data created in the process
    layer_2_deltas = list()
    layer_1_value = list()

    # initialize the hidden value
    layer_1_value.append(np.ones(hidden_dim) * 0.1)

    # propagation
    for position in range(binary_dim):
        # get data from the opposite direction
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        # label
        y = np.array([[c[binary_dim - position - 1]]]).T
        # hidden layer
        # output add hidden value
        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_value[-1], synapse_h))
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))
        # error
        layer_2_error = y - layer_2
        layer_2_deltas.append(layer_2_error * sigmoid_output_to_derivative(layer_2))
        overall += np.abs(layer_2_error[0])
        # prediction
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        # store updated hidden value
        layer_1_value.append(layer_1)

    future_layer_1_delta = np.zeros(hidden_dim)

    # backpropagation
    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])
        layer_1 = layer_1_value[-position-1]
        prev_layer_1 = layer_1_value[-position-2]

        layer_2_delta = layer_2_deltas[-position-1]
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) +
                         layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # update
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

        synapse_0 += synapse_0_update * alpha
        synapse_1 += synapse_1_update * alpha
        synapse_h += synapse_h_update * alpha

        synapse_0_update *= 0
        synapse_1_update *= 0
        synapse_h_update *= 0

    if j%800 == 0:
        print(a, b, c, d, sum(np.equal(c, d)) == 8)