import math, timeit
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops


def create_placeholders(n_x, n_y):
    """
    create placeholders for input and output
    :param n_x: size of input
    :param n_y: size of output
    :return:
    """
    X = tf.placeholder(tf.float32, shape=[n_x, None], name="X")
    Y = tf.placeholder(tf.float32, shape=[n_y, None], name="Y")
    return X, Y


def initialize_parameters(layer_dims, l2_lambda):
    """
    Initializes parameters to build a neural network
    """
    tf.set_random_seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network
    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable('W' + str(l), [layer_dims[l], layer_dims[l-1]],
                                                   initializer=tf.contrib.layers.xavier_initializer(seed=1),
                                                   regularizer=tf.contrib.layers.l2_regularizer(scale=l2_lambda))
        parameters['b' + str(l)] = tf.get_variable('b' + str(l), [layer_dims[l], 1],
                                                   initializer=tf.zeros_initializer())
    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation
    """
    L = len(parameters)//2
    cache = {}
    for i in range(1, L+1):
        if i == 1:
            cache["Z"+str(i)] = tf.add(tf.matmul(parameters["W"+str(i)], X), parameters["b"+str(i)])
        else:
            cache["Z" + str(i)] = tf.add(tf.matmul(parameters["W" + str(i)],
                                                   cache["A"+str(i-1)]), parameters["b" + str(i)])
        cache["A"+str(i)] = tf.nn.relu(cache["Z"+str(i)])
    print cache["Z"+str(L)]
    return cache["Z"+str(L)]


def compute_cost(ZL, Y):
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[1]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def tf_model(X_train, Y_train, layer_dims, X_test=None, Y_test=None, l2_lambda=0.1, learning_rate = 0.0001,
             num_epochs=1500, print_cost=True, device_name="cpu"):

    if device_name == "cpu":
        device_name = "/cpu:0"
    else:
        device_name = "/gpu:0"

    with tf.device(device_name):
        ops.reset_default_graph()
        tf.set_random_seed(1)
        (n_x, m) = X_train.shape
        n_y = Y_train.shape[0]
        costs = []

        X, Y = create_placeholders(n_x, n_y)
        parameters = initialize_parameters(layer_dims, l2_lambda)
        ZL = forward_propagation(X, parameters)
        cross_entropy_cost = compute_cost(ZL, Y)
        l2_reg_cost = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_cost = cross_entropy_cost + tf.reduce_sum(l2_reg_cost)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_cost)
        init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        start = timeit.default_timer()
        for epoch in range(num_epochs):
            epoch_cost = 0.0
            # num_minibatches = int(m/minibatch_size)
            # seed = seed + 1
            # minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            # for minibatch in minibatches:
                # Select a minibatch
                # (minibatch_X, minibatch_Y) = minibatch
            _, minibatch_cost = sess.run([optimizer, total_cost], feed_dict={X: X_train, Y: Y_train})
            epoch_cost += minibatch_cost
            if print_cost is True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost is True and epoch % 5 == 0:
                costs.append(epoch_cost)
            if epoch_cost <= 0.01:
                break
        stop = timeit.default_timer()
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        Y_hat = tf.greater(ZL, 0.5)
        correct_prediction = tf.equal(Y_hat, tf.equal(Y,1.0))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Device name: {}\nTraining time: {} secs".format(device_name,stop-start))
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        if X_test is not None:
            print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

    return parameters
