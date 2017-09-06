from nn_toolkit import *

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    np.random.seed(10)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]


    for i in range(num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, activation="tanh")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")

        cost = compute_cost(A2, Y)

        dA2 = -(np.divide(Y, A2) - np.divide(1-Y, 1-A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="tanh")

        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2
        # print "X: {}\n Y: {}".format(X, Y)
        # print "W1: {}\nb1: {}\nW2: {}\nb2: {}".format(W1, b1, W2, b2)
        # print "Z1 = W1X+b1 = {}".format(np.dot(W1, X)+b1)
        # print "Z2 = W2A1+b2 = {}".format(np.dot(W2, A1)+b2)
        #
        # print "A1: {}\nA2: {}\nCost: {}".format(A1, A2, cost)
        #
        # print "dW1: {}\ndb1: {}\ndW2: {}\ndb2: {}".format(dW1, db1, dW2, db2)
        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]


        if print_cost and i % 500 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 500 == 0:
            costs.append(cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters