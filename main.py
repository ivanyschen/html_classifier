from two_layer_model import two_layer_model
from L_layer_model import L_layer_model
from html_processing import *

lines = file_to_lines("data/htmls", "1.html")
X_t, Y_t, X_f, Y_f = lines_to_vec(lines)
m = np.min([X_t.shape[1], X_f.shape[1]])
X= np.concatenate((X_t[:, :m+1], X_f[:, :m+1]), axis=1)
Y = np.concatenate((Y_t[:, :m+1], Y_f[:, :m+1]), axis=1)
mask = np.all(X == 0, axis=1)
X = X[~mask]
X = X - np.average(X, axis=1).reshape((X.shape[0], 1))
X = X/((1.0/X.shape[1])*np.sum(X*X,axis=1, keepdims=True))

# para = two_layer_model(X[:, :], Y[:, :], (X.shape[0], 5, 1), num_iterations = 100000, print_cost=True)
layers_dims = [X.shape[0], 5, 1]
para = L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=100000, print_cost=True)
