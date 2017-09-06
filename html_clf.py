from neural_net_from_scratch.L_layer_model import L_layer_model
import pickle
from neural_net_from_scratch.nn_toolkit import *
from sklearn.model_selection import train_test_split

XT_obj = open("data/XT_html_vs_another_letters.pkl",'rb')
X_t = pickle.load(XT_obj)
XT_obj.close()
XF_obj = open("data/XF_html_vs_another_letters.pkl",'rb')
X_f = pickle.load(XF_obj)
XF_obj.close()
YT_obj = open("data/YT_html_vs_another_letters.pkl",'rb')
Y_t = pickle.load(YT_obj)
YT_obj.close()
YF_obj = open("data/YF_html_vs_another_letters.pkl",'rb')
Y_f = pickle.load(YF_obj)
YF_obj.close()

m = np.min([X_t.shape[1], X_f.shape[1]])
X = np.concatenate((X_t[:, :m+1], X_f[:, :m+1]), axis=1)
Y = np.concatenate((Y_t[:, :m+1], Y_f[:, :m+1]), axis=1)


mask_row = np.all(X == 0, axis=1)
X = X[~mask_row, :]    # eliminate rows with all zeros
X, unique_ind = np.unique(X, return_index=True, axis=1)
Y = Y[:, unique_ind]
mask_col = np.all(X == 0, axis=0)
X = X[:, ~mask_col]    # eliminate cols with all zeros
Y = Y[:, ~mask_col]

X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size=0.15, random_state=42)
X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.T
Y_test = Y_test.T

MU = np.average(X_train, axis=1).reshape((X.shape[0], 1))
X_train = X_train-MU
SIGMA_2 = ((1.0/X.shape[1])*np.sum(X_train*X_train, axis=1, keepdims=True))
X_train = X_train/SIGMA_2

layers_dims = [X.shape[0], 5, 1]
para = L_layer_model(X_train[:, :], Y_train[:, :], layers_dims, learning_rate=0.001, num_iterations=10000000, print_cost=True, lambd=0.5)
print ("Prediction on the training set")
predict(X_train, Y_train, para)

# X_test = X_test - MU
# X_test = X_test/SIGMA_2
# print ("Prediction on the test set")
# predict(X_test, Y_test, para)
#
# ## data from gist
# XT_obj = open("data/XT_html_vs_another_gist.pkl",'rb')
# X_t_gist = pickle.load(XT_obj)
# XT_obj.close()
# XF_obj = open("data/XF_html_vs_another_gist.pkl",'rb')
# X_f_gist = pickle.load(XF_obj)
# XF_obj.close()
# YT_obj = open("data/YT_html_vs_another_gist.pkl",'rb')
# Y_t_gist = pickle.load(YT_obj)
# YT_obj.close()
# YF_obj = open("data/YF_html_vs_another_gist.pkl",'rb')
# Y_f_gist = pickle.load(YF_obj)
# YF_obj.close()
# X_gist = np.concatenate((X_t_gist, X_f_gist), axis=1)
# Y_gist = np.concatenate((Y_t_gist, Y_f_gist), axis=1)
# X_gist = X_gist-MU
# X_gist = X_gist/SIGMA_2
# print ("Prediction on gist")
# predict(X_gist, Y_gist, para)