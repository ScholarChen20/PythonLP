import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# index=7
# plt.imshow(train_x_orig[index]) #展示数据集中的图像
# print("y="+str(train_y[0,index])+".It's a "+classes[train_y[0,index]].decode("utf-8"))
m_train=train_x_orig.shape[0] #标记为cat（1）和非cat（0）图像的训练集m_train
num_px=train_x_orig.shape[1]
m_test=test_x_orig.shape[0] #标记为cat或non-cat图像的测试集m_test

#对图像进行重塑和标准化
train_x_flatten=train_x_orig.reshape(train_x_orig.shape[0],-1).T
test_x_flatten=test_x_orig.reshape(test_x_orig.shape[0],-1).T
train_x=train_x_flatten/255
test_x=test_x_flatten/255

def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(1)
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))
    assert(W1.shape==(n_h,n_x))
    assert(b1.shape==(n_h,1))
    assert(W2.shape==(n_y,n_h))
    assert(b2.shape==(n_y,1))
    parameters={"W1":W1,"b1":b1,"W2":W2,"b2":b2}
    return parameters
# def linear_forward(A,W,b):
#     Z= np.dot (A,W)+b
#     assert(Z.shape == (W.shape[0],A.shape[1]))
#     cache=(A,W,b)
#     return Z,cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward (A_prev, W, b)
        A, activation_cache = sigmoid (Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward (A_prev, W, b)
        A, activation_cache = relu (Z)
    assert (A.shape == (W.shape [0], A_prev.shape [1]))
    cache = (linear_cache, activation_cache)
    return A, cache

def computer_cost(AL,Y):
    m=Y.shape[1]
    cost=-1/m*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL),axis=1,keepdims=True)
    cost=np.squeeze(cost)
    assert(cost.shape == ())
    return cost

# def linear_backward (dZ, cache):
#     A_prev, W, b = cache
#     m = A_prev.shape [1]
#     dW = 1 / m * np.dot (dZ, A_prev.T)
#     db = 1 / m * np.sum (dZ, axis=1, keepdims=True)
#     dA_prev = np.dot (W.T, dZ)
#     assert (dA_prev.shape == A_prev.shape)
#     assert (dW.shape == W.shape)
#     assert (db.shape == b.shape)
#     return dA_prev, dW, db

def linear_activation_backward (dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward (dA, activation_cache)
        dA_prev, dW, db = linear_backward (dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward (dA, activation_cache)
        dA_prev, dW, db = linear_backward (dZ, linear_cache)
    return dA_prev, dW, db

def update_parameters (parameters, grads, learning_rate):
    L = len (parameters) // 2  # number of layers in the neural network
    for l in range (L):
        parameters ["W" + str (l + 1)] = parameters ["W" + str (l + 1)] - learning_rate * grads ["dW" + str (l + 1)]
        parameters ["b" + str (l + 1)] = parameters ["b" + str (l + 1)] - learning_rate * grads ["db" + str (l + 1)]
    return parameters

n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
def two_layer_model (X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed (1)
    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape [1]  # number of examples
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters (n_x, n_h, n_y)
    W1 = parameters ["W1"]
    b1 = parameters ["b1"]
    W2 = parameters ["W2"]
    b2 = parameters ["b2"]

    for i in range (0, num_iterations):
        A1, cache1 = linear_activation_forward (X, W1, b1, activation="relu")
        A2, cache2 = linear_activation_forward (A1, W2, b2, activation="sigmoid")
        cost = compute_cost (A2, Y)
        dA2 = - (np.divide (Y, A2) - np.divide (1 - Y, 1 - A2))
        dA1, dW2, db2 = linear_activation_backward (dA2, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward (dA1, cache1, activation="relu")
        grads ['dW1'] = dW1
        grads ['db1'] = db1
        grads ['dW2'] = dW2
        grads ['db2'] = db2

        parameters = update_parameters (parameters, grads, learning_rate)
        W1 = parameters ["W1"]
        b1 = parameters ["b1"]
        W2 = parameters ["W2"]
        b2 = parameters ["b2"]
        if print_cost and i % 100 == 0:
            print ("Cost after iteration {}: {}".format (i, np.squeeze (cost)))
        if print_cost and i % 100 == 0:
            costs.append (cost)
    plt.plot (np.squeeze (costs))
    plt.ylabel ('cost')
    plt.xlabel ('iterations (per tens)')
    plt.title ("Learning rate =" + str (learning_rate))
    plt.show ()
    return parameters


parameters=two_layer_model(train_x,train_y,layers_dims=(n_x,n_h,n_y),num_iterations=2500,print_cost=True)

prediction_train=predict(train_x,train_y,parameters)
prediction_test=predict(test_x,test_y,parameters)