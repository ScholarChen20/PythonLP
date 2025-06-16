import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward



np.random.seed(1)


def initialize_parameters (n_x, n_h, n_y):

    np.random.seed (1)

    W1 = np.random.randn (n_h, n_x) * 0.01
    b1 = np.zeros ((n_h, 1))
    W2 = np.random.randn (n_y, n_h) * 0.01
    b2 = np.zeros ((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters
parameters = initialize_parameters(2,2,1)
print("W1="+str(parameters["W1"]))
print("b1="+str(parameters["b1"]))
print("W2="+str(parameters["W2"]))
print("b2="+str(parameters["b2"]))
#建立正向传播的线性部分。
def linear_forward(A,W,b):
    Z= np.dot (A,W)+b
    assert(Z.shape == (W.shape[0],A.shape[1]))
    cache=(A,W,b)
    return Z,cache
#实现 LINEAR->ACTIVATION 层的正向传播。
def linear_activation_forward (A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward (A_prev, W, b)
        A, activation_cache = sigmoid (Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward (A_prev, W, b)
        A, activation_cache = relu (Z)
    assert (A.shape == (W.shape [0], A_prev.shape [1]))
    cache = (linear_cache, activation_cache)

    return A, cache

#实现上述模型的正向传播
def L_model_forward (X, parameters):
    caches = []
    A = X
    L = len (parameters) // 2  # number of layers in the neural network
    for l in range (1, L):
        A_prev = A
        A, cache = linear_activation_forward (A_prev,
            parameters ['W' + str (l)], parameters ['b' + str (l)], activation="relu")
        caches.append (cache)

    AL, cache = linear_activation_forward (A,
        parameters ['W' + str (L)], parameters ['b' + str (L)], activation="sigmoid")
    caches.append (cache)
    assert (AL.shape == (1, X.shape [1]))
    return AL, caches

def compute_cost(AL,Y):
    m=Y.shape[1]
    cost=-1/m*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL),axis=1,keepdims=True)
    cost=np.squeeze(cost)
    assert(cost.shape==())

    return cost
# Y,AL=compute_cost_test_case()
# print("cost="+str(compute_cost(AL,Y)))

def linear_backward (dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape [1]
    dW = 1 / m * np.dot (dZ, A_prev.T)
    db = 1 / m * np.sum (dZ, axis=1, keepdims=True)
    dA_prev = np.dot (W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

dZ, linear_cache = linear_backward_test_case()
dA_prev, dW, db = linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))

#对于使用l=1,2,....,L,使用梯度下降更新每个w和b的参数
def update_parameters (parameters, grads, learning_rate):
    L = len (parameters) // 2  # number of layers in the neural network
    for l in range (L):
        parameters ["W" + str (l + 1)] = parameters ["W" + str (l + 1)] - learning_rate * grads ["dW" + str (l + 1)]
        parameters ["b" + str (l + 1)] = parameters ["b" + str (l + 1)] - learning_rate * grads ["db" + str (l + 1)]

    return parameters
#测试
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))




