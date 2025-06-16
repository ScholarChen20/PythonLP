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
m_train=train_x_orig.shape[0] #标记为cat（1）和非cat（0）图像的训练集m_train
num_px=train_x_orig.shape[1]
m_test=test_x_orig.shape[0] #标记为cat或non-cat图像的测试集m_test

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

# def initialize_parameters_deep (layer_dims):
#     np.random.seed (3)
#     parameters = {}
#     L = len (layer_dims)  # number of layers in the network
#     for l in range (1, L):
#         parameters ['W' + str (l)] = np.random.randn (layer_dims [l], layer_dims [l - 1]) * 0.01
#         parameters ['b' + str (l)] = np.zeros ((layer_dims [l], 1))
#         assert (parameters ['W' + str (l)].shape == (layer_dims [l], layer_dims [l - 1]))
#         assert (parameters ['b' + str (l)].shape == (layer_dims [l], 1))
#     return parameters
#
# def L_model_forward (X, parameters):
#     caches = []
#     A = X
#     L = len (parameters) // 2  # number of layers in the neural network
#     for l in range (1, L):
#         A_prev = A
#         A, cache = linear_activation_forward (A_prev,
#             parameters ['W' + str (l)], parameters ['b' + str (l)], activation="relu")
#         caches.append (cache)
#     AL, cache = linear_activation_forward (A,
#         parameters ['W' + str (L)], parameters ['b' + str (L)], activation="sigmoid")
#     caches.append (cache)
#     assert (AL.shape == (1, X.shape [1]))
#     return AL, caches
#
# def compute_cost(AL,Y):
#     m=Y.shape[1]
#     cost=- 1 / m * np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
#     cost=np.squeeze(cost)
#     assert(cost.shape == ())
#     return cost
#
# def L_model_backward (AL, Y, caches):
#     grads = {}
#     L = len (caches)  # the number of layers
#     m = AL.shape [1]
#     Y = Y.reshape (AL.shape)  # after this line, Y is the same shape as AL
#     dAL = - (np.divide (Y, AL) - np.divide (1 - Y, 1 - AL))
#     current_cache = caches [L - 1]
#     grads ["dA" + str (L)], grads ["dW" + str (L)], grads [
#         "db" + str (L)] = linear_activation_backward (dAL, current_cache, activation="sigmoid")
#     for l in reversed (range (L - 1)):
#         current_cache = caches [l]
#         dA_prev_temp, dW_temp, db_temp = linear_activation_backward (
#             grads ["dA" + str (l + 2)], current_cache, activation="relu")
#         grads ["dA" + str (l + 1)] = dA_prev_temp
#         grads ["dW" + str (l + 1)] = dW_temp
#         grads ["db" + str (l + 1)] = db_temp
#     return grads
#
# def update_parameters (parameters, grads, learning_rate):
#     #更新梯度下降参数
#     L = len (parameters) // 2  # number of layers in the neural network
#     # 更新参数.
#     for l in range (L):
#         parameters ["W" + str (l + 1)] = parameters ["W" + str (l + 1)] - learning_rate * grads ["dW" + str (l + 1)]
#         parameters ["b" + str (l + 1)] = parameters ["b" + str (l + 1)] - learning_rate * grads ["db" + str (l + 1)]
#     return parameters
layers_dims = [12288, 20, 7, 5, 1]
#
def L_layer_model (X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    np.random.seed (1)
    costs = []  # keep track of cost
    parameters = initialize_parameters_deep (layers_dims)
    for i in range (0, num_iterations):
        AL, caches = L_model_forward (X, parameters)
        cost = compute_cost (AL, Y)
        grads = L_model_backward (AL, Y, caches)
        parameters = update_parameters (parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append (cost)
    # plot the cost
    plt.plot (np.squeeze (costs))
    plt.ylabel ('cost')
    plt.xlabel ('iterations (per tens)')
    plt.title ("Learning rate =" + str (learning_rate))
    plt.show ()
    return parameters

parameters=L_layer_model(train_x,train_y,layers_dims,num_iterations=2500,print_cost=True)
# pred_train=predict(train_x,train_y,parameters)
# pred_test=predict(test_x,test_y,parameters)

# my_image = "my_image.jpg" # change this to the name of your image file
# my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
# fname = my_image
# image = np.array(plt.imread(fname))
# my_image = np.array(Image.fromarray(image).resize(size=(num_px,num_px))).reshape((num_px*num_px*3,1))
# my_predicted_image = predict(my_image, my_label_y, parameters)
#
# plt.imshow(image)
# print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")