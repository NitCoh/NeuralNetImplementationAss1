from forward import relu, softmax

import numpy as np


def linear_backward(dZ, cache):
    """
    Implements the linear part of the backward propagation process for a single layer
    :param dZ: the gradient of the cost with respect to the linear output of the current layer (layer l)
    :param cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    :return:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev = cache["A"]
    W = cache["W"]
    # linear output = W(A_prev) + b
    m = A_prev.shape[-1]
    dW = dZ @ A_prev.T
    db = np.sum(dZ, axis=1) / m
    dA_prev = dZ @ W  # (dL/dZ) @ (dZ/dA_prev)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer.
     The function first computes dZ and then applies the linear_backward function.
    :param dA: post activation gradient of the current layer
    :param cache: contains both the linear cache and the activations cache
    :param activation:
    :return:
    (dA1) dA_prev – Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    (dW2) dW – Gradient of the cost with respect to W (current layer l), same shape as W
    (db2) db – Gradient of the cost with respect to b (current layer l), same shape as b

    Lets assume we derive Layer (2) and it is the last layer:
    In general our forward computation is as follows:
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    dA = (dL/dA2) = (assume L = CE) = (A2 - y) => the input to this function
    cache = [W2, A1, b2, Z2]
    we need to return:
    dA_prev = (dA1) = dL/dA1
    dW = dL/dW2
    db = dL/db
    
    now lets use our functions.
    we know that calling the activation backwards with dL/dA2 return dZ2
    to calculate dW2, db2, we need dZ2, so we call linear_backwards which do to the following calc:
    dW2 = dZ2 @ A1.T
    db2 = dZ2
    dA_prev = dL/dA1 = (dL/dA2=dA in our function) *
                        (dA2/dZ2) *
                         (dZ2/dA1 = W2)
     = (dL/dZ2) * (dZ2/dA1)
     
     Each backward_func suppose to get dL/dA_cur as input
     and chain by multiplication:
     (dL/dA_cur) @ (dA_cur/dZ_cur)
     
     Assume y_hat = softmax(Z_last)
     The last layer has:
     dA_last=dy_hat=(dL/dy_hat) ==== - (1/m) * [sum[1 to m] of (1/y_hat_i)]

    """

    def get_derivative(func_act):
        if func_act == 'relu':
            return relu_backward
        elif func_act == 'softmax':
            return softmax_backward

    backward_func = get_derivative(activation)
    dZ = backward_func(dA, cache)
    dA_prev, dW, db = linear_backward(dZ, cache)

    return dA_prev, dW, db


def softmax_backward(dA, activation_cache):
    """
    Implements backward propagation for a softmax unit
    :param dA: the post-activation gradient
    :param activation_cache: contains Z (stored during the forward propagation)
    :return:
    dZ – gradient of the cost with respect to Z

    MOCK FUNCTION
    """
    return dA * 1


def relu_backward(dA, activation_cache):
    """
    Implements backward propagation for a ReLU unit
    :param dA:  the post-activation gradient
    :param activation_cache: contains Z (stored during the forward propagation)
    :return:
    dZ – gradient of the cost with respect to Z 
    """
    Z = activation_cache["Z"]
    f = np.vectorize(lambda x: 1 if x > 0 else 0)
    return dA * f(Z)


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation process for the entire network.
    The backpropagation for the softmax function should be done only
    once as only the output layers uses it and the RELU
    should be done iteratively over all the remaining layers of the network.
    :param AL: the probabilities vector, the output of the forward propagation (L_model_forward)
    :param Y:  the true labels vector (the "ground truth" - true classifications)
    :param caches: list of caches containing for each layer: a) the linear cache; b) the activation cache
    :return:
    Grads - a dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    m = len(Y)
    reversed_cache = reversed(list(enumerate(caches)))
    dA = (AL - Y) / m  # this is dZ actually.
    for l, cache in reversed_cache:
        # if l == len(caches) - 1:
        #     dA_prev, dW, db = linear_activation_backward(dA, cache, 'softmax')
        # else:
        dA_prev, dW, db = linear_activation_backward(dA, cache, 'relu')
        grads['dA' + f'{str(l)}'] = dA
        grads['dW' + f'{str(l)}'] = dW
        grads['db' + f'{str(l)}'] = db
        dA = dA_prev

    return grads


def Update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using gradient descent (IN-PLACE FUNC)
    :param parameters: a python dictionary containing the DNN architecture’s parameters
    :param grads: a python dictionary containing the gradients (generated by L_model_backward)
    :param learning_rate: learning_rate – the learning rate used to update the parameters (the “alpha”)
    :return:
    parameters – the updated values of the parameters object provided as input
    """

    for i, (W, b) in enumerate(zip(parameters["weights"], parameters["bias"])):
        parameters["weights"][i] = W - learning_rate * grads['dW' + f'{str(i)}']
        parameters["bias"][i] = b - learning_rate * grads['db' + f'{str(i)}']

    return parameters
