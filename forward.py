import numpy as np


def initialize_parameters(layers_dim):
    """
    Initialisation of all layers' weights and biases
    :param layers_dim: list of dimensions for each layer in the network
    (layer 0 is the size of the flattened input, layer L is the output softmax)
    :return:
     parameters: A dictionary containing the initialized W and b parameters of each layer (W1…WL, b1…bL).
    """
    parameters = {
        "weights": [],
        "bias": []
    }
    for pair in zip(layers_dim, layers_dim[1:]):
        W = np.random.randn(*pair).T  # shape: [out_dim, in_dim]

        b = np.zeros((pair[-1], 1))
        parameters["weights"].append(W)
        parameters["bias"].append(b)

    return parameters


def linear_forward(A, W, b):
    """
    The linear part of a layer's forward propagation
    :param A: Output of the previous layer, shape size [previous_layer, 1]
    :param W: weights, shape size [current_layer, previous_layer]
    :param b: bias, shape size [current_layer, 1]
    :return:
    Z – the linear component of the activation function (i.e., the value before applying the non-linear function)
    linear_cache – a dictionary containing A, W, b (stored for making the backpropagation easier to compute)
    """
    Z = W @ A + b
    linear_cache = {
        "W": W,
        "A": A,
        "b": b
    }

    return Z, linear_cache


def softmax(Z):
    """
    Apply the softmax function
    :param Z: the linear component of the activation function
    :return:
    A – the activations of the layer
    activation_cache – returns Z, which will be useful for the backpropagation
    """
    A = np.exp(Z - np.amax(Z, 0, keepdims=True))  # numerical stability
    A = A / A.sum(0, keepdims=True)  # broadcasting
    activation_cache = {
        "Z": Z
    }

    return A, activation_cache


def relu(Z):
    """
    Apply relu function
    :param Z:
    :return:
    A – the activations of the layer
    activation_cache – returns Z, which will be useful for the backpropagation
    """
    f = np.vectorize(lambda x: x if x > 0 else 0, otypes=[float])
    A = f(Z)
    activation_cache = {
        "Z": Z
    }

    return A, activation_cache


def linear_activation_forward(A_prev, W, B, activation):
    """
    The forward propagation for the LINEAR->ACTIVATION layer
    :param A_prev: activations of the previous layer
    :param W: the weights matrix of the current layer
    :param B: the bias vector of the current layer
    :param activation: the activation function to be used (a string, either “softmax” or “relu”)
    :return:
    A – the activations of the current layer
    cache – a joint dictionary containing both linear_cache and activation_cache
    """

    def select_activation(name):
        if name == "softmax":
            return softmax
        elif name == "relu":
            return relu

    Z, linear_cache = linear_forward(A_prev, W, B)
    A, activation_cache = select_activation(activation)(Z)
    cache = {**linear_cache, **activation_cache}

    return A, cache


def L_model_forward(X, parameters, use_batchnorm: bool):
    """
    Forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation
    :param X: the data, numpy array of shape (input size, number of examples)
    :param parameters: the initialized W and b parameters of each layer
    :param use_batchnorm: a boolean flag used to determine whether to apply batchnorm after the activation
     (note that this option needs to be set to “false” in Section 3 and “true” in Section 4).
    :return:
    AL – the last post-activation value
    caches – a list of all the cache objects generated by the linear_forward function
    """

    AL = X
    L = len(parameters["weights"])
    caches = []

    for l, (W, b) in enumerate(zip(parameters["weights"], parameters["bias"])):
        activation = 'relu' if l < L - 1 else 'softmax'
        A, cache = linear_activation_forward(AL, W, b, activation)

        if use_batchnorm and l < L - 1:  # no batch-norm on output layer
            A = apply_batchnorm(A)
        AL = A
        caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    """
    Calculate the cost function defined by the categorical cross-entropy loss.
    :param AL: probability vector corresponding to your label predictions, shape (num_of_classes, number of examples)
    :param Y: the labels vector (i.e. the ground truth)
    :return: cost – the cross-entropy cost
    """

    epsilon = 1e-15
    c, m = AL.shape
    C = expand_y(Y, *AL.shape)
    log_res = np.log(AL + epsilon)
    cost = - np.einsum('ij,ij', log_res, C) / m

    return cost


def apply_batchnorm(A):
    """
    Performs batchnorm on the received activation values of a given layer.
    :param A: The activation values of a given layer
    :return: NA: The normalized activation values, based on the formula learned in class
    """
    mean = np.expand_dims(A.mean(axis=1), axis=-1)
    var = np.expand_dims(A.var(axis=1), axis=-1)
    epsilon = 1e-15
    NA = (A - mean) / np.sqrt(var + epsilon)

    return NA


def expand_y(y, l, m):
    """
    C[i][j] =  example j has label i ? 1 : 0
    :param m:
    :param y: vector of labels size (m, ) where yi belongs to {0...l-1}
    :return: C size (l, m) where l is the num of classes, m is the num of examples in batch.
    """
    C = np.zeros((l, m))
    for i, cl in enumerate(y):
        C[cl, i] = 1
    return C