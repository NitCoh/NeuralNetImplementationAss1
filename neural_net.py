import matplotlib.pyplot as plt

from forward import *
from backward import *


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    """
    Implements a L-layer neural network.
    All layers but the last should have the ReLU activation function,
    and the final layer will apply the softmax activation function.
    The size of the output layer should be equal to the number of labels in the data.
    Please select a batch size that enables your code to run well (i.e. no memory overflows while still running relatively fast).

    :param X:  the input data, a numpy array of shape (height*width , number_of_examples)
    :param Y:  the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param layers_dims: a list containing the dimensions of each layer, including the input
    :param learning_rate:
    :param num_iterations:
    :param batch_size: the number of examples in a single training batch.
    :return:
    parameters – the parameters learnt by the system during the training
                (the same parameters that were updated in the update_parameters function).
    costs – the values of the cost function (calculated by the compute_cost function).
    One value is to be saved after each 100 training iterations (e.g. 3000 iterations -> 30 values). 
    """

    def create_batches(input_data, input_labels, batch_size):
        length_data = input_data.shape[-1]
        indices = np.array(list(range(length_data)))
        np.random.shuffle(indices)

        for i in range(0, length_data, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield input_data[:, batch_indices], input_labels[:, batch_indices]

    cost = -1
    parameters = initialize_parameters(layers_dims)  # dict
    mean_loses = []

    for i in range(num_iterations):
        batch_gen = create_batches(X, Y, batch_size)
        curr_iteration_mean_loss = 0

        for batch_num, (batch, y) in enumerate(batch_gen):
            AL, caches = L_model_forward(batch, parameters, False)
            cost = compute_cost(AL, y)
            grads = L_model_backward(AL, y, caches)
            parameters = Update_parameters(parameters, grads, learning_rate)
            print(f'Iteration {i}, Batch_Num: {batch_num}, Loss: {cost}')
            curr_iteration_mean_loss += cost / batch_size

        mean_loses.append(curr_iteration_mean_loss)
        print(f"Iteration mean loss: {curr_iteration_mean_loss}\n" + "=" * 80)


    plt.plot(list(range(num_iterations)), mean_loses)
    plt.ylabel('Mean Loss')
    plt.xlabel('Iteration #')
    plt.grid()
    plt.show()

    return parameters, cost


def Predict(X, Y, parameters):
    """
    The function receives an input data and the true labels
    and calculates the accuracy of the trained neural network on the data.

    :param X: the input data, a numpy array of shape (height*width, number_of_examples)
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param parameters: a python dictionary containing the DNN architecture’s parameters
    :return:
    accuracy – the accuracy measure of the neural net on the provided data
    (i.e. the percentage of the samples for which the correct label receives the hughest confidence score).
    Use the softmax function to normalize the output values
    """

    y_hat, _ = L_model_forward(X, parameters, False)  # y_hat shape: (num_classes, num_examples)
    y_hat_preds = np.argmax(y_hat, axis=1)
    y_preds = np.argmax(Y, axis=1)

    return (y_hat_preds == y_preds).mean()
