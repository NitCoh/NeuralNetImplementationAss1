import matplotlib.pyplot as plt

from forward import *
from backward import *


# def jack_test(X, y, layers_dims):
#     # gradient for all the net
#     parameters = initialize_parameters(layers_dims)  # dict
#
#     AL, caches = L_model_forward(X, parameters, False)
#     loss_value = compute_cost(AL, y)
#     grads = L_model_backward(AL, y, caches)
#
#     # logits = nn.forward(X)
#     # loss_grad = (np.array([1]*1).reshape(1, 1), None, None)
#     # loss_value = loss.forward(logits, y)
#     loss_grad = loss.backward()
#     _, last_grad = nn.backward(loss_grad)
#     dx, dw, db = last_grad
#     d_for_x = np.random.dirichlet(np.ones(X.shape[0]), size=1).T  # (r, 1)
#     e_0 = 2
#     epsilons = []
#     linear = []
#     quad = []
#     for i in range(10):
#         e_now = ((0.5) ** i) * e_0
#         epsilons.append(e_now)
#         v = d_for_x * e_now
#         f_x = loss_value
#         f_x_pert = loss.forward(nn.forward(X + v), y)
#         jack = dx.T @ v
#         linear.append(np.abs(f_x_pert - f_x))
#         quad.append(np.abs(f_x_pert - f_x - jack.item()))
#
#     plt.xlabel = 'epsilon'
#     plt.ylabel = 'error'
#     plt.plot(epsilons, linear, label='linear-x')
#     plt.legend()
#     plt.show()
#     plt.plot(epsilons, quad, label='quad-x')
#     plt.legend()
#     plt.show()
#     plt.plot(epsilons, [x / y for x, y in zip(quad, linear)], label='quad/linear')
#     plt.legend()
#     plt.show()

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
            yield input_data[:, batch_indices], input_labels[batch_indices]

    epsilon = 1e-15
    costs = []
    valid_accs = []
    iterations_save = []
    parameters = initialize_parameters(layers_dims)  # dict

    (x_train, y_train), (x_valid, y_valid) = split_train(X, Y)

    steps = 0
    last_valid_acc = None
    done = False
    while not done:
        for i in range(num_iterations):
            batch_gen = create_batches(x_train, y_train, batch_size)
            curr_iteration_mean_loss = []

            for batch_num, (batch, y) in enumerate(batch_gen):
                AL, caches = L_model_forward(batch, parameters, True)
                cost = compute_cost(AL, y)
                grads = L_model_backward(AL, y, caches)
                parameters = Update_parameters(parameters, grads, learning_rate)
                curr_iteration_mean_loss.append(cost)

                if steps != 0 and steps % 100 == 0:
                    costs.append(cost)
                    iterations_save.append(steps)
                    valid_acc = Predict(x_valid, y_valid, parameters)
                    valid_accs.append(valid_acc)
                    print(f'Epcoh {i}, step: {steps}, validation acc: {valid_acc}\n' + "=" * 80)


                    is_better = last_valid_acc is None or steps < 90000 or abs(last_valid_acc - valid_acc) >= epsilon

                    if not is_better:
                        done = True
                        break
                    else:
                        last_valid_acc = valid_acc

                steps += 1


            avg_loss = np.average(curr_iteration_mean_loss)
            print(f"Epoch {i} mean loss: {avg_loss}\n" + "=" * 80)
            if done:
                break

    train_acc = Predict(x_train, y_train, parameters)
    print(f'Last training step train accuracy: {train_acc}')

    valid_acc = Predict(x_valid, y_valid, parameters)
    print(f'Last validation step train accuracy: {valid_acc}')
    plt.plot(iterations_save, costs)
    plt.ylabel('Loss')
    plt.xlabel('Iteration #')
    plt.grid()
    plt.show()
    plt.plot(iterations_save, valid_accs)
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Iteration #')
    plt.grid()
    plt.show()

    return parameters, costs


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

    y_hat, _ = L_model_forward(X, parameters, True)  # y_hat shape: (num_classes, num_examples)
    y_hat_preds = np.argmax(y_hat, axis=0)
    # y_preds = np.argmax(Y, axis=1)

    return (y_hat_preds == Y).mean()


def split_train(x_train, y_train):
    """
    Auxillary function to split the train to train-valid
    :param x_train: (features, examples)
    :param y_train:
    :return:
    """
    valid_size = int(x_train.shape[-1] / 5)
    indices = np.array(list(range(x_train.shape[-1])))
    np.random.shuffle(indices)
    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]
    return (x_train[:, train_indices], y_train[train_indices]), (
        x_train[:, valid_indices], y_train[valid_indices])
