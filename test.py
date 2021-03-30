from neural_net import *

X = np.arange(12).reshape((2, 6))
Y = np.array([[0, 1],
              [1, 0],
              [0, 1],
              [1, 0],
              [0, 1],
              [1, 0]]).T
layers_dim = [2, 4, 2]

L_layer_model(X, Y,
              layers_dims=layers_dim,
              learning_rate=1e-4,
              num_iterations=20,
              batch_size=2)