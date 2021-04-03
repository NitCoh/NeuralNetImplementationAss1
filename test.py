from neural_net import *
import tensorflow as tf
import tensorflow_datasets as tfds

# X = np.arange(12).reshape((2, 6))
# Y = np.array([[0, 1],
#               [1, 0],
#               [0, 1],
#               [1, 0],
#               [0, 1],
#               [1, 0]]).T
# layers_dim = [2, 4, 2]
#
# L_layer_model(X, Y,
#               layers_dims=layers_dim,
#               learning_rate=1e-4,
#               num_iterations=500,
#               batch_size=2)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('./data')
x_train = x_train.reshape(x_train.shape[0], -1).T / 255
x_test = x_test.reshape(x_test.shape[0], -1).T / 255

# x_train = np.random.randn(3, 20)
# y_train = np.random.randint(0, 10, 20)

layers_dim = [784, 20, 7, 5, 10]
parameters, costs = L_layer_model(x_train, y_train,
                                  layers_dims=layers_dim,
                                  learning_rate=0.009,
                                  num_iterations=30,
                                  batch_size=64)


test_acc = Predict(x_test, y_test, parameters)
print(f'Last training step test accuracy: {test_acc}')


# print()
# np.random.choice(x_train, valid_size)

def train_with_tensorflow():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('./data')

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(16)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(16)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(7, activation='relu'),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(0.009),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )

# train_with_tensorflow()
