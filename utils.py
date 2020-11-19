import numpy as np


def load_data_mnist(path='data/mnist/MNIST_data.npz'):
    mnist_data = np.load(path)
    train_data = np.array(mnist_data['arr_0']).reshape(-1, 28, 28, 1)
    train_labels = np.array(mnist_data['arr_1'])
    test_data = np.array(mnist_data['arr_2']).reshape(-1, 28, 28, 1)
    test_labels = np.array(mnist_data['arr_3'])
    return train_data, train_labels, test_data, test_labels