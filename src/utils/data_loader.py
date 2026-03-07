"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from keras.datasets import mnist, fashion_mnist

def one_hot_encode(y, num_classes=10):
    """
    Convert class labels to one-hot encoded vectors
    """
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def load_dataset(dataset_name):
    """
    Load MNIST or Fashion-MNIST dataset and preprocess it.
    Returns:
        X_train, y_train, X_test, y_test
    """
    if dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")
    
    # Flatten images (28x28 → 784)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    # Normalize pixel values
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    # One-hot encode labels
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    return X_train, y_train, X_test, y_test