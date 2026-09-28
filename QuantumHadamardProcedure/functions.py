import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def log_transformation(x):
    return np.log2(1 + x)


def gamma_correction(x, gamma=1):
    return np.power(x, gamma)


def gamma_correction_1(x):
    return np.power(x, 0.4)


def gamma_correction_2(x):
    return np.power(x, 2.2)