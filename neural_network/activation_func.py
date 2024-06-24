import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derive_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def relu(x):
    if x < 0:
        return 0.01 * x
    else:
        return x

def derive_relu(x):
    if x < 0:
        return 0.01
    else:
        return 1