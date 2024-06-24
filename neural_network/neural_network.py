import numpy as np
from .activation_func import *

def MSE_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class Neuron:
    def __init__(self, w1, w2, bias, activation_func):
        self.w1 = w1
        self.w2 = w2
        self.bias = bias

        self.a_f = activation_func

    def feedforward(self, x):
        total = self.w1 * x[0] + self.w2 * x[1] + self.bias
        return self.a_f(total)

class NeuralNetwork:
    def __init__(self, activation_func):
        # Weights
        self.weights = np.array([np.random.normal() for _ in range(6)])

        # Biases
        self.biases = np.array([np.random.normal() for _ in range(3)])

        # Activation function
        match activation_func:
            case "sigmoid":
                self.a_f = sigmoid
                self.d_a_f = derive_sigmoid
            case "relu":
                self.a_f = relu
                self.d_a_f = derive_relu
            case _:
                raise NotDefaultActivationFunc

        self.h1 = Neuron(self.weights[0], self.weights[1], self.biases[0], self.a_f)
        self.h2 = Neuron(self.weights[2], self.weights[3], self.biases[1], self.a_f)
        self.o1 = Neuron(self.weights[4], self.weights[5], self.biases[2], self.a_f)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.weights[0] * x[0] + self.weights[1] * x[1] + self.biases[0]
                h1 = self.a_f(sum_h1)

                sum_h2 = self.weights[2] * x[0] + self.weights[3] * x[1] + self.biases[1]
                h2 = self.a_f(sum_h2)

                sum_o1 = self.weights[4] * h1 + self.weights[5] * h2 + self.biases[2]
                o1 = self.a_f(sum_o1)

                y_pred = o1

                d_L_d_y_pred = -2 * (y_true - y_pred)

                # Neuron o1
                d_y_pred_d_w5 = h1 * self.d_a_f(sum_o1)
                d_y_pred_d_w6 = h2 * self.d_a_f(sum_o1)
                d_y_pred_d_b3 = self.d_a_f(sum_o1)

                d_y_pred_d_h1 = self.weights[4] * self.d_a_f(sum_o1)
                d_y_pred_d_h2 = self.weights[5] * self.d_a_f(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * self.d_a_f(sum_h1)
                d_h1_d_w2 = x[1] * self.d_a_f(sum_h1)
                d_h1_d_b1 = self.d_a_f(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * self.d_a_f(sum_h2)
                d_h2_d_w4 = x[1] * self.d_a_f(sum_h2)
                d_h2_d_b2 = self.d_a_f(sum_h2)

                # --- Updatind weights & biases ---

                # Neuron h1
                self.weights[0] -= learn_rate * d_L_d_y_pred * d_y_pred_d_h1 * d_h1_d_w1
                self.weights[1] -= learn_rate * d_L_d_y_pred * d_y_pred_d_h1 * d_h1_d_w2
                self.biases[0] -= learn_rate * d_L_d_y_pred * d_y_pred_d_h1 * d_h1_d_b1

                # Neuron h1
                self.weights[2] -= learn_rate * d_L_d_y_pred * d_y_pred_d_h2 * d_h2_d_w3
                self.weights[3] -= learn_rate * d_L_d_y_pred * d_y_pred_d_h2 * d_h2_d_w4
                self.biases[1] -= learn_rate * d_L_d_y_pred * d_y_pred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.weights[4] -= learn_rate * d_L_d_y_pred * d_y_pred_d_w5
                self.weights[5] -= learn_rate * d_L_d_y_pred * d_y_pred_d_w6
                self.biases[2] -= learn_rate * d_L_d_y_pred * d_y_pred_d_b3

            # --- Calculation total loss ---
            if epoch % 10 == 0:
                y_pred = np.apply_along_axis(self.feedforward, 1, data)
                loss = MSE_loss(all_y_trues, y_pred)
                print("Epoch %d loss: %.3f" % (epoch, loss))