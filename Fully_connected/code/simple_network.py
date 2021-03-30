"""
简单的两层神经网络
"""

import numpy as np
from simple_neural import Neuron


# ------------------------ #
# ---     神经网络      --- #
# ------------------------ #

class OurNeuralNetwork:
    '''
    A neural network with:
      - 2 inputs
      - a hidden layer with 2 neurons (h1, h2)
      - an output layer with 1 neuron (o1)
    Each neuron has the same weights and bias:
      - w = [0, 1]
      - b = 0
    '''
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        # The Neuron class here is from the previous section
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # The inputs for o1 are the outputs from h1 and h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1

network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x)) # 0.721632560951842


# ------------------------ #
# ---     损失函数      --- #
# ------------------------ #
def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()


y_true = np.array([1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0])

print(mse_loss(y_true, y_pred)) # 0.5