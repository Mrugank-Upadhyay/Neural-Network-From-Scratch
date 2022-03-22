import sys
import numpy as np
import matplotlib
import nnfs
from nnfs.datasets import spiral_data


nnfs.init()

X = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

# input data generated by nnfs's spiral dataset
X, y = spiral_data(100, 3)


class Layer_Dense:
    # we want to create a layer with n inputs and n neurons
    # this will also help initialize the shape of our weights array
    def __init__(self, n_inputs, n_neurons):
        # We want to bound our weights to be in between -1 and 1
        # note that for randn, the parameters passed in EQUATE to the shape
        # also note that we set the weights as inputSize * n_neurons so we don't have to transpose
        # when we do the forward pass
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # we want a shape of 1 x n_neurons
        # note for zeros, the first parameter IS the actual shape (so it must be a tuple)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# number of inputs, and neurons
layer1 = Layer_Dense(2, 5)
# remember that since we have 5 neurons in layer 1, we have 5 outputs, and so layer2 has 5 inputs
# layer2 = Layer_Dense(5, 2)

activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)


# print(layer1.output)
print(activation1.output)


# layer2.forward(layer1.output)
# print(layer2.output)
