import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import modules
from modules.datasets import *

modules.init()

class Layer_Dense:
    # we want to create a layer with n inputs and n neurons
    # this will also help initialize the shape of our weights array
    def __init__(self, n_inputs, n_neurons):
        # We want to bound our weights to be in between -1 and 1
        # note that for randn, the parameters passed in EQUATE to the shape
        # also note that we set the weights as inputSize * n_neurons so we don't have to transpose
        # when we do the forward pass
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        
        # we want a shape of 1 x n_neurons
        # note for zeros, the first parameter IS the actual shape (so it must be a tuple)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradient on params
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # We will make a copy of the values since we need to modify the original variable
        self.dinputs = dvalues.copy()
        # We have a gradient of 0 for non-positive inputs
        self.dinputs[self.inputs <= 0] = 0

# Used in the output layer
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            
            # Calculate the Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Calculate the sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # we want to clip to ensure that when we use log, we don't go to inf when the value is 0
        # so we'll keep things clipped to a range of 1e-9 to 1 - 1e-9
        y_pred_clipped = np.clip(y_pred, 1e-9, 1 - 1e-9)
        # create confidences using both normal and one-hot-encoding structures
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # For number of labels, we can just use the number found in the first sample
        labels = len(dvalues[0])

        # Convert sparse labels into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

X, y = spiral_data(samples=100, classes=3)

plt.scatter(X[:,0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossEntropy()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss = loss_function.calculate(activation2.output, y)
print('loss:', loss)

#calculating accuracy
predictions = np.argmax(activation2.output, axis=1)
# if we don't use one-hot encoding
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)
print("acc:", accuracy)