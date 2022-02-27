import sys
import numpy as np
import matplotlib

inputs = [13, 41, 2.3]
weights = [[-17, -19, 10, 31], [11, 15, 25, -16], [-7,  7, 18, -9]]
biases = [3, 4, 2]
 
layer_output = []

for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for input, weight in zip(inputs, neuron_weights):
        neuron_output += input * weight 
    neuron_output += neuron_bias
    layer_output.append(neuron_output)

print(layer_output)
