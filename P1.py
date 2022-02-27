import sys
import numpy as np
import matplotlib

inputs_arr = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights_arr = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

weights_arr2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]
 
inputs = np.array(inputs_arr)
weights1 = np.array(weights_arr)
weights2 = np.array(weights_arr2)

layer1_output = np.dot(inputs, weights1.T) + biases

layer2_output = np.dot(layer1_output, weights2.T) + biases2

print(layer2_output)
