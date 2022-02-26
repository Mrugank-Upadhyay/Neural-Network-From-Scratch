from audioop import bias
import sys
import numpy as np
import matplotlib

inputs = [13, 41, 2.3]
weights = [2.4, 1.3, 4.5]
biases = 3

output = sum([a*b for a,b in zip(inputs, weights)]) + biases
print(output)
