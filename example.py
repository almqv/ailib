#!/usr/bin/env python

# This is a example on how to use this library.
# This is also a "testing" file.

from ailib import ai
import numpy as np

# This network will attempt to invert RGB values...
# i.e. [1, 1, 1] -> [0, 0, 0]
# i.e. white -> black


# NOTE: This function is used for comparing the predicted output and actual output
def invertRGB(inp: np.array):
    out = [1 - inp[0], 1 - inp[1], 1 - inp[2]]
    return np.asarray(out)  # This function can do whatever you want BUT:
    # It can only have 1 argument that is the input array!


# Create a new instace for a network
test = ai.neural_network(correctFuncPointer=invertRGB)
# correctFuncPointer must be assigned to a function otherwise you will not be able to teach the network.

test.generateLayers([3, 3, 3])  # Generate the networks layers
# This will generate the following network:
# (I: Input neuron, N: Hidden neuron, O: Output neuron)
#
#   I   N   O
#   I   N   O
#   I   N   O


# Using the network:
testInput = [1.0, 1.0, 1.0]

# Make the network think about [1, 1, 1] and then assign the output to "thinkTest"
thinkTest = test.think(testInput)
# The actual output should be [0, 0, 0] but we will get something far away from that.
# In order for the network to work; we have to teach it.

# Teaching the network:
test.setTeachTimes(10000)  # Teach the network 10000 times
# Teach the network using stochastic gradient descent (https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
test.teach_sgd(theta=0.001)
# The correctFuncPointer is needed here for it to test itself against it.

# test the network again and see what result it got
teachThinkTest = test.think(testInput)
