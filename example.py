#!/usr/bin/env python

# This is a example on how to use this library.
# It is also a "testing" file.

from ailib import ai
import numpy as np

def invertArray(inp:np.array):  # NOTE: This function is used for comparing the predicted output and actual output
    newArray = [ inp[1], inp[0] ]
    return np.asarray(newArray) # This function can do whatever you want BUT:
                                          # It can only have 1 argument that is the input array!


test = ai.neural_network( correctFuncPointer = invertArray ) # Create a new instace of a network
# correctFuncPointer has to be assigned a function otherwise you will not be able to teach the network.

test.generateLayers( [2, 4, 2] ) # Generate the networks layers
# This will generate the following network:
# (I: Input neuron, N: Hidden neuron, O: Output neuron)
#
#       N
#   I   N   O
#   I   N   O
#       N

# Using the network:
thinkTest = test.think([1, 0.2]) # Make the network think about [1, 0.2] and then assign the output to "thinkTest"

# Teaching the network:
test.setTeachTimes( 1000 ) # Teach the network 1000 times
test.teach_sgd() # Teach the network using sthocatic gradient descent (https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
