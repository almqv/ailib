#!/usr/bin/env python

# This is a example on how to use this library.
# It is also a "testing" file.

from ailib import ai

test = ai.neural_network() # Create a new instace of a network
test.generateLayers( [2, 4, 2] )

# This will generate the following network:
# (I: Input neuron, N: Hidden neuron, O: Output neuron)
#
#       N
#   I   N   O
#   I   N   O
#       N


test.think([1, 0.2])
