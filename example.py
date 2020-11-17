#!/usr/bin/env python

# This is a example on how to use this library.
# It is also a "testing" file.

from ailib import ai
import numpy as np

test = ai.neural_network() # Create a new instace of a network
test.generateLayers( [2, 4, 2] )

# This will generate the following network:
# (I: Input neuron, N: Hidden neuron, O: Output neuron)
#
#       N
#   I   N   O
#   I   N   O
#       N

thinkTest = test.think([1, 0.2]) # Make the network think about [1, 0.2] and then assign the output to "thinkTest"
test.debug( str(thinkTest) ) # Print out the output

test.teach( 1000 ) # Teach the AI 1000 times
