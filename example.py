#!/usr/bin/env python

# This is a example on how to use this library.
# It is also a "testing" file.

from ailib import ai

test = ai.neural_network() # Create a new instace of a network
test.generateLayers( [2, 2] ) # 2 input neurons and 2 output neurons
