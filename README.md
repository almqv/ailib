# AIlib (VERY WIP)
This is a lightweight AI library that works as a wrapper. 

## Installation
 1. `git clone https://github.com/E-Almqvist/ailib.git`
 2. Import it in your project and that is it.
 
## Usage
``` python
from ailib import ai

my_neural_network = ai() # Create an object for your network
my_neural_network.generateLayers( [1, 1] ) # Generate 1 input neuron and 1 output neuron.

my_neural_network.think( [0.2] ) # Make the AI "think" about "0.2" and it will give out 1 output.

# This network does not really do anything usefull. 
# Examples on how to optimize a network with this library will come whenever I have implemented that feature.
```

## Features
(Checked boxes are implemented features and unchecked has not been implemented yet.)

#### Neural Networks
- [x] Thinking
	- [x] Generating neural matrix
	- [x] Prediction
- [ ] Learning
	- [ ] Stochastic gradient descent (SGD)
- [ ] Support for networks of (neural) networks
	- [ ] Input/Output passthrough
- [ ] Support for loading trained networks

#### Other planned features
- [x] Cool debug colors
- [ ] Multicore processing
