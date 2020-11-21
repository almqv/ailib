import numpy as np
from copy import deepcopy as copy

# Prediction stuff


def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Gradient descent stuff


def getErrorDifference(inp: np.array, net1: object, net2: object):
    # Compare the two instances
    res1 = net1.think(inp, showDebug=False)
    err1 = net1.getError(inp, res1)  # get the networks error

    res2 = net2.think(inp, showDebug=False)
    err2 = net2.getError(inp, res2)  # get the second error

    # Return the difference in error
    dErr = err2 - err1
    return dErr, err1


def compareInstanceWeight(network: object, inp: np.array, theta: float, layerIndex: int, neuronIndex_X: int, neuronIndex_Y: int):
    # Create new a instance of the object
    # annoying way to create a new instance of the object
    network2 = copy(network)
    # mutate the second objects neuron
    network2.weights[layerIndex][neuronIndex_X][neuronIndex_Y] += theta
    # compare the two and get the dCost with respect to the weights
    dErr, curErr = getErrorDifference(inp, network, network2)

    return dErr, curErr


def compareInstanceBias(network: object, inp: np.array, theta: float, layerIndex: int, biasIndex: int):
    network2 = copy(network)

    # do the same thing for the bias
    network2.bias[layerIndex][0][biasIndex] += theta
    dErr, curErr = getErrorDifference(inp, network, network2)

    return dErr, curErr


def getChangeInError(network: object, inp: np.array, theta: float, layerIndex: int):
    mirrorObj = copy(network)

    # Fill the buffer with a placeholder so that the dCost can replace it later
    dErr_W = np.zeros(shape=mirrorObj.weights[layerIndex].shape)
    dErr_B = np.zeros(shape=mirrorObj.bias[layerIndex].shape)

    # Get the cost change for the weights
    weightLenX = len(dErr_W)
    weightLenY = len(dErr_W[0])

    for x in range(weightLenX):  # get the dCost for each x,y
        for y in range(weightLenY):
            dErr_W[x][y], curErrWeight = compareInstanceWeight(
                network, inp, theta, layerIndex, x, y)

    # Get the cost change for the biases
    biasLenY = len(dErr_B[0])
    for index in range(biasLenY):
        dErr_B[0][index], curErrBias = compareInstanceBias(
            network, inp, theta, layerIndex, index)

    return dErr_W, dErr_B, (curErrBias + curErrWeight)/2


def gradient(network: object, inp: np.array, theta: float, layerIndex: int = 0, grads: dict = None):
    maxLayer = network.maxLayerIndex - 1
    # Check if the gradient exists, if not then create
    grads = grads or [None] * (network.maxLayerIndex)

    dErr_W, dErr_B, meanCurErr = getChangeInError(
        network, inp, theta, layerIndex)

    # Calculate the gradient for the layer
    weightDer = dErr_W / theta
    biasDer = dErr_B / theta

    # Append the gradients to the list
    grads[layerIndex] = {
        "weight": weightDer,
        "bias": biasDer
    }

    if(layerIndex < maxLayer):
        return gradient(network, inp, theta, layerIndex + 1, grads)
    else:
        return grads, dErr_W, dErr_B, meanCurErr
