import numpy as np
from copy import deepcopy as copy

# Prediction stuff

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Gradient descent stuff

def getErrorDifference( inp:np.array, net1:object, net2:object ):
    # Compare the two instances
    res1 = net1.think( inp, showDebug=False )
    err1 = net1.getError(inp) # get the networks error

    res2 = net2.think( inp, showDebug=False )
    err2 = net2.getError(inp) # get the second error

    # Return the difference in error
    dErr = err2 - err1
    return dErr, err2

def compareInstanceWeight( network:object, inp:np.array, theta:float, layerIndex:int, neuronIndex_X:int, neuronIndex_Y:int ):
    # Create new a instance of the object
    network2 = copy(network) # annoying way to create a new instance of the object

    network2.weights[layerIndex][neuronIndex_X][neuronIndex_Y] += theta # mutate the second objects neuron
    dErr, curErr = getErrorDifference( inp, network, network2 ) # compare the two and get the dCost with respect to the weights

    return dErr, curErr

def compareInstanceBias( network:object, inp:np.array, theta:float, layerIndex:int, biasIndex:int ):
    network2 = copy(network)

    network2.bias[layerIndex][0][biasIndex] += theta # do the same thing for the bias
    dErr, curErr = getErrorDifference( inp, network, network2 )

    return dErr, curErr

def getChangeInCost( network:object, inp:np.array, theta:float, layerIndex:int ):
    mirrorObj = copy(network)

    # Fill the buffer with a placeholder so that the dCost can replace it later
    dCost_W = np.zeros( shape = mirrorObj.weights[layerIndex].shape )
    dCost_B = np.zeros( shape = mirrorObj.bias[layerIndex].shape )

    # Get the cost change for the weights
    weightLenX = len(dCost_W)
    weightLenY = len(dCost_W[0])

    for x in range(weightLenX): # get the dCost for each x,y
        for y in range(weightLenY):
            dCost_W[x][y], curCostWeight = compareInstanceWeight( network, inp, theta, layerIndex, x, y )

    # Get the cost change for the biases
    biasLenY = len(dCost_B[0])
    for index in range(biasLenY):
        dCost_B[0][index], curCostBias = compareInstanceBias( network, inp, theta, layerIndex, index )

    return dCost_W, dCost_B, (curCostBias + curCostWeight)/2

def gradient( network:object, inp:np.array, theta:float, layerIndex:int = 0, grads:dict = None ):
    # Check if grads exists, if not create the buffer
    grads = grads or [None] * ( network.maxLayerIndex - 1 )

    dCost_W, dCost_B, meanCurCost = getChangeInCost( network, inp, theta, layerIndex )

    # Calculate the gradient for the layer
    weightDer = dCost_W / theta
    biasDer = dCost_B / theta

    # Append the gradients to the list
    grads[layerIndex] = {
        "weight": weightDer,
        "bias": biasDer
    }

    if( newLayer <= maxLayer ):
        return gradient( network, inp, theta, layerIndex + 1, grads )
    else:
        return grads, dCost_W, dCost_B, meanCurCost
