import numpy as np
from copy import deepcopy as copy

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def getChangeInCost( network:object, inp:np.array, theta:float, layerIndex:int ):
    mirrorObj = copy(obj)

    # Fill the buffer with a placeholder so that the dCost can replace it later
    dCost_W = np.zeros( shape = mirrorObj.weights[layerIndex].shape )
    dCost_B = np.zeros( shape = mirrorObj.bias[layerIndex].shape )

    # Get the cost change for the weights
    weightLenX = len(dCost_W)
    weightLenY = len(dCost_W[0])

    for x in range(weightLenX): # get the dCost for each x,y
        for y in range(weightLenY):
            dCost_W[x][y], curCostWeight = compareInstanceWeight( obj, inp, theta, layerIndex, x, y )

    # Get the cost change for the biases
    biasLenY = len(dCost_B[0])
    for index in range(biasLenY):
        dCost_B[0][index], curCostBias = compareInstanceBias( obj, inp, theta, layerIndex, index )

    return dCost_W, dCost_B, (curCostBias + curCostWeight)/2

def gradient( network:object, inp:np.array, theta:float, layerIndex:int = 0, grads:dict = None ):
    # Check if grads exists, if not create the buffer
    grads = grads or [None] * ( network.maxLayerIndex - 1 )

    dCost_W, dCost_B, meanCurCost = getChangeInCost( obj, inp, theta, layerIndex )

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
