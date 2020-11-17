import numpy as np

from ailib import ai

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def gradient( inp:np.array, obj:ai.neural_network, theta:float, layerIndex:int = 0, grads:dict = None, obj1:ai.neural_network = None, obj2:ai.neural_network = None ):
    # Check if grads exists, if not create the buffer
    grads = grads or [None] * (maxLayer+1)

    dCost_W, dCost_B, meanCurCost = getChangeInCost( obj, inp, theta, layerIndex )

    # Calculate the gradient for the layer
    weightDer = dCost_W / theta
    biasDer = dCost_B / theta

    # Append the gradients to the list
    grads[layerIndex] = {
        "weight": weightDer,
        "bias": biasDer
    }

    newLayer = layerIndex + 1
    if( newLayer <= maxLayer ):
        return gradient( inp, obj, theta, maxLayer, newLayer, grads, obj1, obj2 )
    else:
        return grads, dCost_W, dCost_B, meanCurCost
