import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def gradient( inp:np.array, obj, theta:float, maxLayer:int, layerIndex: int=0, grads=None, obj1=None, obj2=None ): # Calculate the gradient for that prop
    # Check if grads exists, if not create the buffer
    if( not grads ):
        grads = [None] * (maxLayer+1)

    dCost_W, dCost_B, meanCurCost = getChangeInCost( obj, inp, theta, layerIndex )

    # Calculate the gradient for the layer
    weightDer = propDer( dCost_W, theta )
    biasDer = propDer( dCost_B, theta )

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
