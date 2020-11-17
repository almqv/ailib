import sys
import numpy as np
from copy import deepcopy as copy

import ailib.debug as db
import ailib.func as func


class neural_network:
    def __init__( self, enableDebug:bool = True, weights:np.matrix = None, bias:np.matrix = None, correctFuncPointer = None ):
        self.enableDebug = enableDebug

        self.weights = weights
        self.bias = bias

        # SGD stuff
        self.teachTimes = 100 # amount of times the network will be thaught

        if( correctFuncPointer ):
            self.correctFuncPointer = correctFuncPointer
        else:
            self.debug( "No correct function pointer. The network will be unable to learn.", db.level.warn )


        self.debug( f"Created neural network {self}", db.level.success )

    def debug( self, text:str, level:str = db.level.info, indent:int = 0, end:str = "\n" ):
        if( self.enableDebug ): # Only debug when it is enabled
            db.debug( text, level, indent, end )

    def generateLayers( self, neuronDimensions:list = [ 1, 1 ], offset:float = -0.25 ):
        # The neuronDimensions are the dimensions of the neurons. Each index is a layer and that
        # indices value is the amount of neurons in that layer.
        #
        # The offset is what is added to each weight/bias when randomizing them

        try:
            self.neuronDimensions = neuronDimensions
            self.inputDimensions = self.neuronDimensions[0]
            self.outputDimensions = self.neuronDimensions[-1]

            self.debug( f"Generating layers {neuronDimensions}" )
            layersLen = len(neuronDimensions)
            layerProp = [None] * (layersLen - 1)

            # Generate the weight matrix
            self.debug( "Generating weight matrix...", indent=1 )
            self.weights = copy(layerProp)

            for index, neuronCount in enumerate(neuronDimensions): # Iterate through each layer and append the weights
                if( index > 0 ):
                    self.weights[index - 1] = np.random.rand( neuronDimensions[index-1], neuronCount ) + offset

            self.debug( f"Generated weights matrix: {self.weights}", db.level.success, indent=1 )


            # Generate the bias matrix
            self.debug( "Generating bias matrix...", indent=1 )
            self.bias = copy(layerProp)

            for index, neuronCount in enumerate(neuronDimensions):
                if(index > 0):
                    self.bias[index - 1] = np.random.rand( 1, neuronCount ) + offset

            self.debug( f"Generated bias matrix: {self.bias}", db.level.success, indent=1 )

            self.maxLayerIndex = len(self.weights) # Used when recursivley thinking

        except:
            self.debug( f"{sys.exc_info()}", db.level.fail )

    def loadLayers( self, savefile:str ): # TODO: Load weights and biases from files
        self.debug( "loadLayers: Feature is not implimented yet!", db.level.fail )

    def think( self, inp:np.array, layerIndex:int = 0, maxPropLayer:int = None, showDebug:bool = True ):
        try:
            maxPropLayer = maxPropLayer or self.maxLayerIndex - 1

            if( showDebug ):
                self.debug( f"[{layerIndex}/{maxPropLayer}] Layer thinking: {inp} ..." )

            weightedLayer = np.dot( inp, self.weights[layerIndex] )
            outputLayer = func.sigmoid( np.add(weightedLayer, self.bias[layerIndex]) )

            if( layerIndex < maxPropLayer ):
                return self.think( outputLayer, layerIndex + 1, maxPropLayer, showDebug )
            else:
                return outputLayer

        except:
            self.debug( f"{sys.exc_info()}", db.level.fail )

    def correctFunc( self, inp:np.array ): # Wrapper for the "correct function".
        print(inp)
        return self.correctFuncPointer(inp)

    def getError( self, inp:np.array, predicted:np.array ):
        #try:
            correctOutput = self.correctFunc(predicted) # get the correct answer
            errSum = 0

            for i in range(self.outputDimensions):
                errSum += abs( (predicted[i] - correctOutput[i]) )

            return errSum / self.outputDimensions

        #except:
        #    self.debug( f"{sys.exc_info()}", db.level.fail )

    def setTeachTimes( self, teachTimes:int ):
        self.teachTimes = teachTimes

    def teach_sgd( self, theta:float = 0.001, showDebug:bool = False ): # Teach the network using stochastic gradient descent
        gen = 0 # the generation
        inp = None # input, gets randomized each generation

        while( gen <= self.teachTimes ):
            inp = np.asarray(np.random.rand( 1, self.inputDimensions ))[0] # generate a random input for the network
            grads, dErr_bias, dErr_weights, meanErr = func.gradient( self, inp, theta ) # calculate the gradient

            self.debug( f"Teaching [{gen}/{self.teachTimes}]: Error: {meanErr}", end="\r" )

            gen += 1

        print("\n")

        self.debug( f"Teaching finished!", db.level.success )
