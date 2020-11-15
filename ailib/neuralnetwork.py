import sys
import numpy as np

import ailib.debug as db


class neural_network:
    def __init__( self, enableDebug:bool = True, weights:np.matrix = None, bias:np.matrix = None ):
        self.weights = weights
        self.bias = bias

        self.enableDebug = enableDebug

        self.debug( f"Created neural network {self}", db.level.success )

    def debug( self, text:str, lvl:str = db.level.info, indent:int = 0, end:str = "\n" ):
        if( self.enableDebug ): # Only debug when it is enabled
            db.debug( text, lvl, indent, end )

    def generateLayers( self, neuronDimensions:list = [ 1, 1 ], offset:float = -0.25 ):
        # The neuronDimensions are the dimensions of the neurons. Each index is a layer and that
        # indices value is the amount of neurons in that layer.
        #
        # The offset is what is added to each weight/bias when randomizing them

        try:
            self.debug( f"Generating layers {neuronDimensions}" )

            self.debug( "Generating weight matrix", indent=1 )
            self.weights = [None] * (len(neuronDimensions) - 1)

            for index, neuronCount in enumerate(neuronDimensions):
                if( index > 0 ):
                    self.weights[index - 1] = np.random.rand( neuronDimensions[index-1], neuronCount )

            self.debug( f"Generated weights: {self.weights}", indent=1 )

        except:
            err = sys.exc_info()
            self.debug( f"{err}", db.level.fail )
