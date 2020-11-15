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

        self.debug( f"Generating layers {neuronDimensions}" )

        self.debug( "Generating weight matrix", indent=1 )
        self.weights = []
        print(neuronDimensions)
        for index, neuronCount in enumerate(neuronDimensions):
            if( index > 0 ):
                self.debug( f"{index} : {neuronCount}", indent=2 )

                self.weights[index] = np.random.rand( neuronDimensions[index-1], neuronCount )

            self.debug( f"Weights {self.weights}", indent=2, end="\r" )

        self.debug( f"\nGenerated weights: {self.weights}", indent=1 )
