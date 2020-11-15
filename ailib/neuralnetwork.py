import numpy as np

class neural_network:
    def __init__( self, layers: list=None ):
        self.layers = layers
        print(self)

    def generateLayers( self, layerDimensions: list=[ (1,1), (1,1) ] ):
        print("New layers")
