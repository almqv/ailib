import numpy as np

import ailib.debug as db


class neural_network:
    def __init__( self, enableDebug: bool=True, layers: list=None ):
        self.layers = layers
        self.enableDebug = enableDebug

        self.debug( f"Created neural network {self}" )

    def debug( self, text:str, lvl: str=db.level.info ):
        if( self.enableDebug ):
            db.debug( text, lvl )

    def generateLayers( self, layerDimensions: list=[ (1,1), (1,1) ] ):
        print("New layers")
