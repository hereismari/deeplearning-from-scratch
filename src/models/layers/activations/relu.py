import numpy as np
from models.layers.activations.activation import Activation


class Relu(Activation):
    @staticmethod
    def forward(data):
        return np.maximum(0, data)
    
    
    @staticmethod
    def backward(data):
        relu = Relu.forward(data)
        return (relu > 0) * 1.0